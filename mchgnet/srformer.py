"""SRFormer native decoder coordinates and paper adapters."""
import contextlib, importlib, io, sys, types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .assets import asset
def official_class():
    # Both reviewed upstream libraries use an absolute `layers` namespace.
    # Isolate it in this single-baseline process and include both vendor roots.
    roots = [str(asset('srformer')/'FEDformer/layers'), str(asset('srformer')/'PatchTST/PatchTST_supervised/layers')]
    if 'layers' not in sys.modules:
        package = types.ModuleType('layers')
        package.__path__ = roots
        sys.modules['layers'] = package
    else:
        assert list(sys.modules['layers'].__path__) == roots, 'Conflicting baseline imports'
    if str(asset('srformer')) not in sys.path:
        sys.path.insert(0,str(asset('srformer')))
    module = importlib.import_module('baseline.SRFormer')
    assert str(asset('srformer')) in module.__file__
    return module.SRFormer


@contextlib.contextmanager
def preserve_bn_buffers(module):
    snapshots = [(m, {k:v.clone() for k,v in m.named_buffers(recurse=False)})
                 for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        yield
    finally:
        with torch.no_grad():
            for m, state in snapshots:
                for k,v in state.items():
                    getattr(m,k).copy_(v)


class Native(nn.Module):
    def __init__(self, data, device='cuda:0'):
        super().__init__()
        native = official_class()
        fed = dict(enable=True,version='Fourier',mode_select='full',modes=16,
                   moving_avg=[10,20,50],d_model=64,n_heads=8,embed=256,freq='h',
                   dropout=.1,e_layers=2,d_layers=1,d_ff=256,activation='gelu')
        patch = dict(enable=True,patch_len=16,stride=8,d_model=64,d_ff=256,
                     n_layers=2,n_heads=4,dropout=getattr(data, 'encoder_dropout', 0.))
        led, pd = (np.ones((data.c,3),np.float32),)*2
        if data.name == 'glucose':
            led, pd = data.spectra
        with contextlib.redirect_stdout(io.StringIO()):
            parent = native(len(data.id_map),32,64,data.c,3,data.length,
                torch.tensor(led,device=device),torch.tensor(pd,device=device),
                [64,96,128,160],[512,256,128,1],.1,'concat',fed,patch)
        self.abs_gen, self.fedformer, self.patchtst = parent.abs_gen, parent.fedformer, parent.patchtst
        self.norm_conv, self.norm_f, self.norm_enc = parent.norm_conv, parent.norm_f, parent.norm_enc
        self.register_buffer('led',torch.tensor(led))
        self.register_buffer('pd',torch.tensor(pd))
        self.hardware_spectra = data.name == 'glucose'
        self.channels, self.length, self.dim = data.c, data.length, 3
        feature_dim = data.length//2 + data.normal['statistics'][0].size + data.spec['late_text_dim']
        self.head = nn.Sequential(nn.Linear(feature_dim,512),nn.ReLU(),nn.Dropout(.25),
            nn.Linear(512,256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.25),
            nn.Linear(128,data.outputs))
        self.to(device)

    def expanded(self, wave, person):
        curves = self.abs_gen(person)
        if self.hardware_spectra:
            b = len(wave)
            conv = F.conv1d((curves*self.led).reshape(1,b*self.channels,3),
                self.pd.expand(b,-1,-1).reshape(b*self.channels,1,3),
                padding=1, groups=b*self.channels).reshape(b,self.channels,3)
        else:
            conv = curves
        lo, hi = conv.amin(-1,keepdim=True), conv.amax(-1,keepdim=True)
        fingerprint = self.norm_conv((conv-lo)/(hi-lo+1e-8))
        return (wave.unsqueeze(-1)*fingerprint.unsqueeze(2)).permute(0,1,3,2).reshape(len(wave),-1,self.length)

    def encode(self, wave, person):
        expanded = self.expanded(wave,person)
        # Preserve native dimensions but do not introduce unavailable TH/clock observations.
        x = torch.cat([expanded,torch.zeros_like(wave)],dim=1).transpose(1,2)
        mark = torch.zeros((len(wave),self.length,4), device=wave.device)
        f = self.norm_f(self.fedformer(x,mark,torch.zeros_like(x),mark))
        p = self.norm_enc(self.patchtst(f))
        # This is the exact upstream readout axis, including its trainable LayerNorm.
        return p.mean(dim=2)

    def encode_chunks(self, wave, person, chunk=32, activation_checkpoint=True):
        result = []
        for start in range(0,len(wave),chunk):
            args = wave[start:start+chunk],person[start:start+chunk]
            if self.training and torch.is_grad_enabled() and activation_checkpoint:
                result.append(checkpoint(self.encode,*args,use_reentrant=False,
                    context_fn=lambda:(contextlib.nullcontext(),preserve_bn_buffers(self))))
            else:
                result.append(self.encode(*args))
        return torch.cat(result)


def identity_contrast(features, persons):
    if len(features) < 2:
        return features.sum()*0.
    magnitude = torch.linalg.vector_norm(features,dim=1,keepdim=True)+1e-8
    distance = 1. - (features@features.T)/(magnitude@magnitude.T)
    different = (persons[:,None] != persons[None,:]).to(features.dtype)
    losses = (1.-different)*distance.square()+different*(1.-distance).clamp_min(0.).square()
    offdiag = ~torch.eye(len(features),dtype=torch.bool,device=features.device)
    return losses[offdiag].mean()
class Encoder(Native):
    def __init__(self,data,device):
        super().__init__(data,device)
        del self.head

    def decoder(self,wave,person):
        expanded=self.expanded(wave,person)
        x=torch.cat([expanded,torch.zeros_like(wave)],dim=1).transpose(1,2)
        mark=torch.zeros((len(wave),self.length,4),device=wave.device)
        f=self.norm_f(self.fedformer(x,mark,torch.zeros_like(x),mark))
        return self.norm_enc(self.patchtst(f))

    def regroup(self,p):
        n,length,width=p.shape
        assert width==4*self.channels and length==self.length//2
        optical=p[:,:,:3*self.channels].reshape(n,length,self.channels,3).permute(0,2,3,1)
        auxiliary=p[:,:,3*self.channels:].transpose(1,2).unsqueeze(2)
        return torch.cat([optical,auxiliary],dim=2).flatten(2)

    def encode(self,wave,person):
        p=self.decoder(wave,person)
        # The first part retains the exact native identity-loss summary. The second
        # contains each decoder coordinate exactly once, without copying a global vector.
        return torch.cat([p.mean(2),self.regroup(p).flatten(1)],dim=1)

def stable_complex_tanh(value):
    import torch
    native = value.tanh()
    finite = torch.isfinite(native)
    if bool(finite.all()):
        return native
    assert value.dtype == torch.complex64 and bool(torch.isfinite(value).all())
    bad = ~finite
    # Do not retain the invalid tanh in the returned autograd graph: masking an
    # invalid derivative afterwards would still allow 0*NaN in backward.
    result = torch.empty_like(value)
    result[finite] = value[finite].tanh()
    result[bad] = value[bad].to(torch.complex128).tanh().to(value.dtype)
    assert bool(torch.isfinite(result).all()), 'Complex128 tanh fallback is nonfinite'
    return result

def fourier_block(self, q, k, v, mask):
    batch, length, heads, width = q.shape
    x = q.permute(0, 2, 3, 1)
    frequencies = torch.fft.rfft(x, dim=-1)
    # The reviewed model uses consecutive low modes. Preserve upstream's output
    # mode placement, which uses mode ordinal rather than the selected input index.
    assert self.index == list(range(len(self.index)))
    selected = frequencies[..., :len(self.index)]
    projected = torch.einsum('bhim,hiom->bhom', selected, self.weights1)
    output = torch.zeros(batch, heads, width, length//2+1, device=x.device, dtype=torch.cfloat)
    output[..., :len(self.index)] = projected
    return torch.fft.irfft(output, n=length), None


def fourier_cross(self, q, k, v, mask):
    batch, length, heads, width = q.shape
    assert self.index_q == list(range(len(self.index_q)))
    assert self.index_kv == list(range(len(self.index_kv)))
    query = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)[..., :len(self.index_q)].contiguous()
    key = torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1)[..., :len(self.index_kv)].contiguous()
    attention = torch.einsum('bhex,bhey->bhxy', query, key)
    if self.activation == 'tanh':
        attention = stable_complex_tanh(attention)
    elif self.activation == 'softmax':
        attention = torch.softmax(abs(attention), dim=-1)
        attention = torch.complex(attention, torch.zeros_like(attention))
    else:
        raise ValueError(self.activation)
    context = torch.einsum('bhxy,bhey->bhex', attention, key)
    projected = torch.einsum('bhex,heox->bhox', context, self.weights1)
    output = torch.zeros(batch, heads, width, length//2+1, device=q.device, dtype=torch.cfloat)
    output[..., :len(self.index_q)] = projected
    return torch.fft.irfft(output/self.in_channels/self.out_channels, n=length), None

def install_acceleration():
    module = importlib.import_module('layers.FourierCorrelation')
    module.FourierBlock.forward = fourier_block
    module.FourierCrossAttention.forward = fourier_cross
