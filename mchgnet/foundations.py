"""Pinned official pretrained models. Checkpoints are downloaded separately."""
import collections, pickle, types, argparse, _codecs, sys, importlib.util, hashlib
from functools import lru_cache, partial
import numpy as np
import torch
from torch import nn
from .assets import asset, sha256 as sha
@lru_cache(maxsize=1)
def official_class():
    spec = importlib.util.spec_from_file_location('papagei_transfer_official', asset('papagei')/'models/resnet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResNet1DMoE


def official_model(initialization, seed):
    if initialization not in ['random','pretrained']:
        raise ValueError(initialization)
    # Construction never changes downstream head/dropout RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = official_class()(in_channels=1, base_filters=32, kernel_size=3,
            stride=2, groups=1, n_block=18, n_classes=512, n_experts=3)
    if initialization == 'pretrained':
        if hashlib.md5((asset('papagei')/'papagei_s.pt').read_bytes()).hexdigest() != 'a4cdb32392e2a7b25999128af92813b5':
            raise ValueError('Official weight checksum mismatch')
        state = torch.load((asset('papagei')/'papagei_s.pt'), map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.removeprefix('module.'):v for k,v in state.items()}
        model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)

def load_author_file(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


class RestrictedCheckpoint(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={'_codecs.encode':_codecs.encode,'argparse.Namespace':argparse.Namespace,'collections.OrderedDict':collections.OrderedDict,'numpy.dtype':np.dtype,'numpy.core.multiarray.scalar':np.core.multiarray.scalar,'torch.FloatStorage':torch.FloatStorage,'torch._utils._rebuild_tensor_v2':torch._utils._rebuild_tensor_v2}
  key=module+'.'+name
  if key not in allowed:raise pickle.UnpicklingError('Checkpoint global not allowed: '+key)
  return allowed[key]


def sigma_checkpoint():
 return torch.load(asset('sigma_weights')/'sigma.pth',map_location='cpu',pickle_module=types.SimpleNamespace(Unpickler=RestrictedCheckpoint,__name__='restricted'))['model']


class PapaFeatures(nn.Module):
 def __init__(self,initialization,seed):
  super().__init__();full=official_model(initialization,seed)
  self.first_block_conv=full.first_block_conv;self.first_block_bn=full.first_block_bn;self.first_block_relu=full.first_block_relu
  self.basicblock_list=full.basicblock_list;self.final_bn=full.final_bn;self.final_relu=full.final_relu
 def forward(self,x):
  x=self.first_block_relu(self.first_block_bn(self.first_block_conv(x)))
  for b in self.basicblock_list:x=b(x)
  return self.final_relu(self.final_bn(x)).mean(-1)


@lru_cache(None)
def author_class(name):
 if name=='pulseppg':return load_author_file('_ppgt_pulse',asset('pulseppg')/'pulseppg/nets/ResNet1D/ResNet1D_Net.py').Net
 if name=='anyppg':return load_author_file('_ppgt_any',asset('anyppg')/'load_anyppg/resnet1d.py').Net1D
 if name=='sigmappg':return load_author_file('_ppgt_sigma',asset('sigmappg')/'codebook/modeling_finetune.py').NeuralTransformer
 raise ValueError(name)


def foundation(name,seed):
 audit=dict(encoder=name)
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(seed)
  if name.startswith('papagei_'):
   net=PapaFeatures(name.split('_')[1],seed);dim=512
   audit.update(load='strict full official PaPaGei, original pretraining task heads removed',sampling_hz=125,seconds=6)
  elif name=='anyppg':
   net=author_class(name)(in_channels=1,base_filters=64,ratio=1,filter_list=[64,160,160,400,400,512],m_blocks_list=[2,2,2,3,3,1],kernel_size=3,stride=2,groups_width=16,use_bn=True,use_do=True,verbose=False)
   path=asset('anyppg')/'load_anyppg/anyppg_ckpt.pth';weights=torch.load(path,map_location='cpu',weights_only=True)
   weights=weights.get('state_dict',weights);net.load_state_dict(weights,strict=True);dim=512
   audit.update(weight_sha256=sha(path),load='strict=True',sampling_hz=125,seconds=6)
  elif name=='pulseppg':
   net=author_class(name)(in_channels=1,base_filters=128,kernel_size=11,stride=2,groups=1,n_block=12,finalpool='max',downsample_gap=2,increasefilter_gap=4,use_bn=True,use_do=True,verbose=False)
   path=asset('pulse_weights')/'pulseppg/experiments/out/pulseppg/checkpoint_best.pkl'
   weights=torch.load(path,map_location='cpu',weights_only=True)['net']
   net.load_state_dict(weights,strict=True);dim=512
   audit.update(weight_sha256=sha(path),load='strict=True',sampling_hz=50,seconds=6)
  elif name=='sigmappg':
   weights=sigma_checkpoint();weights={k.removeprefix('student.'):v for k,v in weights.items() if k.startswith('student.')}
   net=author_class(name)(PPG_size=12000,patch_size=100,in_chans=1,out_chans=12,num_classes=0,embed_dim=360,depth=18,num_heads=12,mlp_ratio=4.,qkv_bias=False,qk_norm=partial(nn.LayerNorm,eps=1e-6),drop_rate=.08,attn_drop_rate=.08,drop_path_rate=.15,norm_layer=partial(nn.LayerNorm,eps=1e-6),init_values=.1,use_mean_pooling=True)
   # Preserve all pretrained positional/time tensors; six seconds = three 2s patches at 50Hz.
   net.pos_embed=nn.Parameter(torch.empty_like(weights['pos_embed']))
   net.fc_norm=nn.Identity() # pretrained norm already applied to tokens; no extra untrained normalization.
   mapped={};mapping={}
   for k,v in net.state_dict().items():
    source=k.replace('patch_embed.','patch_embed_conv.').replace('patch_proj.','patch_embed_projection.')
    if source not in weights or weights[source].shape!=v.shape:raise ValueError('SIGMA missing/mismatched backbone weight: '+k)
    mapped[k]=weights[source];mapping[k]=source
   ignored=[]
   for k,v in weights.items():
    if k in mapping.values():continue
    if k.startswith('patch_embed_seq.'):
     alias=k.replace('patch_embed_seq.0.','patch_embed_conv.').replace('patch_embed_seq.1.','patch_embed_projection.')
     if alias not in weights or not torch.equal(v,weights[alias]):raise ValueError('SIGMA alias differs: '+k)
    elif not (k=='mask_token' or k.startswith('lm_head.')):raise ValueError('Unexpected unused SIGMA tensor: '+k)
    ignored.append(k)
   net.load_state_dict(mapped,strict=True);dim=360
   audit.update(weight_sha256=sha(asset('sigma_weights')/'sigma.pth'),load='strict=True after explicit backbone mapping',mapping=mapping,excluded=ignored,sampling_hz=50,seconds=6,patch_samples=100,patch_seconds=2,patches=3,position_policy='all native weights retained; channel index1 and first3 temporal positions',normalization='window minmax [-1,1] after common bandpass; native preprocessing rule')
  else:raise ValueError(name)
 net.requires_grad_(True)
 return net,dim,audit
