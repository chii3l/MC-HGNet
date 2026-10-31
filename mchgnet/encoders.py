"""Channel-preserving adapters for the two non-pretrained time-series backbones."""
import importlib
import importlib.util
import sys
from types import SimpleNamespace
import torch
from torch import nn
from .assets import asset
from .timefilter_gate import DeterministicGateMixin


def load_file(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def vendor(name):
    # One encoder per process: upstream projects use the same absolute 'layers' name.
    root = asset('tslib')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.import_module('layers')
    importlib.import_module('models')
    if name == 'timefilter':
        return importlib.import_module('models.TimeFilter')
    modern = asset('moderntcn')/'ModernTCN-classification'
    load_file('layers.RevIN', modern/'layers/RevIN.py')
    load_file('models.ModernTCN_Layer', modern/'models/ModernTCN_Layer.py')
    return load_file('_mchgnet_modern', modern/'models/ModernTCN.py')


class Encoder(nn.Module):
    def __init__(self, name, channels, length=300):
        super().__init__()
        self.name, self.c, self.length = name, channels, length
        self.auxiliary_loss = None
        if name == 'moderntcn':
            self.stem = nn.Sequential(nn.Conv1d(1,32,8,stride=4),nn.BatchNorm1d(32))
            self.stage = vendor(name).Stage(2,2,31,5,32,32,channels,drop=.1)
            self.projection = nn.Linear(32,64)
        elif name == 'timefilter':
            patch = 20 if length==320 else 25
            self.patches = length//patch
            cfg = SimpleNamespace(task_name='features',seq_len=length,pred_len=64,
                enc_in=channels,c_out=channels,d_model=64,d_ff=128,n_heads=4,e_layers=2,
                dropout=.1,embed='timeF',freq='s',factor=1,activation='gelu',moving_avg=25,
                patch_len=patch,alpha=.1,top_p=.5,pos=True)
            net = vendor(name).Model(cfg)
            self.patch_embedding, self.encoder = net.patch_embed, net.backbone
            for block in self.encoder.blocks:
                gate = block.gnn.graph_learner.mask_moe
                cls = type('DeterministicGate',(DeterministicGateMixin,type(gate)),{})
                with torch.random.fork_rng(devices=[]):
                    replacement = cls(channels,top_p=.5,in_dim=channels*self.patches)
                replacement.load_state_dict(gate.state_dict(),strict=True)
                block.gnn.graph_learner.mask_moe = replacement
            self.register_buffer('masks',net._get_mask('cpu'))
        else:
            raise ValueError(name)

    def forward(self,x):
        n,c,length = x.shape
        if self.name == 'moderntcn':
            z=x.reshape(-1,1,length)
            if length==300:
                z=torch.cat([z,z[:,:,-1:].expand(-1,-1,4)],-1)
            z=self.stem(z)
            return self.projection(self.stage(z.reshape(n,c,32,z.shape[-1])).mean(-1))
        z=self.patch_embedding(x.reshape(n,-1))
        z,self.auxiliary_loss=self.encoder(z,self.masks,.1)
        return z.reshape(n,c,self.patches,64).mean(2)
