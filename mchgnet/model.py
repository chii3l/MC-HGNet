"""Shared current/history encoder, module A, module B and prediction heads."""
from types import MethodType, SimpleNamespace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .relations import (PairRefinement, HistoryAdapter, PairConditionFusion,
    ChannelOnlyFusion, PairOnlyFusion, IdentityCurrentFusion, band_only_history)


class MCHGNet(nn.Module):
    def __init__(self, spec, encoder='moderntcn', variant='full', seed=2026,
                 people=1, encoder_dropout=0., spectra=None):
        super().__init__()
        if variant not in ('baseline','full','no_history','no_pairs','no_graph'):
            raise ValueError(variant)
        self.spec, self.name, self.variant = spec, encoder, variant
        self.c, self.s = len(spec['channels']), len(spec['slots'])
        self.outputs = spec.get('outputs',1)
        self.length = spec.get('input_samples',300)
        self.is_pretrained = encoder in ('pulseppg','papagei_pretrained','anyppg','sigmappg')
        self.active = True
        self.parts = ()
        self.identity = None
        self.activation_checkpointing = True
        if encoder=='srformer':
            from .srformer import Encoder, install_acceleration
            data=SimpleNamespace(name=spec['dataset'],c=self.c,length=self.length,
                id_map={str(i):i for i in range(people)},outputs=self.outputs,spec=spec,
                normal={'statistics':[SimpleNamespace(size=self.c*2)]},
                encoder_dropout=encoder_dropout,spectra=spectra)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.encoder=Encoder(data,'cpu')
            install_acceleration()
            dimension=4*(self.length//2)
        elif self.is_pretrained:
            from .foundations import foundation
            self.encoder,dimension,self.load_audit=foundation(encoder,seed)
        else:
            from .encoders import Encoder
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+2161)
                self.encoder=Encoder(encoder,self.c,self.length)
            dimension=None
        if dimension is not None:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+1499)
                self.projection=nn.Sequential(nn.LayerNorm(dimension,elementwise_affine=False),nn.Linear(dimension,64),nn.GELU())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+1412)
            self.token=nn.Sequential(nn.Linear(66,64),nn.GELU())
            self.readout=nn.Sequential(nn.Linear(self.c*64,64),nn.GELU(),nn.Dropout(.1))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+1009)
            self.context=nn.Sequential(nn.Linear(self.s*33,32),nn.GELU(),nn.Linear(32,64,bias=False))
        def head(width,offset):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+offset)
                return nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,self.outputs))
        if encoder=='srformer':
            self.head,self.wave_head,self.text_head=head(128,1413),head(64,1414),head(64,1415)
        else:
            self.wave_head,self.text_head,self.head=head(64,1414),head(64,1415),head(128,1413)
        if variant!='baseline':
            self.fusion=PairRefinement(spec,seed,'pair_gated_hg')
            history=HistoryAdapter(spec,seed,'history_attention_graph',self.fusion.pair)
            # Preserve the original history-message initialization before replacing
            # only its input layer for the shared 64-coordinate encoder output.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+2026102601)
                history.wave_norm=nn.LayerNorm(64,elementwise_affine=False)
                history.band[0]=nn.Linear(66,32)
            self.fusion.history=history
            oxygen=spec['dataset'].startswith('openox_')
            if oxygen:
                self.fusion=PairConditionFusion(self.fusion,self.s,seed,.1)
            if variant=='no_history':
                del self.fusion.history
            elif variant=='no_pairs':
                for key in ('pairs','metadata','incidence','pair','pair_delta','pair_query',
                            'pair_key','pair_time','pair_condition','pair_null'):
                    delattr(history,key)
                history.forward=MethodType(band_only_history,history)
                self.fusion=IdentityCurrentFusion(history) if oxygen else ChannelOnlyFusion(self.fusion)
                self.fusion.history=history
            elif variant=='no_graph':
                self.fusion=PairOnlyFusion(self.fusion)

    def set_active(self, active):
        self.active=bool(active)
        self.encoder.requires_grad_(self.active)

    def train(self, mode=True):
        super().train(mode)
        if self.is_pretrained:
            if not self.active:
                self.encoder.eval()
            for module in self.encoder.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    def encode(self,wave,person=None):
        """Input uses this encoder's native sample rate, output retains band axis."""
        n=len(wave)
        if self.name=='srformer':
            raw=self.encoder.encode_chunks(wave,person,chunk=32,activation_checkpoint=False)
            width=self.length//2
            native=raw[:,:width]
            tokens=self.projection(raw[:,width:].reshape(n,self.c,4*width))
            return tokens,native
        if not self.is_pretrained:
            return self.encoder(wave),None
        if self.name=='sigmappg':
            lo,hi=wave.amin(-1,keepdim=True),wave.amax(-1,keepdim=True)
            x=(2*(wave-lo)/(hi-lo).clamp_min(1e-6)-1).reshape(-1,1,wave.shape[-1]//100,100)
        else:
            x=wave.reshape(-1,1,wave.shape[-1])
        if self.activation_checkpointing and self.training and self.active:
            features=checkpoint(self.encoder,x,use_reentrant=False,preserve_rng_state=True)
        else:
            features=self.encoder(x)
        return self.projection(features.reshape(n,self.c,-1)),None

    def finish(self,tokens,statistics,fields,pending=None):
        late=fields.flatten(1)
        context=self.context(late)
        nodes=self.token(torch.cat([tokens,statistics],-1))
        if self.variant not in ('baseline','no_history'):
            delta,_=self.fusion.history(pending)
            nodes=nodes+delta
        correction=torch.zeros_like(context)
        if self.variant!='baseline':
            nodes,correction=self.fusion(nodes,statistics,fields,fields[...,32].bool())
        band=self.readout(nodes.flatten(1))
        self.parts=(self.wave_head(band),self.text_head(context))
        return self.head(torch.cat([band,context+correction],-1))

    def auxiliary(self):
        if self.name=='timefilter' and self.encoder.auxiliary_loss is not None:
            return .01*self.encoder.auxiliary_loss.mean()
        return next(self.parameters()).new_zeros(())
