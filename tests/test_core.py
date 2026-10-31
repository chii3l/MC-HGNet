import unittest
from types import MethodType
import numpy as np
import torch
from mchgnet.relations import PairRefinement,HistoryAdapter,band_only_history,PairOnlyFusion,ChannelOnlyFusion
from mchgnet.metrics import zone_a,regression
from mchgnet.preprocessing import signal_views,histories,fit_fold


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.spec=dict(channels=['g','r','ir'],wavelengths=[550,660,880],slots=['age','sex'])

    def test_incident_pair_mean(self):
        f=PairRefinement(self.spec,2026,'pair_gated_hg')
        p=f.pair
        torch.testing.assert_close(p.incidence.sum(1),torch.ones(3))
        values=torch.tensor([[[2.],[4.],[8.]]])
        output=torch.einsum('cp,bpd->bcd',p.incidence,values)
        torch.testing.assert_close(output[0,:,0],torch.tensor([3.,5.,6.]))

    def test_absent_history_is_exactly_zero(self):
        f=PairRefinement(self.spec,2026,'pair_gated_hg')
        h=HistoryAdapter(self.spec,2026,'history_attention_graph',f.pair)
        with torch.no_grad():
            h.band_delta[-1].bias.fill_(7);h.pair_delta[-1].bias.fill_(9)
        data=(torch.randn(2,2,3,512),torch.randn(2,2,3,2),torch.randn(2,2,66),
              torch.zeros(2,1,dtype=torch.bool),torch.ones(2,1))
        residual,present=h(data)
        self.assertEqual(torch.count_nonzero(residual),0)
        self.assertEqual(torch.count_nonzero(present),0)

    def test_history_value_gradient_and_mask(self):
        f=PairRefinement(self.spec,2026,'pair_gated_hg')
        h=HistoryAdapter(self.spec,2026,'history_attention_graph',f.pair)
        with torch.no_grad():
            h.band_delta[-1].weight.normal_(0,.1);h.pair_delta[-1].weight.normal_(0,.1)
        wave=torch.randn(2,2,3,512,requires_grad=True)
        fields=torch.randn(2,2,2,33);fields[...,32]=1
        result,_=h((wave,torch.randn(2,2,3,2),fields.flatten(2),
                    torch.tensor([[True],[False]]),torch.ones(2,1)*.02))
        result.square().sum().backward()
        self.assertGreater(float(wave.grad[0,1].abs().sum()),0)
        self.assertEqual(float(wave.grad[1,1].abs().sum()),0)

    def test_native_rate_duration(self):
        t=np.arange(320)/40
        raw=np.tile((.5+.4*np.sin(2*np.pi*2*t))[None,None],(2,3,1))
        canonical,views,stats=signal_views(raw,True)
        self.assertEqual(canonical.shape,(2,3,320))
        for rate in (50,125):
            x=views[rate][0,0]
            freq=np.fft.rfftfreq(len(x),1/rate)
            self.assertEqual(len(x),8*rate)
            self.assertEqual(freq[1+abs(np.fft.rfft(x))[1:].argmax()],2)

    def test_history_causality_and_split(self):
        import pandas as pd
        f=pd.DataFrame(dict(subject=['A']*5,day=['d']*5,start=[0,10,20,30,40],end=[6,16,26,36,46]))
        h=histories(f,[dict(train=[0,2,3],validation=[1],test=[4])])[0]
        self.assertFalse(h['mask'][0].any());self.assertFalse(h['mask'][1].any())
        self.assertFalse(h['mask'][4].any())
        self.assertEqual(h['index'][3,h['mask'][3]].tolist(),[2])

    def test_sample_equal_metrics_and_units(self):
        m=regression(np.array([5.,10.]),np.array([6.,10.]),'glucose')['glucose']
        self.assertAlmostEqual(m['rmse'],18/np.sqrt(2))
        self.assertEqual(zone_a([50,100,200],[50,100,200]),100.)
        with self.assertRaises(ValueError):zone_a([100],[-1])

    def test_fitting_ignores_held_out_observations(self):
        rng=np.random.default_rng(12)
        embeddings=rng.normal(size=(12,2,40)).astype(np.float32)
        statistics=rng.normal(size=(12,3,2)).astype(np.float32)
        available=np.ones((12,2),bool)
        train=np.arange(8,dtype=np.int64)
        fields,stats,fit=fit_fold(embeddings,available,statistics,train)
        changed=embeddings.copy();changed[8:]=rng.normal(500,100,size=changed[8:].shape)
        changed_stats=statistics.copy();changed_stats[8:]=-10000
        new_fields,new_stats,new_fit=fit_fold(changed,available,changed_stats,train)
        self.assertEqual(fit,new_fit)
        np.testing.assert_array_equal(fields[train],new_fields[train])
        np.testing.assert_array_equal(stats[train],new_stats[train])
        self.assertFalse(np.array_equal(fields[8:],new_fields[8:]))


if __name__=='__main__':unittest.main()
