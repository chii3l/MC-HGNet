"""Deterministic adaptation of the MIT-licensed Time-Series-Library gate."""
import torch
from torch.nn import functional as F

class DeterministicGateMixin:
    # TSLib MIT-licensed gate arithmetic, with ONLY the three-expert cumulative
    # sum moved to CPU. torch 2.1's CUDA cumsum lacks a deterministic kernel.
    # Keep deterministic algorithms enabled for the entire experiment.
    def noisy_top_k_gating(self,x,is_training,noise_epsilon=1e-2):
        clean_logits=self.gate(x)
        if self.noisy_gating and is_training:
            noise_stddev=self.softplus(self.noise(x))+noise_epsilon
            logits=clean_logits+torch.randn_like(clean_logits)*noise_stddev
        else: logits=clean_logits
        logits=self.softmax(logits)
        loss_dynamic=self.cross_entropy(logits)
        sorted_probs,sorted_indices=torch.sort(logits,descending=True)
        cumulative_probs=torch.cumsum(sorted_probs.to('cpu'),dim=-1).to(x.device)
        mask=cumulative_probs>self.top_p
        threshold_indices=mask.long().argmax(dim=-1)
        threshold_mask=F.one_hot(threshold_indices,num_classes=sorted_indices.size(-1)).bool()
        mask=mask & ~threshold_mask
        top_p_mask=torch.zeros_like(mask)
        zero_indices=(mask==0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0],zero_indices[1],sorted_indices[zero_indices[0],zero_indices[1],zero_indices[2]]]=1
        sorted_probs=torch.where(mask,0.0,sorted_probs)
        loss_importance=self.cv_squared(sorted_probs.sum(0))
        return top_p_mask,loss_importance+.1*loss_dynamic
