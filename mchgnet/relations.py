"""Current paper relational operators, with original initialization streams."""
import math
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

class FieldFusion(nn.Module):

    def __init__(self, spec, seed, heterogeneous):
        super().__init__()
        self.heterogeneous = heterogeneous
        self.c, self.s, self.d, self.h = (len(spec['channels']), len(spec['slots']), 24, 4)
        d = self.d
        self.disabled = frozenset()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 20261010)
            self.signal_norm = nn.LayerNorm(64)
            self.signal_in = nn.Linear(64, d)
            self.field_in = nn.Linear(33, d)
            self.signal_id = nn.Parameter(torch.randn(self.c, d) * 0.02)
            self.field_id = nn.Parameter(torch.randn(self.s, d) * 0.02)
            self.metadata = nn.Linear(2, d)
            self.low_norm = nn.LayerNorm(d)
            self.signal_qkv = nn.Linear(d, d * 3)
            self.field_qkv = nn.Linear(d, d * 3)
            self.out = nn.Linear(d, 64)
            self.text_out = nn.Linear(d, 64)
            for layer in (self.out, self.text_out):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            self.gate_logit = nn.Parameter(torch.zeros(()))
        known = torch.tensor([w is not None for w in spec['wavelengths']], dtype=torch.float32)
        wl = torch.tensor([0.0 if w is None else (w - 1100.0) / 500.0 for w in spec['wavelengths']])
        self.register_buffer('channel_metadata', torch.stack([wl, known], -1))
        pair = known[:, None] * known[None, :]
        self.register_buffer('pair_prior', torch.stack([(wl[:, None] - wl[None, :]).abs() * 0.5 * pair, pair, torch.eye(self.c)], -1))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 20261011)
            self.relation_score = nn.Parameter(torch.eye(d // self.h).repeat(3, self.h, 1, 1))
            self.relation_value = nn.Parameter(torch.eye(d).repeat(3, 1, 1))
            self.edge_in = nn.Linear(11, 16)
            self.edge_out = nn.Linear(16, self.h, bias=False)

    def pair_fields(self, statistics):
        b, c, _ = statistics.shape
        dest = statistics[:, :, None, :].expand(-1, -1, c, -1)
        source = statistics[:, None, :, :].expand(-1, c, -1, -1)
        delta = source - dest
        return torch.cat([dest, source, delta, delta.abs(), self.pair_prior[None].expand(b, -1, -1, -1)], -1)

    def split_qkv(self, x, layer):
        b, n, _ = x.shape
        return layer(x).reshape(b, n, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)

    def message(self, query, key, value, valid_source, relation, bias=None):
        query = query @ self.relation_score[relation][None]
        b, h, n, p = value.shape
        flat = value.transpose(1, 2).reshape(b, n, self.d)
        flat = flat @ self.relation_value[relation]
        value = flat.reshape(b, n, h, p).transpose(1, 2)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.d // self.h)
        if bias is not None:
            scores = scores + bias
        valid = valid_source[:, None, None, :]
        weights = scores.masked_fill(~valid, -10000.0).softmax(-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return (weights @ value).transpose(1, 2).flatten(2)

    def forward(self, nodes, statistics, fields, available):
        if 'all' in self.disabled:
            return (nodes, nodes.new_zeros(len(nodes), 64))
        mask = available.bool()
        fields = torch.cat([fields[..., :32] * mask[..., None], mask[..., None].to(fields.dtype)], -1)
        signal = self.signal_in(self.signal_norm(nodes)) + self.signal_id[None]
        signal = signal + self.metadata(self.channel_metadata)[None]
        text = self.field_in(fields) + self.field_id[None]
        signal, text = (F.gelu(signal), F.gelu(text))
        signal, text = (self.low_norm(signal), self.low_norm(text))
        text = text * mask[..., None]
        sq, sk, sv = self.split_qkv(signal, self.signal_qkv)
        tq, tk, tv = self.split_qkv(text, self.field_qkv)
        channels = torch.ones(len(nodes), self.c, dtype=torch.bool, device=nodes.device)
        tc = self.message(sq, tk, tv, mask, 1)
        ct = self.message(tq, sk, sv, channels, 2) * mask[..., None]
        if 'tc' in self.disabled:
            tc = torch.zeros_like(tc)
        if 'ct' in self.disabled:
            ct = torch.zeros_like(ct)
        has_text = mask.any(-1).to(nodes.dtype)
        bias = self.edge_out(F.gelu(self.edge_in(self.pair_fields(statistics)))).permute(0, 3, 1, 2)
        cc = self.message(sq, sk, sv, channels, 0, bias)
        if 'cc' in self.disabled:
            cc = torch.zeros_like(cc)
        signal_message = (cc + tc) / (1 + has_text[:, None, None])
        delta = self.out(signal_message)
        pooled = ct.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        context_delta = self.text_out(pooled) * has_text[:, None]
        if 'ct' in self.disabled:
            context_delta = torch.zeros_like(context_delta)
        gate = torch.sigmoid(self.gate_logit)
        return (nodes + gate * delta, gate * context_delta)

class PairFusion(nn.Module):

    def __init__(self, spec, seed, arm):
        super().__init__()
        assert arm == 'pair_mlp'
        self.arm = arm
        self.c, self.s, self.d = (len(spec['channels']), len(spec['slots']), 24)
        pairs = torch.combinations(torch.arange(self.c), r=2)
        assert len(pairs) > 0
        self.register_buffer('pairs', pairs)
        incidence = (torch.arange(self.c)[:, None, None] == pairs[None]).any(-1).float()
        self.register_buffer('incidence', incidence / incidence.sum(1, keepdim=True))
        adjacency = (pairs[:, None, :, None] == pairs[None, :, None, :]).any(-1).any(-1)
        self.register_buffer('adjacency', adjacency)
        known = torch.tensor([w is not None for w in spec['wavelengths']], dtype=torch.float32)
        wl = torch.tensor([0.0 if w is None else (w - 1100.0) / 500.0 for w in spec['wavelengths']])
        meta = torch.stack([wl, known], -1)
        i, j = pairs.unbind(-1)
        pair_known = known[i] * known[j]
        self.register_buffer('pair_metadata', torch.cat([meta[i], meta[j], ((wl[i] - wl[j]) * pair_known)[:, None], pair_known[:, None]], -1))
        reference = FieldFusion(spec, seed, True)
        target = sum((p.numel() for p in reference.parameters()))
        hidden = max(1, round((target - 4177 - 24 * self.c) / 229))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 20261019)
            self.signal_norm = nn.LayerNorm(64)
            self.pair_encoder = nn.Sequential(nn.Linear(204, hidden), nn.GELU(), nn.Linear(hidden, self.d))
            self.channel_id = nn.Parameter(torch.randn(self.c, self.d) * 0.02)
            self.pair_norm = nn.LayerNorm(self.d)
            self.pair_ffn = nn.Sequential(nn.Linear(self.d, 48), nn.GELU(), nn.Linear(48, self.d))
            self.out = nn.Linear(self.d, 64)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.gate_logit = nn.Parameter(torch.zeros(()))
        actual = sum((p.numel() for p in self.parameters()))
        self.capacity = dict(reference_parameters=target, actual_parameters=actual, relative_difference=abs(actual / target - 1), pair_hidden=hidden, pair_width=self.d, pair_count=len(pairs), propagation='identity', adjacency_only_difference_for_mlp_vs_graph=True, conditional_selector=False)
        assert self.capacity['relative_difference'] <= 0.1, self.capacity

    def encode_pairs(self, nodes, statistics):
        x = self.signal_norm(nodes)
        i, j = self.pairs.unbind(-1)
        features = torch.cat([x[:, i], x[:, j], x[:, i] - x[:, j], statistics[:, i], statistics[:, j], statistics[:, i] - statistics[:, j], self.pair_metadata[None].expand(len(x), -1, -1)], -1)
        return self.pair_norm(self.pair_encoder(features) + self.channel_id[i] + self.channel_id[j])

    def condition_gate(self, pairs, fields, available):
        return pairs.new_ones(len(pairs), pairs.shape[1], 1)

    def propagate(self, pairs, gate):
        value = pairs * gate
        return value

    def forward(self, nodes, statistics, fields, available):
        pairs = self.encode_pairs(nodes, statistics)
        gate = self.condition_gate(pairs, fields, available)
        mixed = self.propagate(pairs, gate)
        message = mixed + 0.1 * self.pair_ffn(mixed)
        channels = torch.einsum('cp,bpd->bcd', self.incidence, message)
        return (nodes + self.gate_logit.sigmoid() * self.out(channels), nodes.new_zeros(len(nodes), 64))

class PairRefinement(nn.Module):

    def __init__(self, spec, seed, arm):
        super().__init__()
        assert arm == 'pair_gated_hg'
        self.arm = arm
        self.select_pairs = False
        self.use_graph = True
        self.disabled = frozenset()
        self.pair = PairFusion(spec, seed, 'pair_mlp')
        self.original = FieldFusion(spec, seed, True)
        self.dropout = nn.Dropout(0.0)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 20261024)
            self.context_in = nn.Linear(33, 8)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 20261026)
            self.graph_gate = nn.Sequential(nn.Linear(96, 16), nn.GELU(), nn.Linear(16, 2))
            nn.init.zeros_(self.graph_gate[-1].weight)
            nn.init.constant_(self.graph_gate[-1].bias, math.log(0.1 / 0.9))
        old = sum((p.numel() for p in self.pair.parameters()))
        actual = sum((p.numel() for p in self.parameters()))
        graph = sum((p.numel() for p in self.original.parameters())) if self.use_graph else 0
        self.capacity = dict(reference_parameters=old, actual_parameters=actual, pair_parameters=old, graph_parameters=graph, selector_and_gate_parameters=actual - old - graph, pair_count=len(self.pair.pairs), pair_encoding='explicit unordered pairs', incident_pair_selection=self.select_pairs, null_message=True if self.select_pairs else False, original_text_graph=self.use_graph, graph_initial_gate=0.1 if self.use_graph else None, residual_dropout=self.dropout.p, pair_to_pair_propagation=False, pair_to_field_feedback=False)

    def condition(self, fields, available):
        valid = available.bool()
        masked = torch.cat([fields[..., :32] * valid[..., None], valid[..., None].to(fields.dtype)], -1)
        pooled = masked.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return F.gelu(self.context_in(pooled)) * valid.any(-1)[:, None]

    def routing(self, nodes, messages, context):
        batch, channels = nodes.shape[:2]
        incident = self.pair.incidence > 0
        degree = incident.sum(-1).to(nodes.dtype)
        channel = self.pair.signal_norm(nodes)
        ctx = context[:, None].expand(-1, channels, -1)
        average = torch.einsum('cp,bpd->bcd', self.pair.incidence, messages)
        weights = self.pair.incidence[None].expand(batch, -1, -1)
        null = nodes.new_zeros(batch, channels)
        pooled = torch.einsum('bcp,bpd->bcd', weights, messages) if self.select_pairs and 'selection' not in self.disabled else average
        gates = nodes.new_zeros(batch, channels, 2)
        if self.use_graph and 'graph' not in self.disabled:
            gates = self.graph_gate(torch.cat([channel, average, ctx], -1)).sigmoid()
        return (pooled, weights, null, gates)

    def forward(self, nodes, statistics, fields, available):
        pairs = self.pair.encode_pairs(nodes, statistics)
        messages = pairs + 0.1 * self.pair.pair_ffn(pairs)
        context = self.condition(fields, available)
        pooled, _, _, gates = self.routing(nodes, self.dropout(messages), context)
        delta = self.pair.gate_logit.sigmoid() * self.pair.out(pooled)
        if 'pairs' in self.disabled:
            delta = torch.zeros_like(delta)
        output, text_delta = (nodes + delta, nodes.new_zeros(len(nodes), 64))
        if self.use_graph and 'graph' not in self.disabled:
            graph_nodes, graph_context = self.original(nodes, statistics, fields, available)
            output = output + gates[..., :1] * self.dropout(graph_nodes - nodes)
            text_delta = gates[..., 1].mean(1)[:, None] * self.dropout(graph_context)
        return (output, text_delta)

class HistoryAdapter(nn.Module):

    def __init__(self, spec, seed, arm, pair):
        super().__init__()
        self.arm = arm
        self.c = len(spec['channels'])
        self.s = len(spec['slots'])
        self.register_buffer('pairs', pair.pairs.clone())
        self.register_buffer('metadata', pair.pair_metadata.clone())
        self.register_buffer('incidence', pair.incidence.clone())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2026101901)
            self.wave_norm = nn.LayerNorm(512, elementwise_affine=False)
            self.band = nn.Sequential(nn.Linear(514, 32), nn.GELU())
            self.pair = nn.Sequential(nn.Linear(108, 32), nn.GELU(), nn.Linear(32, 16), nn.LayerNorm(16))
            self.condition = nn.Sequential(nn.Linear(self.s * 33, 8), nn.GELU())
            self.band_delta = nn.Sequential(nn.Linear(106, 32), nn.GELU(), nn.Linear(32, 64))
            self.pair_delta = nn.Sequential(nn.Linear(58, 32), nn.GELU(), nn.Linear(32, 64))
            for layer in (self.band_delta[-1], self.pair_delta[-1]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            self.band_query = nn.Linear(32, 16, bias=False)
            self.band_key = nn.Linear(32, 16, bias=False)
            self.pair_query = nn.Linear(16, 16, bias=False)
            self.pair_key = nn.Linear(16, 16, bias=False)
            for kind, n in [('band', self.c), ('pair', len(self.pairs))]:
                self.register_parameter(kind + '_time', nn.Parameter(torch.full((n,), math.log(math.expm1(0.1)))))
                self.register_parameter(kind + '_condition', nn.Parameter(torch.full((n,), math.log(math.expm1(0.1)))))
                self.register_parameter(kind + '_null', nn.Parameter(torch.zeros(n)))
        self.last_weights = None
        self.last_graph = None

    def encode(self, wave, stats):
        band = self.band(torch.cat([self.wave_norm(wave), stats], -1))
        i, j = self.pairs.unbind(-1)
        pair = self.pair(torch.cat([band[:, i], band[:, j], band[:, i] - band[:, j], stats[:, i], stats[:, j], stats[:, i] - stats[:, j], self.metadata[None].expand(len(wave), -1, -1)], -1))
        return (band, pair)

    def attention(self, kind, current, past, distance, gap, mask):
        query = getattr(self, kind + '_query')(current)
        key = getattr(self, kind + '_key')(past)
        logits = torch.einsum('bnd,bknd->bnk', query, key) / 4.0
        logits = logits - F.softplus(getattr(self, kind + '_time'))[None, :, None] * torch.log1p(gap)[:, None, :]
        logits = logits - F.softplus(getattr(self, kind + '_condition'))[None, :, None] * distance[:, None, :]
        logits = logits.masked_fill(~mask[:, None, :], -torch.inf)
        null = getattr(self, kind + '_null')[None, :, None].expand(len(current), -1, -1)
        probability = torch.cat([logits, null], -1).softmax(-1)
        return (probability[..., :-1], probability[..., -1])

    def forward(self, data):
        wave, stats, fields, mask, gap = data
        b, l, c, d = wave.shape
        k = l - 1
        p = len(self.pairs)
        band, pair = self.encode(wave.reshape(b * l, c, d), stats.reshape(b * l, c, 2))
        band = band.reshape(b, l, c, 32)
        pair = pair.reshape(b, l, p, 16)
        field = fields.reshape(b, l, self.s, 33)
        field = torch.cat([field[..., :32] * field[..., 32:], field[..., 32:]], -1).flatten(2)
        diff = field[:, :1] - field[:, 1:]
        distance = diff.square().mean(-1)
        bw, bnull = self.attention('band', band[:, 0], band[:, 1:], distance, gap, mask)
        pw, pnull = self.attention('pair', pair[:, 0], pair[:, 1:], distance, gap, mask)
        meta = torch.stack([torch.log1p(gap), torch.log1p(mask.sum(-1).float())[:, None].expand(-1, k)], -1)
        ctx = torch.cat([self.condition(diff), meta], -1)
        cb = band[:, :1].expand(-1, k, -1, -1)
        cp = pair[:, :1].expand(-1, k, -1, -1)
        bm = self.band_delta(torch.cat([cb, band[:, 1:], cb - band[:, 1:], ctx[:, :, None].expand(-1, -1, c, -1)], -1))
        pm = self.pair_delta(torch.cat([cp, pair[:, 1:], cp - pair[:, 1:], ctx[:, :, None].expand(-1, -1, p, -1)], -1))
        db = torch.einsum('bck,bkcd->bcd', bw, bm)
        dp = torch.einsum('bpk,bkpd->bpd', pw, pm)
        present = mask.any(-1).float()
        delta = 0.1 * (db + torch.einsum('cp,bpd->bcd', self.incidence, dp)) * present[:, None, None]
        self.last_weights = (bw.detach(), pw.detach())
        self.last_graph = dict(current_band=band[:, 0].detach(), history_band=band[:, 1:].detach(), current_pair=pair[:, 0].detach(), history_pair=pair[:, 1:].detach(), band_weights=bw.detach(), pair_weights=pw.detach(), band_null=bnull.detach(), pair_null=pnull.detach(), band_message=db.detach(), pair_message=dp.detach(), history_present=present.detach(), history_mask=mask.detach(), gap_days=gap.detach())
        return (delta, present)

class PairConditionCalibration(nn.Module):

    def __init__(self, slots, seed, bound):
        super().__init__()
        self.slots, self.bound, self.heads, self.width = (slots, float(bound), 4, 6)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 2026102803)
            self.field_in = nn.Linear(33, 24)
            self.field_id = nn.Parameter(torch.randn(slots, 24) * 0.02)
            self.field_norm = nn.LayerNorm(24)
            self.query = nn.Linear(24, 24, bias=False)
            self.key = nn.Linear(24, 24, bias=False)
            self.value = nn.Linear(24, 24, bias=False)
            self.null_logit = nn.Parameter(torch.zeros(4))
            self.out = nn.Linear(24, 24)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.last = None

    def forward(self, pairs, messages, fields, available):
        valid = available.bool()
        clean = torch.where(valid[..., None], fields[..., :32], torch.zeros_like(fields[..., :32]))
        clean = torch.cat([clean, valid[..., None].to(fields.dtype)], -1)
        tokens = self.field_norm(F.gelu(self.field_in(clean) + self.field_id[None]))
        tokens = tokens * valid[..., None]

        def heads(value):
            return value.reshape(len(value), -1, self.heads, self.width).transpose(1, 2)
        q, k, v = (heads(self.query(pairs)), heads(self.key(tokens)), heads(self.value(tokens)))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.width)
        scores = scores.masked_fill(~valid[:, None, None], -torch.inf)
        null = self.null_logit[None, :, None, None].expand(len(pairs), -1, pairs.shape[1], 1)
        probability = torch.cat([scores, null], -1).softmax(-1)
        context = probability[..., :-1] @ v
        context = context.transpose(1, 2).reshape_as(pairs)
        gain = self.bound * self.out(context).tanh() * valid.any(-1)[:, None, None]
        adjusted = messages * (1 + gain)
        self.last = dict(gain=gain.detach(), attention=probability.detach(), messages=messages.detach(), pair_correction=(adjusted - messages).detach())
        return adjusted

class PairConditionFusion(nn.Module):

    def __init__(self, original, slots, seed, bound):
        super().__init__()
        self.pair, self.history, self.dropout = (original.pair, original.history, original.dropout)
        self.calibration = PairConditionCalibration(slots, seed, bound)
        self.capacity = dict(current_band_field_graph=False, pair_field_attention=True, conditional_additive_shift=False, gain_bound=bound, shared_encoder=True, current_and_historical_pairs=True, history=True, late_conditions=True)

    def forward(self, nodes, statistics, fields, available):
        pairs = self.pair.encode_pairs(nodes, statistics)
        messages = pairs + 0.1 * self.pair.pair_ffn(pairs)
        adjusted = self.calibration(pairs, messages, fields, available)
        pooled = torch.einsum('cp,bpd->bcd', self.pair.incidence, self.dropout(adjusted))
        delta = self.pair.gate_logit.sigmoid() * self.pair.out(pooled)
        with torch.no_grad():
            correction = torch.einsum('cp,bpd->bcd', self.pair.incidence, adjusted - messages)
            self.calibration.last['band_correction'] = self.pair.gate_logit.sigmoid() * F.linear(correction, self.pair.out.weight)
            self.calibration.last['band_input'] = nodes.detach()
        return (nodes + delta, nodes.new_zeros(len(nodes), 64))

def band_only_history(self, data):
    wave, stats, fields, mask, gap = data
    b, length, channels, _ = wave.shape
    k = length - 1
    band = self.band(torch.cat([self.wave_norm(wave), stats], -1))
    fields = fields.reshape(b, length, self.s, 33)
    fields = torch.cat([fields[..., :32] * fields[..., 32:], fields[..., 32:]], -1).flatten(2)
    difference = fields[:, :1] - fields[:, 1:]
    weights, null = self.attention('band', band[:, 0], band[:, 1:], difference.square().mean(-1), gap, mask)
    meta = torch.stack([torch.log1p(gap), torch.log1p(mask.sum(-1).float())[:, None].expand(-1, k)], -1)
    ctx = torch.cat([self.condition(difference), meta], -1)
    current = band[:, :1].expand(-1, k, -1, -1)
    messages = self.band_delta(torch.cat([current, band[:, 1:], current - band[:, 1:], ctx[:, :, None].expand(-1, -1, channels, -1)], -1))
    result = torch.einsum('bck,bkcd->bcd', weights, messages)
    present = mask.any(-1).float()
    self.last_graph = dict(band_weights=weights.detach(), band_null=null.detach(), band_message=result.detach(), history_present=present.detach(), history_mask=mask.detach(), gap_days=gap.detach())
    return (0.1 * result * present[:, None, None], present)

class ChannelOnlyFusion(nn.Module):

    def __init__(self, original):
        super().__init__()
        self.original = original.original
        self.signal_norm = original.pair.signal_norm
        self.context_in = original.context_in
        first, last = (original.graph_gate[0], original.graph_gate[-1])
        with torch.random.fork_rng(devices=[]):
            self.graph_gate = nn.Sequential(nn.Linear(72, 16), nn.GELU(), nn.Linear(16, 2))
        with torch.no_grad():
            self.graph_gate[0].weight.copy_(torch.cat([first.weight[:, :64], first.weight[:, 88:]], -1))
            self.graph_gate[0].bias.copy_(first.bias)
            self.graph_gate[-1].load_state_dict(last.state_dict())

    def forward(self, nodes, statistics, fields, available):
        valid = available.bool()
        masked = torch.cat([fields[..., :32] * valid[..., None], valid[..., None].to(fields.dtype)], -1)
        pooled = masked.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        context = F.gelu(self.context_in(pooled)) * valid.any(-1)[:, None]
        gates = self.graph_gate(torch.cat([self.signal_norm(nodes), context[:, None].expand(-1, nodes.shape[1], -1)], -1)).sigmoid()
        graph_nodes, graph_context = self.original(nodes, statistics, fields, available)
        return (nodes + gates[..., :1] * (graph_nodes - nodes), gates[..., 1].mean(1)[:, None] * graph_context)

class PairOnlyFusion(nn.Module):

    def __init__(self, original):
        super().__init__()
        self.pair, self.history, self.dropout = (original.pair, original.history, original.dropout)

    def forward(self, nodes, statistics, fields, available):
        pairs = self.pair.encode_pairs(nodes, statistics)
        messages = pairs + 0.1 * self.pair.pair_ffn(pairs)
        pooled = torch.einsum('cp,bpd->bcd', self.pair.incidence, self.dropout(messages))
        delta = self.pair.gate_logit.sigmoid() * self.pair.out(pooled)
        return (nodes + delta, nodes.new_zeros(len(nodes), 64))

class IdentityCurrentFusion(nn.Module):
    """No current pair nodes => no incident field-to-pair modulation either."""

    def __init__(self, history):
        super().__init__()
        self.history = history

    def forward(self, nodes, statistics, fields, available):
        return (nodes, nodes.new_zeros(len(nodes), 64))
