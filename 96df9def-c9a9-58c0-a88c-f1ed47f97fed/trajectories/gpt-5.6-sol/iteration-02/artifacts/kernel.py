"""Chunk-parallel DPLR generalized delta rule."""

import torch
import triton
import triton.language as tl

CHUNK = 16
torch.backends.cuda.matmul.allow_tf32 = False


@triton.jit
def _convert_kernel(src, dst, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + offsets, mask=offsets < n)
    tl.store(dst + offsets, values, mask=offsets < n)



@triton.jit
def _chunk_states_kernel(a, b, k, v, g, initial, states,
                         K: tl.constexpr, V: tl.constexpr,
                         BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk = ok < K
    mv = ov < V
    mh = mk[:, None] & mv[None, :]
    state = tl.load(initial + bh * K * V + ok[:, None] * V + ov[None, :], mask=mh, other=0.0)
    base_s = bh * (BT + 1) * K * V
    tl.store(states + base_s + ok[:, None] * V + ov[None, :], state, mask=mh)
    for t in range(BT):
        base_k = (bh * BT + t) * K
        base_v = (bh * BT + t) * V
        av = tl.load(a + base_k + ok, mask=mk, other=0.0)
        bv = tl.load(b + base_k + ok, mask=mk, other=0.0)
        kv = tl.load(k + base_k + ok, mask=mk, other=0.0)
        gv = tl.load(g + base_k + ok, mask=mk, other=0.0)
        vv = tl.load(v + base_v + ov, mask=mv, other=0.0)
        read = tl.sum(state * av[:, None], axis=0)
        state = state * tl.exp(gv)[:, None] + kv[:, None] * vv[None, :] + bv[:, None] * read[None, :]
        off = base_s + (t + 1) * K * V + ok[:, None] * V + ov[None, :]
        tl.store(states + off, state, mask=mh)


@triton.jit
def _chunk_bwd_kernel(q, k, v, a, b, g, states, dout, dnext,
                      pdq, pdk, pdv, pda, pdb, pdg, dinit,
                      K: tl.constexpr, V: tl.constexpr, SCALE: tl.constexpr,
                      BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
                      NV: tl.constexpr):
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk = ok < K
    mv = ov < V
    mh = mk[:, None] & mv[None, :]
    adj = tl.load(dnext + bh * K * V + ok[:, None] * V + ov[None, :], mask=mh, other=0.0)
    base_s = bh * (BT + 1) * K * V
    for it in range(BT):
        t = BT - 1 - it
        base_k = (bh * BT + t) * K
        base_v = (bh * BT + t) * V
        qv = tl.load(q + base_k + ok, mask=mk, other=0.0)
        kv = tl.load(k + base_k + ok, mask=mk, other=0.0)
        av = tl.load(a + base_k + ok, mask=mk, other=0.0)
        bv = tl.load(b + base_k + ok, mask=mk, other=0.0)
        gv = tl.load(g + base_k + ok, mask=mk, other=0.0)
        vv = tl.load(v + base_v + ov, mask=mv, other=0.0)
        do = tl.load(dout + base_v + ov, mask=mv, other=0.0)
        old = tl.load(states + base_s + t * K * V + ok[:, None] * V + ov[None, :], mask=mh, other=0.0)
        new = tl.load(states + base_s + (t + 1) * K * V + ok[:, None] * V + ov[None, :], mask=mh, other=0.0)
        total = adj + qv[:, None] * do[None, :] * SCALE
        read = tl.sum(old * av[:, None], axis=0)
        dread = tl.sum(total * bv[:, None], axis=0)
        slot = ((bh * NV + iv) * BT + t) * K + ok
        tl.store(pdq + slot, tl.sum(new * do[None, :], axis=1) * SCALE, mask=mk)
        tl.store(pdk + slot, tl.sum(total * vv[None, :], axis=1), mask=mk)
        tl.store(pda + slot, tl.sum(old * dread[None, :], axis=1), mask=mk)
        tl.store(pdb + slot, tl.sum(total * read[None, :], axis=1), mask=mk)
        decay_old = old * tl.exp(gv)[:, None]
        tl.store(pdg + slot, tl.sum(total * decay_old, axis=1), mask=mk)
        tl.store(pdv + (bh * BT + t) * V + ov, tl.sum(total * kv[:, None], axis=0), mask=mv)
        adj = total * tl.exp(gv)[:, None] + av[:, None] * dread[None, :]
    tl.store(dinit + bh * K * V + ok[:, None] * V + ov[None, :], adj, mask=mh)


@triton.jit
def _reduce_partials_kernel(pdq, pdk, pda, pdb, pdg, dq, dk, da, db, dg,
                            n: tl.constexpr, STRIDE: tl.constexpr,
                            NV: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    group = offsets // STRIDE
    local = offsets % STRIDE
    accq = tl.zeros([BLOCK], tl.float32)
    acck = tl.zeros([BLOCK], tl.float32)
    acca = tl.zeros([BLOCK], tl.float32)
    accb = tl.zeros([BLOCK], tl.float32)
    accg = tl.zeros([BLOCK], tl.float32)
    for iv in range(NV):
        pos = (group * NV + iv) * STRIDE + local
        accq += tl.load(pdq + pos, mask=mask, other=0.0)
        acck += tl.load(pdk + pos, mask=mask, other=0.0)
        acca += tl.load(pda + pos, mask=mask, other=0.0)
        accb += tl.load(pdb + pos, mask=mask, other=0.0)
        accg += tl.load(pdg + pos, mask=mask, other=0.0)
    tl.store(dq + offsets, accq, mask=mask)
    tl.store(dk + offsets, acck, mask=mask)
    tl.store(da + offsets, acca, mask=mask)
    tl.store(db + offsets, accb, mask=mask)
    tl.store(dg + offsets, accg, mask=mask)


@triton.jit
def _chunk_fwd_kernel(q, k, v, a, b, g, initial, out, next_state,
                      K: tl.constexpr, V: tl.constexpr, SCALE: tl.constexpr,
                      BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    ot = tl.arange(0, BT)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mt = ot < BT
    mk = ok < K
    mv = ov < V
    base_k = bh * BT * K
    base_v = bh * BT * V
    qv = tl.load(q + base_k + ot[:, None] * K + ok[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
    kv = tl.load(k + base_k + ot[:, None] * K + ok[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
    av = tl.load(a + base_k + ot[:, None] * K + ok[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
    bv = tl.load(b + base_k + ot[:, None] * K + ok[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
    gv = tl.load(g + base_k + ot[:, None] * K + ok[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
    vv = tl.load(v + base_v + ot[:, None] * V + ov[None, :], mask=mt[:, None] & mv[None, :], other=0.0)
    gates = tl.cumsum(gv, axis=0)
    qg = qv * tl.exp(gates) * SCALE
    kg = kv * tl.exp(-gates)
    bg = bv * tl.exp(-gates)
    ag = av * tl.exp(gates - gv)
    aqk = tl.dot(qg, tl.trans(kg), input_precision="ieee")
    aqb = tl.dot(qg, tl.trans(bg), input_precision="ieee")
    aak = tl.dot(ag, tl.trans(kg), input_precision="ieee")
    aab = tl.dot(ag, tl.trans(bg), input_precision="ieee")
    row = ot[:, None]
    col = ot[None, :]
    aqk = tl.where(row >= col, aqk, 0.0)
    aqb = tl.where(row >= col, aqb, 0.0)
    aak = tl.where(row > col, aak, 0.0)
    wy = tl.where(row > col, aab, 0.0)
    for i in range(1, BT):
        select = ot == i
        current = tl.sum(tl.where(select[:, None], wy, 0.0), axis=0)
        current += tl.sum(current[:, None] * wy, axis=0) * (ot < i)
        wy = tl.where(select[:, None], current, wy)
    wy += row == col
    state = tl.load(initial + bh * K * V + ok[:, None] * V + ov[None, :], mask=mk[:, None] & mv[None, :], other=0.0)
    u = tl.dot(tl.dot(wy, aak, input_precision="ieee"), vv, input_precision="ieee")
    w = tl.dot(wy, ag, input_precision="ieee")
    vnew = u + tl.dot(w, state, input_precision="ieee")
    result = tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, vnew, input_precision="ieee")
    result += tl.dot(qg, state, input_precision="ieee")
    tl.store(out + base_v + ot[:, None] * V + ov[None, :], result, mask=mt[:, None] & mv[None, :])
    final_gate = tl.sum(tl.where(ot[:, None] == BT - 1, gates, 0.0), axis=0)
    decay = tl.exp(final_gate[None, :] - gates)
    state = state * tl.exp(final_gate)[:, None]
    state += tl.dot(tl.trans(kg * tl.exp(final_gate)[None, :]), vv, input_precision="ieee")
    state += tl.dot(tl.trans(bg * tl.exp(final_gate)[None, :]), vnew, input_precision="ieee")
    tl.store(next_state + bh * K * V + ok[:, None] * V + ov[None, :], state, mask=mk[:, None] & mv[None, :])



class _Convert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        y = torch.empty(x.shape, dtype=dtype, device=x.device)
        n = x.numel()
        _convert_kernel[(triton.cdiv(n, 256),)](x, y, n=n, BLOCK=256)
        ctx.input_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        dx = torch.empty(dy.shape, dtype=ctx.input_dtype, device=dy.device)
        n = dy.numel()
        _convert_kernel[(triton.cdiv(n, 256),)](dy, dx, n=n, BLOCK=256)
        return dx, None


def _pad_time(x, length):
    if x.shape[1] == length:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, length - x.shape[1]))


def _wy_inverse(lower):
    rows = [lower[..., 0, :]]
    for i in range(1, CHUNK):
        previous = torch.stack(rows, dim=-2)
        correction = torch.einsum("...j,...jk->...k", lower[..., i, :i], previous)
        rows.append(lower[..., i, :] + correction)
    inverse = torch.stack(rows, dim=-2)
    eye = torch.eye(CHUNK, dtype=torch.float32, device=lower.device)
    return inverse + eye


def _chunk_math(q, k, v, a, b, g, state):
    K = q.shape[-1]
    gates = g.cumsum(dim=-2)
    time = torch.arange(CHUNK, device=q.device)
    causal = time[:, None] >= time[None, :]
    strict = time[:, None] > time[None, :]
    q = q * (K ** -0.5)
    gate_delta = gates[..., :, None, :] - gates[..., None, :, :]
    prior_delta = (gates - g)[..., :, None, :] - gates[..., None, :, :]
    ratios = torch.exp(gate_delta.masked_fill(~causal[..., None], 0))
    ratios = ratios * causal[..., None]
    prior_ratios = torch.exp(prior_delta.masked_fill(~strict[..., None], 0))
    prior_ratios = prior_ratios * strict[..., None]
    qk = (q[..., :, None, :] * k[..., None, :, :] * ratios).sum(-1)
    qb = (q[..., :, None, :] * b[..., None, :, :] * ratios).sum(-1)
    ak = (a[..., :, None, :] * k[..., None, :, :] * prior_ratios).sum(-1)
    ab = (a[..., :, None, :] * b[..., None, :, :] * prior_ratios).sum(-1)
    qg = q * gates.exp()
    ag = a * torch.exp(gates - g)
    wy = _wy_inverse(ab)
    vnew = wy @ (ak @ v) + (wy @ ag) @ state
    out = qk @ v + qb @ vnew + qg @ state
    final_gate = gates[..., -1, :]
    decay = torch.exp(final_gate[..., None, :] - gates)
    next_state = state * final_gate.exp()[..., None]
    next_state = next_state + (k * decay).transpose(-1, -2) @ v
    next_state = next_state + (b * decay).transpose(-1, -2) @ vnew
    return out, next_state


class _Chunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a, b, g, state):
        xs = [x.contiguous() for x in (q, k, v, a, b, g, state)]
        q, k, v, a, b, g, state = xs
        B, H, _, K = q.shape
        V = v.shape[-1]
        BK = max(16, triton.next_power_of_2(K))
        BV = min(32, max(16, triton.next_power_of_2(V)))
        if BK == K:
            out = torch.empty_like(v)
            next_state = torch.empty_like(state)
            _chunk_fwd_kernel[(B * H, triton.cdiv(V, BV))](
                q, k, v, a, b, g, state, out, next_state,
                K=K, V=V, SCALE=K ** -0.5, BK=BK, BV=BV, BT=CHUNK,
                num_warps=4,
            )
        else:
            out, next_state = _chunk_math(q, k, v, a, b, g, state)
            out = _Convert.apply(out.contiguous(), torch.float32)
            next_state = _Convert.apply(next_state.contiguous(), torch.float32)
        ctx.save_for_backward(*xs, out, next_state)
        return out, next_state

    @staticmethod
    def backward(ctx, dout, dstate):
        q, k, v, a, b, g, initial, _, _ = ctx.saved_tensors
        q, k, v, a, b, g, dout = [x.contiguous() for x in (q, k, v, a, b, g, dout)]
        B, H, _, K = q.shape
        V = v.shape[-1]
        BH = B * H
        BK = triton.next_power_of_2(K)
        BV = min(128, max(16, triton.next_power_of_2(V)))
        NV = triton.cdiv(V, BV)
        states = torch.empty(BH, CHUNK + 1, K, V, dtype=torch.float32, device=q.device)
        _chunk_states_kernel[(BH, NV)](
            a, b, k, v, g, initial.contiguous(), states,
            K=K, V=V, BK=BK, BV=BV, BT=CHUNK, num_warps=4,
        )
        partial_shape = (BH, NV, CHUNK, K)
        partials = [torch.empty(partial_shape, dtype=torch.float32, device=q.device) for _ in range(5)]
        dv = torch.empty_like(v)
        dinit = torch.empty_like(initial)
        _chunk_bwd_kernel[(BH, NV)](
            q, k, v, a, b, g, states, dout, dstate.contiguous(),
            *partials[:2], dv, *partials[2:], dinit,
            K=K, V=V, SCALE=K ** -0.5, BK=BK, BV=BV, BT=CHUNK, NV=NV,
            num_warps=4,
        )
        results = [torch.empty_like(q) for _ in range(5)]
        n = BH * CHUNK * K
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](
            *partials, *results, n=n, STRIDE=CHUNK * K, NV=NV, BLOCK=256,
        )
        dq, dk, da, db, dg = results
        return dq, dk, dv, da, db, dg, dinit


def _forward_math(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    padded_t = triton.cdiv(T, CHUNK) * CHUNK
    tensors = [_pad_time(x.float(), padded_t) for x in (q, k, v, alpha, beta, gk)]
    chunks = [
        x.view(B, -1, CHUNK, H, x.shape[-1]).permute(0, 3, 1, 2, 4)
        for x in tensors
    ]
    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    outputs = []
    for i in range(padded_t // CHUNK):
        out, state = _Chunk.apply(*(x[:, :, i] for x in chunks), state)
        outputs.append(out)
    out = torch.stack(outputs, dim=2).permute(0, 2, 3, 1, 4).reshape(B, padded_t, H, V)
    return _Convert.apply(out[:, :T].contiguous(), v.dtype)


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _forward_math(q, k, v, alpha, beta, gk)
