import torch
import triton
import triton.language as tl

_BT = 16
_FWD_BV = 32
_BWD_BV = 16


@triton.jit
def _chunk_forward(q, k, v, a, b, g, out, checkpoints, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov, ot = tl.arange(0, BK), iv * BV + tl.arange(0, BV), tl.arange(0, 16)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    eye = ot[:, None] == ot[None, :]
    lower, causal = ot[:, None] > ot[None, :], ot[:, None] >= ot[None, :]
    for ic in range(NC):
        ts = ic * 16 + ot
        mt = ts < T
        base_k = ((ib * T + ts) * H + ih) * K
        base_v = ((ib * T + ts) * H + ih) * V
        mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
        qq = tl.load(q + base_k[:, None] + ok[None, :], mask=mtk, other=0.0).to(tl.float32) * SCALE
        kk = tl.load(k + base_k[:, None] + ok[None, :], mask=mtk, other=0.0).to(tl.float32)
        aa = tl.load(a + base_k[:, None] + ok[None, :], mask=mtk, other=0.0).to(tl.float32)
        bb = tl.load(b + base_k[:, None] + ok[None, :], mask=mtk, other=0.0).to(tl.float32)
        gg = tl.load(g + base_k[:, None] + ok[None, :], mask=mtk, other=0.0).to(tl.float32)
        vv = tl.load(v + base_v[:, None] + ov[None, :], mask=mtv, other=0.0).to(tl.float32)
        gc = tl.cumsum(gg, axis=0)
        qg, ap = qq * tl.exp(gc), aa * tl.exp(gc - gg)
        reciprocal_gate = tl.exp(-tl.maximum(gc, -60.0))
        kn, bn = kk * reciprocal_gate, bb * reciprocal_gate
        aqk = tl.where(causal, tl.dot(qg, tl.trans(kn), input_precision="ieee"), 0.0)
        aqb = tl.where(causal, tl.dot(qg, tl.trans(bn), input_precision="ieee"), 0.0)
        lab = tl.where(lower, tl.dot(ap, tl.trans(bn), input_precision="ieee"), 0.0)
        lak = tl.where(lower, tl.dot(ap, tl.trans(kn), input_precision="ieee"), 0.0)
        inv = lab
        for row_index in tl.static_range(1, 16):
            row_mask = ot == row_index
            row = tl.sum(tl.where(row_mask[:, None], inv, 0.0), axis=0)
            row += tl.sum(row[:, None] * inv, axis=0) * (ot < row_index)
            inv = tl.where(row_mask[:, None], row, inv)
        inv += eye.to(tl.float32)
        u = tl.dot(inv, tl.dot(lak, vv, input_precision="ieee"), input_precision="ieee")
        w = tl.dot(inv, ap, input_precision="ieee")
        reads = u + tl.dot(w, state, input_precision="ieee")
        oo = tl.dot(qg, state, input_precision="ieee") + tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, reads, input_precision="ieee")
        tl.store(out + base_v[:, None] + ov[None, :], oo, mask=mtv)
        cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(checkpoints + cp, state, mask=mk[:, None] & mv[None, :])
        last = tl.minimum((ic + 1) * 16, T) - ic * 16 - 1
        glast = tl.sum(tl.where(ot[:, None] == last, gc, 0.0), axis=0)
        kg, bg = kk * tl.exp(glast[None, :] - gc), bb * tl.exp(glast[None, :] - gc)
        state = state * tl.exp(glast)[:, None] + tl.dot(tl.trans(kg), vv, input_precision="ieee") + tl.dot(tl.trans(bg), reads, input_precision="ieee")


@triton.jit
def _expand_chunk_states(k, v, a, b, g, checkpoints, states, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
    state = tl.load(checkpoints + cp, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    for j in tl.static_range(0, 16):
        t, valid = ic * 16 + j, ic * 16 + j < T
        pk, pv = ((ib * T + t) * H + ih) * K + ok, ((ib * T + t) * H + ih) * V + ov
        sm = (((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(states + sm, state, mask=valid & mk[:, None] & mv[None, :])
        kk = tl.load(k + pk, mask=valid & mk, other=0.0).to(tl.float32)
        aa = tl.load(a + pk, mask=valid & mk, other=0.0).to(tl.float32)
        bb = tl.load(b + pk, mask=valid & mk, other=0.0).to(tl.float32)
        gg = tl.load(g + pk, mask=valid & mk, other=0.0).to(tl.float32)
        vv = tl.load(v + pv, mask=valid & mv, other=0.0).to(tl.float32)
        read = tl.sum(state * aa[:, None], axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _backward_scan(q, k, v, a, b, g, do, states, dqpart, dkpart, dapart, dbpart, dgpart, dv, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BV), tl.float32)
    for z in range(T):
        t = T - 1 - z
        pk, pv = ((ib * T + t) * H + ih) * K + ok, ((ib * T + t) * H + ih) * V + ov
        sm = (((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        old = tl.load(states + sm, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
        qq = tl.load(q + pk, mask=mk, other=0.0).to(tl.float32) * SCALE
        kk = tl.load(k + pk, mask=mk, other=0.0).to(tl.float32)
        aa = tl.load(a + pk, mask=mk, other=0.0).to(tl.float32)
        bb = tl.load(b + pk, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + pk, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        dd = tl.load(do + pv, mask=mv, other=0.0).to(tl.float32)
        decay_old = old * tl.exp(gg)[:, None]
        read = tl.sum(old * aa[:, None], axis=0)
        new = decay_old + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        total = ds + qq[:, None] * dd[None, :]
        dqp = tl.sum(new * dd[None, :], axis=1) * SCALE
        dkp = tl.sum(total * vv[None, :], axis=1)
        dbp = tl.sum(total * read[None, :], axis=1)
        dr = tl.sum(total * bb[:, None], axis=0)
        dap = tl.sum(old * dr[None, :], axis=1)
        dgp = tl.sum(total * decay_old, axis=1)
        dvv = tl.sum(total * kk[:, None], axis=0)
        pp = (iv * (T * H) + ib * T * H + t * H + ih) * K + ok
        tl.store(dqpart + pp, dqp, mask=mk)
        tl.store(dkpart + pp, dkp, mask=mk)
        tl.store(dapart + pp, dap, mask=mk)
        tl.store(dbpart + pp, dbp, mask=mk)
        tl.store(dgpart + pp, dgp, mask=mk)
        tl.store(dv + pv, dvv, mask=mv)
        ds = total * tl.exp(gg)[:, None] + aa[:, None] * dr[None, :]


@triton.jit
def _reduce_v_partials(x, y, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        acc += tl.load(x + iv * N + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y + offs, acc, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V, nc, bk = v.shape[-1], triton.cdiv(T, _BT), max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _FWD_BV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _chunk_forward[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints, T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FWD_BV, SCALE=K ** -0.5, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K = q.shape
        V, nc, bk = v.shape[-1], triton.cdiv(T, _BT), max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BWD_BV)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_chunk_states[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states, T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV, num_warps=4, num_stages=2)
        n = B * T * H * K
        parts = [torch.empty((nv, n), device=q.device, dtype=torch.float32) for _ in range(5)]
        dq, dk, da, db, dg = (torch.empty_like(x) for x in (q, k, alpha, beta, gk))
        dv = torch.empty_like(v)
        _backward_scan[(nv, B * H)](q, k, v, alpha, beta, gk, dout.contiguous(), states, *parts, dv, T, H=H, K=K, V=V, BK=bk, BV=_BWD_BV, SCALE=K ** -0.5, num_warps=4, num_stages=2)
        for part, grad in zip(parts, (dq, dk, da, db, dg)):
            _reduce_v_partials[(triton.cdiv(n, 256),)](part, grad, N=n, NV=nv, BLOCK=256)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
