import math

import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(tl.minimum(x, 60.0), -60.0))


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr, SCALE: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    rows = tl.arange(0, 16)
    state = tl.zeros((BK, BV), tl.float32)
    state_mask = (ok[:, None] < K) & (ov[None, :] < V)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V

    for ic in range(0, NC):
        cp = checkpoints + (((ib * NC + ic) * H + ih) * K * V)
        tl.store(cp + ok[:, None] * V + ov[None, :], state, mask=state_mask)

        tt = ic * 16 + rows
        mt = tt < T
        mk = mt[:, None] & (ok[None, :] < K)
        mv = mt[:, None] & (ov[None, :] < V)
        pk = base_k + tt[:, None] * H * K + ok[None, :]
        pv = base_v + tt[:, None] * H * V + ov[None, :]
        q0 = tl.load(q + pk, mask=mk, other=0.0)
        k0 = tl.load(k + pk, mask=mk, other=0.0)
        a0 = tl.load(a + pk, mask=mk, other=0.0)
        b0 = tl.load(b + pk, mask=mk, other=0.0)
        g0 = tl.load(g + pk, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + pv, mask=mv, other=0.0)
        vv32 = vv.to(tl.float32)
        gc = tl.cumsum(g0, axis=0)
        ge = tl.sum(g0, axis=0)
        off = tl.sum(tl.where(rows[:, None] == 8, gc, 0.0), axis=0)

        qx = q0.to(tl.float32) * _bounded_exp(gc - off[None, :])
        kx = k0.to(tl.float32) * _bounded_exp(off[None, :] - gc)
        bx = b0.to(tl.float32) * _bounded_exp(off[None, :] - gc)
        ax = a0.to(tl.float32) * _bounded_exp(gc - g0 - off[None, :])
        causal = rows[:, None] >= rows[None, :]
        strict = rows[:, None] > rows[None, :]
        aqk = tl.where(causal, tl.dot(qx, tl.trans(kx), input_precision="ieee") * SCALE, 0.0)
        aqb = tl.where(causal, tl.dot(qx, tl.trans(bx), input_precision="ieee") * SCALE, 0.0)
        lak = tl.where(strict, tl.dot(ax, tl.trans(kx), input_precision="ieee"), 0.0)
        lab = tl.where(strict, tl.dot(ax, tl.trans(bx), input_precision="ieee"), 0.0)

        qstate = tl.dot(q0.to(tl.float32) * _bounded_exp(gc), state, input_precision="ieee") * SCALE
        r = tl.dot(a0.to(tl.float32) * _bounded_exp(gc - g0), state, input_precision="ieee")
        r += tl.dot(lak, vv32, input_precision="ieee")
        for j in tl.static_range(0, 16):
            rj = tl.sum(tl.where(rows[:, None] == j, r, 0.0), axis=0)
            coeff = tl.sum(tl.where(rows[:, None] == j, lab, 0.0), axis=0)
            rj += tl.sum(coeff[:, None] * r, axis=0)
            r = tl.where(rows[:, None] == j, rj[None, :], r)

        oo = qstate + tl.dot(aqk, vv32, input_precision="ieee")
        oo += tl.dot(aqb, r, input_precision="ieee")
        tl.store(out + pv, oo, mask=mv)

        kg = k0.to(tl.float32) * _bounded_exp(ge[None, :] - gc)
        bg = b0.to(tl.float32) * _bounded_exp(ge[None, :] - gc)
        state *= _bounded_exp(ge)[:, None]
        state += tl.dot(tl.trans(kg), vv32, input_precision="ieee")
        state += tl.dot(tl.trans(bg), r, input_precision="ieee")


@triton.jit
def _checkpoint_kernel(k, v, a, b, g, checkpoints, states,
                       T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                       V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                       BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    sm = (ok[:, None] < K) & (ov[None, :] < V)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(0, NC):
        cp = checkpoints + (((ib * NC + ic) * H + ih) * K * V)
        tl.store(cp + ok[:, None] * V + ov[None, :], state, mask=sm)
        for j in tl.static_range(0, 16):
            t = ic * 16 + j
            valid = t < T
            pk = ((ib * T + t) * H + ih) * K + ok
            pv = ((ib * T + t) * H + ih) * V + ov
            ps = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
            tl.store(states + ps, state, mask=valid & sm)
            kt = tl.load(k + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            at = tl.load(a + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            bt = tl.load(b + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            gt = tl.load(g + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            vt = tl.load(v + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
            read = tl.sum(state * at[:, None], axis=0)
            updated = state * tl.exp(gt)[:, None] + kt[:, None] * vt[None, :] + bt[:, None] * read[None, :]
            state = tl.where(valid, updated, state)


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                          BV: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    sm = (ok[:, None] < K) & (ov[None, :] < V)
    cp = checkpoints + (((ib * NC + ic) * H + ih) * K * V)
    state = tl.load(cp + ok[:, None] * V + ov[None, :], mask=sm, other=0.0).to(tl.float32)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        pk = ((ib * T + t) * H + ih) * K + ok
        pv = ((ib * T + t) * H + ih) * V + ov
        ps = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(states + ps, state, mask=valid & sm)
        kt = tl.load(k + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        at = tl.load(a + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        bt = tl.load(b + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        gt = tl.load(g + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        vt = tl.load(v + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
        read = tl.sum(state * at[:, None], axis=0)
        state = state * tl.exp(gt)[:, None] + kt[:, None] * vt[None, :] + bt[:, None] * read[None, :]


@triton.jit
def _boundary_adjoint_kernel(q, a, b, g, dout, boundaries,
                             T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                             V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                             BV: tl.constexpr, SCALE: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    sm = (ok[:, None] < K) & (ov[None, :] < V)
    ds = tl.zeros((BK, BV), tl.float32)
    for rr in range(0, NC):
        ic = NC - 1 - rr
        pb = boundaries + (((ib * NC + ic) * H + ih) * K * V)
        tl.store(pb + ok[:, None] * V + ov[None, :], ds, mask=sm)
        for jj in tl.static_range(0, 16):
            j = 15 - jj
            t = ic * 16 + j
            valid = t < T
            pk = ((ib * T + t) * H + ih) * K + ok
            pv = ((ib * T + t) * H + ih) * V + ov
            qt = tl.load(q + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            at = tl.load(a + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            bt = tl.load(b + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            gt = tl.load(g + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            do = tl.load(dout + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
            cur = ds + qt[:, None] * (do[None, :] * SCALE)
            dr = tl.sum(cur * bt[:, None], axis=0)
            prev = cur * tl.exp(gt)[:, None] + at[:, None] * dr[None, :]
            ds = tl.where(valid, prev, ds)


@triton.jit
def _backward_chunks_kernel(q, k, v, a, b, g, dout, states, boundaries,
                            dq_part, dk_part, da_part, db_part, dg_part, dv,
                            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                            V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                            BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    sm = (ok[:, None] < K) & (ov[None, :] < V)
    pb = boundaries + (((ib * NC + ic) * H + ih) * K * V)
    ds = tl.load(pb + ok[:, None] * V + ov[None, :], mask=sm, other=0.0).to(tl.float32)
    for jj in tl.static_range(0, 16):
        j = 15 - jj
        t = ic * 16 + j
        valid = t < T
        pk = ((ib * T + t) * H + ih) * K + ok
        pv = ((ib * T + t) * H + ih) * V + ov
        ps = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        old = tl.load(states + ps, mask=valid & sm, other=0.0).to(tl.float32)
        qt = tl.load(q + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        kt = tl.load(k + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        at = tl.load(a + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        bt = tl.load(b + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        gt = tl.load(g + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
        vt = tl.load(v + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
        do = tl.load(dout + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
        read = tl.sum(old * at[:, None], axis=0)
        new = old * tl.exp(gt)[:, None] + kt[:, None] * vt[None, :] + bt[:, None] * read[None, :]
        cur = ds + qt[:, None] * (do[None, :] * SCALE)
        dq0 = tl.sum(new * do[None, :], axis=1) * SCALE
        dk0 = tl.sum(cur * vt[None, :], axis=1)
        dv0 = tl.sum(cur * kt[:, None], axis=0)
        db0 = tl.sum(cur * read[None, :], axis=1)
        dr = tl.sum(cur * bt[:, None], axis=0)
        da0 = tl.sum(old * dr[None, :], axis=1)
        dg0 = tl.sum(cur * (old * tl.exp(gt)[:, None]), axis=1)
        pi = (((iv * T + t) * H + ih) * K + ok)
        pm = valid & (ok < K)
        tl.store(dq_part + pi, dq0, mask=pm)
        tl.store(dk_part + pi, dk0, mask=pm)
        tl.store(da_part + pi, da0, mask=pm)
        tl.store(db_part + pi, db0, mask=pm)
        tl.store(dg_part + pi, dg0, mask=pm)
        tl.store(dv + pv, dv0, mask=valid & (ov < V))
        prev = cur * tl.exp(gt)[:, None] + at[:, None] * dr[None, :]
        ds = tl.where(valid, prev, ds)


@triton.jit
def _reduce_partials_kernel(src, dst, N: tl.constexpr, NV: tl.constexpr,
                            BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for iv in range(0, NV):
        acc += tl.load(src + iv * N + offs, mask=mask, other=0.0)
    tl.store(dst + offs, acc, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _FBV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV, SCALE=K ** -0.5,
            num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        if V == triton.next_power_of_2(V):
            bv = 16
            nv = triton.cdiv(V, bv)
            num_warps = 8
        else:
            bv = max(16, triton.next_power_of_2(V))
            nv = 1
            num_warps = 4
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        boundaries = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        if V == triton.next_power_of_2(V):
            _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states,
                T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, num_warps=num_warps, num_stages=2)
        else:
            _checkpoint_kernel[(nv, B * H)](k, v, alpha, beta, gk, checkpoints, states,
                T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, num_warps=num_warps, num_stages=2)
        _boundary_adjoint_kernel[(nv, B * H)](q, alpha, beta, gk, dout, boundaries,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, SCALE=K ** -0.5,
            num_warps=num_warps, num_stages=2)
        n = B * T * H * K
        if nv == 1:
            parts = [torch.empty_like(q) for _ in range(5)]
        else:
            parts = [torch.empty((nv, n), device=q.device, dtype=torch.float32) for _ in range(5)]
        dv = torch.empty_like(v)
        _backward_chunks_kernel[(nv, nc, B * H)](q, k, v, alpha, beta, gk, dout,
            states, boundaries, *parts, dv, T=T, H=H, K=K, V=V, NC=nc, NV=nv,
            BK=bk, BV=bv, SCALE=K ** -0.5, num_warps=num_warps, num_stages=2)
        if nv == 1:
            grads = parts
        else:
            grads = [torch.empty_like(q) for _ in range(5)]
            for src, dst in zip(parts, grads):
                _reduce_partials_kernel[(triton.cdiv(n, 256),)](
                    src, dst, N=n, NV=nv, BLOCK=256, num_warps=num_warps)
        dq, dk, da, db, dg = grads
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
