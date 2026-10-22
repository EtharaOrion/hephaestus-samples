import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    rows = tl.arange(0, 16)
    causal = rows[:, None] >= rows[None, :]
    strict = rows[:, None] > rows[None, :]
    eye = rows[:, None] == rows[None, :]

    for ic in range(NC):
        hp = checkpoints + ((bh * NC + ic) * K + kk[:, None]) * V + vv[None, :]
        tl.store(hp, state, mask=(kk[:, None] < K) & (vv[None, :] < V))

        tt = ic * 16 + rows
        xbase = ((ib * T + tt[:, None]) * H + ih) * K + kk[None, :]
        kmask = (tt[:, None] < T) & (kk[None, :] < K)
        q0 = tl.load(q + xbase, mask=kmask, other=0.0).to(tl.float32)
        k0 = tl.load(k + xbase, mask=kmask, other=0.0).to(tl.float32)
        a0 = tl.load(a + xbase, mask=kmask, other=0.0).to(tl.float32)
        b0 = tl.load(b + xbase, mask=kmask, other=0.0).to(tl.float32)
        g0 = tl.load(g + xbase, mask=kmask, other=0.0).to(tl.float32)
        gc = tl.cumsum(g0, axis=0)
        safe_gc = tl.maximum(gc, -60.0)
        eg = tl.exp(safe_gc)
        eig = tl.exp(-safe_gc)
        qg = q0 * eg * scale
        kg = k0 * eig
        bg = b0 * eig
        ag = a0 * tl.exp(tl.maximum(gc - g0, -60.0))

        vp = v + ((ib * T + tt[:, None]) * H + ih) * V + vv[None, :]
        tvmask = (tt[:, None] < T) & (vv[None, :] < V)
        v0 = tl.load(vp, mask=tvmask, other=0.0).to(tl.float32)

        lab = tl.dot(ag, tl.trans(bg), input_precision="ieee")
        lab = tl.where(strict, lab, 0.0)
        inv = tl.where(eye, 1.0, 0.0)
        power = lab
        for _ in tl.static_range(0, 4):
            inv += tl.dot(inv, power, input_precision="ieee")
            power = tl.dot(power, power, input_precision="ieee")

        lak = tl.dot(ag, tl.trans(kg), input_precision="ieee")
        lak = tl.where(strict, lak, 0.0)
        read = tl.dot(ag, state, input_precision="ieee")
        read += tl.dot(lak, v0, input_precision="ieee")
        read = tl.dot(inv, read, input_precision="ieee")

        aqk = tl.dot(qg, tl.trans(kg), input_precision="ieee")
        aqb = tl.dot(qg, tl.trans(bg), input_precision="ieee")
        aqk = tl.where(causal, aqk, 0.0)
        aqb = tl.where(causal, aqb, 0.0)
        tile = tl.dot(qg, state, input_precision="ieee")
        tile += tl.dot(aqk, v0, input_precision="ieee")
        tile += tl.dot(aqb, read, input_precision="ieee")
        op = out + ((ib * T + tt[:, None]) * H + ih) * V + vv[None, :]
        tl.store(op, tile, mask=tvmask)

        state += tl.dot(tl.trans(kg), v0, input_precision="ieee")
        state += tl.dot(tl.trans(bg), read, input_precision="ieee")
        state *= tl.exp(tl.sum(g0, axis=0))[:, None]


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                          BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    hp = checkpoints + ((bh * NC + ic) * K + kk[:, None]) * V + vv[None, :]
    mask2 = (kk[:, None] < K) & (vv[None, :] < V)
    state = tl.load(hp, mask=mask2, other=0.0).to(tl.float32)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        x = ((ib * T + t) * H + ih) * K + kk
        mk = (t < T) & (kk < K)
        k0 = tl.load(k + x, mask=mk, other=0.0).to(tl.float32)
        a0 = tl.load(a + x, mask=mk, other=0.0).to(tl.float32)
        b0 = tl.load(b + x, mask=mk, other=0.0).to(tl.float32)
        e0 = tl.exp(tl.load(g + x, mask=mk, other=0.0).to(tl.float32))
        vp = v + ((ib * T + t) * H + ih) * V + vv
        mv = (t < T) & (vv < V)
        v0 = tl.load(vp, mask=mv, other=0.0).to(tl.float32)
        read = tl.sum(a0[:, None] * state, axis=0)
        state = state * e0[:, None] + k0[:, None] * v0[None, :] + b0[:, None] * read[None, :]
        sp = states + (((ib * T + t) * H + ih) * K + kk[:, None]) * V + vv[None, :]
        tl.store(sp, state, mask=(t < T) & mask2)



@triton.jit
def _boundary_adjoint_kernel(q, a, b, g, dout, boundaries,
                             T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                             V: tl.constexpr, NC: tl.constexpr,
                             BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mask2 = (kk[:, None] < K) & (vv[None, :] < V)
    ds = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    for rc in range(NC):
        ic = NC - 1 - rc
        bp = boundaries + ((bh * NC + ic) * K + kk[:, None]) * V + vv[None, :]
        tl.store(bp, ds, mask=mask2)
        for rj in tl.static_range(0, 16):
            j = 15 - rj
            t = ic * 16 + j
            x = ((ib * T + t) * H + ih) * K + kk
            mk = (t < T) & (kk < K)
            q0 = tl.load(q + x, mask=mk, other=0.0).to(tl.float32)
            a0 = tl.load(a + x, mask=mk, other=0.0).to(tl.float32)
            b0 = tl.load(b + x, mask=mk, other=0.0).to(tl.float32)
            e0 = tl.exp(tl.load(g + x, mask=mk, other=0.0).to(tl.float32))
            do = tl.load(dout + ((ib * T + t) * H + ih) * V + vv,
                         mask=(t < T) & (vv < V), other=0.0).to(tl.float32)
            ds += q0[:, None] * scale * do[None, :]
            dr = tl.sum(b0[:, None] * ds, axis=0)
            ds = ds * e0[:, None] + a0[:, None] * dr[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, states, dout, boundaries, dv,
                     pq, pk, pa, pb, pg,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, NC: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mask2 = (kk[:, None] < K) & (vv[None, :] < V)
    bp = boundaries + ((bh * NC + ic) * K + kk[:, None]) * V + vv[None, :]
    ds = tl.load(bp, mask=mask2, other=0.0).to(tl.float32)
    scale: tl.constexpr = K ** -0.5
    for rj in tl.static_range(0, 16):
        j = 15 - rj
        t = ic * 16 + j
        x = ((ib * T + t) * H + ih) * K + kk
        valid = t < T
        mk = valid & (kk < K)
        q0 = tl.load(q + x, mask=mk, other=0.0).to(tl.float32)
        k0 = tl.load(k + x, mask=mk, other=0.0).to(tl.float32)
        a0 = tl.load(a + x, mask=mk, other=0.0).to(tl.float32)
        b0 = tl.load(b + x, mask=mk, other=0.0).to(tl.float32)
        e0 = tl.exp(tl.load(g + x, mask=mk, other=0.0).to(tl.float32))
        vp = v + ((ib * T + t) * H + ih) * V + vv
        mv = valid & (vv < V)
        v0 = tl.load(vp, mask=mv, other=0.0).to(tl.float32)
        do = tl.load(dout + ((ib * T + t) * H + ih) * V + vv,
                     mask=mv, other=0.0).to(tl.float32)
        snp = states + (((ib * T + t) * H + ih) * K + kk[:, None]) * V + vv[None, :]
        sn = tl.load(snp, mask=valid & mask2, other=0.0).to(tl.float32)
        if t > 0:
            spp = states + (((ib * T + t - 1) * H + ih) * K + kk[:, None]) * V + vv[None, :]
            sp = tl.load(spp, mask=valid & mask2, other=0.0).to(tl.float32)
        else:
            sp = tl.zeros((BK, BV), tl.float32)
        ds += q0[:, None] * scale * do[None, :]
        read = tl.sum(a0[:, None] * sp, axis=0)
        dr = tl.sum(b0[:, None] * ds, axis=0)
        dq0 = tl.sum(sn * do[None, :], axis=1) * scale
        dk0 = tl.sum(ds * v0[None, :], axis=1)
        da0 = tl.sum(sp * dr[None, :], axis=1)
        db0 = tl.sum(ds * read[None, :], axis=1)
        dg0 = tl.sum(ds * sp * e0[:, None], axis=1)
        dvo = tl.sum(ds * k0[:, None], axis=0)
        tl.store(dv + ((ib * T + t) * H + ih) * V + vv, dvo, mask=mv)
        part = iv * (tl.num_programs(2) * T * K) + ((ib * T + t) * H + ih) * K + kk
        tl.store(pq + part, dq0, mask=mk)
        tl.store(pk + part, dk0, mask=mk)
        tl.store(pa + part, da0, mask=mk)
        tl.store(pb + part, db0, mask=mk)
        tl.store(pg + part, dg0, mask=mk)
        ds = ds * e0[:, None] + a0[:, None] * dr[None, :]


@triton.jit
def _reduce_kernel(pq, pk, pa, pb, pg, dq, dk, da, db, dg,
                   N: tl.constexpr, NV: tl.constexpr):
    offs = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offs < N
    xq = tl.zeros((256,), tl.float32)
    xk = tl.zeros((256,), tl.float32)
    xa = tl.zeros((256,), tl.float32)
    xb = tl.zeros((256,), tl.float32)
    xg = tl.zeros((256,), tl.float32)
    for iv in range(NV):
        p = iv * N + offs
        xq += tl.load(pq + p, mask=mask, other=0.0)
        xk += tl.load(pk + p, mask=mask, other=0.0)
        xa += tl.load(pa + p, mask=mask, other=0.0)
        xb += tl.load(pb + p, mask=mask, other=0.0)
        xg += tl.load(pg + p, mask=mask, other=0.0)
    tl.store(dq + offs, xq, mask=mask)
    tl.store(dk + offs, xk, mask=mask)
    tl.store(da + offs, xa, mask=mask)
    tl.store(db + offs, xb, mask=mask)
    tl.store(dg + offs, xg, mask=mask)


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
        checkpoints = torch.empty((B * H, nc, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV,
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
        nv = triton.cdiv(V, _BBV)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BBV,
            num_warps=4, num_stages=2)
        boundaries = torch.empty_like(checkpoints)
        _boundary_adjoint_kernel[(nv, B * H)](
            q, alpha, beta, gk, dout, boundaries,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BBV,
            num_warps=4, num_stages=2)
        n = B * T * H * K
        parts = [torch.empty((nv, n), device=q.device, dtype=torch.float32) for _ in range(5)]
        dv = torch.empty_like(v)
        _backward_kernel[(nv, nc, B * H)](
            q, k, v, alpha, beta, gk, states, dout, boundaries, dv, *parts,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BBV,
            num_warps=4, num_stages=2)
        grads = [torch.empty_like(q) for _ in range(5)]
        _reduce_kernel[(triton.cdiv(n, 256),)](
            *parts, *grads, N=n, NV=nv, num_warps=4)
        dq, dk, da, db, dg = grads
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
