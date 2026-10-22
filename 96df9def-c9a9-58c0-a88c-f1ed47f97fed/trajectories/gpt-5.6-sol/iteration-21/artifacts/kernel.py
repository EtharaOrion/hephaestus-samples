import math

import torch
import triton
import triton.language as tl

_BT = 16
_BV = 32


@triton.jit
def _factor_kernel(q, k, v, a, b, g, qg, kg, bg, w, u, aqk, aqb,
                   T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0)
    ibh = tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ot = ic * 16 + tl.arange(0, 16)
    ok = tl.arange(0, BK)
    mt = ot < T
    mk = ok < K
    basek = (ib * T * H + ot[:, None] * H + ih) * K + ok[None, :]
    xq = tl.load(q + basek, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    xk = tl.load(k + basek, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    xa = tl.load(a + basek, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    xb = tl.load(b + basek, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    xg = tl.load(g + basek, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    gc = tl.cumsum(xg, axis=0)
    last = tl.sum(tl.where((tl.arange(0, 16)[:, None] == tl.minimum(T - ic * 16, 16) - 1) & mk[None, :], gc, 0.), axis=0)
    qtile = xq * tl.exp(gc) * (1.0 / tl.sqrt(K + 0.0))
    ktil = xk * tl.exp(last[None, :] - gc)
    btil = xb * tl.exp(last[None, :] - gc)
    atil = xa * tl.exp(gc - xg)
    tl.store(qg + basek, qtile, mask=mt[:, None] & mk[None, :])
    tl.store(kg + basek, ktil, mask=mt[:, None] & mk[None, :])
    tl.store(bg + basek, btil, mask=mt[:, None] & mk[None, :])

    ii = tl.arange(0, 16)[:, None, None]
    jj = tl.arange(0, 16)[None, :, None]
    gg = tl.exp(gc[:, None, :] - gc[None, :, :])
    valid = mt[:, None, None] & mt[None, :, None] & mk[None, None, :]
    lower_eq = ii >= jj
    lower = ii > jj
    m_qk = tl.sum(tl.where(valid & lower_eq, xq[:, None, :] * xk[None, :, :] * gg, 0.), axis=2) * (1.0 / tl.sqrt(K + 0.0))
    m_qb = tl.sum(tl.where(valid & lower_eq, xq[:, None, :] * xb[None, :, :] * gg, 0.), axis=2) * (1.0 / tl.sqrt(K + 0.0))
    ga = tl.exp((gc - xg)[:, None, :] - gc[None, :, :])
    lab = tl.sum(tl.where(valid & lower, xa[:, None, :] * xb[None, :, :] * ga, 0.), axis=2)
    lak = tl.sum(tl.where(valid & lower, xa[:, None, :] * xk[None, :, :] * ga, 0.), axis=2)

    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    inv = tl.where(rows == cols, 1.0, 0.0) + lab
    power = lab
    for _ in tl.static_range(0, 3):
        power = tl.dot(power, power, input_precision="ieee")
        inv += tl.dot(power, inv, input_precision="ieee")

    aa = tl.dot(inv, lak, input_precision="ieee")
    ww = tl.dot(inv, atil, input_precision="ieee")
    ov = tl.arange(0, 16)
    for iv in range(tl.cdiv(V, 16)):
        vv = iv * 16 + ov
        basev = (ib * T * H + ot[:, None] * H + ih) * V + vv[None, :]
        xv = tl.load(v + basev, mask=mt[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
        uu = tl.dot(aa, xv, input_precision="ieee")
        tl.store(u + basev, uu, mask=mt[:, None] & (vv[None, :] < V))
    tl.store(w + basek, ww, mask=mt[:, None] & mk[None, :])
    ma = mt[:, None] & mt[None, :]
    abase = (ib * tl.cdiv(T, 16) * H + ic * H + ih) * 256
    oo = tl.arange(0, 16)
    offs = oo[:, None] * 16 + oo[None, :]
    tl.store(aqk + abase + offs, m_qk, mask=ma)
    tl.store(aqb + abase + offs, m_qb, mask=ma)


@triton.jit
def _state_kernel(kg, bg, v, w, u, g, h, vn, T: tl.constexpr,
                  H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BK: tl.constexpr, BV: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(tl.cdiv(T, 16)):
        ot = ic * 16 + tl.arange(0, 16)
        mt = ot < T
        ph = h + ((ib * tl.cdiv(T, 16) + ic) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
        tl.store(ph, state, mask=(ok[:, None] < K) & (ov[None, :] < V))
        pk = (ib * T * H + ot[:, None] * H + ih) * K + ok[None, :]
        pv = (ib * T * H + ot[:, None] * H + ih) * V + ov[None, :]
        xk = tl.load(kg + pk, mask=mt[:, None] & (ok[None, :] < K), other=0.)
        xb = tl.load(bg + pk, mask=mt[:, None] & (ok[None, :] < K), other=0.)
        xw = tl.load(w + pk, mask=mt[:, None] & (ok[None, :] < K), other=0.)
        xv = tl.load(v + pv, mask=mt[:, None] & (ov[None, :] < V), other=0.).to(tl.float32)
        xu = tl.load(u + pv, mask=mt[:, None] & (ov[None, :] < V), other=0.).to(tl.float32)
        xvn = tl.dot(xw, state, input_precision="ieee") + xu
        tl.store(vn + pv, xvn, mask=mt[:, None] & (ov[None, :] < V))
        add = tl.dot(tl.trans(xk), xv, input_precision="ieee") + tl.dot(tl.trans(xb), xvn, input_precision="ieee")
        pg = (ib * T * H + ot[:, None] * H + ih) * K + ok[None, :]
        gates = tl.load(g + pg, mask=mt[:, None] & (ok[None, :] < K), other=0.).to(tl.float32)
        decay = tl.exp(tl.sum(gates, axis=0))
        state = state * decay[:, None] + add


@triton.jit
def _output_kernel(qg, v, vn, aqk, aqb, h, out, T: tl.constexpr,
                   H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    ot, ok = ic * 16 + tl.arange(0, 16), tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mt = ot < T
    pk = (ib * T * H + ot[:, None] * H + ih) * K + ok[None, :]
    ph = h + ((ib * tl.cdiv(T, 16) + ic) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
    xq = tl.load(qg + pk, mask=mt[:, None] & (ok[None, :] < K), other=0.)
    xs = tl.load(ph, mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.)
    z = tl.dot(xq, xs, input_precision="ieee")
    oo = tl.arange(0, 16)
    abase = (ib * tl.cdiv(T, 16) * H + ic * H + ih) * 256
    amask = mt[:, None] & ((ic * 16 + oo[None, :]) < T)
    m1 = tl.load(aqk + abase + oo[:, None] * 16 + oo[None, :], mask=amask, other=0.)
    m2 = tl.load(aqb + abase + oo[:, None] * 16 + oo[None, :], mask=amask, other=0.)
    pv = (ib * T * H + ot[:, None] * H + ih) * V + ov[None, :]
    xv = tl.load(v + pv, mask=mt[:, None] & (ov[None, :] < V), other=0.).to(tl.float32)
    xn = tl.load(vn + pv, mask=mt[:, None] & (ov[None, :] < V), other=0.).to(tl.float32)
    z += tl.dot(m1, xv, input_precision="ieee") + tl.dot(m2, xn, input_precision="ieee")
    tl.store(out + pv, z, mask=mt[:, None] & (ov[None, :] < V))


@triton.jit
def _expand_kernel(k, v, a, b, g, states, T: tl.constexpr,
                   H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   BK: tl.constexpr, BV: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    s = tl.zeros((BK, BV), tl.float32)
    for t in range(T):
        ps = states + (ib * T * H + t * H + ih) * K * V + ok[:, None] * V + ov[None, :]
        tl.store(ps, s, mask=(ok[:, None] < K) & (ov[None, :] < V))
        pk = (ib * T * H + t * H + ih) * K + ok
        pv = (ib * T * H + t * H + ih) * V + ov
        xa = tl.load(a + pk, mask=ok < K, other=0.).to(tl.float32)
        xb = tl.load(b + pk, mask=ok < K, other=0.).to(tl.float32)
        xk = tl.load(k + pk, mask=ok < K, other=0.).to(tl.float32)
        xg = tl.load(g + pk, mask=ok < K, other=0.).to(tl.float32)
        xv = tl.load(v + pv, mask=ov < V, other=0.).to(tl.float32)
        read = tl.sum(s * xa[:, None], axis=0)
        s = s * tl.exp(xg)[:, None] + xk[:, None] * xv[None, :] + xb[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, dv, partial,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                     NV: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    ds = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(K + 0.0)
    for rev in range(T):
        t = T - 1 - rev
        pk = (ib * T * H + t * H + ih) * K + ok
        pv = (ib * T * H + t * H + ih) * V + ov
        ps = states + (ib * T * H + t * H + ih) * K * V + ok[:, None] * V + ov[None, :]
        sold = tl.load(ps, mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.).to(tl.float32)
        xq = tl.load(q + pk, mask=ok < K, other=0.).to(tl.float32)
        xk = tl.load(k + pk, mask=ok < K, other=0.).to(tl.float32)
        xa = tl.load(a + pk, mask=ok < K, other=0.).to(tl.float32)
        xb = tl.load(b + pk, mask=ok < K, other=0.).to(tl.float32)
        xg = tl.load(g + pk, mask=ok < K, other=0.).to(tl.float32)
        xv = tl.load(v + pv, mask=ov < V, other=0.).to(tl.float32)
        xd = tl.load(do + pv, mask=ov < V, other=0.).to(tl.float32)
        decay = tl.exp(xg)
        read = tl.sum(sold * xa[:, None], axis=0)
        snew = sold * decay[:, None] + xk[:, None] * xv[None, :] + xb[:, None] * read[None, :]
        ds += xq[:, None] * (scale * xd[None, :])
        dq = tl.sum(snew * xd[None, :], axis=1) * scale
        dk = tl.sum(ds * xv[None, :], axis=1)
        dvv = tl.sum(ds * xk[:, None], axis=0)
        dbb = tl.sum(ds * read[None, :], axis=1)
        dr = tl.sum(ds * xb[:, None], axis=0)
        daa = tl.sum(sold * dr[None, :], axis=1)
        dgg = tl.sum(ds * sold * decay[:, None], axis=1)
        n = (ib * T * H + t * H + ih)
        pp = partial + ((iv * (T * H * tl.num_programs(1) // H) + n) * 5) * K + ok
        tl.store(pp + 0 * K, dq, mask=ok < K)
        tl.store(pp + 1 * K, dk, mask=ok < K)
        tl.store(pp + 2 * K, daa, mask=ok < K)
        tl.store(pp + 3 * K, dbb, mask=ok < K)
        tl.store(pp + 4 * K, dgg, mask=ok < K)
        tl.store(dv + pv, dvv, mask=ov < V)
        ds = ds * decay[:, None] + xa[:, None] * dr[None, :]


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr,
                   K: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * K
    n, kk = offs // K, offs % K
    vals0 = tl.zeros((BLOCK,), tl.float32)
    vals1 = tl.zeros((BLOCK,), tl.float32)
    vals2 = tl.zeros((BLOCK,), tl.float32)
    vals3 = tl.zeros((BLOCK,), tl.float32)
    vals4 = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        p = partial + (iv * N + n) * 5 * K + kk
        vals0 += tl.load(p + 0 * K, mask=mask, other=0.)
        vals1 += tl.load(p + 1 * K, mask=mask, other=0.)
        vals2 += tl.load(p + 2 * K, mask=mask, other=0.)
        vals3 += tl.load(p + 3 * K, mask=mask, other=0.)
        vals4 += tl.load(p + 4 * K, mask=mask, other=0.)
    tl.store(dq + offs, vals0, mask=mask)
    tl.store(dk + offs, vals1, mask=mask)
    tl.store(da + offs, vals2, mask=mask)
    tl.store(db + offs, vals3, mask=mask)
    tl.store(dg + offs, vals4, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nc, nv = triton.cdiv(T, _BT), triton.cdiv(V, _BV)
        qg, kg, bg, w = [
            torch.empty(q.shape, device=q.device, dtype=torch.float32) for _ in range(4)
        ]
        u = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        vn = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        aqk = torch.empty((B, nc, H, 16, 16), device=q.device, dtype=torch.float32)
        aqb = torch.empty_like(aqk)
        h = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _factor_kernel[(nc, B * H)](q, k, v, alpha, beta, gk, qg, kg, bg, w, u, aqk, aqb,
            T=T, H=H, K=K, V=V, BK=bk, num_warps=8, num_stages=2)
        _state_kernel[(nv, B * H)](kg, bg, v, w, u, gk, h, vn,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BV, num_warps=8, num_stages=2)
        _output_kernel[(nv, nc, B * H)](qg, v, vn, aqk, aqb, h, out,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BV, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, h)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, a, b, g, h = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        vpow = triton.next_power_of_2(V)
        bv = 16 if V == vpow else max(16, vpow)
        nc, nv = triton.cdiv(T, 16), triton.cdiv(V, bv)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_kernel[(nv, B * H)](k, v, a, b, g, states,
            T=T, H=H, K=K, V=V, BK=bk, BV=bv, num_warps=4, num_stages=2)
        partial = torch.zeros((nv, B * T * H, 5, K), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _backward_kernel[(nv, B * H)](q, k, v, a, b, g, do.contiguous(), states, dv, partial,
            T=T, H=H, K=K, V=V, BK=bk, BV=bv, NV=nv, num_warps=8, num_stages=2)
        dq, dk, da, db = [torch.empty_like(q) for _ in range(4)]
        dg = torch.empty_like(g)
        n = B * T * H
        _reduce_kernel[(triton.cdiv(n * K, 256),)](partial, dq, dk, da, db, dg,
            N=n, K=K, NV=nv, BLOCK=256, num_warps=8)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
