import math

import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _build_wy(k, g, beta, inv, gc, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    c = tl.program_id(0)
    z = tl.program_id(1)
    b = z // HV
    hv = z % HV
    h = hv // (HV // H)
    i = tl.arange(0, BC)
    d = tl.arange(0, BK)
    t = c * BC + i
    valid = t < T
    gg = tl.load(g + (b * T + t) * HV + hv, mask=valid, other=0.).to(tl.float32)
    G = tl.cumsum(gg, axis=0)
    tl.store(gc + (b * T + t) * HV + hv, G, mask=valid)
    kk = tl.load(k + ((b * T + t[:, None]) * H + h) * K + d[None, :],
                 mask=valid[:, None] & (d[None, :] < K), other=0.).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    j = tl.arange(0, BC)
    bb = tl.load(beta + (b * T + t) * HV + hv, mask=valid, other=0.).to(tl.float32)
    n = tl.where((i[:, None] > j[None, :]) & valid[:, None] & valid[None, :],
                 bb[:, None] * tl.exp(G[:, None] - G[None, :]) * gram, 0.)
    a = tl.where(i[:, None] == j[None, :], 1., 0.)
    for r in range(1, BC):
        coeff = tl.sum(tl.where(i[:, None] == r, n, 0.), axis=0)
        coeff = tl.where(i < r, coeff, 0.)
        row = -tl.sum(coeff[:, None] * a, axis=0)
        row = tl.where(i == r, 1., row)
        a = tl.where(i[:, None] == r, row[None, :], a)
    base = ((b * HV + hv) * tl.cdiv(T, BC) + c) * BC * BC
    tl.store(inv + base + i[:, None] * BC + j[None, :], a,
             mask=valid[:, None] & valid[None, :])


@triton.jit
def _scan_state(k, v, beta, inv, gc, states, xbuf,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    z = p // nv
    vb = p % nv
    b = z // HV
    hv = z % HV
    h = hv // (HV // H)
    rk = tl.arange(0, BK)
    rv = vb * BV + tl.arange(0, BV)
    ri = tl.arange(0, BC)
    s = tl.zeros((BK, BV), tl.float32)
    nc = tl.cdiv(T, BC)
    for c in range(0, nc):
        sb = ((((b * HV + hv) * nc + c) * K + rk[:, None]) * V + rv[None, :]).to(tl.int64)
        tl.store(states + sb, s, mask=(rk[:, None] < K) & (rv[None, :] < V))
        t = c * BC + ri
        vm = t < T
        kk = tl.load(k + ((b * T + t[:, None]) * H + h) * K + rk[None, :],
                     mask=vm[:, None] & (rk[None, :] < K), other=0.).to(tl.float32)
        vv = tl.load(v + ((b * T + t[:, None]) * HV + hv) * V + rv[None, :],
                     mask=vm[:, None] & (rv[None, :] < V), other=0.).to(tl.float32)
        bb = tl.load(beta + (b * T + t) * HV + hv, mask=vm, other=0.).to(tl.float32)
        G = tl.load(gc + (b * T + t) * HV + hv, mask=vm, other=0.)
        pred = tl.dot(kk, s, input_precision="tf32")
        rhs = bb[:, None] * (vv - tl.exp(G)[:, None] * pred)
        ia = ((b * HV + hv) * nc + c) * BC * BC + ri[:, None] * BC + ri[None, :]
        aa = tl.load(inv + ia, mask=vm[:, None] & vm[None, :], other=0.)
        xx = tl.dot(aa, rhs, input_precision="tf32")
        xp = ((b * T + t[:, None]) * HV + hv) * V + rv[None, :]
        tl.store(xbuf + xp, xx, mask=vm[:, None] & (rv[None, :] < V))
        last = tl.sum(vm.to(tl.int32), axis=0) - 1
        ge = tl.sum(tl.where(ri == last, G, 0.), axis=0)
        weights = tl.exp(ge - G) * vm
        s = tl.exp(ge) * s + tl.dot(tl.trans(kk), weights[:, None] * xx, input_precision="tf32")


@triton.jit
def _emit(q, k, gc, states, xbuf, out, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
          SCALE: tl.constexpr):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    c = (p // nv) % tl.cdiv(T, BC)
    z = p // (nv * tl.cdiv(T, BC))
    vb = p % nv
    b = z // HV
    hv = z % HV
    h = hv // (HV // H)
    ri = tl.arange(0, BC)
    rk = tl.arange(0, BK)
    rv = vb * BV + tl.arange(0, BV)
    t = c * BC + ri
    vm = t < T
    qq = tl.load(q + ((b * T + t[:, None]) * H + h) * K + rk[None, :],
                 mask=vm[:, None] & (rk[None, :] < K), other=0.).to(tl.float32) * SCALE
    kk = tl.load(k + ((b * T + t[:, None]) * H + h) * K + rk[None, :],
                 mask=vm[:, None] & (rk[None, :] < K), other=0.).to(tl.float32)
    G = tl.load(gc + (b * T + t) * HV + hv, mask=vm, other=0.)
    nc = tl.cdiv(T, BC)
    sp = ((((b * HV + hv) * nc + c) * K + rk[:, None]) * V + rv[None, :]).to(tl.int64)
    s = tl.load(states + sp, mask=(rk[:, None] < K) & (rv[None, :] < V), other=0.)
    xp = ((b * T + t[:, None]) * HV + hv) * V + rv[None, :]
    xx = tl.load(xbuf + xp, mask=vm[:, None] & (rv[None, :] < V), other=0.)
    dots = tl.dot(qq, tl.trans(kk), input_precision="tf32")
    ratio = tl.exp(G[:, None] - G[None, :])
    m = tl.where((ri[:, None] >= ri[None, :]) & vm[:, None] & vm[None, :], dots * ratio, 0.)
    oo = tl.exp(G)[:, None] * tl.dot(qq, s, input_precision="tf32") + tl.dot(m, xx, input_precision="tf32")
    op = ((b * T + t[:, None]) * HV + hv) * V + rv[None, :]
    tl.store(out + op, oo, mask=vm[:, None] & (rv[None, :] < V))


@triton.jit
def _adjoint(q, k, beta, inv, gc, dout, dstates, drbuf,
             T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
             SCALE: tl.constexpr):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    z = p // nv
    vb = p % nv
    b = z // HV
    hv = z % HV
    h = hv // (HV // H)
    rk = tl.arange(0, BK)
    rv = vb * BV + tl.arange(0, BV)
    ri = tl.arange(0, BC)
    ds = tl.zeros((BK, BV), tl.float32)
    nc = tl.cdiv(T, BC)
    for cc in range(0, nc):
        c = nc - 1 - cc
        ep = ((((b * HV + hv) * nc + c) * K + rk[:, None]) * V + rv[None, :]).to(tl.int64)
        tl.store(dstates + ep, ds, mask=(rk[:, None] < K) & (rv[None, :] < V))
        t = c * BC + ri
        vm = t < T
        qq = tl.load(q + ((b * T + t[:, None]) * H + h) * K + rk[None, :],
                     mask=vm[:, None] & (rk[None, :] < K), other=0.).to(tl.float32) * SCALE
        kk = tl.load(k + ((b * T + t[:, None]) * H + h) * K + rk[None, :],
                     mask=vm[:, None] & (rk[None, :] < K), other=0.).to(tl.float32)
        dd = tl.load(dout + ((b * T + t[:, None]) * HV + hv) * V + rv[None, :],
                     mask=vm[:, None] & (rv[None, :] < V), other=0.).to(tl.float32)
        G = tl.load(gc + (b * T + t) * HV + hv, mask=vm, other=0.)
        bb = tl.load(beta + (b * T + t) * HV + hv, mask=vm, other=0.).to(tl.float32)
        dots = tl.dot(qq, tl.trans(kk), input_precision="tf32")
        ratio = tl.exp(G[:, None] - G[None, :])
        m = tl.where((ri[:, None] >= ri[None, :]) & vm[:, None] & vm[None, :], dots * ratio, 0.)
        dx = tl.dot(tl.trans(m), dd, input_precision="tf32")
        last = tl.sum(vm.to(tl.int32), axis=0) - 1
        ge = tl.sum(tl.where(ri == last, G, 0.), axis=0)
        w = tl.exp(ge - G) * vm
        dx += tl.dot(kk, ds, input_precision="tf32") * w[:, None]
        ia = ((b * HV + hv) * nc + c) * BC * BC + ri[:, None] * BC + ri[None, :]
        aa = tl.load(inv + ia, mask=vm[:, None] & vm[None, :], other=0.)
        dr = tl.dot(tl.trans(aa), dx, input_precision="tf32")
        rp = ((b * T + t[:, None]) * HV + hv) * V + rv[None, :]
        tl.store(drbuf + rp, dr, mask=vm[:, None] & (rv[None, :] < V))
        ds = tl.exp(ge) * ds + tl.dot(tl.trans(qq), tl.exp(G)[:, None] * dd, input_precision="tf32")
        ds -= tl.dot(tl.trans(kk), (bb * tl.exp(G))[:, None] * dr, input_precision="tf32")


@triton.jit
def _chunk_grad(q, k, v, beta, gc, states, dstates, xbuf, drbuf, dout,
                pq, pk, pv, pg, pb, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                BK: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    vb = p % nv
    c = (p // nv) % tl.cdiv(T, BC)
    z = p // (nv * tl.cdiv(T, BC))
    b = z // HV
    hv = z % HV
    h = hv // (HV // H)
    ri = tl.arange(0, BC)
    rj = tl.arange(0, BC)
    rk = tl.arange(0, BK)
    rv = vb * BV + tl.arange(0, BV)
    t = c * BC + ri
    vm = t < T
    qq = tl.load(q + ((b * T + t[:, None]) * H + h) * K + rk[None, :], mask=vm[:, None] & (rk[None, :] < K), other=0.).to(tl.float32) * SCALE
    kk = tl.load(k + ((b * T + t[:, None]) * H + h) * K + rk[None, :], mask=vm[:, None] & (rk[None, :] < K), other=0.).to(tl.float32)
    vv = tl.load(v + ((b * T + t[:, None]) * HV + hv) * V + rv[None, :], mask=vm[:, None] & (rv[None, :] < V), other=0.).to(tl.float32)
    dd = tl.load(dout + ((b * T + t[:, None]) * HV + hv) * V + rv[None, :], mask=vm[:, None] & (rv[None, :] < V), other=0.).to(tl.float32)
    xx = tl.load(xbuf + ((b * T + t[:, None]) * HV + hv) * V + rv[None, :], mask=vm[:, None] & (rv[None, :] < V), other=0.)
    dr = tl.load(drbuf + ((b * T + t[:, None]) * HV + hv) * V + rv[None, :], mask=vm[:, None] & (rv[None, :] < V), other=0.)
    G = tl.load(gc + (b * T + t) * HV + hv, mask=vm, other=0.)
    bb = tl.load(beta + (b * T + t) * HV + hv, mask=vm, other=0.).to(tl.float32)
    nc = tl.cdiv(T, BC)
    sp = ((((b * HV + hv) * nc + c) * K + rk[:, None]) * V + rv[None, :]).to(tl.int64)
    s = tl.load(states + sp, mask=(rk[:, None] < K) & (rv[None, :] < V), other=0.)
    ds1 = tl.load(dstates + sp, mask=(rk[:, None] < K) & (rv[None, :] < V), other=0.)
    E = tl.exp(G)
    pred = tl.dot(kk, s, input_precision="tf32")
    # Output-state and triangular attention paths.
    dq = E[:, None] * tl.dot(dd, tl.trans(s), input_precision="tf32")
    dm = tl.dot(dd, tl.trans(xx), input_precision="tf32")
    ratio = tl.exp(G[:, None] - G[None, :])
    tri = (ri[:, None] >= rj[None, :]) & vm[:, None] & vm[None, :]
    zdm = tl.where(tri, dm * ratio, 0.)
    dq += tl.dot(zdm, kk, input_precision="tf32")
    dk = tl.dot(tl.trans(zdm), qq, input_precision="tf32")
    dg = E * tl.sum(tl.dot(qq, s, input_precision="tf32") * dd, axis=1)
    gate_pair = zdm * tl.dot(qq, tl.trans(kk), input_precision="tf32")
    dg += tl.sum(gate_pair, axis=1) - tl.sum(gate_pair, axis=0)
    # End-state transition and its decayed keys.
    last = tl.sum(vm.to(tl.int32), axis=0) - 1
    ge = tl.sum(tl.where(ri == last, G, 0.), axis=0)
    ew = tl.exp(ge - G) * vm
    dkb = tl.dot(xx, tl.trans(ds1), input_precision="tf32")
    dk += ew[:, None] * dkb
    end_term = ew * tl.sum(dkb * kk, axis=1)
    dg -= end_term
    state_dot = tl.sum(tl.sum(ds1 * s, axis=1), axis=0)
    dg += tl.where(ri == tl.sum(vm.to(tl.int32), axis=0) - 1,
                   tl.sum(end_term, axis=0) + tl.exp(ge) * state_dot, 0.)
    # Solved triangular system and right-hand side.
    dp = -(bb * E)[:, None] * dr
    dk += tl.dot(dp, tl.trans(s), input_precision="tf32")
    db = tl.sum(dr * (vv - E[:, None] * pred), axis=1)
    dg += -bb * E * tl.sum(dr * pred, axis=1)
    dl = -tl.dot(dr, tl.trans(xx), input_precision="tf32")
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    strict = (ri[:, None] > rj[None, :]) & vm[:, None] & vm[None, :]
    wl = tl.where(strict, dl * bb[:, None] * ratio, 0.)
    db += tl.sum(tl.where(strict, dl * ratio * gram, 0.), axis=1)
    dg_pair = wl * gram
    dg += tl.sum(dg_pair, axis=1) - tl.sum(dg_pair, axis=0)
    dk += tl.dot(wl + tl.trans(wl), kk, input_precision="tf32")
    dv = bb[:, None] * dr
    base = ((((b * T + t) * HV + hv) * nv + vb)).to(tl.int64)
    tl.store(pq + base[:, None] * K + rk[None, :], dq * SCALE, mask=vm[:, None] & (rk[None, :] < K))
    tl.store(pk + base[:, None] * K + rk[None, :], dk, mask=vm[:, None] & (rk[None, :] < K))
    tl.store(pv + ((b * T + t[:, None]) * HV + hv) * V + rv[None, :], dv, mask=vm[:, None] & (rv[None, :] < V))
    tl.store(pg + base, dg, mask=vm)
    tl.store(pb + base, db, mask=vm)


@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
               NV: tl.constexpr):
    p = tl.program_id(0)
    d = tl.arange(0, BK)
    n = p // H
    h = p % H
    b = n // T
    t = n % T
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    rep = HV // H
    for r in range(0, rep):
        hv = h * rep + r
        for vb in range(0, NV):
            base = ((((b * T + t) * HV + hv) * NV + vb) * K).to(tl.int64)
            aq += tl.load(pq + base + d, mask=d < K, other=0.)
            ak += tl.load(pk + base + d, mask=d < K, other=0.)
    out = ((b * T + t) * H + h) * K + d
    tl.store(dq + out, aq, mask=d < K)
    tl.store(dk + out, ak, mask=d < K)


@triton.jit
def _reduce_scalar(part, out, T: tl.constexpr, HV: tl.constexpr,
                   NV: tl.constexpr, SUFFIX: tl.constexpr):
    z = tl.program_id(0)
    b = z // HV
    hv = z % HV
    carry = 0.0
    for tt in range(0, T):
        t = T - 1 - tt if SUFFIX else tt
        x = 0.0
        for vb in range(0, NV):
            x += tl.load(part + (((b * T + t) * HV + hv) * NV + vb))
        if SUFFIX:
            carry += x
            x = carry
            if t % BC == 0:
                carry = 0.0
        tl.store(out + (b * T + t) * HV + hv, x)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        BK = triton.next_power_of_2(K)
        nc, nv = triton.cdiv(T, BC), triton.cdiv(V, BV)
        gc = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, HV, nc, BC, BC), device=q.device, dtype=torch.float32)
        states = torch.empty((B, HV, nc, K, V), device=q.device, dtype=torch.float32)
        xbuf = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(nc, B * HV)](k, g, beta, inv, gc, T=T, H=H, HV=HV, K=K, BK=BK)
        _scan_state[(B * HV * nv,)](k, v, beta, inv, gc, states, xbuf, T=T, H=H, HV=HV, K=K, V=V, BK=BK)
        _emit[(B * HV * nc * nv,)](q, k, gc, states, xbuf, out, T=T, H=H, HV=HV, K=K, V=V, BK=BK, SCALE=K ** -0.5)
        ctx.save_for_backward(q, k, v, g, beta, inv, gc, states, xbuf)
        ctx.dims = (B, T, H, HV, K, V, BK, nc, nv)
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        q, k, v, g, beta, inv, gc, states, xbuf = ctx.saved_tensors
        B, T, H, HV, K, V, BK, nc, nv = ctx.dims
        dstates = torch.empty_like(states, dtype=torch.float32)
        dr = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        pq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq, dtype=torch.float32)
        pv = torch.empty_like(v)
        pg = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg, dtype=torch.float32)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        _adjoint[(B * HV * nv,)](q, k, beta, inv, gc, dout, dstates, dr, T=T, H=H, HV=HV, K=K, V=V, BK=BK, SCALE=K ** -0.5)
        _chunk_grad[(B * HV * nc * nv,)](q, k, v, beta, gc, states, dstates, xbuf, dr, dout, pq, pk, pv, pg, pb, T=T, H=H, HV=HV, K=K, V=V, BK=BK, SCALE=K ** -0.5)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T=T, H=H, HV=HV, K=K, BK=BK, NV=nv)
        _reduce_scalar[(B * HV,)](pg, dg, T=T, HV=HV, NV=nv, SUFFIX=True)
        _reduce_scalar[(B * HV,)](pb, db, T=T, HV=HV, NV=nv, SUFFIX=False)
        return dq, dk, pv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
