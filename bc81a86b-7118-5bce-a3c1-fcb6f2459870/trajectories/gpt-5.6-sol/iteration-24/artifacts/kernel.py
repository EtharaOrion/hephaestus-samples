import torch
import triton
import triton.language as tl

BC = 16
BV = 32

@triton.jit
def _mm(a, b):
    return tl.dot(a, b, input_precision="tf32")

@triton.jit
def _build_wy(k, g, beta, gc, inv, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, C: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ih = tl.program_id(1).to(tl.int64)
    ib, ivh = ih // HV, ih % HV
    ikh = ivh // (HV // H)
    t, xk = ic * C + tl.arange(0, C), tl.arange(0, BK)
    mt = t < T
    kp = ((ib * T + t[:, None]) * H + ikh) * K + xk[None, :]
    kk = tl.load(k + kp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + t) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(beta + (ib * T + t) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    ii = tl.arange(0, C)
    lower = ii[:, None] > ii[None, :]
    ed = tl.exp(tl.maximum(tl.minimum(cs[:, None] - cs[None, :], 0.0), -80.0))
    a = tl.where(mt[:, None] & mt[None, :] & lower, _mm(kk, tl.trans(kk)) * bb[:, None] * ed, 0.0)
    r = tl.where(ii[:, None] == ii[None, :], 1.0, 0.0).to(tl.float32)
    for row in range(1, C):
        ar = tl.sum(tl.where(ii[:, None] == row, a, 0.0), axis=0)
        rr = -tl.sum(ar[:, None] * r, axis=0) + (ii == row).to(tl.float32)
        r = tl.where(ii[:, None] == row, rr[None, :], r)
    base = (ih * NC + ic) * C * C
    tl.store(inv + base + ii[:, None] * C + ii[None, :], r)
    tl.store(gc + (ib * T + t) * HV + ivh, cs, mask=mt)

@triton.jit
def _scan_state(k, v, beta, gc, inv, states, updates, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr, C: tl.constexpr, VB: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, ih = p % NV, p // NV
    ib, ivh = ih // HV, ih % HV
    ikh = ivh // (HV // H)
    xk, xv, ci = tl.arange(0, BK), iv * VB + tl.arange(0, VB), tl.arange(0, C)
    state = tl.zeros((BK, VB), tl.float32)
    for ic in range(NC):
        tt, mt = ic * C + ci, ic * C + ci < T
        sp = ((ih * NC + ic) * K + xk[:, None]) * V + xv[None, :]
        tl.store(states + sp, state, mask=(xk[:, None] < K) & (xv[None, :] < V))
        kp = ((ib * T + tt[:, None]) * H + ikh) * K + xk[None, :]
        vp = ((ib * T + tt[:, None]) * HV + ivh) * V + xv[None, :]
        kb = tl.load(k + kp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
        vv = tl.load(v + vp, mask=mt[:, None] & (xv[None, :] < V), other=0.0)
        bb = tl.load(beta + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
        gg = tl.load(gc + (ib * T + tt) * HV + ivh, mask=mt, other=0.0)
        ip = ((ih * NC + ic) * C + ci[:, None]) * C + ci[None, :]
        ri = tl.load(inv + ip)
        eg = tl.exp(tl.maximum(gg, -80.0))
        z = vv.to(tl.float32) * bb[:, None]
        pp = kb.to(tl.float32) * (bb * eg)[:, None]
        u = _mm(ri, z - _mm(pp, state))
        tl.store(updates + vp, u, mask=mt[:, None] & (xv[None, :] < V))
        nvalid = tl.sum(mt.to(tl.int32), axis=0)
        last = tl.sum(tl.where(ci == nvalid - 1, gg, 0.0), axis=0)
        fk = kb.to(tl.float32) * tl.exp(tl.maximum(tl.minimum(last - gg, 0.0), -80.0))[:, None]
        state = state * tl.exp(tl.maximum(last, -80.0)) + _mm(tl.trans(fk), u)

@triton.jit
def _emit(q, k, gc, states, updates, out, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr, C: tl.constexpr, VB: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, z = p % NV, p // NV
    ic, ih = z % NC, z // NC
    ib, ivh = ih // HV, ih % HV
    ikh = ivh // (HV // H)
    ci, xk, xv = tl.arange(0, C), tl.arange(0, BK), iv * VB + tl.arange(0, VB)
    tt, mt = ic * C + ci, ic * C + ci < T
    qp = ((ib * T + tt[:, None]) * H + ikh) * K + xk[None, :]
    qb = tl.load(q + qp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
    kb = tl.load(k + qp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
    gg = tl.load(gc + (ib * T + tt) * HV + ivh, mask=mt, other=0.0)
    sp = ((ih * NC + ic) * K + xk[:, None]) * V + xv[None, :]
    ss = tl.load(states + sp, mask=(xk[:, None] < K) & (xv[None, :] < V), other=0.0)
    up = ((ib * T + tt[:, None]) * HV + ivh) * V + xv[None, :]
    uu = tl.load(updates + up, mask=mt[:, None] & (xv[None, :] < V), other=0.0)
    ii = tl.arange(0, C)
    ed = tl.exp(tl.maximum(tl.minimum(gg[:, None] - gg[None, :], 0.0), -80.0))
    score = tl.where((ii[:, None] >= ii[None, :]) & mt[:, None] & mt[None, :], _mm(qb, tl.trans(kb)) * SCALE * ed, 0.0)
    oo = _mm(qb * (tl.exp(tl.maximum(gg, -80.0)) * SCALE)[:, None], ss) + _mm(score, uu)
    tl.store(out + up, oo, mask=mt[:, None] & (xv[None, :] < V))

@triton.jit
def _adjoint(q, k, beta, gc, inv, dout, dstates, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr, C: tl.constexpr, VB: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, ih = p % NV, p // NV
    ib, ivh = ih // HV, ih % HV
    ikh = ivh // (HV // H)
    xk, xv, ci = tl.arange(0, BK), iv * VB + tl.arange(0, VB), tl.arange(0, C)
    ds = tl.zeros((BK, VB), tl.float32)
    for jj in range(NC):
        ic = NC - 1 - jj
        tt, mt = ic * C + ci, ic * C + ci < T
        sp = ((ih * NC + ic) * K + xk[:, None]) * V + xv[None, :]
        tl.store(dstates + sp, ds, mask=(xk[:, None] < K) & (xv[None, :] < V))
        qp = ((ib * T + tt[:, None]) * H + ikh) * K + xk[None, :]
        qb = tl.load(q + qp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
        kb = tl.load(k + qp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
        bb = tl.load(beta + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
        gg = tl.load(gc + (ib * T + tt) * HV + ivh, mask=mt, other=0.0)
        vp = ((ib * T + tt[:, None]) * HV + ivh) * V + xv[None, :]
        do = tl.load(dout + vp, mask=mt[:, None] & (xv[None, :] < V), other=0.0).to(tl.float32)
        ip = ((ih * NC + ic) * C + ci[:, None]) * C + ci[None, :]
        ri = tl.load(inv + ip)
        eg = tl.exp(tl.maximum(gg, -80.0))
        pp = kb.to(tl.float32) * (bb * eg)[:, None]
        ii = tl.arange(0, C)
        ed = tl.exp(tl.maximum(tl.minimum(gg[:, None] - gg[None, :], 0.0), -80.0))
        score = tl.where((ii[:, None] >= ii[None, :]) & mt[:, None] & mt[None, :], _mm(qb, tl.trans(kb)) * SCALE * ed, 0.0)
        nvalid = tl.sum(mt.to(tl.int32), axis=0)
        last = tl.sum(tl.where(ci == nvalid - 1, gg, 0.0), axis=0)
        fk = kb.to(tl.float32) * tl.exp(tl.maximum(tl.minimum(last - gg, 0.0), -80.0))[:, None]
        du = _mm(tl.trans(score), do) + _mm(fk, ds)
        dy = _mm(tl.trans(ri), du)
        ds = _mm(tl.trans(qb * (eg * SCALE)[:, None]), do) + ds * tl.exp(tl.maximum(last, -80.0)) - _mm(tl.trans(pp), dy)

@triton.jit
def _chunk_grad(q, k, v, beta, gc, inv, states, updates, dout, dstates, pq, pk, dv, pg, pb, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr, C: tl.constexpr, VB: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, z = p % NV, p // NV
    ic, ih = z % NC, z // NC
    ib, ivh = ih // HV, ih % HV
    ikh = ivh // (HV // H)
    ci, xk, xv = tl.arange(0, C), tl.arange(0, BK), iv * VB + tl.arange(0, VB)
    tt, mt = ic * C + ci, ic * C + ci < T
    qkp = ((ib * T + tt[:, None]) * H + ikh) * K + xk[None, :]
    q0 = tl.load(q + qkp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
    k0 = tl.load(k + qkp, mask=mt[:, None] & (xk[None, :] < K), other=0.0)
    qq, kk = q0.to(tl.float32) * SCALE, k0.to(tl.float32)
    vp = ((ib * T + tt[:, None]) * HV + ivh) * V + xv[None, :]
    vv = tl.load(v + vp, mask=mt[:, None] & (xv[None, :] < V), other=0.0).to(tl.float32)
    uu = tl.load(updates + vp, mask=mt[:, None] & (xv[None, :] < V), other=0.0)
    do = tl.load(dout + vp, mask=mt[:, None] & (xv[None, :] < V), other=0.0).to(tl.float32)
    bb = tl.load(beta + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    gg = tl.load(gc + (ib * T + tt) * HV + ivh, mask=mt, other=0.0)
    sp = ((ih * NC + ic) * K + xk[:, None]) * V + xv[None, :]
    ss = tl.load(states + sp, mask=(xk[:, None] < K) & (xv[None, :] < V), other=0.0)
    ds = tl.load(dstates + sp, mask=(xk[:, None] < K) & (xv[None, :] < V), other=0.0)
    ip = ((ih * NC + ic) * C + ci[:, None]) * C + ci[None, :]
    ri = tl.load(inv + ip)
    eg = tl.exp(tl.maximum(gg, -80.0))
    nvalid = tl.sum(mt.to(tl.int32), axis=0)
    last = tl.sum(tl.where(ci == nvalid - 1, gg, 0.0), axis=0)
    ef = tl.exp(tl.maximum(tl.minimum(last - gg, 0.0), -80.0))
    pp = kk * (bb * eg)[:, None]
    ii = tl.arange(0, C)
    causal, strict = ii[:, None] >= ii[None, :], ii[:, None] > ii[None, :]
    ed = tl.exp(tl.maximum(tl.minimum(gg[:, None] - gg[None, :], 0.0), -80.0))
    score = tl.where(causal & mt[:, None] & mt[None, :], _mm(q0, tl.trans(k0)) * SCALE * ed, 0.0)
    du = _mm(tl.trans(score), do) + _mm(kk * ef[:, None], ds)
    dy = _mm(tl.trans(ri), du)
    dm = tl.where(causal & mt[:, None] & mt[None, :], _mm(do, tl.trans(uu)) * ed, 0.0)
    cross = _mm(do, tl.trans(ss))
    dqv = _mm(dm, kk) + cross * eg[:, None]
    dkv = _mm(tl.trans(dm), qq)
    cm = dm * _mm(qq, tl.trans(kk))
    dg0 = tl.sum(cm, axis=1) - tl.sum(cm, axis=0)
    dg0 += tl.sum(cross * (qq * eg[:, None]), axis=1)
    dp = -_mm(dy, tl.trans(ss))
    dkv += dp * (bb * eg)[:, None]
    db0 = tl.sum(dp * (kk * eg[:, None]), axis=1) + tl.sum(dy * vv, axis=1)
    dg0 += tl.sum(dp * pp, axis=1)
    dvv = dy * bb[:, None]
    dl = tl.where(strict, -_mm(dy, tl.trans(uu)), 0.0)
    gram = _mm(k0, tl.trans(k0))
    abase = tl.where(strict & mt[:, None] & mt[None, :], bb[:, None] * ed, 0.0)
    ca = dl * abase * gram
    dg0 += tl.sum(ca, axis=1) - tl.sum(ca, axis=0)
    db0 += tl.sum(dl * ed * gram * strict, axis=1)
    sym = dl * abase
    dkv += _mm(sym + tl.trans(sym), kk)
    dfk = _mm(uu, tl.trans(ds))
    dkv += dfk * ef[:, None]
    cf = tl.sum(dfk * (kk * ef[:, None]), axis=1)
    dg0 -= cf
    dlast = tl.sum(cf, axis=0) + tl.sum(ds * ss) * tl.exp(tl.maximum(last, -80.0))
    nvalid = tl.sum(mt.to(tl.int32), axis=0)
    dg0 += tl.where(ii == nvalid - 1, dlast, 0.0)
    dgout = tl.flip(tl.cumsum(tl.flip(dg0, 0), axis=0), 0)
    part = (ih * NV + iv) * T + tt
    poff = part[:, None] * K + xk[None, :]
    tl.store(pq + poff, dqv * SCALE, mask=mt[:, None] & (xk[None, :] < K))
    tl.store(pk + poff, dkv, mask=mt[:, None] & (xk[None, :] < K))
    tl.store(dv + vp, dvv, mask=mt[:, None] & (xv[None, :] < V))
    tl.store(pg + part, dgout, mask=mt)
    tl.store(pb + part, db0, mask=mt)

@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    xk = tl.arange(0, BK)
    ikh, z = p % H, p // H
    it, ib = z % T, z // T
    aq, ak = tl.zeros((BK,), tl.float32), tl.zeros((BK,), tl.float32)
    R: tl.constexpr = HV // H
    for r in range(R):
        ih = ib * HV + ikh * R + r
        for iv in range(NV):
            off = ((ih * NV + iv) * T + it) * K + xk
            aq += tl.load(pq + off, mask=xk < K, other=0.0)
            ak += tl.load(pk + off, mask=xk < K, other=0.0)
    out = ((ib * T + it) * H + ikh) * K + xk
    tl.store(dq + out, aq, mask=xk < K)
    tl.store(dk + out, ak, mask=xk < K)

@triton.jit
def _reduce_scalar(pg, pb, dg, db, T: tl.constexpr, HV: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    ivh, z = p % HV, p // HV
    it, ib = z % T, z // T
    ag, ab = 0.0, 0.0
    for iv in range(NV):
        off = ((ib * HV + ivh) * NV + iv) * T + it
        ag += tl.load(pg + off)
        ab += tl.load(pb + off)
    out = (ib * T + it) * HV + ivh
    tl.store(dg + out, ag)
    tl.store(db + out, ab)

class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC, NV, BK = triton.cdiv(T, BC), triton.cdiv(V, BV), triton.next_power_of_2(K)
        gc = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, HV, NC, BC, BC), device=q.device, dtype=torch.float32)
        states = torch.empty((B, HV, NC, K, V), device=q.device, dtype=torch.float32)
        updates = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(NC, B * HV)](k, g, beta, gc, inv, T=T, H=H, HV=HV, K=K, NC=NC, BK=BK, C=BC, num_warps=4)
        _scan_state[(B * HV * NV,)](k, v, beta, gc, inv, states, updates, T=T, H=H, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK, C=BC, VB=BV, num_warps=4)
        _emit[(B * HV * NC * NV,)](q, k, gc, states, updates, out, T=T, H=H, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK, C=BC, VB=BV, SCALE=K ** -0.5, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, gc, inv, states, updates)
        ctx.dims = B, T, H, HV, K, V, NC, NV, BK
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, gc, inv, states, updates = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, HV, K, V, NC, NV, BK = ctx.dims
        dstates = torch.empty_like(states, dtype=torch.float32)
        pq = torch.empty((B * HV, NV, T, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq, dtype=torch.float32)
        pg = torch.empty((B * HV, NV, T), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg, dtype=torch.float32)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        _adjoint[(B * HV * NV,)](q, k, beta, gc, inv, dout, dstates, T=T, H=H, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK, C=BC, VB=BV, SCALE=K ** -0.5, num_warps=4)
        _chunk_grad[(B * HV * NC * NV,)](q, k, v, beta, gc, inv, states, updates, dout, dstates, pq, pk, dv, pg, pb, T=T, H=H, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK, C=BC, VB=BV, SCALE=K ** -0.5, num_warps=4)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T=T, H=H, HV=HV, K=K, NV=NV, BK=BK, num_warps=4)
        _reduce_scalar[(B * T * HV,)](pg, pb, dg, db, T=T, HV=HV, NV=NV, num_warps=1)
        return dq, dk, dv, dg, db

def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
