import math

import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = 32


@triton.jit
def _ex(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(80.0, x)))


@triton.jit
def _build_transform(k, g, beta, cs, inv, T: tl.constexpr, H: tl.constexpr,
                     HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
                     BK: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ihx = tl.program_id(1).to(tl.int64)
    ib = ihx // HV
    ihv = ihx - ib * HV
    ih = ihv // (HV // H)
    r = tl.arange(0, BC)
    d = tl.arange(0, BK)
    t = ic * BC + r
    mt = t < T
    base_k = ((ib * T + t) * H + ih) * K
    kk = tl.load(k + base_k[:, None] + d[None, :], mask=mt[:, None] & (d[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    cc = tl.cumsum(gg, axis=0)
    bb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    lower = r[:, None] > r[None, :]
    nn = tl.where(lower & mt[:, None] & mt[None, :],
                  bb[:, None] * _ex(cc[:, None] - cc[None, :]) * gram, 0.0)
    aa = tl.zeros((BC, BC), tl.float32)
    col = tl.arange(0, BC)
    for i in tl.static_range(0, BC):
        row = tl.where(col == i, 1.0, 0.0)
        for j in tl.static_range(0, i):
            prev = tl.sum(aa * ((r == j)[:, None] & (col[None, :] <= j)), axis=0)
            nij = tl.sum(nn * ((r == i)[:, None] & (col[None, :] == j)))
            row -= nij * prev
        aa += tl.where((r == i)[:, None], row[None, :], 0.0)
    off_c = ((ib * NC + ic) * HV + ihv) * BC + r
    off_a = ((ib * NC + ic) * HV + ihv) * BC * BC + r[:, None] * BC + col[None, :]
    tl.store(cs + off_c, cc, mask=mt)
    tl.store(inv + off_a, aa, mask=mt[:, None] & mt[None, :])


@triton.jit
def _state_scan(k, v, beta, cs, inv, states, updates,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                NV: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv = p % NV
    ihx = p // NV
    ib = ihx // HV
    ihv = ihx - ib * HV
    ih = ihv // (HV // H)
    d = tl.arange(0, BK)
    x = iv * VB + tl.arange(0, VB)
    md = d < K
    mx = x < V
    state = tl.zeros((BK, VB), tl.float32)
    rr = tl.arange(0, BC)
    ccidx = tl.arange(0, BC)
    for ic in range(NC):
        tt = ic * BC + rr
        mt = tt < T
        sb = ((((ib * NC + ic) * HV + ihv) * K + d[:, None]) * V + x[None, :])
        tl.store(states + sb, state, mask=md[:, None] & mx[None, :])
        cb = ((ib * NC + ic) * HV + ihv) * BC
        csum = tl.load(cs + cb + rr, mask=mt, other=0.0)
        ab = ((ib * NC + ic) * HV + ihv) * BC * BC
        aa = tl.load(inv + ab + rr[:, None] * BC + ccidx[None, :],
                     mask=mt[:, None] & mt[None, :], other=0.0)
        b = tl.load(beta + (ib * T + tt) * HV + ihv, mask=mt, other=0.0)
        kb = ((ib * T + tt) * H + ih) * K
        kval = tl.load(k + kb[:, None] + d[None, :], mask=mt[:, None] & md[None, :], other=0.0)
        vb = ((ib * T + tt) * HV + ihv) * V
        vval = tl.load(v + vb[:, None] + x[None, :], mask=mt[:, None] & mx[None, :], other=0.0)
        srcv = (vval * b[:, None]).to(v.dtype.element_ty)
        srck = (kval * (b * _ex(csum))[:, None]).to(k.dtype.element_ty)
        al = aa.to(k.dtype.element_ty)
        uu = tl.dot(al, srcv, input_precision="tf32")
        ww = tl.dot(al, srck, input_precision="tf32")
        pp = uu - tl.dot(ww.to(k.dtype.element_ty), state.to(k.dtype.element_ty), input_precision="tf32")
        ub = ((ib * T + tt) * HV + ihv) * V
        tl.store(updates + ub[:, None] + x[None, :], pp, mask=mt[:, None] & mx[None, :])
        end = tl.sum(csum * (rr == tl.minimum(BC - 1, T - 1 - ic * BC)))
        z = kval * _ex(end - csum)[:, None]
        state = state * _ex(end) + tl.dot(tl.trans(z.to(k.dtype.element_ty)), pp.to(k.dtype.element_ty), input_precision="tf32")


@triton.jit
def _emit(q, k, cs, states, updates, out,
          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
          K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
          NV: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr,
          SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv = p % NV
    p //= NV
    ic = p % NC
    ihx = p // NC
    ib = ihx // HV
    ihv = ihx - ib * HV
    ih = ihv // (HV // H)
    r = tl.arange(0, BC)
    d = tl.arange(0, BK)
    x = iv * VB + tl.arange(0, VB)
    tt = ic * BC + r
    mt, md, mx = tt < T, d < K, x < V
    qb = ((ib * T + tt) * H + ih) * K
    qv = tl.load(q + qb[:, None] + d[None, :], mask=mt[:, None] & md[None, :], other=0.0)
    kv = tl.load(k + qb[:, None] + d[None, :], mask=mt[:, None] & md[None, :], other=0.0)
    cb = ((ib * NC + ic) * HV + ihv) * BC
    c = tl.load(cs + cb + r, mask=mt, other=0.0)
    sb = ((((ib * NC + ic) * HV + ihv) * K + d[:, None]) * V + x[None, :])
    s = tl.load(states + sb, mask=md[:, None] & mx[None, :], other=0.0)
    ub = ((ib * T + tt) * HV + ihv) * V
    u = tl.load(updates + ub[:, None] + x[None, :], mask=mt[:, None] & mx[None, :], other=0.0)
    score = tl.dot(qv, tl.trans(kv), input_precision="tf32") * SCALE
    score *= _ex(c[:, None] - c[None, :])
    score = tl.where((r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :], score, 0.0)
    y = (qv * (_ex(c) * SCALE)[:, None]).to(q.dtype.element_ty)
    o = tl.dot(y, s.to(q.dtype.element_ty), input_precision="tf32")
    o += tl.dot(score.to(q.dtype.element_ty), u, input_precision="tf32")
    ob = ((ib * T + tt) * HV + ihv) * V
    tl.store(out + ob[:, None] + x[None, :], o, mask=mt[:, None] & mx[None, :])


@triton.jit
def _adjoint(q, k, beta, cs, inv, states, updates, dout, dstates,
             T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
             NV: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr,
             SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv = p % NV
    ihx = p // NV
    ib = ihx // HV
    ihv = ihx - ib * HV
    ih = ihv // (HV // H)
    d = tl.arange(0, BK)
    x = iv * VB + tl.arange(0, VB)
    rr = tl.arange(0, BC)
    jj = tl.arange(0, BC)
    md, mx = d < K, x < V
    ds = tl.zeros((BK, VB), tl.float32)
    for rev in range(NC):
        ic = NC - 1 - rev
        tt = ic * BC + rr
        mt = tt < T
        db = ((((ib * NC + ic) * HV + ihv) * K + d[:, None]) * V + x[None, :])
        tl.store(dstates + db, ds, mask=md[:, None] & mx[None, :])
        qb = ((ib * T + tt) * H + ih) * K
        qv = tl.load(q + qb[:, None] + d[None, :], mask=mt[:, None] & md[None, :], other=0.0)
        kv = tl.load(k + qb[:, None] + d[None, :], mask=mt[:, None] & md[None, :], other=0.0)
        cb = ((ib * NC + ic) * HV + ihv) * BC
        c = tl.load(cs + cb + rr, mask=mt, other=0.0)
        end = tl.sum(c * (rr == tl.minimum(BC - 1, T - 1 - ic * BC)))
        ab = ((ib * NC + ic) * HV + ihv) * BC * BC
        aa = tl.load(inv + ab + rr[:, None] * BC + jj[None, :], mask=mt[:, None] & mt[None, :], other=0.0)
        b = tl.load(beta + (ib * T + tt) * HV + ihv, mask=mt, other=0.0)
        w = tl.dot(aa.to(k.dtype.element_ty), (kv * (b * _ex(c))[:, None]).to(k.dtype.element_ty), input_precision="tf32")
        ub = ((ib * T + tt) * HV + ihv) * V
        pp = tl.load(updates + ub[:, None] + x[None, :], mask=mt[:, None] & mx[None, :], other=0.0)
        do = tl.load(dout + ub[:, None] + x[None, :], mask=mt[:, None] & mx[None, :], other=0.0)
        score = tl.dot(qv, tl.trans(kv), input_precision="tf32") * SCALE
        score *= _ex(c[:, None] - c[None, :])
        score = tl.where((rr[:, None] >= jj[None, :]) & mt[:, None] & mt[None, :], score, 0.0)
        z = kv * _ex(end - c)[:, None]
        dp = tl.dot(tl.trans(score.to(do.dtype)), do, input_precision="tf32")
        dp += tl.dot(z.to(k.dtype.element_ty), ds.to(k.dtype.element_ty), input_precision="tf32")
        y = qv * (_ex(c) * SCALE)[:, None]
        ds = tl.dot(tl.trans(y.to(do.dtype)), do, input_precision="tf32") + ds * _ex(end)
        ds -= tl.dot(tl.trans(w.to(k.dtype.element_ty)), dp.to(k.dtype.element_ty), input_precision="tf32")


@triton.jit
def _chunk_grad(q, k, v, g, beta, cs, inv, states, updates, dout, dstates,
                dqpart, dkpart, dv, dg, dbeta,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                NV: tl.constexpr, BK: tl.constexpr, VB: tl.constexpr,
                SCALE: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ihx = tl.program_id(1).to(tl.int64)
    ib = ihx // HV
    ihv = ihx - ib * HV
    ih = ihv // (HV // H)
    r = tl.arange(0, BC)
    j = tl.arange(0, BC)
    d = tl.arange(0, BK)
    tt = ic * BC + r
    mt, md = tt < T, d < K
    qb = ((ib * T + tt) * H + ih) * K
    qv = tl.load(q + qb[:, None] + d[None, :], mask=mt[:, None] & md[None, :], other=0.0)
    kv = tl.load(k + qb[:, None] + d[None, :], mask=mt[:, None] & md[None, :], other=0.0)
    cb = ((ib * NC + ic) * HV + ihv) * BC
    c = tl.load(cs + cb + r, mask=mt, other=0.0)
    endrow = tl.minimum(BC - 1, T - 1 - ic * BC)
    end = tl.sum(c * (r == endrow))
    b = tl.load(beta + (ib * T + tt) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    ab = ((ib * NC + ic) * HV + ihv) * BC * BC
    aa = tl.load(inv + ab + r[:, None] * BC + j[None, :], mask=mt[:, None] & mt[None, :], other=0.0)
    ratio = _ex(c[:, None] - c[None, :])
    gram = tl.dot(kv, tl.trans(kv), input_precision="tf32")
    strict = (r[:, None] > j[None, :]) & mt[:, None] & mt[None, :]
    nn = tl.where(strict, b[:, None] * ratio * gram, 0.0)
    score = tl.dot(qv, tl.trans(kv), input_precision="tf32") * SCALE
    score = tl.where((r[:, None] >= j[None, :]) & mt[:, None] & mt[None, :], score * ratio, 0.0)
    xmat = kv * (b * _ex(c))[:, None]
    w = tl.dot(aa.to(k.dtype.element_ty), xmat.to(k.dtype.element_ty), input_precision="tf32")
    z = kv * _ex(end - c)[:, None]
    dA = tl.zeros((BC, BC), tl.float32)
    dM = tl.zeros((BC, BC), tl.float32)
    dZ = tl.zeros((BC, BK), tl.float32)
    dY = tl.zeros((BC, BK), tl.float32)
    dR = tl.zeros((BC, VB), tl.float32)
    dX = tl.zeros((BC, BK), tl.float32)
    dc = tl.zeros((BC,), tl.float32)
    dbe = tl.zeros((BC,), tl.float32)
    for iv in range(NV):
        xoff = iv * VB + tl.arange(0, VB)
        mx = xoff < V
        ub = ((ib * T + tt) * HV + ihv) * V
        pp = tl.load(updates + ub[:, None] + xoff[None, :], mask=mt[:, None] & mx[None, :], other=0.0)
        do = tl.load(dout + ub[:, None] + xoff[None, :], mask=mt[:, None] & mx[None, :], other=0.0)
        vb = tl.load(v + ub[:, None] + xoff[None, :], mask=mt[:, None] & mx[None, :], other=0.0)
        sb = ((((ib * NC + ic) * HV + ihv) * K + d[:, None]) * V + xoff[None, :])
        s = tl.load(states + sb, mask=md[:, None] & mx[None, :], other=0.0)
        dsn = tl.load(dstates + sb, mask=md[:, None] & mx[None, :], other=0.0)
        dp = tl.dot(tl.trans(score.to(do.dtype)), do, input_precision="tf32")
        dp += tl.dot(z.to(k.dtype.element_ty), dsn.to(k.dtype.element_ty), input_precision="tf32")
        dM += tl.dot(do, tl.trans(pp), input_precision="tf32")
        dZ += tl.dot(pp, tl.trans(dsn.to(pp.dtype)), input_precision="tf32")
        dY += tl.dot(do, tl.trans(s.to(do.dtype)), input_precision="tf32")
        du = dp
        dw = -tl.dot(dp, tl.trans(s), input_precision="tf32")
        rrval = vb * b[:, None]
        dA += tl.dot(du, tl.trans(rrval), input_precision="tf32")
        dA += tl.dot(dw, tl.trans(xmat), input_precision="tf32")
        ar = tl.dot(tl.trans(aa.to(dp.dtype)), du, input_precision="tf32")
        ax = tl.dot(tl.trans(aa.to(dw.dtype)), dw, input_precision="tf32")
        tl.store(dv + ub[:, None] + xoff[None, :], ar * b[:, None], mask=mt[:, None] & mx[None, :])
        dbe += tl.sum(ar * vb, axis=1)
        dX += ax
    dN = -tl.dot(tl.trans(aa), dA, input_precision="tf32")
    dN = tl.dot(dN, tl.trans(aa), input_precision="tf32")
    dN = tl.where(strict, dN, 0.0)
    causal = (r[:, None] >= j[None, :]) & mt[:, None] & mt[None, :]
    dscore = tl.where(causal, dM * ratio, 0.0)
    dqv = tl.dot(dscore.to(k.dtype.element_ty), kv, input_precision="tf32") * SCALE
    dqv += dY * (_ex(c) * SCALE)[:, None]
    dkv = tl.dot(tl.trans(dscore.to(q.dtype.element_ty)), qv, input_precision="tf32") * SCALE
    left = dN * (b[:, None] * ratio)
    dkv += tl.dot(left.to(k.dtype.element_ty), kv, input_precision="tf32")
    dkv += tl.dot(tl.trans(left.to(k.dtype.element_ty)), kv, input_precision="tf32")
    dkv += dX * (b * _ex(c))[:, None]
    dkv += dZ * _ex(end - c)[:, None]
    dbe += tl.sum(dX * (kv * _ex(c)[:, None]), axis=1)
    nprod = dN * nn
    dbe += tl.sum(dN * tl.where(strict, ratio * gram, 0.0), axis=1)
    mprod = dM * score
    dc += tl.sum(nprod, axis=1) - tl.sum(nprod, axis=0)
    dc += tl.sum(mprod, axis=1) - tl.sum(mprod, axis=0)
    dc += tl.sum(dX * xmat, axis=1)
    dc += tl.sum(dY * (qv * (_ex(c) * SCALE)[:, None]), axis=1)
    zprod = tl.sum(dZ * z, axis=1)
    dc -= zprod
    dc += tl.sum(zprod) * (r == endrow)
    dgc = tl.sum(tl.where(j[None, :] >= r[:, None], dc[None, :], 0.0), axis=1)
    eb = ((ib * T + tt) * HV + ihv)
    partb = ((ib * T + tt) * HV + ihv) * K
    tl.store(dqpart + partb[:, None] + d[None, :], dqv, mask=mt[:, None] & md[None, :])
    tl.store(dkpart + partb[:, None] + d[None, :], dkv, mask=mt[:, None] & md[None, :])
    tl.store(dg + eb, dgc, mask=mt)
    tl.store(dbeta + eb, dbe, mask=mt)


@triton.jit
def _fold_heads(srcq, srck, dq, dk, N: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    ih = p % H
    bt = p // H
    d = tl.arange(0, BK)
    md = d < K
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    for z in range(HV // H):
        ihv = ih * (HV // H) + z
        off = (bt * HV + ihv) * K + d
        aq += tl.load(srcq + off, mask=md, other=0.0)
        ak += tl.load(srck + off, mask=md, other=0.0)
    out = (bt * H + ih) * K + d
    tl.store(dq + out, aq, mask=md)
    tl.store(dk + out, ak, mask=md)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        nc = triton.cdiv(T, BC)
        nv = triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        cs = torch.empty((B, nc, HV, BC), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, nc, HV, BC, BC), device=q.device, dtype=torch.float32)
        states = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        updates = torch.empty_like(v)
        out = torch.empty_like(v)
        _build_transform[(nc, B * HV)](k, g, beta, cs, inv, T=T, H=H, HV=HV, K=K, NC=nc, BK=bk)
        _state_scan[(B * HV * nv,)](k, v, beta, cs, inv, states, updates, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, VB=BV)
        _emit[(B * HV * nc * nv,)](q, k, cs, states, updates, out, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, VB=BV, SCALE=K ** -0.5)
        ctx.save_for_backward(q, k, v, g, beta, cs, inv, states, updates)
        ctx.dims = (B, T, H, HV, K, V, nc, nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, cs, inv, states, updates = ctx.saved_tensors
        B, T, H, HV, K, V, nc, nv, bk = ctx.dims
        dout = dout.contiguous()
        dstates = torch.empty_like(states)
        dqpart = torch.empty((B, T, HV, K), device=q.device, dtype=q.dtype)
        dkpart = torch.empty((B, T, HV, K), device=q.device, dtype=k.dtype)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dv, dg, db = torch.empty_like(v), torch.empty_like(g), torch.empty_like(beta)
        _adjoint[(B * HV * nv,)](q, k, beta, cs, inv, states, updates, dout, dstates, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, VB=BV, SCALE=K ** -0.5)
        _chunk_grad[(nc, B * HV)](q, k, v, g, beta, cs, inv, states, updates, dout, dstates, dqpart, dkpart, dv, dg, db, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, VB=BV, SCALE=K ** -0.5)
        _fold_heads[(B * T * H,)](dqpart, dkpart, dq, dk, N=B*T, H=H, HV=HV, K=K, BK=bk)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
