import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _build_wy(k, g, beta, gc, inv, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ibh = tl.program_id(1).to(tl.int64)
    ib, ih = ibh // HV, ibh % HV
    iqh = ih // (HV // H)
    r = tl.arange(0, BC)
    x = tl.arange(0, BK)
    t = ic * BC + r
    mt = t < T

    pg = g + (ib * T * HV + t * HV + ih)
    c = tl.cumsum(tl.load(pg, mask=mt, other=0.0).to(tl.float32), axis=0)
    tl.store(gc + ib * T * HV + t * HV + ih, c, mask=mt)

    pk = k + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
    kk = tl.load(pk, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    b = tl.load(beta + ib * T * HV + t * HV + ih, mask=mt, other=0.0).to(tl.float32)
    valid = mt[:, None] & mt[None, :] & (r[:, None] > r[None, :])
    l = tl.where(valid, gram * b[:, None] * tl.exp(c[:, None] - c[None, :]), 0.0)

    ai = -l
    for i in range(1, BC):
        row = tl.sum(tl.where((r == i)[:, None], -l, 0.0), axis=0)
        row += tl.sum(row[:, None] * ai, axis=0)
        row = tl.where(r < i, row, 0.0)
        ai = tl.where((r == i)[:, None], row[None, :], ai)
    ai += (r[:, None] == r[None, :]).to(tl.float32)

    base = ((ib * ((T + BC - 1) // BC) + ic) * HV + ih) * BC * BC
    tl.store(inv + base + r[:, None] * BC + r[None, :], ai,
             mask=mt[:, None] & mt[None, :])


@triton.jit
def _scan_state(k, v, beta, gc, inv, hs, delta, T: tl.constexpr,
                H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                V: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = tl.cdiv(V, BV)
    NC: tl.constexpr = tl.cdiv(T, BC)
    iv = p % NV
    ibh = p // NV
    ib, ih = ibh // HV, ibh % HV
    iqh = ih // (HV // H)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    r = tl.arange(0, BC)
    state = tl.zeros((BK, BV), tl.float32)

    for ic in range(NC):
        t = ic * BC + r
        mt = t < T
        mh = (x[:, None] < K) & (y[None, :] < V)
        sh = ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + y[None, :])
        tl.store(hs + sh, state, mask=mh)

        pk = k + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
        kv = tl.load(pk, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        vv = tl.load(v + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :],
                     mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        c = tl.load(gc + ib * T * HV + t * HV + ih, mask=mt, other=0.0)
        b = tl.load(beta + ib * T * HV + t * HV + ih, mask=mt, other=0.0).to(tl.float32)
        abase = ((ib * NC + ic) * HV + ih) * BC * BC
        a = tl.load(inv + abase + r[:, None] * BC + r[None, :],
                    mask=mt[:, None] & mt[None, :], other=0.0)
        pred = tl.dot(kv, state, input_precision="tf32")
        d = tl.dot(a, b[:, None] * (vv - tl.exp(c)[:, None] * pred), input_precision="tf32")
        tl.store(delta + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :],
                 d, mask=mt[:, None] & (y[None, :] < V))
        ce = tl.sum(tl.where(r == tl.minimum(BC - 1, T - ic * BC - 1), c, 0.0), axis=0)
        kw = kv * tl.exp(ce - c)[:, None]
        state = tl.exp(ce) * state + tl.dot(tl.trans(kw), d, input_precision="tf32")


@triton.jit
def _emit(q, k, gc, hs, delta, out, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
          BK: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = tl.cdiv(V, BV)
    NC: tl.constexpr = tl.cdiv(T, BC)
    iv = p % NV
    z = p // NV
    ic = z % NC
    ibh = z // NC
    ib, ih = ibh // HV, ibh % HV
    iqh = ih // (HV // H)
    r = tl.arange(0, BC)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    t = ic * BC + r
    mt = t < T
    pk = k + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
    pq = q + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
    kv = tl.load(pk, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    qv = tl.load(pq, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
    c = tl.load(gc + ib * T * HV + t * HV + ih, mask=mt, other=0.0)
    sh = ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + y[None, :])
    state = tl.load(hs + sh, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    d = tl.load(delta + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :],
                mask=mt[:, None] & (y[None, :] < V), other=0.0)
    m = tl.dot(qv, tl.trans(kv), input_precision="tf32")
    m *= tl.exp(c[:, None] - c[None, :])
    m = tl.where(mt[:, None] & mt[None, :] & (r[:, None] >= r[None, :]), m, 0.0)
    o = tl.exp(c)[:, None] * tl.dot(qv, state, input_precision="tf32")
    o += tl.dot(m, d, input_precision="tf32")
    po = out + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :]
    tl.store(po, o, mask=mt[:, None] & (y[None, :] < V))


@triton.jit
def _adjoint(q, k, beta, gc, inv, dout, delta, ds_chunks, zbuf,
             T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
             SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = tl.cdiv(V, BV)
    NC: tl.constexpr = tl.cdiv(T, BC)
    iv = p % NV
    ibh = p // NV
    ib, ih = ibh // HV, ibh % HV
    iqh = ih // (HV // H)
    r = tl.arange(0, BC)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    ds = tl.zeros((BK, BV), tl.float32)

    for back in range(NC):
        ic = NC - 1 - back
        t = ic * BC + r
        mt = t < T
        pk = k + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
        pq = q + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
        kv = tl.load(pk, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        qv = tl.load(pq, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
        do = tl.load(dout + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :],
                     mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        c = tl.load(gc + ib * T * HV + t * HV + ih, mask=mt, other=0.0)
        b = tl.load(beta + ib * T * HV + t * HV + ih, mask=mt, other=0.0).to(tl.float32)
        m = tl.dot(qv, tl.trans(kv), input_precision="tf32")
        m *= tl.exp(c[:, None] - c[None, :])
        m = tl.where(mt[:, None] & mt[None, :] & (r[:, None] >= r[None, :]), m, 0.0)
        ce = tl.sum(tl.where(r == tl.minimum(BC - 1, T - ic * BC - 1), c, 0.0), axis=0)
        ddelta = tl.dot(tl.trans(m), do, input_precision="tf32")
        ddelta += tl.exp(ce - c)[:, None] * tl.dot(kv, ds, input_precision="tf32")
        abase = ((ib * NC + ic) * HV + ih) * BC * BC
        a = tl.load(inv + abase + r[:, None] * BC + r[None, :],
                    mask=mt[:, None] & mt[None, :], other=0.0)
        zz = tl.dot(tl.trans(a), ddelta, input_precision="tf32")
        tl.store(zbuf + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :],
                 zz, mask=mt[:, None] & (y[None, :] < V))
        sd = ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + y[None, :])
        tl.store(ds_chunks + sd, ds, mask=(x[:, None] < K) & (y[None, :] < V))
        ds = tl.exp(ce) * ds + tl.dot(tl.trans(tl.exp(c)[:, None] * qv), do, input_precision="tf32")
        ds -= tl.dot(tl.trans((b * tl.exp(c))[:, None] * kv), zz, input_precision="tf32")


@triton.jit
def _chunk_grad(q, k, v, beta, gc, hs, ds_chunks, delta, zbuf,
                dout, pdq, pdk, dv, pdb, pdg, T: tl.constexpr,
                H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                V: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = tl.cdiv(V, BV)
    NC: tl.constexpr = tl.cdiv(T, BC)
    iv = p % NV
    z = p // NV
    ic = z % NC
    ibh = z // NC
    ib, ih = ibh // HV, ibh % HV
    iqh = ih // (HV // H)
    r = tl.arange(0, BC)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    t = ic * BC + r
    mt = t < T
    mk = mt[:, None] & (x[None, :] < K)
    mv = mt[:, None] & (y[None, :] < V)
    pk = k + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
    pq = q + ib * T * H * K + t[:, None] * H * K + iqh * K + x[None, :]
    kv = tl.load(pk, mask=mk, other=0.0).to(tl.float32)
    qv = tl.load(pq, mask=mk, other=0.0).to(tl.float32)
    vv = tl.load(v + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :], mask=mv, other=0.0).to(tl.float32)
    do = tl.load(dout + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :], mask=mv, other=0.0).to(tl.float32)
    d = tl.load(delta + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :], mask=mv, other=0.0)
    zz = tl.load(zbuf + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :], mask=mv, other=0.0)
    c = tl.load(gc + ib * T * HV + t * HV + ih, mask=mt, other=0.0)
    b = tl.load(beta + ib * T * HV + t * HV + ih, mask=mt, other=0.0).to(tl.float32)
    sh = ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + y[None, :])
    state = tl.load(hs + sh, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    ds = tl.load(ds_chunks + sh, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    pred = tl.dot(kv, state, input_precision="tf32")
    res = vv - tl.exp(c)[:, None] * pred

    f = tl.exp(c[:, None] - c[None, :])
    lower = mt[:, None] & mt[None, :] & (r[:, None] >= r[None, :])
    strict = mt[:, None] & mt[None, :] & (r[:, None] > r[None, :])
    od = tl.dot(do, tl.trans(d), input_precision="tf32")
    dq = tl.exp(c)[:, None] * tl.dot(do, tl.trans(state), input_precision="tf32")
    dq += tl.dot(tl.where(lower, f * od, 0.0), kv, input_precision="tf32")
    dq *= SCALE

    dk = tl.dot(tl.trans(tl.where(lower, f * od, 0.0)), qv * SCALE, input_precision="tf32")
    ce = tl.sum(tl.where(r == tl.minimum(BC - 1, T - ic * BC - 1), c, 0.0), axis=0)
    dk += tl.exp(ce - c)[:, None] * tl.dot(d, tl.trans(ds), input_precision="tf32")
    dk -= (b * tl.exp(c))[:, None] * tl.dot(zz, tl.trans(state), input_precision="tf32")

    dl = -tl.dot(zz, tl.trans(d), input_precision="tf32")
    gram = tl.dot(kv, tl.trans(kv), input_precision="tf32")
    lf = tl.where(strict, b[:, None] * f, 0.0)
    dk += tl.dot((dl * lf), kv, input_precision="tf32")
    dk += tl.dot(tl.trans(dl * lf), kv, input_precision="tf32")

    db = tl.sum(zz * res, axis=1) + tl.sum(tl.where(strict, dl * f * gram, 0.0), axis=1)
    dc = tl.exp(c) * tl.sum(do * tl.dot(qv * SCALE, state, input_precision="tf32"), axis=1)
    rout = tl.where(lower, f * tl.dot(qv * SCALE, tl.trans(kv), input_precision="tf32") * od, 0.0)
    dc += tl.sum(rout, axis=1) - tl.sum(rout, axis=0)
    dc -= b * tl.exp(c) * tl.sum(zz * pred, axis=1)
    lc = tl.where(strict, dl * b[:, None] * f * gram, 0.0)
    dc += tl.sum(lc, axis=1) - tl.sum(lc, axis=0)
    end_terms = tl.exp(ce - c) * tl.sum(d * tl.dot(kv, ds, input_precision="tf32"), axis=1)
    dc -= end_terms
    last = tl.minimum(BC - 1, T - ic * BC - 1)
    dc += tl.where(r == last, tl.sum(end_terms, axis=0) + tl.exp(ce) * tl.sum(ds * state), 0.0)
    dg = tl.flip(tl.cumsum(tl.flip(dc, 0), axis=0), 0)

    base = ((ib * T * HV + t * HV + ih) * NV + iv)
    tl.store(pdq + base[:, None] * K + x[None, :], dq, mask=mk)
    tl.store(pdk + base[:, None] * K + x[None, :], dk, mask=mk)
    tl.store(pdb + base, db, mask=mt)
    tl.store(pdg + base, dg, mask=mt)
    tl.store(dv + ib * T * HV * V + t[:, None] * HV * V + ih * V + y[None, :],
             b[:, None] * zz, mask=mv)


@triton.jit
def _reduce_qk(pdq, pdk, dq, dk, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
               BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    it = p % T
    z = p // T
    ih = z % H
    ib = z // H
    x = tl.arange(0, BK)
    NV: tl.constexpr = tl.cdiv(V, BV)
    R: tl.constexpr = HV // H
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    for j in range(R * NV):
        ir = j // NV
        iv = j % NV
        ivh = ih * R + ir
        base = ((ib * T * HV + it * HV + ivh) * NV + iv) * K
        aq += tl.load(pdq + base + x, mask=x < K, other=0.0)
        ak += tl.load(pdk + base + x, mask=x < K, other=0.0)
    outbase = (ib * T * H + it * H + ih) * K
    tl.store(dq + outbase + x, aq, mask=x < K)
    tl.store(dk + outbase + x, ak, mask=x < K)


@triton.jit
def _reduce_scalar(pdb, pdg, db, dg, T: tl.constexpr, HV: tl.constexpr,
                   V: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = tl.cdiv(V, BV)
    accb = 0.0
    accg = 0.0
    for iv in range(NV):
        accb += tl.load(pdb + p * NV + iv)
        accg += tl.load(pdg + p * NV + iv)
    tl.store(db + p, accb)
    tl.store(dg + p, accg)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC, NV = triton.cdiv(T, BC), triton.cdiv(V, BV)
        BK = triton.next_power_of_2(K)
        gc = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, NC, HV, BC, BC), device=q.device, dtype=torch.float32)
        hs = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(NC, B * HV)](k, g, beta, gc, inv, T=T, H=H, HV=HV, K=K, BK=BK)
        _scan_state[(B * HV * NV,)](k, v, beta, gc, inv, hs, delta,
                                    T=T, H=H, HV=HV, K=K, V=V, BK=BK)
        _emit[(B * HV * NC * NV,)](q, k, gc, hs, delta, out,
                                   T=T, H=H, HV=HV, K=K, V=V, BK=BK,
                                   SCALE=K ** -0.5)
        ctx.save_for_backward(q, k, v, beta, gc, inv, hs, delta)
        ctx.shape = (B, T, H, HV, K, V, NC, NV, BK)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, beta, gc, inv, hs, delta = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, HV, K, V, NC, NV, BK = ctx.shape
        ds = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        zbuf = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        pdq = torch.empty((B, T, HV, NV, K), device=q.device, dtype=torch.float32)
        pdk = torch.empty_like(pdq, dtype=torch.float32)
        pdb = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        pdg = torch.empty_like(pdb, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gc)
        _adjoint[(B * HV * NV,)](q, k, beta, gc, inv, dout, delta, ds, zbuf,
                                  T=T, H=H, HV=HV, K=K, V=V, BK=BK,
                                  SCALE=K ** -0.5)
        _chunk_grad[(B * HV * NC * NV,)](q, k, v, beta, gc, hs, ds, delta,
                                         zbuf, dout, pdq, pdk, dv, pdb, pdg,
                                         T=T, H=H, HV=HV, K=K, V=V, BK=BK,
                                         SCALE=K ** -0.5)
        _reduce_qk[(B * T * H,)](pdq, pdk, dq, dk, T=T, H=H, HV=HV,
                                  K=K, V=V, BK=BK)
        _reduce_scalar[(B * T * HV,)](pdb, pdg, db, dg, T=T, HV=HV, V=V)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
