import torch
import triton
import triton.language as tl

C = 16
BV = 16
TC = tl.constexpr(16)


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(x, 0.0)))


@triton.jit
def _build_inverse(k, g, beta, gc, inv, T: tl.constexpr, H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ibh = tl.program_id(1).to(tl.int64)
    ih = ibh % HV
    ib = ibh // HV
    ikh = ih // (HV // H)
    r = tl.arange(0, TC)
    x = tl.arange(0, BK)
    t = ic * TC + r
    mt = t < T
    pk = k + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
    kk = tl.load(pk, mask=mt[:, None] & (x[None, :] < K), other=0.0)
    pg = g + (ib * T + t) * HV + ih
    gg = tl.load(pg, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    pb = beta + (ib * T + t) * HV + ih
    bb = tl.load(pb, mask=mt, other=0.0).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    lower = r[:, None] > r[None, :]
    ee = _bounded_exp(cs[:, None] - cs[None, :])
    ll = tl.where(lower & mt[:, None] & mt[None, :], gram * bb[:, None] * ee, 0.0)
    eye = r[:, None] == r[None, :]
    mm = tl.where(eye, 1.0, 0.0)
    for i in range(TC):
        li = tl.sum(tl.where(r[:, None] == i, ll, 0.0), axis=0)
        row = -tl.sum(li[:, None] * mm, axis=0)
        row += tl.where(r == i, 1.0, 0.0)
        mm = tl.where(r[:, None] == i, row[None, :], mm)
    pgc = gc + (ib * T + t) * HV + ih
    pi = inv + ((ib * ((T + TC - 1) // TC) + ic) * HV + ih) * TC * TC
    tl.store(pgc, cs, mask=mt)
    tl.store(pi + r[:, None] * TC + r[None, :], mm)


@triton.jit
def _state_scan(k, v, beta, gc, inv, states, delta,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                BK: tl.constexpr, BV_: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv = p % ((V + BV_ - 1) // BV_)
    ibh = p // ((V + BV_ - 1) // BV_)
    ih = ibh % HV
    ib = ibh // HV
    ikh = ih // (HV // H)
    r = tl.arange(0, TC)
    x = tl.arange(0, BK)
    z = iv * BV_ + tl.arange(0, BV_)
    mx = x < K
    mz = z < V
    ss = tl.zeros((BK, BV_), tl.float32)
    for ic in range(NC):
        t = ic * TC + r
        mt = t < T
        ps = states + ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + z[None, :])
        tl.store(ps, ss, mask=mx[:, None] & mz[None, :])
        pk = k + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
        kv = tl.load(pk, mask=mt[:, None] & mx[None, :], other=0.0)
        pv = v + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
        vv = tl.load(pv, mask=mt[:, None] & mz[None, :], other=0.0)
        pb = beta + (ib * T + t) * HV + ih
        bb = tl.load(pb, mask=mt, other=0.0).to(tl.float32)
        pc = gc + (ib * T + t) * HV + ih
        cc = tl.load(pc, mask=mt, other=0.0)
        pi = inv + ((ib * NC + ic) * HV + ih) * TC * TC
        mm = tl.load(pi + r[:, None] * TC + r[None, :])
        xx = vv * bb[:, None]
        yy = kv * (bb * _bounded_exp(cc))[:, None]
        uu = tl.dot(mm.to(vv.dtype), xx.to(vv.dtype), input_precision="tf32")
        ww = tl.dot(mm.to(kv.dtype), yy.to(kv.dtype), input_precision="tf32")
        dd = uu - tl.dot(ww.to(kv.dtype), ss.to(kv.dtype), input_precision="tf32")
        pd = delta + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
        tl.store(pd, dd, mask=mt[:, None] & mz[None, :])
        n = tl.minimum(TC, T - ic * TC)
        end = tl.sum(tl.where(r == n - 1, cc, 0.0), axis=0)
        rr = kv * _bounded_exp(end - cc)[:, None]
        ss = ss * _bounded_exp(end) + tl.dot(tl.trans(rr).to(kv.dtype), dd.to(kv.dtype), input_precision="tf32")


@triton.jit
def _emit(q, k, gc, states, delta, out,
          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
          K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
          BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv = p % ((V + BV_ - 1) // BV_)
    p //= ((V + BV_ - 1) // BV_)
    ic = p % NC
    ibh = p // NC
    ih = ibh % HV
    ib = ibh // HV
    ikh = ih // (HV // H)
    r = tl.arange(0, TC)
    x = tl.arange(0, BK)
    z = iv * BV_ + tl.arange(0, BV_)
    t = ic * TC + r
    mt = t < T
    mx = x < K
    mz = z < V
    pq = q + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
    pk = k + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
    qq = tl.load(pq, mask=mt[:, None] & mx[None, :], other=0.0) * SCALE
    kk = tl.load(pk, mask=mt[:, None] & mx[None, :], other=0.0)
    pc = gc + (ib * T + t) * HV + ih
    cc = tl.load(pc, mask=mt, other=0.0)
    ps = states + ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + z[None, :])
    ss = tl.load(ps, mask=mx[:, None] & mz[None, :], other=0.0)
    pd = delta + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
    dd = tl.load(pd, mask=mt[:, None] & mz[None, :], other=0.0)
    causal = r[:, None] >= r[None, :]
    pp = tl.dot(qq, tl.trans(kk), input_precision="tf32")
    pp *= _bounded_exp(cc[:, None] - cc[None, :])
    pp = tl.where(causal & mt[:, None] & mt[None, :], pp, 0.0)
    oo = tl.dot((qq * _bounded_exp(cc)[:, None]).to(qq.dtype), ss.to(qq.dtype), input_precision="tf32")
    oo += tl.dot(pp.to(dd.dtype), dd, input_precision="tf32")
    po = out + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
    tl.store(po, oo, mask=mt[:, None] & mz[None, :])



@triton.jit
def _adjoint_scan(q, k, beta, dout, gc, inv, dstates,
                  T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                  BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + BV_ - 1) // BV_
    iv = p % nv
    ibh = p // nv
    ih = ibh % HV
    ib = ibh // HV
    ikh = ih // (HV // H)
    r = tl.arange(0, TC)
    x = tl.arange(0, BK)
    z = iv * BV_ + tl.arange(0, BV_)
    mx = x < K
    mz = z < V
    ds = tl.zeros((BK, BV_), tl.float32)
    for rev in range(NC):
        ic = NC - 1 - rev
        t = ic * TC + r
        mt = t < T
        pds = dstates + ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + z[None, :])
        tl.store(pds, ds, mask=mx[:, None] & mz[None, :])
        pq = q + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
        pk = k + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
        qq = tl.load(pq, mask=mt[:, None] & mx[None, :], other=0.0) * SCALE
        kk = tl.load(pk, mask=mt[:, None] & mx[None, :], other=0.0)
        po = dout + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
        do = tl.load(po, mask=mt[:, None] & mz[None, :], other=0.0)
        pc = gc + (ib * T + t) * HV + ih
        cc = tl.load(pc, mask=mt, other=0.0)
        pb = beta + (ib * T + t) * HV + ih
        bb = tl.load(pb, mask=mt, other=0.0).to(tl.float32)
        pi = inv + ((ib * NC + ic) * HV + ih) * TC * TC
        mm = tl.load(pi + r[:, None] * TC + r[None, :])
        causal = (r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :]
        pp = tl.where(causal, tl.dot(qq, tl.trans(kk), input_precision="tf32") *
                      _bounded_exp(cc[:, None] - cc[None, :]), 0.0)
        dd = tl.dot(tl.trans(pp).to(do.dtype), do, input_precision="tf32")
        local = tl.dot(tl.trans(qq * _bounded_exp(cc)[:, None]).to(do.dtype), do,
                       input_precision="tf32")
        n = tl.minimum(TC, T - ic * TC)
        end = tl.sum(tl.where(r == n - 1, cc, 0.0), axis=0)
        rr = kk * _bounded_exp(end - cc)[:, None]
        dd += tl.dot(rr.to(ds.dtype), ds.to(rr.dtype), input_precision="tf32")
        yy = kk * (bb * _bounded_exp(cc))[:, None]
        ww = tl.dot(mm.to(kk.dtype), yy.to(kk.dtype), input_precision="tf32")
        ds = local + ds * _bounded_exp(end) - tl.dot(tl.trans(ww).to(dd.dtype), dd,
                                                      input_precision="tf32")


@triton.jit
def _reverse(q, k, v, g, beta, dout, gc, inv, states, delta,
             dstates, pqg, pkg, pvg, pgg, pbg,
             T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
             BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + BV_ - 1) // BV_
    iv = p % nv
    p //= nv
    ic = p % NC
    ibh = p // NC
    ih = ibh % HV
    ib = ibh // HV
    ikh = ih // (HV // H)
    r = tl.arange(0, TC)
    x = tl.arange(0, BK)
    z = iv * BV_ + tl.arange(0, BV_)
    mx = x < K
    mz = z < V
    t = ic * TC + r
    mt = t < T
    pq = q + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
    pk = k + ((ib * T + t[:, None]) * H + ikh) * K + x[None, :]
    qq0 = tl.load(pq, mask=mt[:, None] & mx[None, :], other=0.0)
    qq = qq0 * SCALE
    kk = tl.load(pk, mask=mt[:, None] & mx[None, :], other=0.0)
    pv = v + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
    vv = tl.load(pv, mask=mt[:, None] & mz[None, :], other=0.0)
    po = dout + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
    do = tl.load(po, mask=mt[:, None] & mz[None, :], other=0.0)
    pd = delta + ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
    dd = tl.load(pd, mask=mt[:, None] & mz[None, :], other=0.0)
    ps = states + ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + z[None, :])
    ss = tl.load(ps, mask=mx[:, None] & mz[None, :], other=0.0)
    pds = dstates + ((((ib * NC + ic) * HV + ih) * K + x[:, None]) * V + z[None, :])
    ds = tl.load(pds, mask=mx[:, None] & mz[None, :], other=0.0)
    pc = gc + (ib * T + t) * HV + ih
    cc = tl.load(pc, mask=mt, other=0.0)
    pb = beta + (ib * T + t) * HV + ih
    bb = tl.load(pb, mask=mt, other=0.0).to(tl.float32)
    pi = inv + ((ib * NC + ic) * HV + ih) * TC * TC
    mm = tl.load(pi + r[:, None] * TC + r[None, :])

    ec = _bounded_exp(cc)
    causal = (r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :]
    ep = _bounded_exp(cc[:, None] - cc[None, :])
    pp = tl.where(causal, tl.dot(qq, tl.trans(kk), input_precision="tf32") * ep, 0.0)
    dp = tl.dot(do, tl.trans(dd).to(do.dtype), input_precision="tf32")
    ddd = tl.dot(tl.trans(pp).to(do.dtype), do, input_precision="tf32")
    weighted = tl.where(causal, dp * ep, 0.0)
    dqq = tl.dot(weighted.to(kk.dtype), kk, input_precision="tf32")
    dkk = tl.dot(tl.trans(weighted).to(qq.dtype), qq, input_precision="tf32")
    dcc = tl.sum(dp * pp, axis=1) - tl.sum(dp * pp, axis=0)
    base_dq = tl.dot(do, tl.trans(ss).to(do.dtype), input_precision="tf32")
    dqq += base_dq * ec[:, None]
    dcc += tl.sum(base_dq * (qq * ec[:, None]), axis=1)

    n = tl.minimum(TC, T - ic * TC)
    end = tl.sum(tl.where(r == n - 1, cc, 0.0), axis=0)
    er = _bounded_exp(end - cc)
    rr = kk * er[:, None]
    ddd += tl.dot(rr.to(ds.dtype), ds.to(rr.dtype), input_precision="tf32")
    drr = tl.dot(dd.to(ds.dtype), tl.trans(ds).to(dd.dtype), input_precision="tf32")
    dkk += drr * er[:, None]
    dcc -= tl.sum(drr * rr, axis=1)
    dend = tl.sum(drr * rr)
    dend += tl.sum(ds * (ss * _bounded_exp(end)))

    xx = vv * bb[:, None]
    yy = kk * (bb * ec)[:, None]
    du = ddd
    dw = -tl.dot(ddd.to(ss.dtype), tl.trans(ss).to(ddd.dtype), input_precision="tf32")
    dm = tl.dot(du, tl.trans(xx), input_precision="tf32")
    dm += tl.dot(dw, tl.trans(yy), input_precision="tf32")
    dx = tl.dot(tl.trans(mm).to(du.dtype), du, input_precision="tf32")
    dy = tl.dot(tl.trans(mm).to(dw.dtype), dw, input_precision="tf32")
    dvv = dx * bb[:, None]
    dbb = tl.sum(dx * vv, axis=1)
    dkk += dy * (bb * ec)[:, None]
    dbb += tl.sum(dy * (kk * ec[:, None]), axis=1)
    dcc += tl.sum(dy * yy, axis=1)

    dl = -tl.dot(tl.trans(mm), tl.dot(dm, tl.trans(mm), input_precision="tf32"), input_precision="tf32")
    strict = (r[:, None] > r[None, :]) & mt[:, None] & mt[None, :]
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    el = _bounded_exp(cc[:, None] - cc[None, :])
    wa = tl.where(strict, dl * (bb[:, None] * el), 0.0)
    dkk += tl.dot(wa.to(kk.dtype), kk, input_precision="tf32")
    dkk += tl.dot(tl.trans(wa).to(kk.dtype), kk, input_precision="tf32")
    dbb += tl.sum(tl.where(strict, dl * gram * el, 0.0), axis=1)
    dll = tl.where(strict, dl * gram * bb[:, None] * el, 0.0)
    dcc += tl.sum(dll, axis=1) - tl.sum(dll, axis=0)
    n = tl.minimum(TC, T - ic * TC)
    dcc += tl.where(r == n - 1, dend, 0.0)
    dgg = tl.sum(dcc, axis=0) - tl.cumsum(dcc, axis=0) + dcc

    baseq = (((ib * T + t[:, None]) * HV + ih) * ((V + BV_ - 1) // BV_) + iv) * K + x[None, :]
    tl.store(pqg + baseq, dqq * SCALE, mask=mt[:, None] & mx[None, :])
    tl.store(pkg + baseq, dkk, mask=mt[:, None] & mx[None, :])
    basev = ((ib * T + t[:, None]) * HV + ih) * V + z[None, :]
    tl.store(pvg + basev, dvv, mask=mt[:, None] & mz[None, :])
    bases = ((ib * T + t) * HV + ih) * ((V + BV_ - 1) // BV_) + iv
    tl.store(pgg + bases, dgg, mask=mt)
    tl.store(pbg + bases, dbb, mask=mt)

@triton.jit
def _reduce_qk(src, dst, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, BK)
    it = p % T
    p //= T
    ih = p % H
    ib = p // H
    acc = tl.zeros((BK,), tl.float32)
    for j in range(HV // H):
        ivh = ih * (HV // H) + j
        for iv in range(NV):
            base = (((ib * T + it) * HV + ivh) * NV + iv) * K
            acc += tl.load(src + base + x, mask=x < K, other=0.0)
    tl.store(dst + ((ib * T + it) * H + ih) * K + x, acc, mask=x < K)


@triton.jit
def _reduce_scalar(src, dst, N: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    acc = 0.0
    for iv in range(NV):
        acc += tl.load(src + p * NV + iv)
    tl.store(dst + p, acc)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC, NV = triton.cdiv(T, C), triton.cdiv(V, BV)
        BK = triton.next_power_of_2(K)
        gc = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, NC, HV, C, C), device=q.device, dtype=torch.float32)
        states = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_inverse[(NC, B * HV)](k, g, beta, gc, inv, T=T, H=H, HV=HV, K=K, BK=BK, num_warps=4)
        _state_scan[(B * HV * NV,)](k, v, beta, gc, inv, states, delta, T=T, H=H, HV=HV,
                                    K=K, V=V, NC=NC, BK=BK, BV_=BV, num_warps=8)
        _emit[(B * HV * NC * NV,)](q, k, gc, states, delta, out, T=T, H=H, HV=HV,
                                    K=K, V=V, NC=NC, BK=BK, BV_=BV,
                                    SCALE=K ** -0.5, num_warps=8)
        ctx.save_for_backward(q, k, v, g, beta, gc, inv, states, delta)
        ctx.dims = (B, T, H, HV, K, V, NC, NV, BK)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, gc, inv, states, delta = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, HV, K, V, NC, NV, BK = ctx.dims
        pq = torch.empty((B, T, HV, NV, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq, dtype=torch.float32)
        dv = torch.empty_like(v)
        pg = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg, dtype=torch.float32)
        dstates = torch.empty_like(states, dtype=torch.float32)
        _adjoint_scan[(B * HV * NV,)](q, k, beta, dout, gc, inv, dstates,
                                      T=T, H=H, HV=HV, K=K, V=V, NC=NC,
                                      BK=BK, BV_=BV, SCALE=K ** -0.5, num_warps=4)
        _reverse[(B * HV * NC * NV,)](q, k, v, g, beta, dout, gc, inv, states, delta,
                                       dstates, pq, pk, dv, pg, pb, T=T, H=H, HV=HV,
                                       K=K, V=V, NC=NC, BK=BK, BV_=BV,
                                       SCALE=K ** -0.5, num_warps=4)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dg = torch.empty_like(g, dtype=torch.float32)
        db = torch.empty_like(beta)
        _reduce_qk[(B * H * T,)](pq, dq, T=T, H=H, HV=HV, K=K, NV=NV, BK=BK, num_warps=4)
        _reduce_qk[(B * H * T,)](pk, dk, T=T, H=H, HV=HV, K=K, NV=NV, BK=BK, num_warps=4)
        _reduce_scalar[(B * T * HV,)](pg, dg, N=B * T * HV, NV=NV)
        _reduce_scalar[(B * T * HV,)](pb, db, N=B * T * HV, NV=NV)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
