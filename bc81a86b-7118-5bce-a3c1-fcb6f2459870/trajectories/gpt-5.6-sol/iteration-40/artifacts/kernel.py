import torch
import triton
import triton.language as tl

BC = 16
BV = 32


@triton.jit
def _build_wy(k, g, beta, cs, inv, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ibh = tl.program_id(1).to(tl.int64)
    ib = ibh // HV
    ihv = ibh - ib * HV
    ih = ihv // (HV // H)
    r = tl.arange(0, 16)
    x = tl.arange(0, BK)
    t = ic * 16 + r
    mt = t < T
    pg = g + (ib * T + t) * HV + ihv
    gv = tl.load(pg, mask=mt, other=0.0).to(tl.float32)
    cv = tl.cumsum(gv, axis=0)
    cv = tl.where(mt, cv, 0.0)
    tl.store(cs + (ib * T + t) * HV + ihv, cv, mask=mt)
    pk = k + ((ib * T + t[:, None]) * H + ih) * K + x[None, :]
    kv = tl.load(pk, mask=mt[:, None] & (x[None, :] < K), other=0.0)
    gram = tl.dot(kv, tl.trans(kv), input_precision="tf32")
    bb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    valid = mt[:, None] & mt[None, :]
    lower = r[:, None] > r[None, :]
    ratio = tl.exp(tl.maximum(tl.minimum(cv[:, None] - cv[None, :], 10.0), -10.0))
    a = tl.where(valid & lower, gram * bb[:, None] * ratio, 0.0)
    eye = r[:, None] == r[None, :]
    m = tl.where(eye, 1.0, 0.0)
    for i in range(1, 16):
        ar = tl.sum(tl.where((r == i)[:, None], a, 0.0), axis=0)
        nr = -tl.sum(ar[:, None] * m, axis=0)
        nr = tl.where(r == i, 1.0, tl.where(r < i, nr, 0.0))
        m = tl.where((r == i)[:, None], nr[None, :], m)
    pm = inv + (((ib * ((T + 16 - 1) // 16) + ic) * HV + ihv) * 16 + r[:, None]) * 16 + r[None, :]
    tl.store(pm, m)


@triton.jit
def _state_scan(k, v, beta, cs, inv, states, dvals,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + 32 - 1) // 32
    iv = p % nv
    ibh = p // nv
    ib = ibh // HV
    ihv = ibh - ib * HV
    ih = ihv // (HV // H)
    nc: tl.constexpr = (T + 16 - 1) // 16
    kk = tl.arange(0, BK)
    vv = iv * 32 + tl.arange(0, 32)
    rr = tl.arange(0, 16)
    state = tl.zeros((BK, 32), tl.float32)
    for ic in range(nc):
        tt = ic * 16 + rr
        mt = tt < T
        ps = states + ((((ib * nc + ic) * HV + ihv) * K + kk[:, None]) * V + vv[None, :])
        tl.store(ps, state, mask=(kk[:, None] < K) & (vv[None, :] < V))
        c = tl.load(cs + (ib * T + tt) * HV + ihv, mask=mt, other=0.0)
        kval = tl.load(k + ((ib * T + tt[:, None]) * H + ih) * K + kk[None, :],
                       mask=mt[:, None] & (kk[None, :] < K), other=0.0)
        xmat = kval * tl.exp(tl.maximum(tl.minimum(c, 10.0), -10.0))[:, None]
        zmat = kval * tl.exp(tl.maximum(tl.minimum(-c, 10.0), -10.0))[:, None]
        val = tl.load(v + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :],
                      mask=mt[:, None] & (vv[None, :] < V), other=0.0)
        b = tl.load(beta + (ib * T + tt) * HV + ihv, mask=mt, other=0.0)
        pred = tl.dot(xmat.to(k.dtype.element_ty), state.to(k.dtype.element_ty), input_precision="tf32")
        e = (val - pred) * b[:, None]
        pm = inv + (((ib * nc + ic) * HV + ihv) * 16 + rr[:, None]) * 16 + rr[None, :]
        m = tl.load(pm)
        d = tl.dot(m.to(k.dtype.element_ty), e.to(k.dtype.element_ty), input_precision="tf32")
        pd = dvals + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :]
        tl.store(pd, d, mask=mt[:, None] & (vv[None, :] < V))
        state += tl.dot(tl.trans(zmat).to(k.dtype.element_ty), d.to(k.dtype.element_ty), input_precision="tf32")
        last = tl.minimum(16 - 1, T - 1 - ic * 16)
        ce = tl.sum(tl.where(rr == last, c, 0.0), axis=0)
        state *= tl.exp(tl.maximum(tl.minimum(ce, 10.0), -10.0))


@triton.jit
def _emit(q, k, cs, states, dvals, out, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + 32 - 1) // 32
    ic = (p // nv) % ((T + 16 - 1) // 16)
    iv = p % nv
    ibh = p // (nv * ((T + 16 - 1) // 16))
    ib = ibh // HV
    ihv = ibh - ib * HV
    ih = ihv // (HV // H)
    rr = tl.arange(0, 16); kk = tl.arange(0, BK); vv = iv * 32 + tl.arange(0, 32)
    tt = ic * 16 + rr; mt = tt < T
    c = tl.load(cs + (ib * T + tt) * HV + ihv, mask=mt, other=0.0)
    ev = tl.exp(tl.maximum(tl.minimum(c, 10.0), -10.0))
    qv = tl.load(q + ((ib * T + tt[:, None]) * H + ih) * K + kk[None, :], mask=mt[:, None] & (kk[None, :] < K), other=0.0)
    kv = tl.load(k + ((ib * T + tt[:, None]) * H + ih) * K + kk[None, :], mask=mt[:, None] & (kk[None, :] < K), other=0.0)
    qe = qv * (ev / tl.sqrt(float(K)))[:, None]
    z = kv / ev[:, None]
    ps = states + ((((ib * ((T + 16 - 1) // 16) + ic) * HV + ihv) * K + kk[:, None]) * V + vv[None, :])
    s = tl.load(ps, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.0)
    d = tl.load(dvals + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :], mask=mt[:, None] & (vv[None, :] < V), other=0.0)
    r = tl.dot(qe.to(q.dtype.element_ty), tl.trans(z).to(q.dtype.element_ty), input_precision="tf32")
    r = tl.where(rr[:, None] >= rr[None, :], r, 0.0)
    o = tl.dot(qe.to(q.dtype.element_ty), s.to(q.dtype.element_ty), input_precision="tf32") + tl.dot(r.to(q.dtype.element_ty), d.to(q.dtype.element_ty), input_precision="tf32")
    tl.store(out + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :], o, mask=mt[:, None] & (vv[None, :] < V))


@triton.jit
def _backward(q, k, v, g, beta, cs, inv, states, dvals, dout,
              pq, pkout, pv, pg, pb, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + 32 - 1) // 32
    iv = p % nv; ibh = p // nv
    ib = ibh // HV; ihv = ibh - ib * HV; ih = ihv // (HV // H)
    nc: tl.constexpr = (T + 16 - 1) // 16
    rr = tl.arange(0, 16); kk = tl.arange(0, BK); vv = iv * 32 + tl.arange(0, 32)
    ds = tl.zeros((BK, 32), tl.float32)
    for rev in range(nc):
        ic = nc - 1 - rev; tt = ic * 16 + rr; mt = tt < T
        c = tl.load(cs + (ib * T + tt) * HV + ihv, mask=mt, other=0.0)
        ep = tl.exp(tl.maximum(tl.minimum(c, 10.0), -10.0)); en = 1.0 / ep
        q0 = tl.load(q + ((ib * T + tt[:, None]) * H + ih) * K + kk[None, :], mask=mt[:, None] & (kk[None, :] < K), other=0.0)
        k0 = tl.load(k + ((ib * T + tt[:, None]) * H + ih) * K + kk[None, :], mask=mt[:, None] & (kk[None, :] < K), other=0.0)
        qs = q0 / tl.sqrt(float(K)); qe = qs * ep[:, None]; x = k0 * ep[:, None]; z = k0 * en[:, None]
        val = tl.load(v + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :], mask=mt[:, None] & (vv[None, :] < V), other=0.0)
        do = tl.load(dout + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :], mask=mt[:, None] & (vv[None, :] < V), other=0.0)
        d = tl.load(dvals + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :], mask=mt[:, None] & (vv[None, :] < V), other=0.0)
        ps = states + ((((ib * nc + ic) * HV + ihv) * K + kk[:, None]) * V + vv[None, :])
        s = tl.load(ps, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.0)
        b = tl.load(beta + (ib * T + tt) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
        pm = inv + (((ib * nc + ic) * HV + ihv) * 16 + rr[:, None]) * 16 + rr[None, :]
        m = tl.load(pm)
        rmat = tl.dot(qe.to(q.dtype.element_ty), tl.trans(z).to(q.dtype.element_ty), input_precision="tf32")
        tri = rr[:, None] >= rr[None, :]
        rmat = tl.where(tri, rmat, 0.0)
        dr = tl.dot(do, tl.trans(d).to(q.dtype.element_ty), input_precision="tf32"); dr = tl.where(tri, dr, 0.0)
        dqe = tl.dot(do, tl.trans(s).to(q.dtype.element_ty), input_precision="tf32") + tl.dot(dr.to(q.dtype.element_ty), z.to(q.dtype.element_ty), input_precision="tf32")
        dz = tl.dot(tl.trans(dr).to(q.dtype.element_ty), qe.to(q.dtype.element_ty), input_precision="tf32")
        dd = tl.dot(tl.trans(rmat).to(q.dtype.element_ty), do, input_precision="tf32")
        last = tl.minimum(16 - 1, T - 1 - ic * 16)
        ce = tl.sum(tl.where(rr == last, c, 0.0), axis=0)
        ee = tl.exp(tl.maximum(tl.minimum(ce, 10.0), -10.0))
        dd += ee * tl.dot(z.to(q.dtype.element_ty), ds.to(q.dtype.element_ty), input_precision="tf32")
        dz += ee * tl.dot(d.to(q.dtype.element_ty), tl.trans(ds).to(q.dtype.element_ty), input_precision="tf32")
        base = s + tl.dot(tl.trans(z).to(q.dtype.element_ty), d.to(q.dtype.element_ty), input_precision="tf32")
        dee = tl.sum(ds * base)
        ds0 = ee * ds + tl.dot(tl.trans(qe).to(q.dtype.element_ty), do, input_precision="tf32")
        pred = tl.dot(x.to(q.dtype.element_ty), s.to(q.dtype.element_ty), input_precision="tf32")
        e = (val - pred) * b[:, None]
        dm = tl.dot(dd.to(q.dtype.element_ty), tl.trans(e).to(q.dtype.element_ty), input_precision="tf32")
        de = tl.dot(tl.trans(m).to(q.dtype.element_ty), dd.to(q.dtype.element_ty), input_precision="tf32")
        y = de * b[:, None]
        dvv = y
        dbv = tl.sum(de * (val - pred), axis=1)
        dx = -tl.dot(y.to(q.dtype.element_ty), tl.trans(s).to(q.dtype.element_ty), input_precision="tf32")
        ds0 -= tl.dot(tl.trans(x).to(q.dtype.element_ty), y.to(q.dtype.element_ty), input_precision="tf32")
        da = -tl.dot(tl.dot(tl.trans(m).to(q.dtype.element_ty), dm.to(q.dtype.element_ty), input_precision="tf32").to(q.dtype.element_ty), tl.trans(m).to(q.dtype.element_ty), input_precision="tf32")
        strict = rr[:, None] > rr[None, :]; da = tl.where(strict, da, 0.0)
        gram = tl.dot(x.to(q.dtype.element_ty), tl.trans(z).to(q.dtype.element_ty), input_precision="tf32")
        dbv += tl.sum(da * gram, axis=1)
        pp = da * b[:, None]
        dx += tl.dot(pp.to(q.dtype.element_ty), z.to(q.dtype.element_ty), input_precision="tf32")
        dz += tl.dot(tl.trans(pp).to(q.dtype.element_ty), x.to(q.dtype.element_ty), input_precision="tf32")
        dqv = dqe * (ep / tl.sqrt(float(K)))[:, None]
        dkv = dx * ep[:, None] + dz * en[:, None]
        dc = tl.sum(dqe * qe + dx * x - dz * z, axis=1)
        dc += tl.where(rr == last, dee * ee, 0.0)
        dgout = tl.flip(tl.cumsum(tl.flip(dc, 0), axis=0), 0)
        basep = ((((ib * T + tt) * HV + ihv) * nv + iv) * K)
        tl.store(pq + basep[:, None] + kk[None, :], dqv, mask=mt[:, None] & (kk[None, :] < K))
        tl.store(pkout + basep[:, None] + kk[None, :], dkv, mask=mt[:, None] & (kk[None, :] < K))
        scal = (((ib * T + tt) * HV + ihv) * nv + iv)
        tl.store(pg + scal, dgout, mask=mt)
        tl.store(pb + scal, dbv, mask=mt)
        tl.store(pv + ((ib * T + tt[:, None]) * HV + ihv) * V + vv[None, :], dvv, mask=mt[:, None] & (vv[None, :] < V))
        ds = ds0


@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, BK: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64); x = tl.arange(0, BK)
    ih = p % H; bt = p // H; ib = bt // T; t = bt - ib * T
    rep: tl.constexpr = HV // H
    aq = tl.zeros((BK,), tl.float32); ak = tl.zeros((BK,), tl.float32)
    for j in range(rep):
        ihv = ih * rep + j
        for iv in range(NV):
            off = ((((ib * T + t) * HV + ihv) * NV + iv) * K + x)
            aq += tl.load(pq + off, mask=x < K, other=0.0)
            ak += tl.load(pk + off, mask=x < K, other=0.0)
    tl.store(dq + ((ib * T + t) * H + ih) * K + x, aq, mask=x < K)
    tl.store(dk + ((ib * T + t) * H + ih) * K + x, ak, mask=x < K)


@triton.jit
def _reduce_scalar(pg, pb, dg, db, N: tl.constexpr, NV: tl.constexpr, BNV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64); iv = tl.arange(0, BNV)
    mask = iv < NV
    tl.store(dg + p, tl.sum(tl.load(pg + p * NV + iv, mask=mask, other=0.0), axis=0))
    tl.store(db + p, tl.sum(tl.load(pb + p * NV + iv, mask=mask, other=0.0), axis=0))


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape; HV, V = v.shape[2:]
        nc = triton.cdiv(T, BC); nv = triton.cdiv(V, BV); bk = triton.next_power_of_2(K)
        cs = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, nc, HV, BC, BC), device=q.device, dtype=torch.float32)
        states = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        dvals = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(nc, B * HV)](k, g, beta, cs, inv, T=T, H=H, HV=HV, K=K, BK=bk, num_warps=4)
        _state_scan[(B * HV * nv,)](k, v, beta, cs, inv, states, dvals, T=T, H=H, HV=HV, K=K, V=V, BK=bk, num_warps=4)
        _emit[(B * HV * nc * nv,)](q, k, cs, states, dvals, out, T=T, H=H, HV=HV, K=K, V=V, BK=bk, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, cs, inv, states, dvals)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, cs, inv, states, dvals = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape; HV, V = v.shape[2:]; nv = triton.cdiv(V, BV); bk = triton.next_power_of_2(K)
        pq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq); pg = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32); pb = torch.empty_like(pg)
        dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v); dg = torch.empty_like(g); db = torch.empty_like(beta)
        _backward[(B * HV * nv,)](q, k, v, g, beta, cs, inv, states, dvals, dout, pq, pk, dv, pg, pb, T=T, H=H, HV=HV, K=K, V=V, BK=bk, num_warps=4)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T=T, H=H, HV=HV, K=K, BK=bk, NV=nv)
        _reduce_scalar[(B * T * HV,)](pg, pb, dg, db, N=B*T*HV, NV=nv, BNV=triton.next_power_of_2(nv))
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
