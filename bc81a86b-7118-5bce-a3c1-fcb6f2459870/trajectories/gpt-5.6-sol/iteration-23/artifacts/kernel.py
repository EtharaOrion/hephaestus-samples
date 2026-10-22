import torch
import triton
import triton.language as tl

C = 16
BV = 32


@triton.jit
def _make_inverse(k, g, beta, inv, T, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, IP: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ibh = tl.program_id(1).to(tl.int64)
    ib, ih = ibh // HV, ibh % HV
    ikh = ih // (HV // H)
    ot = ic * BC + tl.arange(0, BC)
    ok = tl.arange(0, BK)
    m = ot < T
    p = ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :]
    x = tl.load(k + p, mask=m[:, None] & (ok[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + ot) * HV + ih, mask=m, other=0.0).to(tl.float32)
    bb = tl.load(beta + (ib * T + ot) * HV + ih, mask=m, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    ii = tl.arange(0, BC)
    gram = tl.dot(x, tl.trans(x), input_precision=IP)
    lower = ii[:, None] > ii[None, :]
    valid = m[:, None] & m[None, :]
    mat = tl.where(lower & valid, gram * bb[:, None] * tl.exp(tl.where(lower & valid, cs[:, None] - cs[None, :], 0.0)), 0.0)
    eye = ii[:, None] == ii[None, :]
    ai = tl.where(eye, 1.0, 0.0)
    for r in range(1, BC):
        lr = tl.sum(tl.where((ii == r)[:, None], mat, 0.0), axis=0)
        row = -tl.sum(lr[:, None] * ai, axis=0) + (ii == r).to(tl.float32)
        ai = tl.where((ii == r)[:, None], row, ai)
    pa = (((ib * T + ot[:, None]) * HV + ih) * BC + ii[None, :])
    tl.store(inv + pa, ai, mask=m[:, None])


@triton.jit
def _scan_state(k, v, g, beta, inv, states, delta, T, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                NC: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
                VB: tl.constexpr, IP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv = pid % nv
    ibh = pid // nv
    ib, ih = ibh // HV, ibh % HV
    ikh = ih // (HV // H)
    ok, ov, oi = tl.arange(0, BK), iv * VB + tl.arange(0, VB), tl.arange(0, BC)
    st = tl.zeros((BK, VB), tl.float32)
    for ic in range(NC):
        ot = ic * BC + oi
        mt = ot < T
        ps = ((((ib * NC + ic) * HV + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(states + ps, st, mask=(ok[:, None] < K) & (ov[None, :] < V))
        pk = ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :]
        kk = tl.load(k + pk, mask=mt[:, None] & (ok[None, :] < K), other=0.0)
        vv = tl.load(v + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :],
                     mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        gg = tl.load(g + (ib * T + ot) * HV + ih, mask=mt, other=0.0).to(tl.float32)
        bb = tl.load(beta + (ib * T + ot) * HV + ih, mask=mt, other=0.0).to(tl.float32)
        cs = tl.cumsum(gg, axis=0)
        aa = tl.load(inv + (((ib * T + ot[:, None]) * HV + ih) * BC + oi[None, :]),
                     mask=mt[:, None], other=0.0)
        pred = tl.dot(kk, st.to(kk.dtype), input_precision=IP) * tl.exp(cs)[:, None]
        xx = bb[:, None] * (vv - pred)
        dd = tl.dot(aa.to(kk.dtype), xx.to(kk.dtype), input_precision=IP)
        pd = ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :]
        tl.store(delta + pd, dd, mask=mt[:, None] & (ov[None, :] < V))
        count = tl.sum(mt.to(tl.int32), axis=0)
        last = tl.sum(tl.where(oi == count - 1, cs, 0.0), axis=0)
        weights = tl.where(mt, tl.exp(last - cs), 0.0)
        st = st * tl.exp(last) + tl.dot(tl.trans(kk), (dd * weights[:, None]).to(kk.dtype), input_precision=IP)


@triton.jit
def _emit(q, k, g, states, delta, out, T, H: tl.constexpr, HV: tl.constexpr,
          K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
          BC: tl.constexpr, VB: tl.constexpr, SCALE: tl.constexpr, IP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv, z = pid % nv, pid // nv
    ic, ibh = z % NC, z // NC
    ib, ih = ibh // HV, ibh % HV
    ikh = ih // (HV // H)
    oi, ok, ov = tl.arange(0, BC), tl.arange(0, BK), iv * VB + tl.arange(0, VB)
    ot, mt = ic * BC + oi, ic * BC + oi < T
    qq = tl.load(q + ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :],
                 mask=mt[:, None] & (ok[None, :] < K), other=0.0) * SCALE
    kk = tl.load(k + ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :],
                 mask=mt[:, None] & (ok[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + ot) * HV + ih, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    st = tl.load(states + ((((ib * NC + ic) * HV + ih) * K + ok[:, None]) * V + ov[None, :]),
                 mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.0)
    dd = tl.load(delta + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :],
                 mask=mt[:, None] & (ov[None, :] < V), other=0.0)
    base = tl.dot(qq, st.to(qq.dtype), input_precision=IP) * tl.exp(cs)[:, None]
    score = tl.dot(qq, tl.trans(kk), input_precision=IP)
    score *= tl.where((oi[:, None] >= oi[None, :]) & mt[:, None] & mt[None, :],
                      tl.exp(tl.where((oi[:, None] >= oi[None, :]) & mt[:, None] & mt[None, :], cs[:, None] - cs[None, :], 0.0)), 0.0)
    oo = base + tl.dot(score.to(dd.dtype), dd, input_precision=IP)
    tl.store(out + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :], oo,
             mask=mt[:, None] & (ov[None, :] < V))


@triton.jit
def _adjoint(q, k, g, beta, inv, dout, dstates, T, H: tl.constexpr,
             HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
             NC: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
             VB: tl.constexpr, SCALE: tl.constexpr, IP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv, ibh = pid % nv, pid // nv
    ib, ih = ibh // HV, ibh % HV
    ikh = ih // (HV // H)
    oi, ok, ov = tl.arange(0, BC), tl.arange(0, BK), iv * VB + tl.arange(0, VB)
    ds = tl.zeros((BK, VB), tl.float32)
    for jr in range(NC):
        ic = NC - 1 - jr
        ot, mt = ic * BC + oi, ic * BC + oi < T
        pds = ((((ib * NC + ic) * HV + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(dstates + pds, ds, mask=(ok[:, None] < K) & (ov[None, :] < V))
        qq = tl.load(q + ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :], mask=mt[:, None] & (ok[None, :] < K), other=0.0) * SCALE
        kk = tl.load(k + ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :], mask=mt[:, None] & (ok[None, :] < K), other=0.0)
        do = tl.load(dout + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :], mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        gg = tl.load(g + (ib * T + ot) * HV + ih, mask=mt, other=0.0).to(tl.float32)
        bb = tl.load(beta + (ib * T + ot) * HV + ih, mask=mt, other=0.0).to(tl.float32)
        cs = tl.cumsum(gg, axis=0)
        count = tl.sum(mt.to(tl.int32), axis=0)
        last = tl.sum(tl.where(oi == count - 1, cs, 0.0), axis=0)
        aa = tl.load(inv + (((ib * T + ot[:, None]) * HV + ih) * BC + oi[None, :]), mask=mt[:, None], other=0.0)
        score = tl.dot(qq, tl.trans(kk), input_precision=IP)
        rr = score * tl.where((oi[:, None] >= oi[None, :]) & mt[:, None] & mt[None, :], tl.exp(tl.where((oi[:, None] >= oi[None, :]) & mt[:, None] & mt[None, :], cs[:, None] - cs[None, :], 0.0)), 0.0)
        z = kk * tl.exp(last - cs)[:, None]
        ddelta = tl.dot(tl.trans(rr).to(do.dtype), do, input_precision=IP) + tl.dot(z, ds.to(z.dtype), input_precision=IP)
        dx = tl.dot(tl.trans(aa).to(ddelta.dtype), ddelta, input_precision=IP)
        ds = ds * tl.exp(last) + tl.dot(tl.trans(qq), (do * tl.exp(cs)[:, None]).to(qq.dtype), input_precision=IP)
        ds -= tl.dot(tl.trans(kk), (dx * (bb * tl.exp(cs))[:, None]).to(kk.dtype), input_precision=IP)


@triton.jit
def _chunk_grad(q, k, v, g, beta, inv, states, delta, dstates, dout,
                pq, pkout, dv, pdc, pdb, T, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                NV: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
                VB: tl.constexpr, SCALE: tl.constexpr, IP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    iv, z = pid % NV, pid // NV
    ic, ibh = z % NC, z // NC
    ib, ih = ibh // HV, ibh % HV
    ikh = ih // (HV // H)
    oi, ok, ov = tl.arange(0, BC), tl.arange(0, BK), iv * VB + tl.arange(0, VB)
    ot, mt = ic * BC + oi, ic * BC + oi < T
    mk, mv = ok < K, ov < V
    qq0 = tl.load(q + ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
    qq = qq0 * SCALE
    kk = tl.load(k + ((ib * T + ot[:, None]) * H + ikh) * K + ok[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
    vv = tl.load(v + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :], mask=mt[:, None] & mv[None, :], other=0.0)
    do = tl.load(dout + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :], mask=mt[:, None] & mv[None, :], other=0.0)
    dd = tl.load(delta + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :], mask=mt[:, None] & mv[None, :], other=0.0)
    st = tl.load(states + ((((ib * NC + ic) * HV + ih) * K + ok[:, None]) * V + ov[None, :]), mask=mk[:, None] & mv[None, :], other=0.0)
    ds = tl.load(dstates + ((((ib * NC + ic) * HV + ih) * K + ok[:, None]) * V + ov[None, :]), mask=mk[:, None] & mv[None, :], other=0.0)
    gg = tl.load(g + (ib * T + ot) * HV + ih, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(beta + (ib * T + ot) * HV + ih, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    count = tl.sum(mt.to(tl.int32), axis=0)
    last = tl.sum(tl.where(oi == count - 1, cs, 0.0), axis=0)
    aa = tl.load(inv + (((ib * T + ot[:, None]) * HV + ih) * BC + oi[None, :]), mask=mt[:, None], other=0.0)
    e = tl.where(mt[:, None] & mt[None, :], tl.exp(tl.where((oi[:, None] >= oi[None, :]) & mt[:, None] & mt[None, :], cs[:, None] - cs[None, :], 0.0)), 0.0)
    low = oi[:, None] >= oi[None, :]
    score = tl.dot(qq, tl.trans(kk), input_precision=IP)
    rr = tl.where(low, score * e, 0.0)
    zmat = kk * tl.exp(last - cs)[:, None]
    pred = tl.dot(kk, st.to(kk.dtype), input_precision=IP) * tl.exp(cs)[:, None]
    xx = bb[:, None] * (vv - pred)
    ddelta = tl.dot(tl.trans(rr).to(do.dtype), do, input_precision=IP) + tl.dot(zmat, ds.to(zmat.dtype), input_precision=IP)
    dx = tl.dot(tl.trans(aa).to(ddelta.dtype), ddelta, input_precision=IP)
    dl = -tl.dot(dx.to(dd.dtype), tl.trans(dd), input_precision=IP)
    dl = tl.where((oi[:, None] > oi[None, :]) & mt[:, None] & mt[None, :], dl, 0.0)
    dr = tl.dot(do, tl.trans(dd), input_precision=IP)
    dr = tl.where(low & mt[:, None] & mt[None, :], dr, 0.0)
    dz = tl.dot(dd, tl.trans(ds).to(dd.dtype), input_precision=IP)
    dy = -dx * bb[:, None]
    dinner = dr * e
    gram = tl.dot(kk, tl.trans(kk), input_precision=IP)
    ll = tl.where(oi[:, None] > oi[None, :], gram * bb[:, None] * e, 0.0)
    dgmat = dl * (bb[:, None] * e)
    dqv = tl.dot(do, tl.trans(st).to(do.dtype), input_precision=IP) * tl.exp(cs)[:, None]
    dqv += tl.dot(dinner.to(kk.dtype), kk, input_precision=IP)
    dqv *= SCALE
    dkv = tl.dot(dy, tl.trans(st).to(dy.dtype), input_precision=IP) * tl.exp(cs)[:, None]
    dkv += tl.dot(dgmat.to(kk.dtype), kk, input_precision=IP) + tl.dot(tl.trans(dgmat).to(kk.dtype), kk, input_precision=IP)
    dkv += tl.dot(tl.trans(dinner).to(qq.dtype), qq, input_precision=IP)
    dkv += dz * tl.exp(last - cs)[:, None]
    tl.store(pq + ((((ib * T + ot[:, None]) * HV + ih) * NV + iv) * K + ok[None, :]), dqv, mask=mt[:, None] & mk[None, :])
    tl.store(pkout + ((((ib * T + ot[:, None]) * HV + ih) * NV + iv) * K + ok[None, :]), dkv, mask=mt[:, None] & mk[None, :])
    tl.store(dv + ((ib * T + ot[:, None]) * HV + ih) * V + ov[None, :], dx * bb[:, None], mask=mt[:, None] & mv[None, :])
    db = tl.sum(dx * (vv - pred), axis=1)
    db += tl.sum(dl * gram * e, axis=1)
    base_o = tl.dot(qq, st.to(qq.dtype), input_precision=IP) * tl.exp(cs)[:, None]
    dc = tl.sum(do * base_o, axis=1) + tl.sum(dy * pred, axis=1)
    lr = dl * ll
    dc += tl.sum(lr, axis=1) - tl.sum(lr, axis=0)
    rprod = dr * rr
    dc += tl.sum(rprod, axis=1) - tl.sum(rprod, axis=0)
    zp = tl.sum(dz * zmat, axis=1)
    dc -= zp
    dc += tl.where(oi == tl.sum(mt.to(tl.int32), axis=0) - 1, tl.sum(zp, axis=0) + tl.sum(ds * (st * tl.exp(last)), axis=None), 0.0)
    ps = ((ib * T + ot) * HV + ih) * NV + iv
    tl.store(pdc + ps, dc, mask=mt)
    tl.store(pdb + ps, db, mask=mt)


@triton.jit
def _reduce_qk(pq, pk, dq, dk, T, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    it, z = pid % T, pid // T
    ih, ib = z % H, z // H
    ok = tl.arange(0, BK)
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    groups = HV // H
    for r in range(groups):
        vh = ih * groups + r
        for iv in range(NV):
            p = ((((ib * T + it) * HV + vh) * NV + iv) * K + ok)
            aq += tl.load(pq + p, mask=ok < K, other=0.0)
            ak += tl.load(pk + p, mask=ok < K, other=0.0)
    outp = ((ib * T + it) * H + ih) * K + ok
    tl.store(dq + outp, aq, mask=ok < K)
    tl.store(dk + outp, ak, mask=ok < K)


@triton.jit
def _reduce_scalar(pdc, pdb, dg, db, T, HV: tl.constexpr,
                   NV: tl.constexpr, BC: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    ih, z = pid % HV, pid // HV
    it, ib = z % T, z // T
    pos = it % BC
    accg = 0.0
    accb = 0.0
    for j in range(BC):
        jt = it - pos + j
        valid = (j >= pos) & (jt < T)
        for iv in range(NV):
            p = ((ib * T + jt) * HV + ih) * NV + iv
            accg += tl.load(pdc + p, mask=valid, other=0.0)
            if j == pos:
                accb += tl.load(pdb + p)
    p = (ib * T + it) * HV + ih
    tl.store(dg + p, accg)
    tl.store(db + p, accb)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        nc, nv = triton.cdiv(T, C), triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        ip = "ieee" if q.dtype == torch.float32 else "tf32"
        inv = torch.empty((B, T, HV, C), device=q.device, dtype=torch.float32)
        states = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty_like(v)
        out = torch.empty_like(v)
        _make_inverse[(nc, B * HV)](k, g, beta, inv, T, H=H, HV=HV, K=K, BK=bk, BC=C, IP=ip, num_warps=4)
        _scan_state[(B * HV * nv,)](k, v, g, beta, inv, states, delta, T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, BC=C, VB=BV, IP=ip, num_warps=4)
        _emit[(B * HV * nc * nv,)](q, k, g, states, delta, out, T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, BC=C, VB=BV, SCALE=K ** -0.5, IP=ip, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, inv, states, delta)
        ctx.dims = (B, T, H, HV, K, V, nc, nv, bk, ip)
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        q, k, v, g, beta, inv, states, delta = ctx.saved_tensors
        B, T, H, HV, K, V, nc, nv, bk, ip = ctx.dims
        dstates = torch.empty_like(states, dtype=torch.float32)
        pq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        pdc = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32)
        pdb = torch.empty_like(pdc)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        _adjoint[(B * HV * nv,)](q, k, g, beta, inv, dout, dstates, T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, BC=C, VB=BV, SCALE=K ** -0.5, IP=ip, num_warps=4)
        _chunk_grad[(B * HV * nc * nv,)](q, k, v, g, beta, inv, states, delta, dstates, dout, pq, pk, dv, pdc, pdb, T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, BC=C, VB=BV, SCALE=K ** -0.5, IP=ip, num_warps=4)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T, H=H, HV=HV, K=K, NV=nv, BK=bk, num_warps=4)
        _reduce_scalar[(B * T * HV,)](pdc, pdb, dg, db, T, HV=HV, NV=nv, BC=C)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
