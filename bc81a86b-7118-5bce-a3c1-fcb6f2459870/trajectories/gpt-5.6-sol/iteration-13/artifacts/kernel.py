import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(16)

@triton.jit
def _mm(a, b, dtype: tl.constexpr):
    return tl.dot(a.to(dtype), b.to(dtype), input_precision='tf32')



@triton.jit
def _build_wy(k, g, beta, gc, inv, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ih = tl.program_id(1).to(tl.int64)
    b, hv = ih // HV, ih % HV
    h = hv // (HV // H)
    t, x = ic * C + tl.arange(0, C), tl.arange(0, BK)
    mt = t < T
    pk = ((b*T+t[:, None])*H+h)*K+x[None, :]
    kk = tl.load(k+pk, mask=mt[:, None] & (x[None, :] < K), other=0.)
    pz = (b*T+t)*HV+hv
    gg = tl.load(g+pz, mask=mt, other=0.).to(tl.float32)
    bb = tl.load(beta+pz, mask=mt, other=0.).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    gram = tl.dot(kk, tl.trans(kk), input_precision='tf32')
    row, col = tl.arange(0, C)[:, None], tl.arange(0, C)[None, :]
    low = (row > col) & mt[:, None] & mt[None, :]
    aa = tl.where(low, gram * bb[:, None] * tl.exp(cs[:, None]-cs[None, :]), 0.)
    mm = -aa
    for i in range(2, C):
        ai = tl.sum(tl.where(row == i, -aa, 0.), axis=0)
        ai = tl.where(tl.arange(0, C) < i, ai, 0.)
        ri = ai + tl.sum(ai[:, None] * mm, axis=0)
        mm = tl.where(row == i, ri, mm)
    mm += row == col
    pi = ((b*T+t[:, None])*HV+hv)*C+tl.arange(0, C)[None, :]
    tl.store(inv+pi, mm, mask=mt[:, None])
    tl.store(gc+pz, cs, mask=mt)


@triton.jit
def _scan_state(k, v, gc, beta, inv, states, delta, NC, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, ih = pid % nv, pid // nv
    b, hv = ih // HV, ih % HV
    h = hv // (HV // H)
    x, y = tl.arange(0, BK), iv*BV+tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        t = ic*C+tl.arange(0, C)
        mt = t < T
        ps = ((((b*NC+ic)*HV+hv)*K+x[:, None])*V+y[None, :])
        tl.store(states+ps, state, mask=(x[:, None] < K) & (y[None, :] < V))
        pk = ((b*T+t[:, None])*H+h)*K+x[None, :]
        kk = tl.load(k+pk, mask=mt[:, None] & (x[None, :] < K), other=0.)
        pz = (b*T+t)*HV+hv
        cs = tl.load(gc+pz, mask=mt, other=0.)
        bb = tl.load(beta+pz, mask=mt, other=0.).to(tl.float32)
        vv = tl.load(v+pz[:, None]*V+y[None, :], mask=mt[:, None] & (y[None, :] < V), other=0.)
        last = tl.minimum(C-1, T-ic*C-1)
        end = tl.sum(tl.where(tl.arange(0, C) == last, cs, 0.), axis=0)
        ke = kk*tl.exp(cs)[:, None]
        rhs = (vv-_mm(ke, state, k.dtype.element_ty))*bb[:, None]
        pm = ((b*T+t[:, None])*HV+hv)*C+tl.arange(0, C)[None, :]
        mm = tl.load(inv+pm, mask=mt[:, None], other=0.)
        dd = _mm(mm, rhs, k.dtype.element_ty)
        tl.store(delta+pz[:, None]*V+y[None, :], dd, mask=mt[:, None] & (y[None, :] < V))
        state = tl.exp(end)*state + _mm(tl.trans(kk*tl.exp(end-cs)[:, None]), dd, k.dtype.element_ty)


@triton.jit
def _emit(q, k, gc, states, delta, out, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, z = pid % nv, pid // nv
    ic, ih = z % NC, z // NC
    b, hv = ih // HV, ih % HV
    h = hv // (HV // H)
    t, x, y = ic*C+tl.arange(0, C), tl.arange(0, BK), iv*BV+tl.arange(0, BV)
    mt = t < T
    pk = ((b*T+t[:, None])*H+h)*K+x[None, :]
    qq = tl.load(q+pk, mask=mt[:, None] & (x[None, :] < K), other=0.)*SCALE
    kk = tl.load(k+pk, mask=mt[:, None] & (x[None, :] < K), other=0.)
    pz = (b*T+t)*HV+hv
    cs = tl.load(gc+pz, mask=mt, other=0.)
    ps = ((((b*NC+ic)*HV+hv)*K+x[:, None])*V+y[None, :])
    ss = tl.load(states+ps, mask=(x[:, None] < K) & (y[None, :] < V), other=0.)
    dd = tl.load(delta+pz[:, None]*V+y[None, :], mask=mt[:, None] & (y[None, :] < V), other=0.)
    dots = _mm(qq, tl.trans(kk), q.dtype.element_ty)
    ii = tl.arange(0, C)
    ca = (ii[:, None] >= ii[None, :]) & mt[:, None] & mt[None, :]
    ll = tl.where(ca, dots*tl.exp(cs[:, None]-cs[None, :]), 0.)
    oo = _mm(qq*tl.exp(cs)[:, None], ss, q.dtype.element_ty) + _mm(ll, dd, q.dtype.element_ty)
    tl.store(out+pz[:, None]*V+y[None, :], oo, mask=mt[:, None] & (y[None, :] < V))


@triton.jit
def _reverse(q, k, v, gc, beta, inv, states, delta, dout, dqp, dkp, dgp, dbp, dv, NC, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, ih = pid % nv, pid // nv
    b, hv = ih // HV, ih % HV
    h = hv // (HV // H)
    x, y, ii = tl.arange(0, BK), iv*BV+tl.arange(0, BV), tl.arange(0, C)
    row, col = ii[:, None], ii[None, :]
    ds = tl.zeros((BK, BV), tl.float32)
    for ir in range(NC):
        ic = NC-1-ir
        t, mt = ic*C+ii, ic*C+ii < T
        pk = ((b*T+t[:, None])*H+h)*K+x[None, :]
        q0 = tl.load(q+pk, mask=mt[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
        qq = q0*SCALE
        kk = tl.load(k+pk, mask=mt[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
        pz = (b*T+t)*HV+hv
        cs = tl.load(gc+pz, mask=mt, other=0.)
        bb = tl.load(beta+pz, mask=mt, other=0.).to(tl.float32)
        vv = tl.load(v+pz[:, None]*V+y[None, :], mask=mt[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
        dd = tl.load(delta+pz[:, None]*V+y[None, :], mask=mt[:, None] & (y[None, :] < V), other=0.)
        do = tl.load(dout+pz[:, None]*V+y[None, :], mask=mt[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
        ps = ((((b*NC+ic)*HV+hv)*K+x[:, None])*V+y[None, :])
        s0 = tl.load(states+ps, mask=(x[:, None] < K) & (y[None, :] < V), other=0.)
        pm = ((b*T+t[:, None])*HV+hv)*C+ii[None, :]
        mm = tl.load(inv+pm, mask=mt[:, None], other=0.)
        last = tl.minimum(C-1, T-ic*C-1)
        end = tl.sum(tl.where(ii == last, cs, 0.), axis=0)
        ee, ff = tl.exp(cs), tl.exp(end-cs)
        ke, kf, qe = kk*ee[:, None], kk*ff[:, None], qq*ee[:, None]
        dots = _mm(qq, tl.trans(kk), q.dtype.element_ty)
        ca = (row >= col) & mt[:, None] & mt[None, :]
        ratio = tl.where(ca, tl.exp(cs[:, None]-cs[None, :]), 0.)
        ll = dots*ratio
        dqe = _mm(do, tl.trans(s0), q.dtype.element_ty)
        dss = _mm(tl.trans(qe), do, q.dtype.element_ty) + tl.exp(end)*ds
        dld = _mm(do, tl.trans(dd), q.dtype.element_ty)
        ddd = _mm(tl.trans(ll), do, q.dtype.element_ty) + _mm(kf, ds, q.dtype.element_ty)
        dkf = _mm(dd, tl.trans(ds), q.dtype.element_ty)
        pred = _mm(ke, s0, q.dtype.element_ty)
        rr = (vv-pred)*bb[:, None]
        dm = _mm(ddd, tl.trans(rr), q.dtype.element_ty)
        dr = _mm(tl.trans(mm), ddd, q.dtype.element_ty)
        da = -_mm(_mm(tl.trans(mm), dm, q.dtype.element_ty), tl.trans(mm), q.dtype.element_ty)
        strict = (row > col) & mt[:, None] & mt[None, :]
        da = tl.where(strict, da, 0.)
        dp = -dr*bb[:, None]
        dke = _mm(dp, tl.trans(s0), q.dtype.element_ty)
        dss += _mm(tl.trans(ke), dp, q.dtype.element_ty)
        dl0 = dld*ratio
        dqq = dqe*ee[:, None] + _mm(dl0, kk, q.dtype.element_ty)
        dkk = _mm(tl.trans(dl0), qq, q.dtype.element_ty)
        gram = _mm(kk, tl.trans(kk), q.dtype.element_ty)
        ar = tl.where(strict, tl.exp(cs[:, None]-cs[None, :]), 0.)
        coef = da*bb[:, None]*ar
        dkk += _mm(coef, kk, q.dtype.element_ty) + _mm(tl.trans(coef), kk, q.dtype.element_ty)
        dkk += dke*ee[:, None] + dkf*ff[:, None]
        db = tl.sum(dr*(vv-pred), axis=1) + tl.sum(da*gram*ar, axis=1)
        dg = tl.sum(dqe*qe, axis=1)
        gl = dld*ll
        dg += tl.sum(gl, axis=1)-tl.sum(gl, axis=0)
        ga = da*(gram*bb[:, None]*ar)
        dg += tl.sum(ga, axis=1)-tl.sum(ga, axis=0)
        dg += tl.sum(dke*ke, axis=1)
        gf = tl.sum(dkf*kf, axis=1)
        dg -= gf
        state_term = tl.sum(ds*(tl.exp(end)*s0), axis=1)
        dend = tl.sum(gf, axis=0) + tl.sum(state_term, axis=0)
        dg += tl.where(ii == last, dend, 0.)
        dg = tl.cumsum(dg, axis=0, reverse=True)
        pb = (((b*T+t)*HV+hv)*nv+iv)
        tl.store(dqp+pb[:, None]*K+x[None, :], dqq*SCALE, mask=mt[:, None] & (x[None, :] < K))
        tl.store(dkp+pb[:, None]*K+x[None, :], dkk, mask=mt[:, None] & (x[None, :] < K))
        tl.store(dgp+pb, dg, mask=mt)
        tl.store(dbp+pb, db, mask=mt)
        tl.store(dv+pz[:, None]*V+y[None, :], dr*bb[:, None], mask=mt[:, None] & (y[None, :] < V))
        ds = dss


@triton.jit
def _sum_qk(dqp, dkp, dq, dk, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    h, z = pid % H, pid // H
    t, b = z % T, z // T
    x, nv, rep = tl.arange(0, BK), tl.cdiv(V, BV), HV // H
    aq, ak = tl.zeros((BK,), tl.float32), tl.zeros((BK,), tl.float32)
    for r in range(rep):
        hv = h*rep+r
        for iv in range(nv):
            base = (((b*T+t)*HV+hv)*nv+iv)*K
            aq += tl.load(dqp+base+x, mask=x < K, other=0.)
            ak += tl.load(dkp+base+x, mask=x < K, other=0.)
    out = ((b*T+t)*H+h)*K+x
    tl.store(dq+out, aq, mask=x < K)
    tl.store(dk+out, ak, mask=x < K)


@triton.jit
def _sum_scalar(dgp, dbp, dg, db, T: tl.constexpr, HV: tl.constexpr, V: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    hv, z = pid % HV, pid // HV
    t, b = z % T, z // T
    nv, ag, ab = tl.cdiv(V, BV), 0., 0.
    for iv in range(nv):
        p = ((b*T+t)*HV+hv)*nv+iv
        ag += tl.load(dgp+p)
        ab += tl.load(dbp+p)
    p = (b*T+t)*HV+hv
    tl.store(dg+p, ag)
    tl.store(db+p, ab)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        NC, NV, BK = triton.cdiv(T, C), triton.cdiv(V, BV), triton.next_power_of_2(K)
        gc = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, T, HV, C), device=q.device, dtype=torch.float32)
        states = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(NC, B*HV)](k, g, beta, gc, inv, T=T, H=H, HV=HV, K=K, BK=BK, num_warps=4)
        _scan_state[(B*HV*NV,)](k, v, gc, beta, inv, states, delta, NC, T=T, H=H, HV=HV, K=K, V=V, BK=BK, num_warps=8)
        _emit[(B*HV*NC*NV,)](q, k, gc, states, delta, out, T=T, H=H, HV=HV, K=K, V=V, NC=NC, BK=BK, SCALE=K**-0.5, num_warps=4)
        ctx.save_for_backward(q, k, v, beta, gc, inv, states, delta)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, beta, gc, inv, states, delta = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        NC, NV, BK = triton.cdiv(T, C), triton.cdiv(V, BV), triton.next_power_of_2(K)
        dqp = torch.empty((B, T, HV, NV, K), device=q.device, dtype=torch.float32)
        dkp = torch.empty_like(dqp)
        dgp = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        dbp = torch.empty_like(dgp)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg, db = torch.empty_like(gc), torch.empty_like(beta)
        _reverse[(B*HV*NV,)](q, k, v, gc, beta, inv, states, delta, dout, dqp, dkp, dgp, dbp, dv, NC, T=T, H=H, HV=HV, K=K, V=V, BK=BK, SCALE=K**-0.5, num_warps=4)
        _sum_qk[(B*T*H,)](dqp, dkp, dq, dk, T=T, H=H, HV=HV, K=K, V=V, BK=BK, num_warps=4)
        _sum_scalar[(B*T*HV,)](dgp, dbp, dg, db, T=T, HV=HV, V=V, num_warps=1)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
