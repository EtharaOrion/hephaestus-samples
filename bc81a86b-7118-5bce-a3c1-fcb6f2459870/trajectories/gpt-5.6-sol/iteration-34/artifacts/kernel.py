import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _mm(a, b):
    return tl.dot(a, b, input_precision="tf32")


@triton.jit
def _build_wy(kp, gp, bp, pp, wp, cp, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0)
    z = tl.program_id(1)
    b = z // HV
    hv = z - b * HV
    h = hv // (HV // H)
    rr = tl.arange(0, BC)
    kk = tl.arange(0, BK)
    tt = ic * BC + rr
    valid = tt < T
    ko = ((b * T + tt[:, None]) * H + h) * K + kk[None, :]
    x = tl.load(kp + ko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
    go = (b * T + tt) * HV + hv
    gg = tl.load(gp + go, mask=valid, other=0.).to(tl.float32)
    cc = tl.cumsum(gg, axis=0)
    be = tl.load(bp + go, mask=valid, other=0.).to(tl.float32)
    decay = tl.exp(cc)
    gram = _mm(x, tl.trans(x)) * be[:, None] * tl.exp(cc[:, None] - cc[None, :])
    gram = tl.where((rr[:, None] > rr[None, :]) & valid[:, None] & valid[None, :], gram, 0.)
    inv = tl.where(rr[:, None] == rr[None, :], 1., 0.)
    for i in range(1, BC):
        row = tl.sum(tl.where(rr[:, None] == i, gram, 0.), axis=0)
        correction = tl.sum(row[:, None] * inv, axis=0)
        inv = tl.where(rr[:, None] == i, inv - correction[None, :], inv)
    inv = tl.where(valid[:, None] & valid[None, :], inv, 0.)
    weighted_k = x * (be * decay)[:, None]
    w = _mm(inv, weighted_k)
    po = (((b * ((T + BC - 1) // BC) + ic) * HV + hv) * BC + rr[:, None]) * BC + rr[None, :]
    wo = ((b * T + tt[:, None]) * HV + hv) * K + kk[None, :]
    tl.store(pp + po, inv)
    tl.store(wp + wo, w, mask=valid[:, None] & (kk[None, :] < K))
    tl.store(cp + go, cc, mask=valid)


@triton.jit
def _state_scan(kp, vp, gp, bp, pp, wp, sp, up, T: tl.constexpr,
                H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    vb = pid % nv
    z = pid // nv
    b = z // HV
    hv = z - b * HV
    h = hv // (HV // H)
    kk = tl.arange(0, BK)
    vv = vb * BV + tl.arange(0, BV)
    rr = tl.arange(0, BC)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        tt = ic * BC + rr
        valid = tt < T
        so = ((((b * NC + ic) * HV + hv) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
        tl.store(sp + so, state, mask=(kk[:, None] < K) & (vv[None, :] < V))
        po = (((b * NC + ic) * HV + hv) * BC + rr[:, None]) * BC + rr[None, :]
        inv = tl.load(pp + po)
        vo = ((b * T + tt[:, None]) * HV + hv) * V + vv[None, :]
        val = tl.load(vp + vo, mask=valid[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
        bo = (b * T + tt) * HV + hv
        beta = tl.load(bp + bo, mask=valid, other=0.).to(tl.float32)
        base = _mm(inv, val * beta[:, None])
        wo = ((b * T + tt[:, None]) * HV + hv) * K + kk[None, :]
        w = tl.load(wp + wo, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
        u = base - _mm(w, state)
        uo = ((b * T + tt[:, None]) * HV + hv) * V + vv[None, :]
        tl.store(up + uo, u, mask=valid[:, None] & (vv[None, :] < V))
        co = (b * T + tt) * HV + hv
        c = tl.load(gp + co, mask=valid, other=0.).to(tl.float32)
        endc = tl.sum(tl.where(valid & (rr == tl.minimum(BC - 1, T - ic * BC - 1)), c, 0.))
        r = tl.exp(endc - c)
        ko = ((b * T + tt[:, None]) * H + h) * K + kk[None, :]
        x = tl.load(kp + ko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
        state = tl.exp(endc) * state + _mm(tl.trans(x * r[:, None]), u)


@triton.jit
def _emit(qp, kp, gp, sp, up, op, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
          NC: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    vb = pid % nv
    xpid = pid // nv
    ic = xpid % NC
    z = xpid // NC
    b = z // HV
    hv = z - b * HV
    h = hv // (HV // H)
    rr = tl.arange(0, BC)
    kk = tl.arange(0, BK)
    vv = vb * BV + tl.arange(0, BV)
    tt = ic * BC + rr
    valid = tt < T
    qo = ((b * T + tt[:, None]) * H + h) * K + kk[None, :]
    ko = qo
    q = tl.load(qp + qo, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32) * SCALE
    k = tl.load(kp + ko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
    co = (b * T + tt) * HV + hv
    c = tl.load(gp + co, mask=valid, other=0.).to(tl.float32)
    so = ((((b * NC + ic) * HV + hv) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
    state = tl.load(sp + so, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.)
    uo = ((b * T + tt[:, None]) * HV + hv) * V + vv[None, :]
    u = tl.load(up + uo, mask=valid[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
    score = _mm(q * tl.exp(c)[:, None], state)
    pair = _mm(q, tl.trans(k)) * tl.exp(c[:, None] - c[None, :])
    pair = tl.where((rr[:, None] >= rr[None, :]) & valid[:, None] & valid[None, :], pair, 0.)
    out = score + _mm(pair, u)
    oo = ((b * T + tt[:, None]) * HV + hv) * V + vv[None, :]
    tl.store(op + oo, out, mask=valid[:, None] & (vv[None, :] < V))


@triton.jit
def _output_source(qp, kp, gp, wp, dop, ap, T: tl.constexpr,
                   H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                   SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    vb = pid % nv
    xpid = pid // nv
    ic = xpid % NC
    z = xpid // NC
    b = z // HV
    hv = z - b * HV
    h = hv // (HV // H)
    rr = tl.arange(0, BC); kk = tl.arange(0, BK); vv = vb * BV + tl.arange(0, BV)
    tt = ic * BC + rr; valid = tt < T
    qko = ((b * T + tt[:, None]) * H + h) * K + kk[None, :]
    q = tl.load(qp + qko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32) * SCALE
    k = tl.load(kp + qko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
    co = (b * T + tt) * HV + hv
    c = tl.load(gp + co, mask=valid, other=0.).to(tl.float32)
    doo = ((b * T + tt[:, None]) * HV + hv) * V + vv[None, :]
    do = tl.load(dop + doo, mask=valid[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
    pair = _mm(q, tl.trans(k)) * tl.exp(c[:, None] - c[None, :])
    pair = tl.where((rr[:, None] >= rr[None, :]) & valid[:, None] & valid[None, :], pair, 0.)
    du = _mm(tl.trans(pair), do)
    wo = ((b * T + tt[:, None]) * HV + hv) * K + kk[None, :]
    w = tl.load(wp + wo, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
    src = _mm(tl.trans(q * tl.exp(c)[:, None]), do) - _mm(tl.trans(w), du)
    ao = ((((b * NC + ic) * HV + hv) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
    tl.store(ap + ao, src, mask=(kk[:, None] < K) & (vv[None, :] < V))


@triton.jit
def _adjoint_scan(kp, gp, wp, ap, T: tl.constexpr, H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   NC: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV; vb = pid % nv; z = pid // nv
    b = z // HV; hv = z - b * HV; h = hv // (HV // H)
    rr = tl.arange(0, BC); kk = tl.arange(0, BK); vv = vb * BV + tl.arange(0, BV)
    ds = tl.zeros((BK, BV), tl.float32)
    for step in range(NC):
        ic = NC - 1 - step; tt = ic * BC + rr; valid = tt < T
        ao = ((((b * NC + ic) * HV + hv) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
        src = tl.load(ap + ao, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.)
        tl.store(ap + ao, ds, mask=(kk[:, None] < K) & (vv[None, :] < V))
        co = (b * T + tt) * HV + hv
        c = tl.load(gp + co, mask=valid, other=0.).to(tl.float32)
        last = tl.minimum(BC - 1, T - ic * BC - 1)
        ce = tl.sum(tl.where(valid & (rr == last), c, 0.))
        ko = ((b * T + tt[:, None]) * H + h) * K + kk[None, :]
        k = tl.load(kp + ko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
        wo = ((b * T + tt[:, None]) * HV + hv) * K + kk[None, :]
        w = tl.load(wp + wo, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
        zeta = _mm(k * tl.exp(ce - c)[:, None], ds)
        ds = tl.exp(ce) * ds - _mm(tl.trans(w), zeta) + src


@triton.jit
def _chunk_grad(qp, kp, vp, gp, gcp, bp, sp, up, dop, ap, dqpart, dkpart,
                dvp, dgpart, dbpart, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr,
                SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    vb = pid % NV; xpid = pid // NV; ic = xpid % NC; z = xpid // NC
    b = z // HV; hv = z - b * HV; h = hv // (HV // H)
    rr = tl.arange(0, BC); kk = tl.arange(0, BK); vv = vb * BV + tl.arange(0, BV)
    tt = ic * BC + rr; valid = tt < T
    qko = ((b * T + tt[:, None]) * H + h) * K + kk[None, :]
    q = tl.load(qp + qko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32) * SCALE
    k = tl.load(kp + qko, mask=valid[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
    vo = ((b * T + tt[:, None]) * HV + hv) * V + vv[None, :]
    val = tl.load(vp + vo, mask=valid[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
    u = tl.load(up + vo, mask=valid[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
    do = tl.load(dop + vo, mask=valid[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
    co = (b * T + tt) * HV + hv
    glog = tl.load(gcp + co, mask=valid, other=0.).to(tl.float32)
    beta = tl.load(bp + co, mask=valid, other=0.).to(tl.float32)
    so = ((((b * NC + ic) * HV + hv) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
    s0 = tl.load(sp + so, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.)
    ds = tl.load(ap + so, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.)
    dq = tl.zeros((BC, BK), tl.float32); dk = tl.zeros((BC, BK), tl.float32)
    dgv = tl.zeros((BC,), tl.float32); dbv = tl.zeros((BC,), tl.float32)
    # Reconstruct each decayed state directly from the chunk start and WY updates.
    for rev in range(BC):
        i = BC - 1 - rev
        rowmask = rr == i
        active = ic * BC + i < T
        gi = tl.sum(tl.where(rowmask, glog, 0.))
        bi = tl.sum(tl.where(rowmask, beta, 0.))
        qi = tl.sum(tl.where(rowmask[:, None], q, 0.), axis=0)
        ki = tl.sum(tl.where(rowmask[:, None], k, 0.), axis=0)
        ui = tl.sum(tl.where(rowmask[:, None], u, 0.), axis=0)
        doi = tl.sum(tl.where(rowmask[:, None], do, 0.), axis=0)
        vali = tl.sum(tl.where(rowmask[:, None], val, 0.), axis=0)
        coeff = tl.where((rr < i) & valid, tl.exp(gi - glog), 0.)
        sp_i = tl.exp(gi) * s0 + _mm(tl.trans(k * coeff[:, None]), u)
        pred = tl.sum(ki[:, None] * sp_i, axis=0)
        delta = bi * (vali - pred)
        s_i = sp_i + ki[:, None] * delta[None, :]
        dq_i = tl.sum(s_i * doi[None, :], axis=1) * SCALE
        ds += qi[:, None] * doi[None, :]
        ddelta = tl.sum(ki[:, None] * ds, axis=0)
        dk_i = tl.sum(ds * delta[None, :], axis=1)
        dpred = -bi * ddelta
        dk_i += tl.sum(sp_i * dpred[None, :], axis=1)
        dv_i = bi * ddelta
        db_i = tl.sum(ddelta * (vali - pred), axis=0)
        dsp = ds + ki[:, None] * dpred[None, :]
        dg_i = tl.sum(dsp * sp_i)
        raw_gi = tl.load(gp + ((b * T + ic * BC + i) * HV + hv), mask=active, other=0.).to(tl.float32)
        ds = tl.exp(raw_gi) * dsp
        dq = tl.where(rowmask[:, None], dq_i[None, :], dq)
        dk = tl.where(rowmask[:, None], dk_i[None, :], dk)
        dgv = tl.where(rowmask, dg_i, dgv)
        dbv = tl.where(rowmask, db_i, dbv)
        dvo = ((b * T + ic * BC + i) * HV + hv) * V + vv
        tl.store(dvp + dvo, dv_i, mask=active & (vv < V))
    partbase = ((((b * T + tt[:, None]) * HV + hv) * NV + vb) * K + kk[None, :]).to(tl.int64)
    tl.store(dqpart + partbase, dq, mask=valid[:, None] & (kk[None, :] < K))
    tl.store(dkpart + partbase, dk, mask=valid[:, None] & (kk[None, :] < K))
    sbase = (((b * T + tt) * HV + hv) * NV + vb).to(tl.int64)
    tl.store(dgpart + sbase, dgv, mask=valid)
    tl.store(dbpart + sbase, dbv, mask=valid)


@triton.jit
def _reduce_qk(dqp, dkp, dqo, dko, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
               BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64); kk = tl.arange(0, BK)
    h = pid % H; x = pid // H; t = x % T; b = x // T
    group = HV // H
    dq = tl.zeros((BK,), tl.float32); dk = tl.zeros((BK,), tl.float32)
    for j in range(group):
        hv = h * group + j
        for vb in range(NV):
            o = ((((b * T + t) * HV + hv) * NV + vb) * K + kk).to(tl.int64)
            dq += tl.load(dqp + o, mask=kk < K, other=0.)
            dk += tl.load(dkp + o, mask=kk < K, other=0.)
    out = ((b * T + t) * H + h) * K + kk
    tl.store(dqo + out, dq, mask=kk < K); tl.store(dko + out, dk, mask=kk < K)


@triton.jit
def _reduce_scalar(dgp, dbp, dgo, dbo, N: tl.constexpr, NV: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    dg = 0.; db = 0.
    for vb in range(NV):
        dg += tl.load(dgp + i * NV + vb); db += tl.load(dbp + i * NV + vb)
    tl.store(dgo + i, dg); tl.store(dbo + i, db)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape; HV = v.shape[2]; V = v.shape[-1]
        nc = triton.cdiv(T, BC); nv = triton.cdiv(V, BV); bk = triton.next_power_of_2(K)
        p = torch.empty((B, nc, HV, BC, BC), device=q.device, dtype=torch.float32)
        w = torch.empty((B, T, HV, K), device=q.device, dtype=torch.float32)
        gc = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        states = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        u = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(nc, B * HV)](k, g, beta, p, w, gc, T=T, H=H, HV=HV, K=K, BK=bk)
        _state_scan[(B * HV * nv,)](k, v, gc, beta, p, w, states, u, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk)
        _emit[(B * HV * nc * nv,)](q, k, gc, states, u, out, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, SCALE=K ** -0.5)
        ctx.save_for_backward(q, k, v, g, beta, gc, w, states, u)
        ctx.dims = (B, T, H, HV, K, V, nc, nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, gc, w, states, u = ctx.saved_tensors
        B, T, H, HV, K, V, nc, nv, bk = ctx.dims
        dout = dout.contiguous()
        adj = torch.empty_like(states, dtype=torch.float32)
        _output_source[(B * HV * nc * nv,)](q, k, gc, w, dout, adj, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, SCALE=K ** -0.5)
        _adjoint_scan[(B * HV * nv,)](k, gc, w, adj, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk)
        shape = (B, T, HV, nv, K)
        dqp = torch.empty(shape, device=q.device, dtype=torch.float32)
        dkp = torch.empty_like(dqp)
        dsp = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32)
        dbp = torch.empty_like(dsp)
        dv = torch.empty_like(v)
        _chunk_grad[(B * HV * nc * nv,)](q, k, v, g, gc, beta, states, u, dout, adj, dqp, dkp, dv, dsp, dbp, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, SCALE=K ** -0.5)
        dq = torch.empty_like(q); dk = torch.empty_like(k)
        dg = torch.empty_like(g); db = torch.empty_like(beta)
        _reduce_qk[(B * T * H,)](dqp, dkp, dq, dk, T=T, H=H, HV=HV, K=K, NV=nv, BK=bk)
        _reduce_scalar[(B * T * HV,)](dsp, dbp, dg, db, N=B*T*HV, NV=nv)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
