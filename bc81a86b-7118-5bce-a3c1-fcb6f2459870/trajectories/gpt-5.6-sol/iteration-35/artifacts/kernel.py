import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(80.0, x)))


@triton.jit
def _build_inverse(g, beta, k, csum, inv, T: tl.constexpr, H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0)
    z = tl.program_id(1)
    b = z // HV
    hv = z - b * HV
    hk = hv // (HV // H)
    r = tl.arange(0, BC)
    d = tl.arange(0, BK)
    t = ic * BC + r
    valid = t < T
    go = (b * T + t) * HV + hv
    ko = ((b * T + t[:, None]) * H + hk) * K + d[None, :]
    gg = tl.load(g + go, mask=valid, other=0.0).to(tl.float32)
    cc = tl.cumsum(gg, axis=0)
    tl.store(csum + go, cc, mask=valid)
    kk = tl.load(k + ko, mask=valid[:, None] & (d[None, :] < K), other=0.0)
    bb = tl.load(beta + go, mask=valid, other=0.0).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    rr = r[:, None]
    ss = r[None, :]
    lower = tl.where((rr > ss) & valid[:, None] & valid[None, :], bb[:, None] * gram, 0.0)
    eye = tl.where(rr == ss, 1.0, 0.0)
    pinv = eye
    for i in range(1, BC):
        li = tl.sum(tl.where(rr == i, lower, 0.0), axis=0)
        coeff = tl.sum(li[:, None] * pinv, axis=0)
        row = tl.where(r < i, -coeff, tl.where(r == i, 1.0, 0.0))
        pinv = tl.where(rr == i, row[None, :], pinv)
    base = ((b * ((T + BC - 1) // BC) + ic) * HV + hv) * BC * BC
    tl.store(inv + base + rr * BC + ss, pinv)


@triton.jit
def _state_scan(k, v, csum, beta, inv, states, updates,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = (V + BV - 1) // BV
    iv = pid % NV
    z = pid // NV
    b = z // HV
    hv = z - b * HV
    hk = hv // (HV // H)
    d = tl.arange(0, BK)
    x = tl.arange(0, BV)
    r = tl.arange(0, BC)
    nc: tl.constexpr = (T + BC - 1) // BC
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(nc):
        tt = ic * BC + r
        tm = tt < T
        vm = iv * BV + x < V
        hbase = ((((b * nc + ic) * HV + hv) * ((V + BV - 1) // BV) + iv) * K) * BV
        tl.store(states + hbase + d[:, None] * BV + x[None, :], state,
                 mask=(d[:, None] < K) & vm[None, :])
        ko = ((b * T + tt[:, None]) * H + hk) * K + d[None, :]
        vo = ((b * T + tt[:, None]) * HV + hv) * V + iv * BV + x[None, :]
        so = (b * T + tt) * HV + hv
        kk = tl.load(k + ko, mask=tm[:, None] & (d[None, :] < K), other=0.0).to(tl.float32)
        vv = tl.load(v + vo, mask=tm[:, None] & vm[None, :], other=0.0).to(tl.float32)
        cc = tl.load(csum + so, mask=tm, other=0.0)
        bb = tl.load(beta + so, mask=tm, other=0.0).to(tl.float32)
        pred = tl.dot(kk, state, input_precision="tf32")
        rhs = bb[:, None] * (_bounded_exp(-cc)[:, None] * vv - pred)
        pbase = ((b * nc + ic) * HV + hv) * BC * BC
        rr = r[:, None]
        ss = r[None, :]
        pp = tl.load(inv + pbase + rr * BC + ss)
        uu = tl.dot(pp, rhs, input_precision="tf32")
        tl.store(updates + vo, uu, mask=tm[:, None] & vm[None, :])
        raw = state + tl.dot(tl.trans(kk), uu, input_precision="tf32")
        last = tl.minimum(BC - 1, T - 1 - ic * BC)
        ce = tl.sum(tl.where(r == last, cc, 0.0), axis=0)
        state = raw * _bounded_exp(ce)


@triton.jit
def _emit(q, k, csum, states, updates, out,
          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
          K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = (V + BV - 1) // BV
    nc: tl.constexpr = (T + BC - 1) // BC
    iv = pid % NV
    z = pid // NV
    ic = z % nc
    z = z // nc
    b = z // HV
    hv = z - b * HV
    hk = hv // (HV // H)
    r = tl.arange(0, BC)
    d = tl.arange(0, BK)
    x = tl.arange(0, BV)
    tt = ic * BC + r
    tm = tt < T
    vm = iv * BV + x < V
    qo = ((b * T + tt[:, None]) * H + hk) * K + d[None, :]
    ko = ((b * T + tt[:, None]) * H + hk) * K + d[None, :]
    vo = ((b * T + tt[:, None]) * HV + hv) * V + iv * BV + x[None, :]
    so = (b * T + tt) * HV + hv
    qq = tl.load(q + qo, mask=tm[:, None] & (d[None, :] < K), other=0.0).to(tl.float32) * SCALE
    kk = tl.load(k + ko, mask=tm[:, None] & (d[None, :] < K), other=0.0).to(tl.float32)
    uu = tl.load(updates + vo, mask=tm[:, None] & vm[None, :], other=0.0)
    hbase = ((((b * nc + ic) * HV + hv) * NV + iv) * K) * BV
    state = tl.load(states + hbase + d[:, None] * BV + x[None, :],
                    mask=(d[:, None] < K) & vm[None, :], other=0.0)
    score = tl.dot(qq, tl.trans(kk), input_precision="tf32")
    score = tl.where(r[:, None] >= r[None, :], score, 0.0)
    raw = tl.dot(qq, state, input_precision="tf32") + tl.dot(score, uu, input_precision="tf32")
    cc = tl.load(csum + so, mask=tm, other=0.0)
    tl.store(out + vo, raw * _bounded_exp(cc)[:, None], mask=tm[:, None] & vm[None, :])


@triton.jit
def _adjoint_scan(q, k, v, beta, csum, inv, states, updates, dout, dstates,
                  T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = (V + BV - 1) // BV
    iv = pid % NV
    z = pid // NV
    b = z // HV
    hv = z - b * HV
    hk = hv // (HV // H)
    r = tl.arange(0, BC)
    d = tl.arange(0, BK)
    x = tl.arange(0, BV)
    nc: tl.constexpr = (T + BC - 1) // BC
    adj = tl.zeros((BK, BV), tl.float32)
    for jc in range(nc):
        ic = nc - 1 - jc
        tt = ic * BC + r
        tm = tt < T
        vm = iv * BV + x < V
        db = ((((b * nc + ic) * HV + hv) * NV + iv) * K) * BV
        tl.store(dstates + db + d[:, None] * BV + x[None, :], adj,
                 mask=(d[:, None] < K) & vm[None, :])
        qo = ((b * T + tt[:, None]) * H + hk) * K + d[None, :]
        ko = ((b * T + tt[:, None]) * H + hk) * K + d[None, :]
        vo = ((b * T + tt[:, None]) * HV + hv) * V + iv * BV + x[None, :]
        so = (b * T + tt) * HV + hv
        qq = tl.load(q + qo, mask=tm[:, None] & (d[None, :] < K), other=0.0).to(tl.float32) * SCALE
        kk = tl.load(k + ko, mask=tm[:, None] & (d[None, :] < K), other=0.0).to(tl.float32)
        uu = tl.load(updates + vo, mask=tm[:, None] & vm[None, :], other=0.0)
        do = tl.load(dout + vo, mask=tm[:, None] & vm[None, :], other=0.0).to(tl.float32)
        cc = tl.load(csum + so, mask=tm, other=0.0)
        bb = tl.load(beta + so, mask=tm, other=0.0).to(tl.float32)
        dy = do * _bounded_exp(cc)[:, None]
        score = tl.dot(qq, tl.trans(kk), input_precision="tf32")
        score = tl.where(r[:, None] >= r[None, :], score, 0.0)
        last = tl.minimum(BC - 1, T - 1 - ic * BC)
        ce = tl.sum(tl.where(r == last, cc, 0.0), axis=0)
        ee = _bounded_exp(ce)
        dx = tl.dot(tl.trans(score), dy, input_precision="tf32") + ee * tl.dot(kk, adj, input_precision="tf32")
        pbase = ((b * nc + ic) * HV + hv) * BC * BC
        pp = tl.load(inv + pbase + r[:, None] * BC + r[None, :])
        dz = tl.dot(tl.trans(pp), dx, input_precision="tf32")
        adj = tl.dot(tl.trans(qq), dy, input_precision="tf32") + ee * adj - tl.dot(tl.trans(kk), bb[:, None] * dz, input_precision="tf32")


@triton.jit
def _chunk_grad(q, k, v, beta, csum, inv, states, updates, dout, dstates,
                pq, pk, dv, pscalar, out,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = (V + BV - 1) // BV
    nc: tl.constexpr = (T + BC - 1) // BC
    iv = pid % NV
    z = pid // NV
    ic = z % nc
    z = z // nc
    b = z // HV
    hv = z - b * HV
    hk = hv // (HV // H)
    r = tl.arange(0, BC)
    d = tl.arange(0, BK)
    x = tl.arange(0, BV)
    tt = ic * BC + r
    tm = tt < T
    vm = iv * BV + x < V
    qo = ((b * T + tt[:, None]) * H + hk) * K + d[None, :]
    ko = ((b * T + tt[:, None]) * H + hk) * K + d[None, :]
    vo = ((b * T + tt[:, None]) * HV + hv) * V + iv * BV + x[None, :]
    so = (b * T + tt) * HV + hv
    qq = tl.load(q + qo, mask=tm[:, None] & (d[None, :] < K), other=0.0).to(tl.float32) * SCALE
    kk = tl.load(k + ko, mask=tm[:, None] & (d[None, :] < K), other=0.0).to(tl.float32)
    vv = tl.load(v + vo, mask=tm[:, None] & vm[None, :], other=0.0).to(tl.float32)
    uu = tl.load(updates + vo, mask=tm[:, None] & vm[None, :], other=0.0)
    do = tl.load(dout + vo, mask=tm[:, None] & vm[None, :], other=0.0).to(tl.float32)
    cc = tl.load(csum + so, mask=tm, other=0.0)
    bb = tl.load(beta + so, mask=tm, other=0.0).to(tl.float32)
    hbase = ((((b * nc + ic) * HV + hv) * NV + iv) * K) * BV
    state = tl.load(states + hbase + d[:, None] * BV + x[None, :],
                    mask=(d[:, None] < K) & vm[None, :], other=0.0)
    adj = tl.load(dstates + hbase + d[:, None] * BV + x[None, :],
                  mask=(d[:, None] < K) & vm[None, :], other=0.0)
    dy = do * _bounded_exp(cc)[:, None]
    score = tl.dot(qq, tl.trans(kk), input_precision="tf32")
    tril = r[:, None] >= r[None, :]
    score = tl.where(tril, score, 0.0)
    dscore = tl.where(tril, tl.dot(dy, tl.trans(uu), input_precision="tf32"), 0.0)
    dqv = (tl.dot(dy, tl.trans(state), input_precision="tf32") +
           tl.dot(dscore, kk, input_precision="tf32")) * SCALE
    last = tl.minimum(BC - 1, T - 1 - ic * BC)
    ce = tl.sum(tl.where(r == last, cc, 0.0), axis=0)
    ee = _bounded_exp(ce)
    dx = tl.dot(tl.trans(score), dy, input_precision="tf32") + ee * tl.dot(kk, adj, input_precision="tf32")
    pbase = ((b * nc + ic) * HV + hv) * BC * BC
    pp = tl.load(inv + pbase + r[:, None] * BC + r[None, :])
    dz = tl.dot(tl.trans(pp), dx, input_precision="tf32")
    pred = tl.dot(kk, state, input_precision="tf32")
    rv = _bounded_exp(-cc)[:, None] * vv
    residual = rv - pred
    dA = -tl.dot(dz, tl.trans(uu), input_precision="tf32")
    strict = r[:, None] > r[None, :]
    mm = tl.where(strict, dA * bb[:, None], 0.0)
    dscore = tl.dot(dy, tl.trans(uu), input_precision="tf32")
    dscore = tl.where(tril, dscore, 0.0)
    dkv = tl.dot(tl.trans(dscore), qq, input_precision="tf32")
    dkv += ee * tl.dot(uu, tl.trans(adj), input_precision="tf32")
    dkv -= (bb[:, None] * tl.dot(dz, tl.trans(state), input_precision="tf32"))
    dkv += tl.dot(mm, kk, input_precision="tf32") + tl.dot(tl.trans(mm), kk, input_precision="tf32")
    dvv = bb[:, None] * _bounded_exp(-cc)[:, None] * dz
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    dbv = tl.sum(dz * residual, axis=1) + tl.sum(tl.where(strict, dA * gram, 0.0), axis=1)
    raw = tl.dot(qq, state, input_precision="tf32") + tl.dot(score, uu, input_precision="tf32")
    dcv = tl.sum(do * (raw * _bounded_exp(cc)[:, None]), axis=1)
    dcv -= tl.sum(dz * (bb[:, None] * rv), axis=1)
    end_state = ee * (state + tl.dot(tl.trans(kk), uu, input_precision="tf32"))
    edge = tl.sum(tl.sum(adj * end_state, axis=1), axis=0)
    dcv += tl.where(r == last, edge, 0.0)
    dgv = tl.sum(tl.where(r[None, :] >= r[:, None], dcv[None, :], 0.0), axis=1)
    partbase = (((b * T + tt[:, None]) * HV + hv) * NV + iv) * K + d[None, :]
    tl.store(pq + partbase, dqv, mask=tm[:, None] & (d[None, :] < K))
    tl.store(pk + partbase, dkv, mask=tm[:, None] & (d[None, :] < K))
    tl.store(dv + vo, dvv, mask=tm[:, None] & vm[None, :])
    sb = (((b * T + tt) * HV + hv) * NV + iv) * 2
    tl.store(pscalar + sb, dgv, mask=tm)
    tl.store(pscalar + sb + 1, dbv, mask=tm)


@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BK)
    hk = pid % H
    z = pid // H
    t = z % T
    b = z // T
    NV: tl.constexpr = (V + BV - 1) // BV
    REP: tl.constexpr = HV // H
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    for j in range(REP):
        hv = hk * REP + j
        for iv in range(NV):
            base = (((b * T + t) * HV + hv) * NV + iv) * K
            aq += tl.load(pq + base + d, mask=d < K, other=0.0)
            ak += tl.load(pk + base + d, mask=d < K, other=0.0)
    base = ((b * T + t) * H + hk) * K
    tl.store(dq + base + d, aq, mask=d < K)
    tl.store(dk + base + d, ak, mask=d < K)


@triton.jit
def _reduce_scalar(ps, dg, db, T: tl.constexpr, HV: tl.constexpr, V: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    NV: tl.constexpr = (V + BV - 1) // BV
    a = 0.0
    z = 0.0
    for iv in range(NV):
        base = (pid * NV + iv) * 2
        a += tl.load(ps + base)
        z += tl.load(ps + base + 1)
    tl.store(dg + pid, a)
    tl.store(db + pid, z)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        nc = triton.cdiv(T, BC)
        nv = triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        csum = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        inv = torch.empty((B, nc, HV, BC, BC), device=q.device, dtype=torch.float32)
        states = torch.empty((B, nc, HV, nv, K, BV), device=q.device, dtype=torch.float32)
        updates = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_inverse[(nc, B * HV)](g, beta, k, csum, inv, T, H, HV, K, bk)
        _state_scan[(B * HV * nv,)](k, v, csum, beta, inv, states, updates, T, H, HV, K, V, bk)
        _emit[(B * HV * nc * nv,)](q, k, csum, states, updates, out, T, H, HV, K, V, bk, K ** -0.5)
        ctx.save_for_backward(q, k, v, beta, csum, inv, states, updates, out)
        ctx.meta = (B, T, H, HV, K, V, nc, nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, beta, csum, inv, states, updates, out = ctx.saved_tensors
        B, T, H, HV, K, V, nc, nv, bk = ctx.meta
        dout = dout.contiguous()
        dstates = torch.empty_like(states, dtype=torch.float32)
        pq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq, dtype=torch.float32)
        dv = torch.empty_like(v)
        ps = torch.empty((B, T, HV, nv, 2), device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dg = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        db = torch.empty_like(beta)
        scale = K ** -0.5
        _adjoint_scan[(B * HV * nv,)](q, k, v, beta, csum, inv, states, updates, dout, dstates,
                                      T, H, HV, K, V, bk, scale)
        _chunk_grad[(B * HV * nc * nv,)](q, k, v, beta, csum, inv, states, updates, dout, dstates,
                                         pq, pk, dv, ps, out, T, H, HV, K, V, bk, scale)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T, H, HV, K, V, bk)
        _reduce_scalar[(B * T * HV,)](ps, dg, db, T, HV, V)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
