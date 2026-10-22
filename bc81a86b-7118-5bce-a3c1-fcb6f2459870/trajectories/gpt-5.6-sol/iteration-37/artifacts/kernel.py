import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _safe_exp(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(0.0, x)))


@triton.jit
def _build_wy(k, g, beta, inv, w, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
              IP: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ibh = tl.program_id(1).to(tl.int64)
    b, h = ibh // HV, ibh % HV
    kh = h // (HV // H)
    r = tl.arange(0, BC)
    x = tl.arange(0, BK)
    t = ic * BC + r
    mt = t < T
    kp = k + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
    kb = tl.load(kp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    gp = g + (b * T + t) * HV + h
    bp = beta + (b * T + t) * HV + h
    gb = tl.load(gp, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(bp, mask=mt, other=0.0).to(tl.float32)
    c = tl.cumsum(gb, axis=0)
    gram = tl.dot(kb, tl.trans(kb), input_precision=IP)
    rr = r[:, None]
    cc = r[None, :]
    lower = (rr > cc) & mt[:, None] & mt[None, :]
    ratio = _safe_exp(c[:, None] - c[None, :])
    a = tl.where(lower, bb[:, None] * ratio * gram, 0.0)
    eye = (rr == cc).to(tl.float32)
    m = eye
    for i in range(BC):
        ai = tl.sum(tl.where(rr == i, a, 0.0), axis=0)
        row = tl.where(r == i, 1.0, 0.0) - tl.sum(ai[:, None] * m, axis=0)
        m = tl.where(rr == i, row[None, :], m)
    e = _safe_exp(c)
    xb = kb.to(tl.float32) * (bb * e)[:, None]
    wb = tl.dot(m, xb, input_precision=IP)
    ip = inv + (((b * ((T + BC - 1) // BC) + ic) * HV + h) * BC + r[:, None]) * BC + r[None, :]
    wp = w + ((b * T + t[:, None]) * HV + h) * K + x[None, :]
    tl.store(ip, m, mask=mt[:, None] & mt[None, :])
    tl.store(wp, wb, mask=mt[:, None] & (x[None, :] < K))


@triton.jit
def _state_scan(k, v, g, beta, inv, w, states, delta,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                BK: tl.constexpr, IP: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    ibh, iv = p // nv, p % nv
    b, h = ibh // HV, ibh % HV
    kh = h // (HV // H)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    r = tl.arange(0, BC)
    s = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        t = ic * BC + r
        mt = t < T
        sp = states + ((((b * NC + ic) * HV + h) * K + x[:, None]) * V + y[None, :])
        tl.store(sp, s, mask=(x[:, None] < K) & (y[None, :] < V))
        ip = inv + (((b * NC + ic) * HV + h) * BC + r[:, None]) * BC + r[None, :]
        mb = tl.load(ip, mask=mt[:, None] & mt[None, :], other=0.0)
        vp = v + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
        vb = tl.load(vp, mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        bp = beta + (b * T + t) * HV + h
        bb = tl.load(bp, mask=mt, other=0.0).to(tl.float32)
        u = tl.dot(mb.to(vb.dtype), (vb * bb[:, None]).to(vb.dtype), input_precision=IP)
        wp = w + ((b * T + t[:, None]) * HV + h) * K + x[None, :]
        wb = tl.load(wp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        d = u - tl.dot(wb, s.to(wb.dtype), input_precision=IP)
        dp = delta + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
        tl.store(dp, d, mask=mt[:, None] & (y[None, :] < V))
        gp = g + (b * T + t) * HV + h
        c = tl.cumsum(tl.load(gp, mask=mt, other=0.0).to(tl.float32), axis=0)
        last = tl.sum(tl.where(r == BC - 1, c, 0.0), axis=0)
        # Invalid tail lanes carry zero gates, so the final lane equals the chunk sum.
        eend = _safe_exp(last)
        kp = k + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
        kb = tl.load(kp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        pend = kb.to(tl.float32) * _safe_exp(last - c)[:, None]
        s = eend * s + tl.dot(tl.trans(pend), d, input_precision=IP)


@triton.jit
def _emit(q, k, g, states, delta, out, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
          NC: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr,
          IP: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    iv, z = p % nv, p // nv
    ic, ibh = z % NC, z // NC
    b, h = ibh // HV, ibh % HV
    kh = h // (HV // H)
    r = tl.arange(0, BC)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    t = ic * BC + r
    mt = t < T
    qp = q + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
    kp = k + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
    qb = tl.load(qp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
    kb = tl.load(kp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    gp = g + (b * T + t) * HV + h
    c = tl.cumsum(tl.load(gp, mask=mt, other=0.0).to(tl.float32), axis=0)
    sp = states + ((((b * NC + ic) * HV + h) * K + x[:, None]) * V + y[None, :])
    sb = tl.load(sp, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    dp = delta + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
    db = tl.load(dp, mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
    cross = tl.dot(qb, sb, input_precision=IP) * _safe_exp(c)[:, None]
    rr, cc = r[:, None], r[None, :]
    ratio = _safe_exp(c[:, None] - c[None, :])
    score = tl.dot(qb, tl.trans(kb), input_precision=IP)
    score = tl.where((rr >= cc) & mt[:, None] & mt[None, :], score * ratio, 0.0)
    ob = cross + tl.dot(score.to(db.dtype), db, input_precision=IP)
    op = out + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
    tl.store(op, ob, mask=mt[:, None] & (y[None, :] < V))


@triton.jit
def _adjoint_scan(q, k, g, w, dout, adj, T: tl.constexpr, H: tl.constexpr,
                  HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  NC: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr,
                  IP: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    ibh, iv = p // nv, p % nv
    b, h = ibh // HV, ibh % HV
    kh = h // (HV // H)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    r = tl.arange(0, BC)
    ds = tl.zeros((BK, BV), tl.float32)
    for iz in range(NC):
        ic = NC - 1 - iz
        t = ic * BC + r
        mt = t < T
        ap = adj + ((((b * NC + ic) * HV + h) * K + x[:, None]) * V + y[None, :])
        tl.store(ap, ds, mask=(x[:, None] < K) & (y[None, :] < V))
        qp = q + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
        kp = k + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
        wp = w + ((b * T + t[:, None]) * HV + h) * K + x[None, :]
        dop = dout + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
        qb = tl.load(qp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
        kb = tl.load(kp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        wb = tl.load(wp, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        do = tl.load(dop, mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        gp = g + (b * T + t) * HV + h
        c = tl.cumsum(tl.load(gp, mask=mt, other=0.0).to(tl.float32), axis=0)
        last = tl.sum(tl.where(r == BC - 1, c, 0.0), axis=0)
        ratio = _safe_exp(c[:, None] - c[None, :])
        score = tl.dot(qb, tl.trans(kb), input_precision=IP)
        score = tl.where((r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :], score * ratio, 0.0)
        pend = kb.to(tl.float32) * _safe_exp(last - c)[:, None]
        dd = tl.dot(tl.trans(score), do, input_precision=IP) + tl.dot(pend, ds, input_precision=IP)
        ds = tl.dot(tl.trans(qb), do * _safe_exp(c)[:, None], input_precision=IP) \
             + _safe_exp(last) * ds - tl.dot(tl.trans(wb), dd, input_precision=IP)


@triton.jit
def _chunk_grad(q, k, v, g, beta, inv, w, states, delta, dout, adj,
                pq, pk, pg, pb, dv, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                NC: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr,
                IP: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    iv, z = p % nv, p // nv
    ic, ibh = z % NC, z // NC
    b, h = ibh // HV, ibh % HV
    kh = h // (HV // H)
    r = tl.arange(0, BC)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    t = ic * BC + r
    mt = t < T
    mx = x < K
    my = y < V
    qp = q + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
    kp = k + ((b * T + t[:, None]) * H + kh) * K + x[None, :]
    vp = v + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
    wp = w + ((b * T + t[:, None]) * HV + h) * K + x[None, :]
    dp = delta + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
    dop = dout + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
    qb = tl.load(qp, mask=mt[:, None] & mx[None, :], other=0.0).to(tl.float32) * SCALE
    kb = tl.load(kp, mask=mt[:, None] & mx[None, :], other=0.0).to(tl.float32)
    vb = tl.load(vp, mask=mt[:, None] & my[None, :], other=0.0).to(tl.float32)
    wb = tl.load(wp, mask=mt[:, None] & mx[None, :], other=0.0).to(tl.float32)
    db = tl.load(dp, mask=mt[:, None] & my[None, :], other=0.0).to(tl.float32)
    do = tl.load(dop, mask=mt[:, None] & my[None, :], other=0.0).to(tl.float32)
    sp = states + ((((b * NC + ic) * HV + h) * K + x[:, None]) * V + y[None, :])
    ap = adj + ((((b * NC + ic) * HV + h) * K + x[:, None]) * V + y[None, :])
    s = tl.load(sp, mask=mx[:, None] & my[None, :], other=0.0)
    ds = tl.load(ap, mask=mx[:, None] & my[None, :], other=0.0)
    gp = g + (b * T + t) * HV + h
    bp = beta + (b * T + t) * HV + h
    gb = tl.load(gp, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(bp, mask=mt, other=0.0).to(tl.float32)
    c = tl.cumsum(gb, axis=0)
    last = tl.sum(tl.where(r == BC - 1, c, 0.0), axis=0)
    rr, cc = r[:, None], r[None, :]
    ratio = _safe_exp(c[:, None] - c[None, :])
    lower_eq = (rr >= cc) & mt[:, None] & mt[None, :]
    qk = tl.dot(qb, tl.trans(kb), input_precision=IP)
    score = tl.where(lower_eq, qk * ratio, 0.0)
    pend = kb.to(tl.float32) * _safe_exp(last - c)[:, None]
    dd = tl.dot(tl.trans(score), do, input_precision=IP) + tl.dot(pend, ds, input_precision=IP)
    dr = tl.dot(do, tl.trans(db), input_precision=IP)
    dr = tl.where(lower_eq, dr, 0.0)
    dqb = tl.dot((dr * ratio).to(kb.dtype), kb, input_precision=IP)
    dqb += tl.dot(do, tl.trans(s), input_precision=IP) * _safe_exp(c)[:, None]
    dkb = tl.dot(tl.trans((dr * ratio).to(qb.dtype)), qb, input_precision=IP)
    contrib_r = dr * ratio * qk
    dc = tl.sum(contrib_r, axis=1) - tl.sum(contrib_r, axis=0)
    dc += tl.sum(do * tl.dot(qb, s, input_precision=IP), axis=1) * _safe_exp(c)
    dpend = tl.dot(db, tl.trans(ds), input_precision=IP)
    pend_fac = _safe_exp(last - c)
    dkb += dpend * pend_fac[:, None]
    pc = tl.sum(dpend * kb, axis=1) * pend_fac
    dc -= pc
    dc += tl.where(r == BC - 1, tl.sum(pc, axis=0), 0.0)
    dw = -tl.dot(dd, tl.trans(s), input_precision=IP)
    du = dd
    ip = inv + (((b * NC + ic) * HV + h) * BC + r[:, None]) * BC + r[None, :]
    mb = tl.load(ip, mask=mt[:, None] & mt[None, :], other=0.0)
    dx = tl.dot(tl.trans(mb), dw, input_precision=IP)
    dz = tl.dot(tl.trans(mb), du, input_precision=IP)
    e = _safe_exp(c)
    xb = kb.to(tl.float32) * (bb * e)[:, None]
    zb = vb.to(tl.float32) * bb[:, None]
    dm = tl.dot(dw, tl.trans(xb), input_precision=IP) + tl.dot(du, tl.trans(zb), input_precision=IP)
    dn = -tl.dot(tl.trans(mb), tl.dot(dm, tl.trans(mb), input_precision=IP), input_precision=IP)
    da = tl.where((rr > cc) & mt[:, None] & mt[None, :], dn, 0.0)
    gram = tl.dot(kb, tl.trans(kb), input_precision=IP)
    ar = ratio * gram
    dkb += dx * (bb * e)[:, None]
    dkb += tl.dot((da * (bb[:, None] * ratio)).to(kb.dtype), kb, input_precision=IP)
    dkb += tl.dot(tl.trans((da * (bb[:, None] * ratio)).to(kb.dtype)), kb, input_precision=IP)
    dval = dz * bb[:, None]
    dbeta = tl.sum(dx * (e[:, None] * kb), axis=1) + tl.sum(dz * vb, axis=1)
    dbeta += tl.sum(da * ar, axis=1)
    ac = da * (bb[:, None] * ratio * gram)
    dc += tl.sum(ac, axis=1) - tl.sum(ac, axis=0)
    dc += tl.sum(dx * xb, axis=1)
    dgout = tl.sum(dc, axis=0) - tl.cumsum(dc, axis=0) + dc
    base = (((b * T + t) * HV + h) * nv + iv)
    pqp = pq + (base[:, None] * K + x[None, :])
    pkp = pk + (base[:, None] * K + x[None, :])
    psp = pg + base
    pbp = pb + base
    dvp = dv + ((b * T + t[:, None]) * HV + h) * V + y[None, :]
    tl.store(pqp, dqb * SCALE, mask=mt[:, None] & mx[None, :])
    tl.store(pkp, dkb, mask=mt[:, None] & mx[None, :])
    tl.store(psp, dgout, mask=mt)
    tl.store(pbp, dbeta, mask=mt)
    tl.store(dvp, dval, mask=mt[:, None] & my[None, :])


@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
               BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, BK)
    kh = p % H
    z = p // H
    t, b = z % T, z // T
    groups = HV // H
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    for j in range(groups):
        vh = kh * groups + j
        for iv in range(NV):
            base = (((b * T + t) * HV + vh) * NV + iv) * K
            aq += tl.load(pq + base + x, mask=x < K, other=0.0)
            ak += tl.load(pk + base + x, mask=x < K, other=0.0)
    out = ((b * T + t) * H + kh) * K + x
    tl.store(dq + out, aq, mask=x < K)
    tl.store(dk + out, ak, mask=x < K)


@triton.jit
def _reduce_scalar(pg, pb, dg, dbeta, N: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    a = tl.zeros((), tl.float32)
    b = tl.zeros((), tl.float32)
    for iv in range(NV):
        a += tl.load(pg + p * NV + iv)
        b += tl.load(pb + p * NV + iv)
    tl.store(dg + p, a)
    tl.store(dbeta + p, b)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        nc = triton.cdiv(T, BC)
        nv = triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        ip = "ieee" if q.dtype == torch.float32 else "tf32"
        inv = torch.empty((B, nc, HV, BC, BC), device=q.device, dtype=torch.float32)
        w = torch.empty((B, T, HV, K), device=q.device, dtype=q.dtype)
        states = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(nc, B * HV)](k, g, beta, inv, w, T=T, H=H, HV=HV, K=K, BK=bk, IP=ip)
        _state_scan[(B * HV * nv,)](k, v, g, beta, inv, w, states, delta,
            T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, IP=ip)
        _emit[(B * HV * nc * nv,)](q, k, g, states, delta, out, T=T, H=H,
            HV=HV, K=K, V=V, NC=nc, BK=bk, SCALE=K ** -0.5, IP=ip)
        ctx.save_for_backward(q, k, v, g, beta, inv, w, states, delta)
        ctx.meta = (B, T, H, HV, K, V, nc, nv, bk, ip)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, inv, w, states, delta = ctx.saved_tensors
        B, T, H, HV, K, V, nc, nv, bk, ip = ctx.meta
        dout = dout.contiguous()
        adj = torch.empty_like(states)
        pq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        pg = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        dbeta = torch.empty_like(beta)
        scale = K ** -0.5
        _adjoint_scan[(B * HV * nv,)](q, k, g, w, dout, adj, T=T, H=H,
            HV=HV, K=K, V=V, NC=nc, BK=bk, SCALE=scale, IP=ip)
        _chunk_grad[(B * HV * nc * nv,)](q, k, v, g, beta, inv, w, states,
            delta, dout, adj, pq, pk, pg, pb, dv, T=T, H=H, HV=HV, K=K,
            V=V, NC=nc, BK=bk, SCALE=scale, IP=ip)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T=T, H=H, HV=HV, K=K, NV=nv, BK=bk)
        _reduce_scalar[(B * T * HV,)](pg, pb, dg, dbeta, N=B*T*HV, NV=nv)
        return dq, dk, dv, dg, dbeta


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
