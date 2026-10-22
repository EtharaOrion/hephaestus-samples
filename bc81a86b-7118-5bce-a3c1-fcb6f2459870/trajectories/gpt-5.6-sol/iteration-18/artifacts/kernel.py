import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _safe_exp(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(0.0, x)))


@triton.jit
def _make_wy(q, k, g, beta, inv, gcum, T: tl.constexpr, H: tl.constexpr,
             HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
             BK: tl.constexpr):
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    rr = tl.arange(0, C)
    kk = tl.arange(0, BK)
    tt = c * C + rr
    valid = tt < T
    b64 = b.to(tl.int64)
    tt64 = tt.to(tl.int64)
    go = (b64 * T + tt64) * HV + vh
    gv = tl.load(g + go, mask=valid, other=0.0).to(tl.float32)
    cs = tl.cumsum(gv, axis=0)
    tl.store(gcum + ((b64 * HV + vh) * NC + c) * C + rr, cs)
    ko = ((b64 * T + tt64[:, None]) * H + kh) * K + kk[None, :]
    km = valid[:, None] & (kk[None, :] < K)
    kv = tl.load(k + ko, mask=km, other=0.0)
    gram = tl.dot(kv, tl.trans(kv), input_precision="tf32")
    bv = tl.load(beta + go, mask=valid, other=0.0).to(tl.float32)
    lower = rr[:, None] > rr[None, :]
    mat = tl.where(lower, bv[:, None] * _safe_exp(cs[:, None] - cs[None, :]) * gram, 0.0)
    cols = tl.arange(0, C)
    a = tl.where(rr[:, None] == cols[None, :], 1.0, 0.0).to(tl.float32)
    for i in range(C):
        row = tl.where(cols == i, 1.0, 0.0).to(tl.float32)
        for j in range(i):
            lij = tl.sum(tl.where((rr[:, None] == i) & (cols[None, :] == j), mat, 0.0))
            prev = tl.sum(tl.where(rr[:, None] == j, a, 0.0), axis=0)
            row -= lij * prev
        a = tl.where(rr[:, None] == i, row[None, :], a)
    ao = (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * C + cols[None, :]
    tl.store(inv + ao, a)


@triton.jit
def _scan_state(k, v, beta, inv, gcum, states, delta,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                NV: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nv = pid % NV
    bh = pid // NV
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    rr = tl.arange(0, C)
    kk = tl.arange(0, BK)
    vv = nv * BV + tl.arange(0, BV)
    b64 = b.to(tl.int64)
    state = tl.zeros((BK, BV), tl.float32)
    for c in tl.range(0, NC):
        tt = c * C + rr
        valid = tt < T
        tt64 = tt.to(tl.int64)
        so = ((((b64 * HV + vh) * NC + c) * K + kk[:, None]) * V + vv[None, :])
        sm = (kk[:, None] < K) & (vv[None, :] < V)
        tl.store(states + so, state, mask=sm)
        ko = ((b64 * T + tt64[:, None]) * H + kh) * K + kk[None, :]
        km = valid[:, None] & (kk[None, :] < K)
        kval = tl.load(k + ko, mask=km, other=0.0)
        vo = ((b64 * T + tt64[:, None]) * HV + vh) * V + vv[None, :]
        vm = valid[:, None] & (vv[None, :] < V)
        vval = tl.load(v + vo, mask=vm, other=0.0).to(tl.float32)
        bo = (b64 * T + tt64) * HV + vh
        bet = tl.load(beta + bo, mask=valid, other=0.0).to(tl.float32)
        gg = tl.load(gcum + ((b64 * HV + vh) * NC + c) * C + rr)
        ek = kval.to(tl.float32) * _safe_exp(gg)[:, None]
        pred = tl.dot(ek, state, input_precision="tf32")
        x = bet[:, None] * (vval - pred)
        aa = tl.load(inv + (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * C + rr[None, :])
        dd = tl.dot(aa, x, input_precision="tf32")
        do = (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * V + vv[None, :]
        tl.store(delta + do, dd, mask=vm)
        glast = tl.sum(tl.where(rr == C - 1, gg, 0.0))
        p = kval.to(tl.float32) * _safe_exp(glast - gg)[:, None]
        state = _safe_exp(glast) * state + tl.dot(tl.trans(p), dd, input_precision="tf32")


@triton.jit
def _emit(q, k, gcum, states, delta, out,
          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
          K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
          NV: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    nv = pid % NV
    z = pid // NV
    c = z % NC
    bh = z // NC
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    rr = tl.arange(0, C)
    kk = tl.arange(0, BK)
    vv = nv * BV + tl.arange(0, BV)
    tt = c * C + rr
    valid = tt < T
    b64 = b.to(tl.int64)
    tt64 = tt.to(tl.int64)
    qoff = ((b64 * T + tt64[:, None]) * H + kh) * K + kk[None, :]
    km = valid[:, None] & (kk[None, :] < K)
    qv = tl.load(q + qoff, mask=km, other=0.0)
    kv = tl.load(k + qoff, mask=km, other=0.0)
    gg = tl.load(gcum + ((b64 * HV + vh) * NC + c) * C + rr)
    s = tl.load(states + ((((b64 * HV + vh) * NC + c) * K + kk[:, None]) * V + vv[None, :]),
                mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.0)
    dd = tl.load(delta + (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * V + vv[None, :],
                 mask=valid[:, None] & (vv[None, :] < V), other=0.0)
    cross = tl.dot(qv, tl.trans(kv), input_precision="tf32") * SCALE
    tri = rr[:, None] >= rr[None, :]
    weights = tl.where(tri, _safe_exp(gg[:, None] - gg[None, :]) * cross, 0.0)
    base = tl.dot(qv.to(tl.float32) * (_safe_exp(gg) * SCALE)[:, None], s, input_precision="tf32")
    result = base + tl.dot(weights, dd, input_precision="tf32")
    oo = ((b64 * T + tt64[:, None]) * HV + vh) * V + vv[None, :]
    tl.store(out + oo, result, mask=valid[:, None] & (vv[None, :] < V))


@triton.jit
def _adjoint_scan(q, k, beta, inv, gcum, delta, dout, zbuf,
                  T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                  NV: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    nv = pid % NV
    bh = pid // NV
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    rr = tl.arange(0, C)
    kk = tl.arange(0, BK)
    vv = nv * BV + tl.arange(0, BV)
    b64 = b.to(tl.int64)
    zstate = tl.zeros((BK, BV), tl.float32)
    for rc in tl.range(0, NC):
        c = NC - 1 - rc
        tt = c * C + rr
        valid = tt < T
        tt64 = tt.to(tl.int64)
        zo = ((((b64 * HV + vh) * NC + c) * K + kk[:, None]) * V + vv[None, :])
        zm = (kk[:, None] < K) & (vv[None, :] < V)
        tl.store(zbuf + zo, zstate, mask=zm)
        qoff = ((b64 * T + tt64[:, None]) * H + kh) * K + kk[None, :]
        km = valid[:, None] & (kk[None, :] < K)
        qv = tl.load(q + qoff, mask=km, other=0.0).to(tl.float32) * SCALE
        kf = tl.load(k + qoff, mask=km, other=0.0).to(tl.float32)
        vo = ((b64 * T + tt64[:, None]) * HV + vh) * V + vv[None, :]
        vm = valid[:, None] & (vv[None, :] < V)
        od = tl.load(dout + vo, mask=vm, other=0.0).to(tl.float32)
        bet = tl.load(beta + (b64 * T + tt64) * HV + vh, mask=valid, other=0.0).to(tl.float32)
        gg = tl.load(gcum + ((b64 * HV + vh) * NC + c) * C + rr)
        aa = tl.load(inv + (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * C + rr[None, :])
        dd = tl.load(delta + (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * V + vv[None, :], mask=vm, other=0.0)
        glast = tl.sum(tl.where(rr == C - 1, gg, 0.0))
        p = kf * _safe_exp(glast - gg)[:, None]
        tri = rr[:, None] >= rr[None, :]
        gram = tl.dot(qv, tl.trans(kf), input_precision="tf32")
        rmat = tl.where(tri, _safe_exp(gg[:, None] - gg[None, :]) * gram, 0.0)
        ddel = tl.dot(p, zstate, input_precision="tf32") + tl.dot(tl.trans(rmat), od, input_precision="tf32")
        dx = tl.dot(tl.trans(aa), ddel, input_precision="tf32")
        dpred = -bet[:, None] * dx
        eq = qv * _safe_exp(gg)[:, None]
        ek = kf * _safe_exp(gg)[:, None]
        zstate = _safe_exp(glast) * zstate + tl.dot(tl.trans(eq), od, input_precision="tf32") + tl.dot(tl.trans(ek), dpred, input_precision="tf32")


@triton.jit
def _chunk_grad(q, k, v, beta, inv, gcum, states, delta, dout, zbuf, dv, pq, pk, ps,
             T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
             NV: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    nv = pid % NV
    z = pid // NV
    c = z % NC
    bh = z // NC
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    rr = tl.arange(0, C)
    kk = tl.arange(0, BK)
    vv = nv * BV + tl.arange(0, BV)
    b64 = b.to(tl.int64)
    zstate = tl.load(zbuf + ((((b64 * HV + vh) * NC + c) * K + kk[:, None]) * V + vv[None, :]),
                     mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.0)
    tt = c * C + rr
    valid = tt < T
    tt64 = tt.to(tl.int64)
    qoff = ((b64 * T + tt64[:, None]) * H + kh) * K + kk[None, :]
    km = valid[:, None] & (kk[None, :] < K)
    qraw = tl.load(q + qoff, mask=km, other=0.0)
    kval = tl.load(k + qoff, mask=km, other=0.0)
    qv = qraw.to(tl.float32) * SCALE
    kf = kval.to(tl.float32)
    vo = ((b64 * T + tt64[:, None]) * HV + vh) * V + vv[None, :]
    vm = valid[:, None] & (vv[None, :] < V)
    val = tl.load(v + vo, mask=vm, other=0.0).to(tl.float32)
    od = tl.load(dout + vo, mask=vm, other=0.0).to(tl.float32)
    bo = (b64 * T + tt64) * HV + vh
    bet = tl.load(beta + bo, mask=valid, other=0.0).to(tl.float32)
    gg = tl.load(gcum + ((b64 * HV + vh) * NC + c) * C + rr)
    aa = tl.load(inv + (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * C + rr[None, :])
    st = tl.load(states + ((((b64 * HV + vh) * NC + c) * K + kk[:, None]) * V + vv[None, :]),
                 mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.0)
    dd = tl.load(delta + (((b64 * HV + vh) * NC + c) * C + rr[:, None]) * V + vv[None, :], mask=vm, other=0.0)
    ei = _safe_exp(gg)
    glast = tl.sum(tl.where(rr == C - 1, gg, 0.0))
    ec = _safe_exp(glast)
    fend = _safe_exp(glast - gg)
    eq = qv * ei[:, None]
    ek = kf * ei[:, None]
    p = kf * fend[:, None]
    gram_qk = tl.dot(qraw, tl.trans(kval), input_precision="tf32") * SCALE
    tri = rr[:, None] >= rr[None, :]
    rmat = tl.where(tri, _safe_exp(gg[:, None] - gg[None, :]) * gram_qk, 0.0)
    dp = tl.dot(dd, tl.trans(zstate), input_precision="tf32")
    ddel = tl.dot(p, zstate, input_precision="tf32") + tl.dot(tl.trans(rmat), od, input_precision="tf32")
    dr = tl.dot(od, tl.trans(dd), input_precision="tf32")
    dx = tl.dot(tl.trans(aa), ddel, input_precision="tf32")
    dl = -tl.dot(dx, tl.trans(dd), input_precision="tf32")
    deq = tl.dot(od, tl.trans(st), input_precision="tf32")
    pred = tl.dot(ek, st, input_precision="tf32")
    dpred = -bet[:, None] * dx
    dek = tl.dot(dpred, tl.trans(st), input_precision="tf32")
    dqv = ei[:, None] * deq
    dkv = ei[:, None] * dek + fend[:, None] * dp
    dgvec = tl.sum(deq * eq, axis=1) + tl.sum(dek * ek, axis=1) - tl.sum(dp * p, axis=1)
    dglast = tl.sum(dp * p) + tl.sum(zstate * (ec * st))
    dbv = tl.sum(dx * (val - pred), axis=1)
    tl.store(dv + vo, bet[:, None] * dx, mask=vm)
    er = tl.where(tri, _safe_exp(gg[:, None] - gg[None, :]), 0.0)
    cr = dr * er
    dqv += tl.dot(cr, kf, input_precision="tf32")
    dkv += tl.dot(tl.trans(cr), qv, input_precision="tf32")
    gr = dr * rmat
    dgvec += tl.sum(gr, axis=1) - tl.sum(gr, axis=0)
    strict = rr[:, None] > rr[None, :]
    gram_kk = tl.dot(kval, tl.trans(kval), input_precision="tf32")
    el = tl.where(strict, _safe_exp(gg[:, None] - gg[None, :]), 0.0)
    cl = dl * (bet[:, None] * el)
    dkv += tl.dot(cl, kf, input_precision="tf32") + tl.dot(tl.trans(cl), kf, input_precision="tf32")
    lval = bet[:, None] * el * gram_kk
    gl = dl * lval
    dgvec += tl.sum(gl, axis=1) - tl.sum(gl, axis=0)
    dbv += tl.sum(dl * el * gram_kk, axis=1)
    dgvec += tl.where(rr == C - 1, dglast, 0.0)
    dgstep = tl.sum(tl.where(rr[None, :] >= rr[:, None], dgvec[None, :], 0.0), axis=1)
    part = (((((b64 * HV + vh) * NC + c) * C + rr[:, None]) * NV + nv) * K + kk[None, :])
    pm = valid[:, None] & (kk[None, :] < K)
    tl.store(pq + part, dqv * SCALE, mask=pm)
    tl.store(pk + part, dkv, mask=pm)
    sp = (((((b64 * HV + vh) * NC + c) * C + rr) * NV + nv) * 2)
    tl.store(ps + sp, dgstep, mask=valid)
    tl.store(ps + sp + 1, dbv, mask=valid)

@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
               NV: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    z = pid // H
    t = z % T
    b = z // T
    c = t // C
    r = t - c * C
    kk = tl.arange(0, BK)
    sq = tl.zeros((BK,), tl.float32)
    sk = tl.zeros((BK,), tl.float32)
    group = HV // H
    b64 = b.to(tl.int64)
    for j in range(group):
        vh = h * group + j
        for nv in range(NV):
            off = (((((b64 * HV + vh) * NC + c) * C + r) * NV + nv) * K + kk)
            sq += tl.load(pq + off, mask=kk < K, other=0.0)
            sk += tl.load(pk + off, mask=kk < K, other=0.0)
    out = ((b64 * T + t) * H + h) * K + kk
    tl.store(dq + out, sq, mask=kk < K)
    tl.store(dk + out, sk, mask=kk < K)


@triton.jit
def _reduce_scalars(ps, dg, dbeta, T: tl.constexpr, HV: tl.constexpr,
                    NC: tl.constexpr, NV: tl.constexpr):
    pid = tl.program_id(0)
    vh = pid % HV
    z = pid // HV
    t = z % T
    b = z // T
    c = t // C
    r = t - c * C
    sg = 0.0
    sb = 0.0
    b64 = b.to(tl.int64)
    for nv in range(NV):
        off = (((((b64 * HV + vh) * NC + c) * C + r) * NV + nv) * 2)
        sg += tl.load(ps + off)
        sb += tl.load(ps + off + 1)
    out = (b64 * T + t) * HV + vh
    tl.store(dg + out, sg)
    tl.store(dbeta + out, sb)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        nc = triton.cdiv(T, C)
        nv = triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        inv = torch.empty((B, HV, nc, C, C), device=q.device, dtype=torch.float32)
        gcum = torch.empty((B, HV, nc, C), device=q.device, dtype=torch.float32)
        states = torch.empty((B, HV, nc, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty((B, HV, nc, C, V), device=q.device, dtype=torch.float32)
        out = torch.empty((B, T, HV, V), device=q.device, dtype=v.dtype)
        _make_wy[(nc, B * HV)](q, k, g, beta, inv, gcum, T=T, H=H, HV=HV, K=K, NC=nc, BK=bk, num_warps=4)
        _scan_state[(B * HV * nv,)](k, v, beta, inv, gcum, states, delta, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, num_warps=4)
        _emit[(B * HV * nc * nv,)](q, k, gcum, states, delta, out, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, SCALE=K ** -0.5, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, inv, gcum, states, delta)
        ctx.dims = (B, T, H, HV, K, V, nc, nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, inv, gcum, states, delta = ctx.saved_tensors
        B, T, H, HV, K, V, nc, nv, bk = ctx.dims
        dout = dout.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg, dbeta = torch.empty_like(g), torch.empty_like(beta)
        pq = torch.empty((B, HV, nc, C, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty((B, HV, nc, C, nv, K), device=q.device, dtype=torch.float32)
        ps = torch.empty((B, HV, nc, C, nv, 2), device=q.device, dtype=torch.float32)
        zbuf = torch.empty((B, HV, nc, K, V), device=q.device, dtype=torch.float32)
        _adjoint_scan[(B * HV * nv,)](q, k, beta, inv, gcum, delta, dout, zbuf, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, SCALE=K ** -0.5, num_warps=4)
        _chunk_grad[(B * HV * nc * nv,)](q, k, v, beta, inv, gcum, states, delta, dout, zbuf, dv, pq, pk, ps, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, BK=bk, SCALE=K ** -0.5, num_warps=4)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T=T, H=H, HV=HV, K=K, NC=nc, NV=nv, BK=bk, num_warps=4)
        _reduce_scalars[(B * T * HV,)](ps, dg, dbeta, T=T, HV=HV, NC=nc, NV=nv, num_warps=4)
        return dq, dk, dv, dg, dbeta


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
