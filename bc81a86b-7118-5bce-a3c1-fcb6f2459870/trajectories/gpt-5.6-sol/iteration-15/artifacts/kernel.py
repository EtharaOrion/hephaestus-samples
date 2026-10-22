import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(16)


@triton.jit
def _make_wy(k, g, beta, hbuf, abuf, pbuf, T: tl.constexpr, H: tl.constexpr,
             HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr):
    ic = tl.program_id(0)
    z = tl.program_id(1)
    b = z // HV
    vh = z % HV
    kh = vh // (HV // H)
    rr = tl.arange(0, C)[:, None]
    cc = tl.arange(0, C)[None, :]
    kk = tl.arange(0, BK)[None, :]
    t = ic * C + tl.arange(0, C)
    valid = t < T
    go = (b * T + t) * HV + vh
    gv = tl.load(g + go, mask=valid, other=0.0).to(tl.float32)
    hs = tl.cumsum(gv, axis=0)
    tl.store(hbuf + z * T + t, hs, mask=valid)
    ko = ((b * T + t[:, None]) * H + kh) * K + kk
    kv = tl.load(k + ko, mask=valid[:, None] & (kk < K), other=0.0).to(tl.float32)
    gram = tl.dot(kv, tl.trans(kv), input_precision='tf32')
    bv = tl.load(beta + go, mask=valid, other=0.0).to(tl.float32)
    ed = tl.exp(hs[:, None] - hs[None, :])
    lm = tl.where((rr > cc) & valid[:, None] & valid[None, :], bv[:, None] * ed * gram, 0.0)
    ai = (rr == cc).to(tl.float32)
    for i in tl.static_range(0, C):
        li = tl.sum(tl.where(rr == i, lm, 0.0), axis=0)
        li = tl.where((tl.arange(0, C) < i) & valid, li, 0.0)
        row = -tl.sum(li[:, None] * ai, axis=0) + (tl.arange(0, C) == i).to(tl.float32)
        ai = tl.where(rr == i, row[None, :], ai)
    tl.store(abuf + ((z * NC + ic) * C + rr) * C + cc, ai)
    pv = tl.dot(ai, bv[:, None] * tl.exp(hs)[:, None] * kv, input_precision='tf32')
    tl.store(pbuf + ((z * NC + ic) * C + rr) * K + kk, pv, mask=kk < K)


@triton.jit
def _state_scan(k, v, beta, hbuf, abuf, pbuf, xbuf, sbuf,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                NC: tl.constexpr, NV: tl.constexpr):
    pid = tl.program_id(0)
    iv = pid % NV
    z = pid // NV
    b = z // HV
    vh = z % HV
    kh = vh // (HV // H)
    rk = tl.arange(0, BK)[:, None]
    kvcol = tl.arange(0, BK)[None, :]
    cv = iv * BV + tl.arange(0, BV)[None, :]
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        tl.store(sbuf + ((z * NC + ic) * K + rk) * V + cv, state, mask=(rk < K) & (cv < V))
        tt = ic * C + tl.arange(0, C)
        valid = tt < T
        rr = tl.arange(0, C)[:, None]
        cc = tl.arange(0, C)[None, :]
        aa = tl.load(abuf + ((z * NC + ic) * C + rr) * C + cc)
        vv = tl.load(v + ((b * T + tt[:, None]) * HV + vh) * V + cv,
                     mask=valid[:, None] & (cv < V), other=0.0).to(tl.float32)
        be = tl.load(beta + (b * T + tt) * HV + vh, mask=valid, other=0.0).to(tl.float32)
        uu = tl.dot(aa, be[:, None] * vv, input_precision='tf32')
        pp = tl.load(pbuf + ((z * NC + ic) * C + rr) * K + kvcol, mask=kvcol < K, other=0.0)
        xx = uu - tl.dot(pp, state, input_precision='tf32')
        tl.store(xbuf + (z * T + tt[:, None]) * V + cv, xx, mask=valid[:, None] & (cv < V))
        kval = tl.load(k + ((b * T + tt[:, None]) * H + kh) * K + kvcol,
                      mask=valid[:, None] & (kvcol < K), other=0.0).to(tl.float32)
        hs = tl.load(hbuf + z * T + tt, mask=valid, other=0.0)
        last = tl.minimum(C - 1, T - 1 - ic * C)
        he = tl.sum(tl.where(tl.arange(0, C) == last, hs, 0.0), axis=0)
        state = tl.exp(he) * state + tl.dot(tl.trans(kval * tl.exp(he - hs)[:, None]), xx, input_precision='tf32')


@triton.jit
def _emit(q, k, hbuf, xbuf, sbuf, out, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
          NC: tl.constexpr, NV: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    iv = pid % NV
    p = pid // NV
    ic = p % NC
    z = p // NC
    b = z // HV
    vh = z % HV
    kh = vh // (HV // H)
    tt = ic * C + tl.arange(0, C)
    valid = tt < T
    kk = tl.arange(0, BK)[None, :]
    cv = iv * BV + tl.arange(0, BV)[None, :]
    qo = ((b * T + tt[:, None]) * H + kh) * K + kk
    qv = tl.load(q + qo, mask=valid[:, None] & (kk < K), other=0.0).to(tl.float32) * SCALE
    kval = tl.load(k + qo, mask=valid[:, None] & (kk < K), other=0.0).to(tl.float32)
    hs = tl.load(hbuf + z * T + tt, mask=valid, other=0.0)
    rk = tl.arange(0, BK)[:, None]
    ss = tl.load(sbuf + ((z * NC + ic) * K + rk) * V + cv, mask=(rk < K) & (cv < V), other=0.0)
    xx = tl.load(xbuf + (z * T + tt[:, None]) * V + cv, mask=valid[:, None] & (cv < V), other=0.0)
    gram = tl.dot(qv, tl.trans(kval), input_precision='tf32')
    ri = tl.arange(0, C)[:, None]
    rj = tl.arange(0, C)[None, :]
    rmat = tl.where((ri >= rj) & valid[:, None] & valid[None, :],
                    gram * tl.exp(hs[:, None] - hs[None, :]), 0.0)
    oo = tl.exp(hs)[:, None] * tl.dot(qv, ss, input_precision='tf32') + tl.dot(rmat, xx, input_precision='tf32')
    tl.store(out + (z * T + tt[:, None]) * V + cv, oo, mask=valid[:, None] & (cv < V))


@triton.jit
def _reverse(q, k, v, beta, do, hbuf, abuf, pbuf, xbuf, sbuf,
             dqpart, dkpart, dvpart, dgpart, dbpart,
             T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
             NC: tl.constexpr, NV: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    iv = pid % NV
    z = pid // NV
    b = z // HV
    vh = z % HV
    kh = vh // (HV // H)
    ks = tl.arange(0, BK)
    cvs = iv * BV + tl.arange(0, BV)
    dsn = tl.zeros((BK, BV), tl.float32)
    for step in range(NC):
        ic = NC - 1 - step
        ts = ic * C + tl.arange(0, C)
        valid = ts < T
        rr = tl.arange(0, C)[:, None]
        cc = tl.arange(0, C)[None, :]
        km = tl.arange(0, BK)[None, :]
        cv = cvs[None, :]
        qo = ((b * T + ts[:, None]) * H + kh) * K + km
        qq = tl.load(q + qo, mask=valid[:, None] & (km < K), other=0.0).to(tl.float32) * SCALE
        kval = tl.load(k + qo, mask=valid[:, None] & (km < K), other=0.0).to(tl.float32)
        vo = ((b * T + ts[:, None]) * HV + vh) * V + cv
        oo = (z * T + ts[:, None]) * V + cv
        vv = tl.load(v + vo, mask=valid[:, None] & (cv < V), other=0.0).to(tl.float32)
        gg = tl.load(do + oo, mask=valid[:, None] & (cv < V), other=0.0).to(tl.float32)
        xx = tl.load(xbuf + (z * T + ts[:, None]) * V + cv, mask=valid[:, None] & (cv < V), other=0.0)
        hs = tl.load(hbuf + z * T + ts, mask=valid, other=0.0)
        be = tl.load(beta + (b * T + ts) * HV + vh, mask=valid, other=0.0).to(tl.float32)
        ss = tl.load(sbuf + ((z * NC + ic) * K + ks[:, None]) * V + cvs[None, :],
                     mask=(ks[:, None] < K) & (cvs[None, :] < V), other=0.0)
        last = tl.minimum(C - 1, T - 1 - ic * C)
        he = tl.sum(tl.where(tl.arange(0, C) == last, hs, 0.0), axis=0)
        ke = kval * tl.exp(he - hs)[:, None]
        dke = tl.dot(xx, tl.trans(dsn), input_precision='tf32')
        dx = tl.dot(ke, dsn, input_precision='tf32')
        dstate = tl.exp(he) * dsn
        gram = tl.dot(qq, tl.trans(kval), input_precision='tf32')
        tri = (rr >= cc) & valid[:, None] & valid[None, :]
        ee = tl.exp(hs[:, None] - hs[None, :])
        rmat = tl.where(tri, gram * ee, 0.0)
        dr = tl.dot(gg, tl.trans(xx), input_precision='tf32')
        dx += tl.dot(tl.trans(rmat), gg, input_precision='tf32')
        qstate = tl.dot(qq, ss, input_precision='tf32')
        dqv = tl.exp(hs)[:, None] * tl.dot(gg, tl.trans(ss), input_precision='tf32')
        dstate += tl.dot(tl.trans(qq), tl.exp(hs)[:, None] * gg, input_precision='tf32')
        dre = tl.where(tri, dr * ee, 0.0)
        dqv += tl.dot(dre, kval, input_precision='tf32')
        dkk = tl.dot(tl.trans(dre), qq, input_precision='tf32') + dke * tl.exp(he - hs)[:, None]
        dh = tl.sum(gg * (tl.exp(hs)[:, None] * qstate), axis=1)
        dh += tl.sum(dre * gram, axis=1) - tl.sum(dre * gram, axis=0)
        dh += (tl.arange(0, C) == last) * tl.sum(dsn * (tl.exp(he) * ss))
        zke = tl.sum(dke * ke, axis=1)
        dh -= zke
        dh += (tl.arange(0, C) == last) * tl.sum(zke)
        pp = tl.load(pbuf + ((z * NC + ic) * C + rr) * K + km, mask=km < K, other=0.0)
        dp = -tl.dot(dx, tl.trans(ss), input_precision='tf32')
        dstate -= tl.dot(tl.trans(pp), dx, input_precision='tf32')
        aa = tl.load(abuf + ((z * NC + ic) * C + rr) * C + cc)
        dbv = tl.dot(tl.trans(aa), dx, input_precision='tf32')
        dbk = tl.dot(tl.trans(aa), dp, input_precision='tf32')
        bvmat = be[:, None] * vv
        bkmat = be[:, None] * tl.exp(hs)[:, None] * kval
        da = tl.dot(dx, tl.trans(bvmat), input_precision='tf32') + tl.dot(dp, tl.trans(bkmat), input_precision='tf32')
        dn = -tl.dot(tl.trans(aa), tl.dot(da, tl.trans(aa), input_precision='tf32'), input_precision='tf32')
        strict = (rr > cc) & valid[:, None] & valid[None, :]
        kg = tl.dot(kval, tl.trans(kval), input_precision='tf32')
        dlbase = tl.where(strict, dn * ee, 0.0)
        dkk += tl.dot(dlbase * be[:, None], kval, input_precision='tf32')
        dkk += tl.dot(tl.trans(dlbase * be[:, None]), kval, input_precision='tf32')
        lterm = dlbase * kg * be[:, None]
        dh += tl.sum(lterm, axis=1) - tl.sum(lterm, axis=0)
        dbeta = tl.sum(dbv * vv, axis=1) + tl.sum(dbk * (tl.exp(hs)[:, None] * kval), axis=1)
        dbeta += tl.sum(tl.where(strict, dn * ee * kg, 0.0), axis=1)
        dvv = dbv * be[:, None]
        dkk += dbk * (be * tl.exp(hs))[:, None]
        dh += tl.sum(dbk * bkmat, axis=1)
        dsn = dstate
        dh = tl.cumsum(dh, axis=0, reverse=True)
        baseq = ((z * NV + iv) * T + ts[:, None]) * BK + km
        tl.store(dqpart + baseq, dqv * SCALE, mask=valid[:, None] & (km < K))
        tl.store(dkpart + baseq, dkk, mask=valid[:, None] & (km < K))
        basev = ((z * NV + iv) * T + ts[:, None]) * V + cv
        tl.store(dvpart + basev, dvv, mask=valid[:, None] & (cv < V))
        bases = (z * NV + iv) * T + ts
        tl.store(dgpart + bases, dh, mask=valid)
        tl.store(dbpart + bases, dbeta, mask=valid)


@triton.jit
def _reduce_qk(inp, out, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, BK: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0)
    t = p % T
    bh = p // T
    kh = bh % H
    b = bh // H
    kk = tl.arange(0, BK)
    acc = tl.zeros((BK,), tl.float32)
    R = HV // H
    for r in range(R):
        z = b * HV + kh * R + r
        for iv in range(NV):
            acc += tl.load(inp + ((z * NV + iv) * T + t) * BK + kk, mask=kk < K, other=0.0)
    tl.store(out + ((b * T + t) * H + kh) * K + kk, acc, mask=kk < K)


@triton.jit
def _copy_v(inp, out, T: tl.constexpr, HV: tl.constexpr, V: tl.constexpr,
            BV0: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0)
    vv = tl.arange(0, BV0)
    t = p % T
    z = p // T
    iv = vv // BV
    lane = vv % BV
    off = ((z * NV + iv) * T + t) * V + iv * BV + lane
    val = tl.load(inp + off, mask=(iv < NV) & (iv * BV + lane < V), other=0.0)
    b = z // HV
    vh = z % HV
    tl.store(out + ((b * T + t) * HV + vh) * V + vv, val, mask=vv < V)


@triton.jit
def _sum_scalar(inp, out, T: tl.constexpr, HV: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0)
    t = p % T
    z = p // T
    iv = tl.arange(0, 64)
    val = tl.load(inp + (z * NV + iv) * T + t, mask=iv < NV, other=0.0)
    b = z // HV
    vh = z % HV
    tl.store(out + (b * T + t) * HV + vh, tl.sum(val, axis=0))


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC, NV, BK = triton.cdiv(T, C), triton.cdiv(V, BV), triton.next_power_of_2(K)
        dev = q.device
        h = torch.empty((B * HV, T), device=dev, dtype=torch.float32)
        a = torch.empty((B * HV, NC, C, C), device=dev, dtype=torch.float32)
        p = torch.empty((B * HV, NC, C, K), device=dev, dtype=torch.float32)
        x = torch.empty((B * HV, T, V), device=dev, dtype=torch.float32)
        s = torch.empty((B * HV, NC, K, V), device=dev, dtype=torch.float32)
        out = torch.empty((B * HV, T, V), device=dev, dtype=v.dtype)
        _make_wy[(NC, B * HV)](k, g, beta, h, a, p, T, H, HV, K, BK, NC, num_warps=4)
        _state_scan[(B * HV * NV,)](k, v, beta, h, a, p, x, s, T, H, HV, K, V, BK, NC, NV, num_warps=8)
        _emit[(B * HV * NC * NV,)](q, k, h, x, s, out, T, H, HV, K, V, BK, NC, NV, K ** -0.5, num_warps=8)
        ctx.save_for_backward(q, k, v, beta, h, a, p, x, s)
        ctx.dims = (B, T, H, HV, K, V, BK, NC, NV)
        return out.view(B, HV, T, V).transpose(1, 2)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, beta, h, a, p, x, s = ctx.saved_tensors
        B, T, H, HV, K, V, BK, NC, NV = ctx.dims
        dev = q.device
        do = dout.transpose(1, 2).contiguous().view(B * HV, T, V)
        shape = (B * HV, NV, T, BK)
        dqp = torch.empty(shape, device=dev, dtype=torch.float32)
        dkp = torch.empty(shape, device=dev, dtype=torch.float32)
        dvp = torch.empty((B * HV, NV, T, V), device=dev, dtype=torch.float32)
        dgp = torch.empty((B * HV, NV, T), device=dev, dtype=torch.float32)
        dbp = torch.empty((B * HV, NV, T), device=dev, dtype=torch.float32)
        _reverse[(B * HV * NV,)](q, k, v, beta, do, h, a, p, x, s, dqp, dkp, dvp, dgp, dbp,
                                 T, H, HV, K, V, BK, NC, NV, K ** -0.5, num_warps=8)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg, db = torch.empty((B, T, HV), device=dev, dtype=torch.float32), torch.empty_like(beta)
        _reduce_qk[(B * H * T,)](dqp, dq, T, H, HV, K, BK, NV)
        _reduce_qk[(B * H * T,)](dkp, dk, T, H, HV, K, BK, NV)
        _copy_v[(B * HV * T,)](dvp, dv, T, HV, V, triton.next_power_of_2(V), NV)
        _sum_scalar[(B * HV * T,)](dgp, dg, T, HV, NV)
        _sum_scalar[(B * HV * T,)](dbp, db, T, HV, NV)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
