import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 16


@triton.jit
def _wy_build(q, k, g, beta, inv_out, w_out, T: tl.constexpr,
              HK: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
              BK: tl.constexpr, C: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nc = tl.cdiv(T, C)
    vh = p % HV
    z = p // HV
    c = z % nc
    b = z // nc
    kh = vh // (HV // HK)
    i = tl.arange(0, C)
    x = tl.arange(0, BK)
    t = c * C + i
    tm = t < T

    ko = ((b * T + t[:, None]) * HK + kh) * K + x[None, :]
    kk = tl.load(k + ko, mask=tm[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
    gg = tl.load(g + (b * T + t) * HV + vh, mask=tm, other=0.).to(tl.float32)
    bb = tl.load(beta + (b * T + t) * HV + vh, mask=tm, other=0.).to(tl.float32)
    G = tl.cumsum(gg, axis=0)
    rows = i[:, None]
    cols = i[None, :]
    lower = (rows > cols) & tm[:, None] & tm[None, :]
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    ratio = tl.exp(tl.maximum(tl.minimum(G[:, None] - G[None, :], 0.), -80.))
    a = tl.where(lower, bb[:, None] * ratio * gram, 0.)
    eye = rows == cols
    inv = tl.where(eye, 1., 0.)
    for r in range(C):
        ar = tl.sum(tl.where(rows == r, a, 0.), axis=0)
        solved = -tl.sum(ar[:, None] * inv, axis=0)
        new_row = tl.where(i < r, solved, tl.where(i == r, 1., 0.))
        inv = tl.where(rows == r, new_row[None, :], inv)

    E = tl.exp(tl.maximum(tl.minimum(G, 0.), -80.))
    xx = bb[:, None] * E[:, None] * kk
    ww = tl.dot(inv, xx, input_precision="tf32")
    io = ((b * HV + vh) * nc + c) * C * C + rows * C + cols
    tl.store(inv_out + io, inv, mask=tm[:, None] & tm[None, :])
    wo = ((b * HV + vh) * T + t[:, None]) * K + x[None, :]
    tl.store(w_out + wo, ww, mask=tm[:, None] & (x[None, :] < K))


@triton.jit
def _values_build(v, beta, inv, u, T: tl.constexpr, HV: tl.constexpr,
                  V: tl.constexpr, C: tl.constexpr, BV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    nc = tl.cdiv(T, C)
    vb = p % nv
    z = p // nv
    c = z % nc
    bh = z // nc
    b = bh // HV
    vh = bh % HV
    i = tl.arange(0, C)
    j = tl.arange(0, C)
    y = vb * BV + tl.arange(0, BV)
    t = c * C + i
    tm = t < T
    io = ((bh * nc + c) * C + i[:, None]) * C + j[None, :]
    aa = tl.load(inv + io, mask=tm[:, None] & tm[None, :], other=0.)
    vo = ((b * T + t[:, None]) * HV + vh) * V + y[None, :]
    vv = tl.load(v + vo, mask=tm[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
    bb = tl.load(beta + (b * T + t) * HV + vh, mask=tm, other=0.).to(tl.float32)
    uu = tl.dot(aa, bb[:, None] * vv, input_precision="tf32")
    uo = ((bh * T + t[:, None]) * V + y[None, :])
    tl.store(u + uo, uu, mask=tm[:, None] & (y[None, :] < V))


@triton.jit
def _state_scan(k, g, w, u, starts, delta, T: tl.constexpr,
                HK: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                C: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    vb = p % nv
    bh = p // nv
    b = bh // HV
    vh = bh % HV
    kh = vh // (HV // HK)
    nc = tl.cdiv(T, C)
    x = tl.arange(0, BK)
    y = vb * BV + tl.arange(0, BV)
    i = tl.arange(0, C)
    state = tl.zeros((BK, BV), tl.float32)
    for c in range(nc):
        so = (((bh * nc + c) * K + x[:, None]) * V + y[None, :])
        tl.store(starts + so, state, mask=(x[:, None] < K) & (y[None, :] < V))
        t = c * C + i
        tm = t < T
        wo = ((bh * T + t[:, None]) * K + x[None, :])
        ww = tl.load(w + wo, mask=tm[:, None] & (x[None, :] < K), other=0.)
        uo = ((bh * T + t[:, None]) * V + y[None, :])
        uu = tl.load(u + uo, mask=tm[:, None] & (y[None, :] < V), other=0.)
        dd = uu - tl.dot(ww, state, input_precision="tf32")
        tl.store(delta + uo, dd, mask=tm[:, None] & (y[None, :] < V))
        go = (b * T + t) * HV + vh
        gg = tl.load(g + go, mask=tm, other=0.).to(tl.float32)
        G = tl.cumsum(gg, axis=0)
        last = tl.sum(gg, axis=0)
        ko = ((b * T + t[:, None]) * HK + kh) * K + x[None, :]
        kk = tl.load(k + ko, mask=tm[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
        rr = tl.exp(tl.maximum(tl.minimum(last - G, 0.), -80.))
        state = tl.exp(tl.maximum(tl.minimum(last, 0.), -80.)) * state + tl.dot(
            tl.trans(rr[:, None] * kk), dd, input_precision="tf32")


@triton.jit
def _output_chunks(q, k, g, starts, delta, out, T: tl.constexpr,
                   HK: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                   C: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    nc = tl.cdiv(T, C)
    vb = p % nv
    z = p // nv
    c = z % nc
    bh = z // nc
    b = bh // HV
    vh = bh % HV
    kh = vh // (HV // HK)
    i = tl.arange(0, C)
    x = tl.arange(0, BK)
    y = vb * BV + tl.arange(0, BV)
    t = c * C + i
    tm = t < T
    qo = ((b * T + t[:, None]) * HK + kh) * K + x[None, :]
    qs = tl.load(q + qo, mask=tm[:, None] & (x[None, :] < K), other=0.).to(tl.float32) * SCALE
    kk = tl.load(k + qo, mask=tm[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
    gg = tl.load(g + (b * T + t) * HV + vh, mask=tm, other=0.).to(tl.float32)
    G = tl.cumsum(gg, axis=0)
    E = tl.exp(tl.maximum(tl.minimum(G, 0.), -80.))
    so = (((bh * nc + c) * K + x[:, None]) * V + y[None, :])
    ss = tl.load(starts + so, mask=(x[:, None] < K) & (y[None, :] < V), other=0.)
    do = ((bh * T + t[:, None]) * V + y[None, :])
    dd = tl.load(delta + do, mask=tm[:, None] & (y[None, :] < V), other=0.)
    causal = i[:, None] >= i[None, :]
    ratio = tl.exp(tl.maximum(tl.minimum(G[:, None] - G[None, :], 0.), -80.))
    scores = tl.where(causal, tl.dot(qs, tl.trans(kk), input_precision="tf32") * ratio, 0.)
    oo = E[:, None] * tl.dot(qs, ss, input_precision="tf32") + tl.dot(scores, dd, input_precision="tf32")
    out_off = ((b * T + t[:, None]) * HV + vh) * V + y[None, :]
    tl.store(out + out_off, oo, mask=tm[:, None] & (y[None, :] < V))


@triton.jit
def _chunk_reverse(q, k, v, g, beta, inv, starts, delta, dout,
                   pq, pk, pv, pg, pb, T: tl.constexpr,
                   HK: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr,
                   BV: tl.constexpr, C: tl.constexpr, SCALE: tl.constexpr,
                   SD_B: tl.constexpr, SD_T: tl.constexpr,
                   SD_H: tl.constexpr, SD_V: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    vb = p % nv
    bh = p // nv
    b = bh // HV
    vh = bh % HV
    kh = vh // (HV // HK)
    nc = tl.cdiv(T, C)
    x = tl.arange(0, BK)
    yy = tl.arange(0, BV)
    y = vb * BV + yy
    i = tl.arange(0, C)
    rows = i[:, None]
    cols = i[None, :]
    dstate = tl.zeros((BK, BV), tl.float32)

    for rc in range(nc):
        c = nc - 1 - rc
        t = c * C + i
        tm = t < T
        qo = ((b * T + t[:, None]) * HK + kh) * K + x[None, :]
        qs = tl.load(q + qo, mask=tm[:, None] & (x[None, :] < K), other=0.).to(tl.float32) * SCALE
        kk = tl.load(k + qo, mask=tm[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
        vo = ((b * T + t[:, None]) * HV + vh) * V + y[None, :]
        vv = tl.load(v + vo, mask=tm[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
        dd = tl.load(delta + ((bh * T + t[:, None]) * V + y[None, :]),
                     mask=tm[:, None] & (y[None, :] < V), other=0.)
        do = b * SD_B + t[:, None] * SD_T + vh * SD_H + y[None, :] * SD_V
        oo = tl.load(dout + do, mask=tm[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
        gg = tl.load(g + (b * T + t) * HV + vh, mask=tm, other=0.).to(tl.float32)
        bb = tl.load(beta + (b * T + t) * HV + vh, mask=tm, other=0.).to(tl.float32)
        G = tl.cumsum(gg, axis=0)
        last = tl.sum(gg, axis=0)
        E = tl.exp(tl.maximum(tl.minimum(G, 0.), -80.))
        EL = tl.exp(tl.maximum(tl.minimum(last, 0.), -80.))
        lower_eq = (rows >= cols) & tm[:, None] & tm[None, :]
        strict = (rows > cols) & tm[:, None] & tm[None, :]
        ratio = tl.exp(tl.maximum(tl.minimum(G[:, None] - G[None, :], 0.), -80.))
        gram_qk = tl.dot(qs, tl.trans(kk), input_precision="tf32")
        score = tl.where(lower_eq, ratio * gram_qk, 0.)
        dc = tl.where(lower_eq, tl.dot(oo, tl.trans(dd), input_precision="tf32"), 0.)
        weighted_dc = dc * ratio

        so = (((bh * nc + c) * K + x[:, None]) * V + y[None, :])
        ss = tl.load(starts + so, mask=(x[:, None] < K) & (y[None, :] < V), other=0.)
        dqs = E[:, None] * tl.dot(oo, tl.trans(ss), input_precision="tf32")
        dqs += tl.dot(weighted_dc, kk, input_precision="tf32")
        dkk = tl.dot(tl.trans(weighted_dc), qs, input_precision="tf32")
        dD = tl.dot(tl.trans(score), oo, input_precision="tf32")
        dS0 = EL * dstate + tl.dot(tl.trans(E[:, None] * qs), oo, input_precision="tf32")
        dGc = tl.sum(dc * score, axis=1) - tl.sum(dc * score, axis=0)
        dGc += tl.sum(oo * (E[:, None] * tl.dot(qs, ss, input_precision="tf32")), axis=1)

        rend = tl.exp(tl.maximum(tl.minimum(last - G, 0.), -80.))
        kdS = tl.dot(kk, dstate, input_precision="tf32")
        dD += rend[:, None] * kdS
        dkk += rend[:, None] * tl.dot(dd, tl.trans(dstate), input_precision="tf32")
        hs = rend * tl.sum(kdS * dd, axis=1)
        dGc -= hs
        state_inner = tl.sum(tl.sum(dstate * ss, axis=1), axis=0)
        dlast = tl.sum(hs, axis=0) + EL * state_inner

        io = ((bh * nc + c) * C + rows) * C + cols
        aa = tl.load(inv + io, mask=tm[:, None] & tm[None, :], other=0.)
        Y = tl.dot(tl.trans(aa), dD, input_precision="tf32")
        dA = tl.where(strict, -tl.dot(Y, tl.trans(dd), input_precision="tf32"), 0.)
        pred = E[:, None] * tl.dot(kk, ss, input_precision="tf32")
        pvval = bb[:, None] * Y
        db = tl.sum(Y * (vv - pred), axis=1)
        dkk -= bb[:, None] * E[:, None] * tl.dot(Y, tl.trans(ss), input_precision="tf32")
        dGc -= tl.sum(Y * (bb[:, None] * pred), axis=1)
        dS0 -= tl.dot(tl.trans(bb[:, None] * E[:, None] * kk), Y, input_precision="tf32")

        gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
        ar = tl.where(strict, ratio * gram, 0.)
        db += tl.sum(dA * ar, axis=1)
        da_ratio = dA * ratio
        dkk += bb[:, None] * tl.dot(da_ratio, kk, input_precision="tf32")
        dkk += tl.dot(tl.trans(da_ratio * bb[:, None]), kk, input_precision="tf32")
        dAg = dA * (bb[:, None] * ar)
        dGc += tl.sum(dAg, axis=1) - tl.sum(dAg, axis=0)
        dGc += tl.where(i == C - 1, dlast, 0.)
        dgpart = tl.sum(tl.where(rows >= cols, dGc[:, None], 0.), axis=0)

        po = ((((b * NV + vb) * HV + vh) * T + t[:, None]) * K + x[None, :])
        tl.store(pq + po, dqs * SCALE, mask=tm[:, None] & (x[None, :] < K))
        tl.store(pk + po, dkk, mask=tm[:, None] & (x[None, :] < K))
        tl.store(pv + vo, pvval, mask=tm[:, None] & (y[None, :] < V))
        spo = ((b * NV + vb) * HV + vh) * T + t
        tl.store(pg + spo, dgpart, mask=tm)
        tl.store(pb + spo, db, mask=tm)
        dstate = dS0


@triton.jit
def _sum_qk(parts_q, parts_k, dq, dk, T: tl.constexpr, HK: tl.constexpr,
            HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
            BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    t = p % T
    z = p // T
    h = z % HK
    b = z // HK
    x = tl.arange(0, BK)
    sq = tl.zeros((BK,), tl.float32)
    sk = tl.zeros((BK,), tl.float32)
    rep = HV // HK
    for r in range(rep):
        vh = h * rep + r
        for vb in range(NV):
            off = (((b * NV + vb) * HV + vh) * T + t) * K + x
            sq += tl.load(parts_q + off, mask=x < K, other=0.)
            sk += tl.load(parts_k + off, mask=x < K, other=0.)
    out = ((b * T + t) * HK + h) * K + x
    tl.store(dq + out, sq, mask=x < K)
    tl.store(dk + out, sk, mask=x < K)


@triton.jit
def _sum_scalars(parts_g, parts_b, dg, db, T: tl.constexpr,
                 HV: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    t = p % T
    z = p // T
    h = z % HV
    b = z // HV
    sg = 0.
    sb = 0.
    for vb in range(NV):
        off = ((b * NV + vb) * HV + h) * T + t
        sg += tl.load(parts_g + off)
        sb += tl.load(parts_b + off)
    out = (b * T + t) * HV + h
    tl.store(dg + out, sg)
    tl.store(db + out, sb)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, HK, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        if HV % HK:
            raise ValueError("the number of value heads must be divisible by key heads")
        C = CHUNK
        BK = triton.next_power_of_2(K)
        BV = VALUE_BLOCK
        NC = triton.cdiv(T, C)
        NV = triton.cdiv(V, BV)
        device = q.device
        inv = torch.empty((B, HV, NC, C, C), device=device, dtype=torch.float32)
        w = torch.empty((B, HV, T, K), device=device, dtype=torch.float32)
        u = torch.empty((B, HV, T, V), device=device, dtype=torch.float32)
        starts = torch.empty((B, HV, NC, K, V), device=device, dtype=torch.float32)
        delta = torch.empty((B, HV, T, V), device=device, dtype=torch.float32)
        out = torch.empty((B, T, HV, V), device=device, dtype=v.dtype)
        _wy_build[(B * HV * NC,)](q, k, g, beta, inv, w, T=T, HK=HK, HV=HV,
                                  K=K, BK=BK, C=C, num_warps=4)
        _values_build[(B * HV * NC * NV,)](v, beta, inv, u, T=T, HV=HV, V=V,
                                           C=C, BV=BV, num_warps=4)
        _state_scan[(B * HV * NV,)](k, g, w, u, starts, delta, T=T, HK=HK, HV=HV,
                                    K=K, V=V, BK=BK, BV=BV, C=C, num_warps=8)
        _output_chunks[(B * HV * NC * NV,)](
            q, k, g, starts, delta, out, T=T, HK=HK, HV=HV, K=K, V=V,
            BK=BK, BV=BV, C=C, SCALE=K ** -0.5, num_warps=8)
        ctx.save_for_backward(q, k, v, g, beta, inv, starts, delta)
        ctx.dims = (B, T, HK, HV, K, V, BK, BV, C, NV)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, inv, starts, delta = ctx.saved_tensors
        B, T, HK, HV, K, V, BK, BV, C, NV = ctx.dims
        device = q.device
        pq = torch.empty((B, NV, HV, T, K), device=device, dtype=torch.float32)
        pk = torch.empty((B, NV, HV, T, K), device=device, dtype=torch.float32)
        pg = torch.empty((B, NV, HV, T), device=device, dtype=torch.float32)
        pb = torch.empty((B, NV, HV, T), device=device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        _chunk_reverse[(B * HV * NV,)](
            q, k, v, g, beta, inv, starts, delta, dout, pq, pk, dv, pg, pb,
            T=T, HK=HK, HV=HV, K=K, V=V, NV=NV, BK=BK, BV=BV, C=C,
            SCALE=K ** -0.5, SD_B=dout.stride(0), SD_T=dout.stride(1),
            SD_H=dout.stride(2), SD_V=dout.stride(3), num_warps=4)
        _sum_qk[(B * HK * T,)](pq, pk, dq, dk, T=T, HK=HK, HV=HV, K=K,
                               NV=NV, BK=BK, num_warps=4)
        _sum_scalars[(B * HV * T,)](pg, pb, dg, db, T=T, HV=HV, NV=NV, num_warps=4)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
