import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 16


@triton.jit
def _wy_build(kp, gp, bp, mp, wp,
              T: tl.constexpr, B: tl.constexpr, HQ: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
              BK: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    ic = pid % NC
    z = pid // NC
    ih = z % HV
    ib = z // HV
    kh = ih // (HV // HQ)
    r = tl.arange(0, C)
    c = tl.arange(0, C)
    d = tl.arange(0, BK)
    t = ic * C + r
    ok = t < T
    ko = (ib * T * HQ + t * HQ + kh).to(tl.int64) * K
    x = tl.load(kp + ko[:, None] + d[None, :],
                mask=ok[:, None] & (d[None, :] < K), other=0.0).to(tl.float32)
    go = (ib * T * HV + t * HV + ih).to(tl.int64)
    raw = tl.load(gp + go, mask=ok, other=0.0).to(tl.float32)
    gate = tl.cumsum(raw, axis=0)
    beta = tl.load(bp + go, mask=ok, other=0.0).to(tl.float32)
    gram = tl.dot(x, tl.trans(x), input_precision="tf32")
    strict = (r[:, None] > c[None, :]) & ok[:, None] & ok[None, :]
    a = tl.where(strict, beta[:, None] * tl.exp(gate[:, None] - gate[None, :]) * gram, 0.0)
    inv = tl.where(r[:, None] == c[None, :], 1.0, 0.0)
    for i in range(1, C):
        for j in range(i):
            aij = tl.sum(tl.sum(tl.where((r[:, None] == i) & (c[None, :] == j), a, 0.0), axis=1), axis=0)
            row = tl.sum(tl.where(r[:, None] == j, inv, 0.0), axis=0)
            inv = tl.where(r[:, None] == i, inv - aij * row[None, :], inv)
    y = x * (beta * tl.exp(gate))[:, None]
    w = tl.dot(inv, y, input_precision="tf32")
    mo = ((z * NC + ic).to(tl.int64) * C + r[:, None]) * C + c[None, :]
    tl.store(mp + mo, inv)
    wo = (z.to(tl.int64) * T + t) * K
    tl.store(wp + wo[:, None] + d[None, :], w,
             mask=ok[:, None] & (d[None, :] < K))


@triton.jit
def _value_build(vp, bp, mp, up,
                 T: tl.constexpr, B: tl.constexpr, HV: tl.constexpr,
                 V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                 BV: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    iv = pid % NV
    pid //= NV
    ic = pid % NC
    z = pid // NC
    ih = z % HV
    ib = z // HV
    r = tl.arange(0, C)
    c = tl.arange(0, C)
    d = iv * BV + tl.arange(0, BV)
    t = ic * C + r
    ok = t < T
    mo = ((z * NC + ic).to(tl.int64) * C + r[:, None]) * C + c[None, :]
    inv = tl.load(mp + mo)
    io = (ib * T * HV + t * HV + ih).to(tl.int64)
    beta = tl.load(bp + io, mask=ok, other=0.0).to(tl.float32)
    val = tl.load(vp + io[:, None] * V + d[None, :],
                  mask=ok[:, None] & (d[None, :] < V), other=0.0).to(tl.float32)
    u = tl.dot(inv, val * beta[:, None], input_precision="tf32")
    uo = (z.to(tl.int64) * T + t) * V
    tl.store(up + uo[:, None] + d[None, :], u,
             mask=ok[:, None] & (d[None, :] < V))


@triton.jit
def _state_forward(kp, gp, wp, up, hp, dp,
                   T: tl.constexpr, B: tl.constexpr, HQ: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr,
                   BV: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    iv = pid % NV
    z = pid // NV
    ih = z % HV
    ib = z // HV
    kh = ih // (HV // HQ)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    r = tl.arange(0, C)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        ho = (((z * NC + ic).to(tl.int64) * K + x[:, None]) * V + y[None, :])
        tl.store(hp + ho, state, mask=(x[:, None] < K) & (y[None, :] < V))
        t = ic * C + r
        ok = t < T
        ko = (ib * T * HQ + t * HQ + kh).to(tl.int64) * K
        key = tl.load(kp + ko[:, None] + x[None, :],
                      mask=ok[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        zo = (z.to(tl.int64) * T + t)
        w = tl.load(wp + zo[:, None] * K + x[None, :],
                    mask=ok[:, None] & (x[None, :] < K), other=0.0)
        u = tl.load(up + zo[:, None] * V + y[None, :],
                    mask=ok[:, None] & (y[None, :] < V), other=0.0)
        delta = u - tl.dot(w, state, input_precision="tf32")
        tl.store(dp + zo[:, None] * V + y[None, :], delta,
                 mask=ok[:, None] & (y[None, :] < V))
        go = (ib * T * HV + t * HV + ih).to(tl.int64)
        gate = tl.cumsum(tl.load(gp + go, mask=ok, other=0.0).to(tl.float32), axis=0)
        last = tl.sum(tl.where(r == C - 1, gate, 0.0), axis=0)
        state = tl.exp(last) * state + tl.dot(
            tl.trans(key * tl.exp(last - gate)[:, None]), delta, input_precision="tf32")


@triton.jit
def _output_forward(qp, kp, gp, hp, dp, op,
                    scale: tl.constexpr, T: tl.constexpr, B: tl.constexpr,
                    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    iv = pid % NV
    pid //= NV
    ic = pid % NC
    z = pid // NC
    ih = z % HV
    ib = z // HV
    kh = ih // (HV // HQ)
    r = tl.arange(0, C)
    c = tl.arange(0, C)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    t = ic * C + r
    ok = t < T
    qo = (ib * T * HQ + t * HQ + kh).to(tl.int64) * K
    q = tl.load(qp + qo[:, None] + x[None, :],
                mask=ok[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * scale
    key = tl.load(kp + qo[:, None] + x[None, :],
                  mask=ok[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    go = (ib * T * HV + t * HV + ih).to(tl.int64)
    gate = tl.cumsum(tl.load(gp + go, mask=ok, other=0.0).to(tl.float32), axis=0)
    ho = (((z * NC + ic).to(tl.int64) * K + x[:, None]) * V + y[None, :])
    state = tl.load(hp + ho, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    zo = (z.to(tl.int64) * T + t)
    delta = tl.load(dp + zo[:, None] * V + y[None, :],
                    mask=ok[:, None] & (y[None, :] < V), other=0.0)
    score = tl.dot(q, tl.trans(key), input_precision="tf32")
    causal = (r[:, None] >= c[None, :]) & ok[:, None] & ok[None, :]
    score = tl.where(causal, score * tl.exp(gate[:, None] - gate[None, :]), 0.0)
    out = tl.exp(gate)[:, None] * tl.dot(q, state, input_precision="tf32")
    out += tl.dot(score, delta, input_precision="tf32")
    oo = (ib * T * HV + t * HV + ih).to(tl.int64) * V
    tl.store(op + oo[:, None] + y[None, :], out,
             mask=ok[:, None] & (y[None, :] < V))


@triton.jit
def _reverse_chunks(qp, kp, vp, gp, bp, dop, mp, wp, hp, dp,
                    qpp, kpp, dvp, spp, dosb, dost, dosh, dosv,
                    scale: tl.constexpr, T: tl.constexpr, B: tl.constexpr,
                    HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    iv = pid % NV
    z = pid // NV
    ih = z % HV
    ib = z // HV
    kh = ih // (HV // HQ)
    r = tl.arange(0, C)
    c = tl.arange(0, C)
    x = tl.arange(0, BK)
    y = iv * BV + tl.arange(0, BV)
    dstate = tl.zeros((BK, BV), tl.float32)
    for rev in range(NC):
        ic = NC - 1 - rev
        t = ic * C + r
        ok = t < T
        qo = (ib * T * HQ + t * HQ + kh).to(tl.int64) * K
        q = tl.load(qp + qo[:, None] + x[None, :],
                    mask=ok[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * scale
        key = tl.load(kp + qo[:, None] + x[None, :],
                      mask=ok[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        io = (ib * T * HV + t * HV + ih).to(tl.int64)
        raw = tl.load(gp + io, mask=ok, other=0.0).to(tl.float32)
        gate = tl.cumsum(raw, axis=0)
        beta = tl.load(bp + io, mask=ok, other=0.0).to(tl.float32)
        zo = z.to(tl.int64) * T + t
        val = tl.load(vp + io[:, None] * V + y[None, :],
                      mask=ok[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        doff = ib * dosb + t[:, None] * dost + ih * dosh + y[None, :] * dosv
        dout = tl.load(dop + doff,
                       mask=ok[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        delta = tl.load(dp + zo[:, None] * V + y[None, :],
                        mask=ok[:, None] & (y[None, :] < V), other=0.0)
        w = tl.load(wp + zo[:, None] * K + x[None, :],
                    mask=ok[:, None] & (x[None, :] < K), other=0.0)
        ho = (((z * NC + ic).to(tl.int64) * K + x[:, None]) * V + y[None, :])
        state = tl.load(hp + ho, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
        mo = ((z * NC + ic).to(tl.int64) * C + r[:, None]) * C + c[None, :]
        inv = tl.load(mp + mo)

        last = tl.sum(tl.where(r == C - 1, gate, 0.0), axis=0)
        eend = tl.exp(last - gate) * ok
        gram_qk = tl.dot(q, tl.trans(key), input_precision="tf32")
        causal = (r[:, None] >= c[None, :]) & ok[:, None] & ok[None, :]
        ez = tl.where(causal, tl.exp(gate[:, None] - gate[None, :]), 0.0)
        score = gram_qk * ez
        kdstate = tl.dot(key, dstate, input_precision="tf32")
        ddelta = tl.dot(tl.trans(score), dout, input_precision="tf32")
        ddelta += eend[:, None] * kdstate
        dscore = tl.dot(dout, tl.trans(delta), input_precision="tf32")
        dgram_qk = tl.where(causal, dscore * ez, 0.0)

        dq = tl.exp(gate)[:, None] * tl.dot(dout, tl.trans(state), input_precision="tf32")
        dq += tl.dot(dgram_qk, key, input_precision="tf32")
        dk = tl.dot(tl.trans(dgram_qk), q, input_precision="tf32")
        dk += eend[:, None] * tl.dot(delta, tl.trans(dstate), input_precision="tf32")
        dw = -tl.dot(ddelta, tl.trans(state), input_precision="tf32")
        dxv = tl.dot(tl.trans(inv), ddelta, input_precision="tf32")
        dyk = tl.dot(tl.trans(inv), dw, input_precision="tf32")
        xv = val * beta[:, None]
        yk = key * (beta * tl.exp(gate))[:, None]
        dinv = tl.dot(ddelta, tl.trans(xv), input_precision="tf32")
        dinv += tl.dot(dw, tl.trans(yk), input_precision="tf32")
        dtri = -tl.dot(tl.dot(tl.trans(inv), dinv, input_precision="tf32"),
                       tl.trans(inv), input_precision="tf32")
        strict = (r[:, None] > c[None, :]) & ok[:, None] & ok[None, :]
        dtri = tl.where(strict, dtri, 0.0)
        gram_kk = tl.dot(key, tl.trans(key), input_precision="tf32")
        ea = tl.where(strict, tl.exp(gate[:, None] - gate[None, :]), 0.0)
        dc = dtri * beta[:, None] * ea
        dk += tl.dot(dc, key, input_precision="tf32")
        dk += tl.dot(tl.trans(dc), key, input_precision="tf32")
        dk += dyk * (beta * tl.exp(gate))[:, None]
        dbeta = tl.sum(dxv * val, axis=1)
        dbeta += tl.sum(dyk * (tl.exp(gate)[:, None] * key), axis=1)
        dbeta += tl.sum(dtri * ea * gram_kk, axis=1)

        dgate = tl.sum(dout * (tl.exp(gate)[:, None] * tl.dot(q, state, input_precision="tf32")), axis=1)
        dgate += tl.sum(dyk * yk, axis=1)
        zterm = dscore * score
        dgate += tl.sum(zterm, axis=1) - tl.sum(zterm, axis=0)
        aterm = dtri * beta[:, None] * ea * gram_kk
        dgate += tl.sum(aterm, axis=1) - tl.sum(aterm, axis=0)
        end_term = eend * tl.sum(kdstate * delta, axis=1)
        dgate -= end_term
        dgate += tl.where(r == C - 1, tl.sum(end_term, axis=0), 0.0)
        base_term = tl.exp(last) * tl.sum(dstate * state)
        dgate += tl.where(r == C - 1, base_term, 0.0)
        dg = tl.cumsum(dgate, axis=0, reverse=True)

        dstate = tl.exp(last) * dstate
        dstate += tl.dot(tl.trans(q * tl.exp(gate)[:, None]), dout, input_precision="tf32")
        dstate -= tl.dot(tl.trans(w), ddelta, input_precision="tf32")
        dv = dxv * beta[:, None]
        po = ((((iv * B + ib) * HV + ih).to(tl.int64) * T + t) * K)
        tl.store(qpp + po[:, None] + x[None, :], dq * scale,
                 mask=ok[:, None] & (x[None, :] < K))
        tl.store(kpp + po[:, None] + x[None, :], dk,
                 mask=ok[:, None] & (x[None, :] < K))
        dvo = (ib * T * HV + t * HV + ih).to(tl.int64) * V
        tl.store(dvp + dvo[:, None] + y[None, :], dv,
                 mask=ok[:, None] & (y[None, :] < V))
        so = ((((iv * B + ib) * HV + ih).to(tl.int64) * T + t) * 2)
        tl.store(spp + so, dg, mask=ok)
        tl.store(spp + so + 1, dbeta, mask=ok)


@triton.jit
def _sum_qk(qpp, kpp, dqp, dkp,
            T: tl.constexpr, B: tl.constexpr, HQ: tl.constexpr,
            HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
            BK: tl.constexpr):
    pid = tl.program_id(0)
    it = pid % T
    pid //= T
    kh = pid % HQ
    ib = pid // HQ
    x = tl.arange(0, BK)
    dq = tl.zeros((BK,), tl.float32)
    dk = tl.zeros((BK,), tl.float32)
    for r in range(HV // HQ):
        ih = kh * (HV // HQ) + r
        for iv in range(NV):
            off = ((((iv * B + ib) * HV + ih).to(tl.int64) * T + it) * K + x)
            dq += tl.load(qpp + off, mask=x < K, other=0.0)
            dk += tl.load(kpp + off, mask=x < K, other=0.0)
    out = (ib * T * HQ + it * HQ + kh).to(tl.int64) * K + x
    tl.store(dqp + out, dq, mask=x < K)
    tl.store(dkp + out, dk, mask=x < K)


@triton.jit
def _sum_scalars(spp, dgp, dbp,
                 T: tl.constexpr, B: tl.constexpr, HV: tl.constexpr,
                 NV: tl.constexpr):
    pid = tl.program_id(0)
    it = pid % T
    z = pid // T
    ih = z % HV
    ib = z // HV
    dg = 0.0
    db = 0.0
    for iv in range(NV):
        off = ((((iv * B + ib) * HV + ih).to(tl.int64) * T + it) * 2)
        dg += tl.load(spp + off)
        db += tl.load(spp + off + 1)
    out = ib * T * HV + it * HV + ih
    tl.store(dgp + out, dg)
    tl.store(dbp + out, db)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, HQ, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC = triton.cdiv(T, CHUNK)
        NV = triton.cdiv(V, VALUE_BLOCK)
        BK = triton.next_power_of_2(K)
        inv = torch.empty((B, HV, NC, CHUNK, CHUNK), device=q.device, dtype=torch.float32)
        w = torch.empty((B, HV, T, K), device=q.device, dtype=torch.float32)
        u = torch.empty((B, HV, T, V), device=q.device, dtype=torch.float32)
        states = torch.empty((B, HV, NC, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty((B, HV, T, V), device=q.device, dtype=torch.float32)
        out = torch.empty((B, T, HV, V), device=q.device, dtype=v.dtype)
        _wy_build[(B * HV * NC,)](k, g, beta, inv, w, T=T, B=B, HQ=HQ, HV=HV, K=K, NC=NC, BK=BK, C=CHUNK, num_warps=4)
        _value_build[(B * HV * NC * NV,)](v, beta, inv, u, T=T, B=B, HV=HV, V=V, NC=NC, NV=NV, BV=VALUE_BLOCK, C=CHUNK, num_warps=4)
        _state_forward[(B * HV * NV,)](k, g, w, u, states, delta, T=T, B=B, HQ=HQ, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK, BV=VALUE_BLOCK, C=CHUNK, num_warps=8)
        _output_forward[(B * HV * NC * NV,)](q, k, g, states, delta, out, scale=K ** -0.5, T=T, B=B, HQ=HQ, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK, BV=VALUE_BLOCK, C=CHUNK, num_warps=8)
        ctx.save_for_backward(q, k, v, g, beta, inv, w, states, delta)
        ctx.meta = (B, T, HQ, HV, K, V, NC, NV, BK)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, inv, w, states, delta = ctx.saved_tensors
        B, T, HQ, HV, K, V, NC, NV, BK = ctx.meta
        qpart = torch.empty((NV, B, HV, T, K), device=q.device, dtype=torch.float32)
        kpart = torch.empty((NV, B, HV, T, K), device=q.device, dtype=torch.float32)
        scalars = torch.empty((NV, B, HV, T, 2), device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        dbeta = torch.empty_like(beta)
        _reverse_chunks[(B * HV * NV,)](q, k, v, g, beta, dout, inv, w, states, delta, qpart, kpart, dv, scalars, *dout.stride(), scale=K ** -0.5, T=T, B=B, HQ=HQ, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK, BV=VALUE_BLOCK, C=CHUNK, num_warps=4)
        _sum_qk[(B * HQ * T,)](qpart, kpart, dq, dk, T=T, B=B, HQ=HQ, HV=HV, K=K, NV=NV, BK=BK, num_warps=4)
        _sum_scalars[(B * HV * T,)](scalars, dg, dbeta, T=T, B=B, HV=HV, NV=NV, num_warps=4)
        return dq, dk, dv, dg, dbeta


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
