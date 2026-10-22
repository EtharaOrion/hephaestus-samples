import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _build_wy(k, v, g, beta, inv, w, u, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              BK: tl.constexpr, NC: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    b, hv = bh // HV, bh % HV
    h = hv // (HV // H)
    it = ic * C + tl.arange(0, C)
    ok = tl.arange(0, BK)
    mt = it < T
    mk = ok < K
    p_k = k + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
    kk = tl.load(p_k, mask=mt[:, None] & mk[None, :], other=0.0)
    gg = tl.load(g + (b * T + it) * HV + hv, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(beta + (b * T + it) * HV + hv, mask=mt, other=0.0).to(tl.float32)
    G = tl.cumsum(gg, axis=0)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    ii = tl.arange(0, C)
    lower = (ii[:, None] > ii[None, :]) & mt[:, None] & mt[None, :]
    decay_pair = tl.exp(tl.where(lower, G[:, None] - G[None, :], 0.0))
    M = tl.where(lower, gram * bb[:, None] * decay_pair, 0.0)
    P = tl.where(ii[:, None] == ii[None, :], 1.0, 0.0)
    for row in range(1, C):
        mr = tl.sum(tl.where(ii[:, None] == row, M, 0.0), axis=0)
        pr = -tl.sum(mr[:, None] * P, axis=0) + (ii == row)
        P = tl.where(ii[:, None] == row, pr[None, :], P)
    pinv = inv + ((b * NC + ic) * HV + hv) * C * C + ii[:, None] * C + ii[None, :]
    tl.store(pinv, P)
    x = kk * (bb * tl.exp(G))[:, None]
    ww = tl.dot(P.to(kk.dtype), x.to(kk.dtype), input_precision="tf32")
    pw = w + ((b * T + it[:, None]) * HV + hv) * K + ok[None, :]
    tl.store(pw, ww, mask=mt[:, None] & mk[None, :])
    for iv in range(tl.cdiv(V, BV)):
        ov = iv * BV + tl.arange(0, BV)
        mv = ov < V
        pv = v + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
        vv = tl.load(pv, mask=mt[:, None] & mv[None, :], other=0.0)
        uu = tl.dot(P.to(vv.dtype), (vv * bb[:, None]).to(vv.dtype), input_precision="tf32")
        pu = u + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
        tl.store(pu, uu, mask=mt[:, None] & mv[None, :])


@triton.jit
def _scan_state(k, g, w, u, states, delta, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                BK: tl.constexpr, NC: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    bh, iv = pid // nv, pid % nv
    b, hv = bh // HV, bh % HV
    h = hv // (HV // H)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    S = tl.zeros((BK, BV), tl.float32)
    ii = tl.arange(0, C)
    for ic in range(NC):
        it = ic * C + ii
        mt = it < T
        ps = states + ((((b * NC + ic) * HV + hv) * K + ok[:, None]) * V + ov[None, :])
        tl.store(ps, S, mask=mk[:, None] & mv[None, :])
        pk = k + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
        pw = w + ((b * T + it[:, None]) * HV + hv) * K + ok[None, :]
        pu = u + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
        kb = tl.load(pk, mask=mt[:, None] & mk[None, :], other=0.0)
        wb = tl.load(pw, mask=mt[:, None] & mk[None, :], other=0.0)
        ub = tl.load(pu, mask=mt[:, None] & mv[None, :], other=0.0)
        D = ub - tl.dot(wb, S.to(wb.dtype), input_precision="tf32")
        pd = delta + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
        tl.store(pd, D, mask=mt[:, None] & mv[None, :])
        G = tl.cumsum(tl.load(g + (b * T + it) * HV + hv, mask=mt, other=0.0), axis=0)
        last = tl.sum(tl.where(ii == tl.minimum(C - 1, T - ic * C - 1), G, 0.0), axis=0)
        N = kb * tl.exp(last - G)[:, None]
        S = S * tl.exp(last) + tl.dot(tl.trans(N), D.to(N.dtype), input_precision="tf32")


@triton.jit
def _emit(q, k, g, delta, states, out, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
          BK: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    ic, rem = pid // (HV * nv), pid % (HV * nv)
    b = ic // NC
    ic = ic % NC
    hv, iv = rem // nv, rem % nv
    h = hv // (HV // H)
    ii, ok = tl.arange(0, C), tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    it = ic * C + ii
    mt, mk, mv = it < T, ok < K, ov < V
    pq = q + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
    pk = k + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
    qb = tl.load(pq, mask=mt[:, None] & mk[None, :], other=0.0)
    kb = tl.load(pk, mask=mt[:, None] & mk[None, :], other=0.0)
    G = tl.cumsum(tl.load(g + (b * T + it) * HV + hv, mask=mt, other=0.0), axis=0)
    ps = states + ((((b * NC + ic) * HV + hv) * K + ok[:, None]) * V + ov[None, :])
    S = tl.load(ps, mask=mk[:, None] & mv[None, :], other=0.0)
    pd = delta + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
    D = tl.load(pd, mask=mt[:, None] & mv[None, :], other=0.0)
    causal = (ii[:, None] >= ii[None, :]) & mt[:, None] & mt[None, :]
    pair_decay = tl.exp(tl.where(causal, G[:, None] - G[None, :], 0.0))
    R = tl.where(causal, tl.dot(qb, tl.trans(kb), input_precision="tf32") * SCALE * pair_decay, 0.0)
    O = tl.dot((qb * (tl.exp(G) * SCALE)[:, None]).to(qb.dtype), S.to(qb.dtype), input_precision="tf32")
    O += tl.dot(R.to(D.dtype), D, input_precision="tf32")
    po = out + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
    tl.store(po, O, mask=mt[:, None] & mv[None, :])


@triton.jit
def _adjoint_scan_k(q, k, g, w, delta, dout, dstates, T: tl.constexpr, H: tl.constexpr,
                    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BK: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr,
                    DS0: tl.constexpr, DS1: tl.constexpr, DS2: tl.constexpr, DS3: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    bh, iv = pid // nv, pid % nv
    b, hv = bh // HV, bh % HV
    h = hv // (HV // H)
    ii, ok = tl.arange(0, C), tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    dS = tl.zeros((BK, BV), tl.float32)
    for jc in range(NC):
        ic = NC - 1 - jc
        it = ic * C + ii
        mt = it < T
        pe = dstates + ((((b * NC + ic) * HV + hv) * K + ok[:, None]) * V + ov[None, :])
        tl.store(pe, dS, mask=mk[:, None] & mv[None, :])
        pq = q + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
        pk = k + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
        pw = w + ((b * T + it[:, None]) * HV + hv) * K + ok[None, :]
        po = dout + b * DS0 + it[:, None] * DS1 + hv * DS2 + ov[None, :] * DS3
        qb = tl.load(pq, mask=mt[:, None] & mk[None, :], other=0.0)
        kb = tl.load(pk, mask=mt[:, None] & mk[None, :], other=0.0)
        wb = tl.load(pw, mask=mt[:, None] & mk[None, :], other=0.0)
        do = tl.load(po, mask=mt[:, None] & mv[None, :], other=0.0)
        G = tl.cumsum(tl.load(g + (b * T + it) * HV + hv, mask=mt, other=0.0), axis=0)
        last = tl.sum(tl.where(ii == tl.minimum(C - 1, T - ic * C - 1), G, 0.0), axis=0)
        N = kb * tl.exp(last - G)[:, None]
        causal = (ii[:, None] >= ii[None, :]) & mt[:, None] & mt[None, :]
        pair_decay = tl.exp(tl.where(causal, G[:, None] - G[None, :], 0.0))
        R = tl.where(causal, tl.dot(qb, tl.trans(kb), input_precision="tf32") * SCALE * pair_decay, 0.0)
        dD = tl.dot(N, dS.to(N.dtype), input_precision="tf32")
        dD += tl.dot(tl.trans(R).to(do.dtype), do, input_precision="tf32")
        dS = dS * tl.exp(last)
        qe = (qb * (tl.exp(G) * SCALE)[:, None]).to(qb.dtype)
        dS += tl.dot(tl.trans(qe), do.to(qb.dtype), input_precision="tf32")
        dS -= tl.dot(tl.trans(wb), dD.to(wb.dtype), input_precision="tf32")


@triton.jit
def _chunk_grad(q, k, v, g, beta, inv, w, u, delta, states, dstates, dout,
                pdq, pdk, dv, pdg, pdb, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                BK: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr,
                DS0: tl.constexpr, DS1: tl.constexpr, DS2: tl.constexpr, DS3: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    ic, rem = pid // (HV * nv), pid % (HV * nv)
    b, ic = ic // NC, ic % NC
    hv, iv = rem // nv, rem % nv
    h = hv // (HV // H)
    ii, ok = tl.arange(0, C), tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    it = ic * C + ii
    mt, mk, mv = it < T, ok < K, ov < V
    pq = q + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
    pk = k + ((b * T + it[:, None]) * H + h) * K + ok[None, :]
    pv = v + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
    pw = w + ((b * T + it[:, None]) * HV + hv) * K + ok[None, :]
    pu = u + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
    pdel = delta + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :]
    qb = tl.load(pq, mask=mt[:, None] & mk[None, :], other=0.0)
    kb = tl.load(pk, mask=mt[:, None] & mk[None, :], other=0.0)
    vb = tl.load(pv, mask=mt[:, None] & mv[None, :], other=0.0)
    wb = tl.load(pw, mask=mt[:, None] & mk[None, :], other=0.0)
    ub = tl.load(pu, mask=mt[:, None] & mv[None, :], other=0.0)
    D = tl.load(pdel, mask=mt[:, None] & mv[None, :], other=0.0)
    pdo = dout + b * DS0 + it[:, None] * DS1 + hv * DS2 + ov[None, :] * DS3
    do = tl.load(pdo, mask=mt[:, None] & mv[None, :], other=0.0)
    ps = states + ((((b * NC + ic) * HV + hv) * K + ok[:, None]) * V + ov[None, :])
    pe = dstates + ((((b * NC + ic) * HV + hv) * K + ok[:, None]) * V + ov[None, :])
    S = tl.load(ps, mask=mk[:, None] & mv[None, :], other=0.0)
    dE = tl.load(pe, mask=mk[:, None] & mv[None, :], other=0.0)
    G = tl.cumsum(tl.load(g + (b * T + it) * HV + hv, mask=mt, other=0.0), axis=0)
    last = tl.sum(tl.where(ii == tl.minimum(C - 1, T - ic * C - 1), G, 0.0), axis=0)
    bb = tl.load(beta + (b * T + it) * HV + hv, mask=mt, other=0.0).to(tl.float32)
    P = tl.load(inv + ((b * NC + ic) * HV + hv) * C * C + ii[:, None] * C + ii[None, :])
    N = kb * tl.exp(last - G)[:, None]
    Qe = qb * (tl.exp(G) * SCALE)[:, None]
    tri = (ii[:, None] >= ii[None, :]) & mt[:, None] & mt[None, :]
    pair_decay = tl.exp(tl.where(tri, G[:, None] - G[None, :], 0.0))
    R = tl.where(tri, tl.dot(qb, tl.trans(kb), input_precision="tf32") * SCALE * pair_decay, 0.0)
    dD = tl.dot(N, dE.to(N.dtype), input_precision="tf32") + tl.dot(tl.trans(R).to(do.dtype), do, input_precision="tf32")
    dU = dD
    dW = -tl.dot(dD, tl.trans(S.to(dD.dtype)), input_precision="tf32")
    X = kb * (bb * tl.exp(G))[:, None]
    Y = vb * bb[:, None]
    dA = tl.dot(dU, tl.trans(Y.to(dU.dtype)), input_precision="tf32") + tl.dot(dW, tl.trans(X.to(dW.dtype)), input_precision="tf32")
    dY = tl.dot(tl.trans(P).to(dU.dtype), dU, input_precision="tf32")
    dX = tl.dot(tl.trans(P).to(dW.dtype), dW, input_precision="tf32")
    dM = -tl.dot(tl.trans(P), tl.dot(dA, tl.trans(P), input_precision="tf32"), input_precision="tf32")
    dM = tl.where(ii[:, None] > ii[None, :], dM, 0.0)
    gram = tl.dot(kb, tl.trans(kb), input_precision="tf32")
    fac = bb[:, None] * pair_decay
    dgram = dM * fac
    dk = tl.dot(dgram.to(kb.dtype), kb, input_precision="tf32") + tl.dot(tl.trans(dgram).to(kb.dtype), kb, input_precision="tf32")
    dk += dX * (bb * tl.exp(G))[:, None]
    dN = tl.dot(D, tl.trans(dE.to(D.dtype)), input_precision="tf32")
    dk += dN * tl.exp(last - G)[:, None]
    dR = tl.dot(do, tl.trans(D.to(do.dtype)), input_precision="tf32")
    dbase = dR * tl.exp(G[:, None] - G[None, :]) * SCALE
    dbase = tl.where(tri, dbase, 0.0)
    dq = tl.dot(dbase.to(kb.dtype), kb, input_precision="tf32")
    dk += tl.dot(tl.trans(dbase).to(qb.dtype), qb, input_precision="tf32")
    dQe = tl.dot(do, tl.trans(S.to(do.dtype)), input_precision="tf32")
    dq += dQe * (tl.exp(G) * SCALE)[:, None]
    dgG = tl.sum(dbase * tl.dot(qb, tl.trans(kb), input_precision="tf32"), axis=1) - tl.sum(dbase * tl.dot(qb, tl.trans(kb), input_precision="tf32"), axis=0)
    dgG += tl.sum(dQe * Qe, axis=1) + tl.sum(dX * X, axis=1)
    dgG += tl.sum(dM * (gram * fac), axis=1) - tl.sum(dM * (gram * fac), axis=0)
    zN = tl.sum(dN * N, axis=1)
    dgG -= zN
    dlast = tl.sum(zN) + tl.sum(dE * (S * tl.exp(last)))
    dgG += tl.where(ii == tl.minimum(C - 1, T - ic * C - 1), dlast, 0.0)
    dg = tl.flip(tl.cumsum(tl.flip(dgG, dim=0), axis=0), dim=0)
    db = tl.sum(dY * vb, axis=1) + tl.sum(dX * (kb * tl.exp(G)[:, None]), axis=1)
    db += tl.sum(dM * gram * pair_decay, axis=1)
    dpbase = (((b * T + it) * HV + hv) * nv + iv)
    pqo = pdq + dpbase[:, None] * K + ok[None, :]
    pko = pdk + dpbase[:, None] * K + ok[None, :]
    pgo = pdg + dpbase
    pbo = pdb + dpbase
    tl.store(pqo, dq, mask=mt[:, None] & mk[None, :])
    tl.store(pko, dk, mask=mt[:, None] & mk[None, :])
    tl.store(pgo, dg, mask=mt)
    tl.store(pbo, db, mask=mt)
    tl.store(dv + ((b * T + it[:, None]) * HV + hv) * V + ov[None, :], dY * bb[:, None], mask=mt[:, None] & mv[None, :])


@triton.jit
def _reduce_qk(pdq, pdk, dq, dk, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    bth = pid // H
    h = pid % H
    b = bth // T
    t = bth % T
    ok = tl.arange(0, BK)
    mk = ok < K
    nv = tl.cdiv(V, BV)
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    rep = HV // H
    for r in range(rep):
        hv = h * rep + r
        for iv in range(nv):
            base = ((((b * T + t) * HV + hv) * nv + iv) * K)
            aq += tl.load(pdq + base + ok, mask=mk, other=0.0)
            ak += tl.load(pdk + base + ok, mask=mk, other=0.0)
    tl.store(dq + ((b * T + t) * H + h) * K + ok, aq, mask=mk)
    tl.store(dk + ((b * T + t) * H + h) * K + ok, ak, mask=mk)


@triton.jit
def _reduce_scalar(px, out, N: tl.constexpr, NV: tl.constexpr, BNV: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    iv = tl.arange(0, BNV)
    x = tl.load(px + pid * NV + iv, mask=iv < NV, other=0.0)
    tl.store(out + pid, tl.sum(x, axis=0))


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        NC, BK = triton.cdiv(T, C), triton.next_power_of_2(K)
        nv = triton.cdiv(V, BV)
        inv = torch.empty((B, NC, HV, C, C), device=q.device, dtype=torch.float32)
        w = torch.empty((B, T, HV, K), device=q.device, dtype=q.dtype)
        u = torch.empty_like(v)
        delta = torch.empty_like(v)
        states = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(NC, B * HV)](k, v, g, beta, inv, w, u, T=T, H=H, HV=HV, K=K, V=V, BK=BK, NC=NC, num_warps=4)
        _scan_state[(B * HV * nv,)](k, g, w, u, states, delta, T=T, H=H, HV=HV, K=K, V=V, BK=BK, NC=NC, num_warps=4)
        _emit[(B * NC * HV * nv,)](q, k, g, delta, states, out, T=T, H=H, HV=HV, K=K, V=V, BK=BK, NC=NC, SCALE=K ** -0.5, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, inv, w, u, delta, states)
        ctx.dims = (B, T, H, HV, K, V, NC, BK, nv)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, inv, w, u, delta, states = ctx.saved_tensors
        B, T, H, HV, K, V, NC, BK, nv = ctx.dims
        dstates = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        pdq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pdk = torch.empty_like(pdq)
        pdg = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32)
        pdb = torch.empty_like(pdg)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dv, dg, db = torch.empty_like(v), torch.empty_like(g), torch.empty_like(beta)
        ds0, ds1, ds2, ds3 = dout.stride()
        _adjoint_scan_k[(B * HV * nv,)](q, k, g, w, delta, dout, dstates, T=T, H=H, HV=HV, K=K, V=V, BK=BK, NC=NC, SCALE=K ** -0.5, DS0=ds0, DS1=ds1, DS2=ds2, DS3=ds3, num_warps=4)
        _chunk_grad[(B * NC * HV * nv,)](q, k, v, g, beta, inv, w, u, delta, states, dstates, dout, pdq, pdk, dv, pdg, pdb, T=T, H=H, HV=HV, K=K, V=V, BK=BK, NC=NC, SCALE=K ** -0.5, DS0=ds0, DS1=ds1, DS2=ds2, DS3=ds3, num_warps=4)
        _reduce_qk[(B * T * H,)](pdq, pdk, dq, dk, T=T, H=H, HV=HV, K=K, V=V, BK=BK, num_warps=4)
        bnv = triton.next_power_of_2(nv)
        _reduce_scalar[(B * T * HV,)](pdg, dg, N=B*T*HV, NV=nv, BNV=bnv, num_warps=1)
        _reduce_scalar[(B * T * HV,)](pdb, db, N=B*T*HV, NV=nv, BNV=bnv, num_warps=1)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
