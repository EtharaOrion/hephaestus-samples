import torch
import triton
import triton.language as tl

C = 16
BV = 32


@triton.jit
def _build_wy(k, v, g, beta, inv, w, x, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              BK: tl.constexpr, BC: tl.constexpr, VB: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ihv = tl.program_id(1).to(tl.int64)
    ib = ihv // HV
    ivh = ihv - ib * HV
    ikh = ivh // (HV // H)
    tt = ic * BC + tl.arange(0, BC)
    ii = tl.arange(0, BC)
    kk = tl.arange(0, BK)
    mt = tt < T
    kp = k + ((ib * T + tt[:, None]) * H + ikh) * K + kk[None, :]
    kb = tl.load(kp, mask=mt[:, None] & (kk[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(beta + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    gram = tl.dot(kb, tl.trans(kb), input_precision="tf32")
    lower = (ii[:, None] > ii[None, :]) & mt[:, None] & mt[None, :]
    decay = tl.exp(tl.where(lower, cs[:, None] - cs[None, :], 0.0))
    ll = tl.where(lower, gram * bb[:, None] * decay, 0.0)
    aa = tl.where(ii[:, None] == ii[None, :], 1.0, 0.0)
    for row in range(1, BC):
        lr = tl.sum(tl.where(ii[:, None] == row, ll, 0.0), axis=0)
        ar = -tl.sum(lr[:, None] * aa, axis=0) + (ii == row)
        aa = tl.where(ii[:, None] == row, ar[None, :], aa)
    ap = inv + (((ib * ((T + BC - 1) // BC) + ic) * HV + ivh) * BC + ii[:, None]) * BC + ii[None, :]
    tl.store(ap, aa, mask=mt[:, None] & mt[None, :])
    z = (kb * (bb * tl.exp(cs))[:, None]).to(kb.dtype)
    wb = tl.dot(aa.to(kb.dtype), z, input_precision="tf32")
    wp = w + ((ib * T + tt[:, None]) * HV + ivh) * K + kk[None, :]
    tl.store(wp, wb, mask=mt[:, None] & (kk[None, :] < K))
    vv = tl.arange(0, VB)
    for jv in range(tl.cdiv(V, VB)):
        ov = jv * VB + vv
        vp = v + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        vb = tl.load(vp, mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        xb = tl.dot(aa.to(vb.dtype), (vb * bb[:, None]).to(vb.dtype), input_precision="tf32")
        xp = x + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        tl.store(xp, xb, mask=mt[:, None] & (ov[None, :] < V))


@triton.jit
def _scan_state(k, g, w, x, hs, y, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                NC: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
                VB: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + VB - 1) // VB
    jv = pid % nv
    z = pid // nv
    ivh = z % HV
    ib = z // HV
    ikh = ivh // (HV // H)
    ik = tl.arange(0, BK)
    ov = jv * VB + tl.arange(0, VB)
    it = tl.arange(0, BC)
    state = tl.zeros((BK, VB), tl.float32)
    for ic in range(NC):
        tt = ic * BC + it
        mt = tt < T
        hp = hs + ((((ib * NC + ic) * HV + ivh) * K + ik[:, None]) * V + ov[None, :])
        tl.store(hp, state, mask=(ik[:, None] < K) & (ov[None, :] < V))
        wp = w + ((ib * T + tt[:, None]) * HV + ivh) * K + ik[None, :]
        wb = tl.load(wp, mask=mt[:, None] & (ik[None, :] < K), other=0.0)
        xp = x + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        xb = tl.load(xp, mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        yy = xb - tl.dot(wb, state.to(wb.dtype), input_precision="tf32")
        yp = y + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        tl.store(yp, yy, mask=mt[:, None] & (ov[None, :] < V))
        gp = g + (ib * T + tt) * HV + ivh
        gg = tl.load(gp, mask=mt, other=0.0).to(tl.float32)
        cs = tl.cumsum(gg, axis=0)
        total = tl.sum(gg, axis=0)
        kp = k + ((ib * T + tt[:, None]) * H + ikh) * K + ik[None, :]
        kb = tl.load(kp, mask=mt[:, None] & (ik[None, :] < K), other=0.0)
        ek = (kb * tl.exp(total - cs)[:, None]).to(kb.dtype)
        state = state * tl.exp(total) + tl.dot(tl.trans(ek), yy.to(ek.dtype), input_precision="tf32")


@triton.jit
def _emit(q, k, g, hs, y, out, T: tl.constexpr, H: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
          BK: tl.constexpr, BC: tl.constexpr, VB: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + VB - 1) // VB
    jv = pid % nv
    z = pid // nv
    ic = z % NC
    z //= NC
    ivh = z % HV
    ib = z // HV
    ikh = ivh // (HV // H)
    it = tl.arange(0, BC)
    ik = tl.arange(0, BK)
    ov = jv * VB + tl.arange(0, VB)
    tt = ic * BC + it
    mt = tt < T
    qp = q + ((ib * T + tt[:, None]) * H + ikh) * K + ik[None, :]
    kp = k + ((ib * T + tt[:, None]) * H + ikh) * K + ik[None, :]
    qb = tl.load(qp, mask=mt[:, None] & (ik[None, :] < K), other=0.0) * SCALE
    kb = tl.load(kp, mask=mt[:, None] & (ik[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    hp = hs + ((((ib * NC + ic) * HV + ivh) * K + ik[:, None]) * V + ov[None, :])
    hb = tl.load(hp, mask=(ik[:, None] < K) & (ov[None, :] < V), other=0.0)
    yp = y + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
    yy = tl.load(yp, mask=mt[:, None] & (ov[None, :] < V), other=0.0)
    mm = tl.dot(qb, tl.trans(kb), input_precision="tf32")
    causal = (it[:, None] >= it[None, :]) & mt[:, None] & mt[None, :]
    mm *= tl.exp(tl.where(causal, cs[:, None] - cs[None, :], 0.0))
    mm = tl.where(causal, mm, 0.0)
    oo = tl.dot((qb * tl.exp(cs)[:, None]).to(qb.dtype), hb.to(qb.dtype), input_precision="tf32")
    oo += tl.dot(mm.to(yy.dtype), yy, input_precision="tf32")
    op = out + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
    tl.store(op, oo, mask=mt[:, None] & (ov[None, :] < V))


@triton.jit
def _adjoint(q, k, g, w, dout, dhs, T: tl.constexpr, H: tl.constexpr,
             HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
             NC: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
             VB: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + VB - 1) // VB
    jv = pid % nv
    z = pid // nv
    ivh = z % HV
    ib = z // HV
    ikh = ivh // (HV // H)
    it = tl.arange(0, BC)
    ik = tl.arange(0, BK)
    ov = jv * VB + tl.arange(0, VB)
    dh = tl.zeros((BK, VB), tl.float32)
    for jr in range(NC):
        ic = NC - 1 - jr
        tt = ic * BC + it
        mt = tt < T
        ep = dhs + ((((ib * NC + ic) * HV + ivh) * K + ik[:, None]) * V + ov[None, :])
        tl.store(ep, dh, mask=(ik[:, None] < K) & (ov[None, :] < V))
        qp = q + ((ib * T + tt[:, None]) * H + ikh) * K + ik[None, :]
        kp = k + ((ib * T + tt[:, None]) * H + ikh) * K + ik[None, :]
        qb = tl.load(qp, mask=mt[:, None] & (ik[None, :] < K), other=0.0) * SCALE
        kb = tl.load(kp, mask=mt[:, None] & (ik[None, :] < K), other=0.0)
        wb = tl.load(w + ((ib * T + tt[:, None]) * HV + ivh) * K + ik[None, :], mask=mt[:, None] & (ik[None, :] < K), other=0.0)
        do = tl.load(dout + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :], mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        gg = tl.load(g + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
        cs = tl.cumsum(gg, axis=0)
        total = tl.sum(gg, axis=0)
        causal = (it[:, None] >= it[None, :]) & mt[:, None] & mt[None, :]
        mm = tl.dot(qb, tl.trans(kb), input_precision="tf32") * tl.exp(tl.where(causal, cs[:, None] - cs[None, :], 0.0))
        mm = tl.where(causal, mm, 0.0)
        ek = (kb * tl.exp(total - cs)[:, None]).to(kb.dtype)
        dy = tl.dot(tl.trans(mm.to(do.dtype)), do, input_precision="tf32") + tl.dot(ek, dh.to(ek.dtype), input_precision="tf32")
        dh = dh * tl.exp(total)
        dh += tl.dot(tl.trans((qb * tl.exp(cs)[:, None]).to(do.dtype)), do, input_precision="tf32")
        dh -= tl.dot(tl.trans(wb), dy.to(wb.dtype), input_precision="tf32")


@triton.jit
def _chunk_grad(q, k, v, g, beta, inv, w, x, hs, y, dout, dhs,
                pq, pk, dv, dg, db, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                NC: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
                VB: tl.constexpr, SCALE: tl.constexpr):
    ic = tl.program_id(0).to(tl.int64)
    ihv = tl.program_id(1).to(tl.int64)
    ib = ihv // HV
    ivh = ihv - ib * HV
    ikh = ivh // (HV // H)
    it = tl.arange(0, BC)
    ik = tl.arange(0, BK)
    tt = ic * BC + it
    mt = tt < T
    qp0 = q + ((ib * T + tt[:, None]) * H + ikh) * K + ik[None, :]
    kp0 = k + ((ib * T + tt[:, None]) * H + ikh) * K + ik[None, :]
    qb = tl.load(qp0, mask=mt[:, None] & (ik[None, :] < K), other=0.0) * SCALE
    kb = tl.load(kp0, mask=mt[:, None] & (ik[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(beta + (ib * T + tt) * HV + ivh, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    total = tl.sum(gg, axis=0)
    tri = (it[:, None] >= it[None, :]) & mt[:, None] & mt[None, :]
    rr = tl.exp(tl.where(tri, cs[:, None] - cs[None, :], 0.0))
    strict = (it[:, None] > it[None, :]) & mt[:, None] & mt[None, :]
    gram = tl.dot(kb, tl.trans(kb), input_precision="tf32")
    mm = tl.where(tri, tl.dot(qb, tl.trans(kb), input_precision="tf32") * rr, 0.0)
    ll = tl.where(strict, gram * bb[:, None] * rr, 0.0)
    aa = tl.load(inv + (((ib * NC + ic) * HV + ivh) * BC + it[:, None]) * BC + it[None, :], mask=mt[:, None] & mt[None, :], other=0.0)
    wb = tl.load(w + ((ib * T + tt[:, None]) * HV + ivh) * K + ik[None, :], mask=mt[:, None] & (ik[None, :] < K), other=0.0)
    zb = kb * (bb * tl.exp(cs))[:, None]
    dqa = tl.zeros((BC, BK), tl.float32)
    dka = tl.zeros((BC, BK), tl.float32)
    dca = tl.zeros((BC,), tl.float32)
    dba = tl.zeros((BC,), tl.float32)
    da = tl.zeros((BC, BC), tl.float32)
    vv = tl.arange(0, VB)
    for jv in range(tl.cdiv(V, VB)):
        ov = jv * VB + vv
        mv = ov < V
        vp = v + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        vb = tl.load(vp, mask=mt[:, None] & mv[None, :], other=0.0)
        xp = x + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        xb = tl.load(xp, mask=mt[:, None] & mv[None, :], other=0.0)
        yp = y + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        yy = tl.load(yp, mask=mt[:, None] & mv[None, :], other=0.0)
        op = dout + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :]
        do = tl.load(op, mask=mt[:, None] & mv[None, :], other=0.0)
        hp = hs + ((((ib * NC + ic) * HV + ivh) * K + ik[:, None]) * V + ov[None, :])
        hb = tl.load(hp, mask=(ik[:, None] < K) & mv[None, :], other=0.0)
        ep = dhs + ((((ib * NC + ic) * HV + ivh) * K + ik[:, None]) * V + ov[None, :])
        de = tl.load(ep, mask=(ik[:, None] < K) & mv[None, :], other=0.0)
        ek = (kb * tl.exp(total - cs)[:, None]).to(kb.dtype)
        dy = tl.dot(tl.trans(mm.to(do.dtype)), do, input_precision="tf32") + tl.dot(ek, de.to(ek.dtype), input_precision="tf32")
        dm = tl.dot(do, tl.trans(yy), input_precision="tf32")
        dek = tl.dot(yy, tl.trans(de.to(yy.dtype)), input_precision="tf32")
        dw = -tl.dot(dy, tl.trans(hb), input_precision="tf32")
        rr0 = vb * bb[:, None]
        da += tl.dot(dy, tl.trans(rr0.to(dy.dtype)), input_precision="tf32") + tl.dot(dw, tl.trans(zb.to(dw.dtype)), input_precision="tf32")
        dr = tl.dot(tl.trans(aa.to(dy.dtype)), dy, input_precision="tf32")
        dz = tl.dot(tl.trans(aa.to(dw.dtype)), dw, input_precision="tf32")
        tl.store(dv + ((ib * T + tt[:, None]) * HV + ivh) * V + ov[None, :], dr * bb[:, None], mask=mt[:, None] & mv[None, :])
        dba += tl.sum(dr * vb, axis=1) + tl.sum(dz * kb * tl.exp(cs)[:, None], axis=1)
        dca += tl.sum(dz * zb, axis=1)
        dqa += tl.dot((do * tl.exp(cs)[:, None]).to(hb.dtype), tl.trans(hb), input_precision="tf32")
        dmr = tl.where(tri, dm * rr, 0.0)
        dqa += tl.dot(dmr.to(kb.dtype), kb, input_precision="tf32")
        dka += tl.dot(tl.trans(dmr.to(qb.dtype)), qb, input_precision="tf32")
        sm = dm * mm
        dca += tl.sum(sm, axis=1) - tl.sum(sm, axis=0)
        dka += dek * tl.exp(total - cs)[:, None]
        se = tl.sum(dek * (kb * tl.exp(total - cs)[:, None]), axis=1)
        dca -= se
        dca += tl.where(it == BC - 1, tl.sum(se, axis=0), 0.0)
        dka += dz * (bb * tl.exp(cs))[:, None]
        direct = tl.sum((tl.dot(do, tl.trans(hb.to(do.dtype)), input_precision="tf32") * qb), axis=1) * tl.exp(cs)
        dca += direct
    dl = -tl.dot(tl.trans(aa), tl.dot(da, tl.trans(aa), input_precision="tf32"), input_precision="tf32")
    dl = tl.where(strict, dl, 0.0)
    coeff = dl * bb[:, None] * rr
    dka += tl.dot(coeff.to(kb.dtype), kb, input_precision="tf32")
    dka += tl.dot(tl.trans(coeff.to(kb.dtype)), kb, input_precision="tf32")
    dba += tl.sum(dl * gram * rr, axis=1)
    sl = dl * ll
    dca += tl.sum(sl, axis=1) - tl.sum(sl, axis=0)
    dgc = tl.flip(tl.cumsum(tl.flip(dca, 0), axis=0), 0)
    tl.store(pq + ((ib * T + tt[:, None]) * HV + ivh) * K + ik[None, :], dqa * SCALE, mask=mt[:, None] & (ik[None, :] < K))
    tl.store(pk + ((ib * T + tt[:, None]) * HV + ivh) * K + ik[None, :], dka, mask=mt[:, None] & (ik[None, :] < K))
    tl.store(dg + (ib * T + tt) * HV + ivh, dgc, mask=mt)
    tl.store(db + (ib * T + tt) * HV + ivh, dba, mask=mt)


@triton.jit
def _reduce_heads(pq, pk, dq, dk, N: tl.constexpr, H: tl.constexpr,
                  HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    ih = pid % H
    n = pid // H
    ik = tl.arange(0, BK)
    sq = tl.zeros((BK,), tl.float32)
    sk = tl.zeros((BK,), tl.float32)
    R: tl.constexpr = HV // H
    for r in range(R):
        ivh = ih * R + r
        sq += tl.load(pq + (n * HV + ivh) * K + ik, mask=ik < K, other=0.0)
        sk += tl.load(pk + (n * HV + ivh) * K + ik, mask=ik < K, other=0.0)
    tl.store(dq + (n * H + ih) * K + ik, sq, mask=ik < K)
    tl.store(dk + (n * H + ih) * K + ik, sk, mask=ik < K)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        nc = triton.cdiv(T, C)
        bk = triton.next_power_of_2(K)
        nv = triton.cdiv(V, BV)
        inv = torch.empty((B, nc, HV, C, C), device=q.device, dtype=torch.float32)
        w = torch.empty((B, T, HV, K), device=q.device, dtype=q.dtype)
        x = torch.empty_like(v)
        hs = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        y = torch.empty_like(v)
        out = torch.empty_like(v)
        _build_wy[(nc, B * HV)](k, v, g, beta, inv, w, x, T=T, H=H, HV=HV, K=K, V=V, BK=bk, BC=C, VB=BV, num_warps=4)
        _scan_state[(B * HV * nv,)](k, g, w, x, hs, y, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, BC=C, VB=BV, num_warps=4)
        _emit[(B * HV * nc * nv,)](q, k, g, hs, y, out, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, BC=C, VB=BV, SCALE=K ** -0.5, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, inv, w, x, hs, y)
        ctx.dims = (B, T, H, HV, K, V, nc, bk, nv)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, inv, w, x, hs, y = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, HV, K, V, nc, bk, nv = ctx.dims
        dhs = torch.empty_like(hs)
        pq = torch.empty((B, T, HV, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        _adjoint[(B * HV * nv,)](q, k, g, w, dout, dhs, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, BC=C, VB=BV, SCALE=K ** -0.5, num_warps=4)
        _chunk_grad[(nc, B * HV)](q, k, v, g, beta, inv, w, x, hs, y, dout, dhs, pq, pk, dv, dg, db, T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk, BC=C, VB=BV, SCALE=K ** -0.5, num_warps=4)
        _reduce_heads[(B * T * H,)](pq, pk, dq, dk, N=B*T, H=H, HV=HV, K=K, BK=bk, num_warps=4)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
