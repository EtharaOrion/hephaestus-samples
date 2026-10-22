
import torch
import triton
import triton.language as tl

C = 32
BV = 16


@triton.jit
def _make_inverse(k, g, beta, gc, inv, T: tl.constexpr, H: tl.constexpr,
                  HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
                  BK: tl.constexpr, BC: tl.constexpr):
    ic = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // HV
    ihv = ibh - ib * HV
    ih = ihv // (HV // H)
    t = ic * BC + tl.arange(0, BC)
    x = tl.arange(0, BK)
    mt = t < T
    base_k = ((ib * T + t) * H + ih) * K
    kk = tl.load(k + base_k[:, None] + x[None, :],
                 mask=mt[:, None] & (x[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    gg = tl.cumsum(gg, axis=0)
    bb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    row = tl.arange(0, BC)[:, None]
    col = tl.arange(0, BC)[None, :]
    valid = mt[:, None] & mt[None, :]
    lower = row > col
    mat = tl.where(lower & valid, gram * bb[:, None] * tl.exp(gg[:, None] - gg[None, :]), 0.0)
    mi = tl.zeros((BC, BC), tl.float32)
    for i in range(1, BC):
        r = tl.sum(tl.where((tl.arange(0, BC) == i)[:, None], -mat, 0.0), axis=0)
        r = tl.where(tl.arange(0, BC) < i, r, 0.0)
        r += tl.sum(r[:, None] * mi, axis=0)
        mi = tl.where((tl.arange(0, BC) == i)[:, None], r, mi)
    mi += row == col
    aoff = (((ib * NC + ic) * HV + ihv) * BC + row) * BC + col
    tl.store(inv + aoff, mi, mask=valid)
    tl.store(gc + (ib * T + t) * HV + ihv, gg, mask=mt)


@triton.jit
def _forward_scan(k, v, gc, beta, inv, checkpoints,
                  T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                  BK: tl.constexpr, BC: tl.constexpr, VB: tl.constexpr,
                  SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv = p % nv
    p //= nv
    ihv = p % HV
    ib = p // HV
    ih = ihv // (HV // H)
    x = tl.arange(0, BK)
    y = iv * VB + tl.arange(0, VB)
    r = tl.arange(0, BC)
    state = tl.zeros((BK, VB), tl.float32)
    for ic in range(NC):
        t = ic * BC + r
        mt = t < T
        cp = (((ib * HV + ihv) * NC + ic) * K + x[:, None]) * V + y[None, :]
        tl.store(checkpoints + cp, state, mask=(x[:, None] < K) & (y[None, :] < V))
        ko = ((ib * T + t[:, None]) * H + ih) * K + x[None, :]
        vo = ((ib * T + t[:, None]) * HV + ihv) * V + y[None, :]
        sk = tl.load(k + ko, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        sv = tl.load(v + vo, mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        sg = tl.load(gc + (ib * T + t) * HV + ihv, mask=mt, other=0.0)
        sb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
        ao = (((ib * NC + ic) * HV + ihv) * BC + r[:, None]) * BC + r[None, :]
        ma = tl.load(inv + ao, mask=mt[:, None] & mt[None, :], other=0.0)
        pred = tl.dot(sk, state, input_precision="tf32") * tl.exp(sg)[:, None]
        rhs = (sv.to(tl.float32) - pred) * sb[:, None]
        delta = tl.dot(ma, rhs, input_precision="tf32")
        nvalid = tl.minimum(BC, T - ic * BC)
        glast = tl.sum(tl.where(r == nvalid - 1, sg, 0.0), axis=0)
        weighted = delta * tl.exp(glast - sg)[:, None]
        state = state * tl.exp(glast) + tl.dot(tl.trans(sk), weighted, input_precision="tf32")


@triton.jit
def _emit_chunks(q, k, v, gc, beta, inv, checkpoints, out,
                 T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                 K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                 BK: tl.constexpr, BC: tl.constexpr, VB: tl.constexpr,
                 SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv = p % nv
    p //= nv
    ic = p % NC
    p //= NC
    ihv = p % HV
    ib = p // HV
    ih = ihv // (HV // H)
    x = tl.arange(0, BK)
    y = iv * VB + tl.arange(0, VB)
    r = tl.arange(0, BC)
    t = ic * BC + r
    mt = t < T
    ko = ((ib * T + t[:, None]) * H + ih) * K + x[None, :]
    vo = ((ib * T + t[:, None]) * HV + ihv) * V + y[None, :]
    sk = tl.load(k + ko, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    sq = tl.load(q + ko, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    sv = tl.load(v + vo, mask=mt[:, None] & (y[None, :] < V), other=0.0)
    sg = tl.load(gc + (ib * T + t) * HV + ihv, mask=mt, other=0.0)
    sb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    ao = (((ib * NC + ic) * HV + ihv) * BC + r[:, None]) * BC + r[None, :]
    ma = tl.load(inv + ao, mask=mt[:, None] & mt[None, :], other=0.0)
    cp = (((ib * HV + ihv) * NC + ic) * K + x[:, None]) * V + y[None, :]
    state = tl.load(checkpoints + cp, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    pred = tl.dot(sk, state, input_precision="tf32") * tl.exp(sg)[:, None]
    delta = tl.dot(ma, (sv.to(tl.float32) - pred) * sb[:, None], input_precision="tf32")
    dots = tl.dot(sq, tl.trans(sk), input_precision="tf32") * SCALE
    tri = (r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :]
    weights = tl.where(tri, dots * tl.exp(sg[:, None] - sg[None, :]), 0.0)
    result = tl.dot(sq, state, input_precision="tf32") * (SCALE * tl.exp(sg))[:, None]
    result += tl.dot(weights, delta, input_precision="tf32")
    tl.store(out + vo, result, mask=mt[:, None] & (y[None, :] < V))



@triton.jit
def _adjoint_scan(q, k, gc, beta, inv, dout, adjoints,
                  T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                  BK: tl.constexpr, BC: tl.constexpr, VB: tl.constexpr,
                  SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv = p % nv
    p //= nv
    ihv = p % HV
    ib = p // HV
    ih = ihv // (HV // H)
    x = tl.arange(0, BK)
    y = iv * VB + tl.arange(0, VB)
    r = tl.arange(0, BC)
    carry = tl.zeros((BK, VB), tl.float32)
    for rev in range(NC):
        ic = NC - 1 - rev
        ap = (((ib * HV + ihv) * NC + ic) * K + x[:, None]) * V + y[None, :]
        tl.store(adjoints + ap, carry, mask=(x[:, None] < K) & (y[None, :] < V))
        t = ic * BC + r
        mt = t < T
        ko = ((ib * T + t[:, None]) * H + ih) * K + x[None, :]
        vo = ((ib * T + t[:, None]) * HV + ihv) * V + y[None, :]
        sk = tl.load(k + ko, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        sq = tl.load(q + ko, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
        sd = tl.load(dout + vo, mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        sg = tl.load(gc + (ib * T + t) * HV + ihv, mask=mt, other=0.0)
        sb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
        ao = (((ib * NC + ic) * HV + ihv) * BC + r[:, None]) * BC + r[None, :]
        ma = tl.load(inv + ao, mask=mt[:, None] & mt[None, :], other=0.0)
        tri = (r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :]
        weights = tl.where(tri, tl.dot(sq, tl.trans(sk), input_precision="tf32") *
                           tl.exp(sg[:, None] - sg[None, :]), 0.0)
        ddelta = tl.dot(tl.trans(weights), sd, input_precision="tf32")
        nvalid = tl.minimum(BC, T - ic * BC)
        glast = tl.sum(tl.where(r == nvalid - 1, sg, 0.0), axis=0)
        z = sk * tl.exp(glast - sg)[:, None]
        ddelta += tl.dot(z, carry, input_precision="tf32")
        dx = tl.dot(tl.trans(ma), ddelta, input_precision="tf32")
        carry = tl.dot(tl.trans(sq * tl.exp(sg)[:, None]), sd, input_precision="tf32")
        carry += tl.exp(glast) * tl.load(adjoints + ap,
                                        mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
        carry += tl.dot(tl.trans(sk * tl.exp(sg)[:, None]), -dx * sb[:, None],
                        input_precision="tf32")

@triton.jit
def _reverse_chunks(q, k, v, gc, beta, inv, checkpoints, adjoints, dout,
                    pdq, pdk, dv, pdg, pdb,
                    T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                    BK: tl.constexpr, BC: tl.constexpr, VB: tl.constexpr,
                    SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, VB)
    iv = p % nv
    p //= nv
    ic = p % NC
    p //= NC
    ihv = p % HV
    ib = p // HV
    ih = ihv // (HV // H)
    x = tl.arange(0, BK)
    y = iv * VB + tl.arange(0, VB)
    r = tl.arange(0, BC)
    t = ic * BC + r
    mt = t < T
    ko = ((ib * T + t[:, None]) * H + ih) * K + x[None, :]
    vo = ((ib * T + t[:, None]) * HV + ihv) * V + y[None, :]
    sk = tl.load(k + ko, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    sq = tl.load(q + ko, mask=mt[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    sv = tl.load(v + vo, mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
    sd = tl.load(dout + vo, mask=mt[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
    sg = tl.load(gc + (ib * T + t) * HV + ihv, mask=mt, other=0.0)
    sb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    cp = (((ib * HV + ihv) * NC + ic) * K + x[:, None]) * V + y[None, :]
    s0 = tl.load(checkpoints + cp, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    ap = (((ib * HV + ihv) * NC + ic) * K + x[:, None]) * V + y[None, :]
    state_bar = tl.load(adjoints + ap, mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    ao = (((ib * NC + ic) * HV + ihv) * BC + r[:, None]) * BC + r[None, :]
    ma = tl.load(inv + ao, mask=mt[:, None] & mt[None, :], other=0.0)
    eg = tl.exp(sg)
    pred = tl.dot(sk, s0, input_precision="tf32") * eg[:, None]
    rhs = (sv.to(tl.float32) - pred) * sb[:, None]
    delta = tl.dot(ma, rhs, input_precision="tf32")
    qq = sq * SCALE
    dots = tl.dot(qq, tl.trans(sk), input_precision="tf32")
    tri = (r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :]
    ew = tl.exp(sg[:, None] - sg[None, :])
    weights = tl.where(tri, dots * ew, 0.0)

    ddelta = tl.dot(tl.trans(weights), sd, input_precision="tf32")
    dp = tl.dot(sd, tl.trans(delta), input_precision="tf32")
    dqq = tl.dot(dp * tl.where(tri, ew, 0.0), sk, input_precision="tf32")
    dk_part = tl.dot(tl.trans(dp * tl.where(tri, ew, 0.0)), qq, input_precision="tf32")
    dgc = tl.sum(dp * weights, axis=1) - tl.sum(dp * weights, axis=0)

    cross_q = tl.dot(sd, tl.trans(s0), input_precision="tf32") * eg[:, None]
    dqq += cross_q
    dgc += tl.sum(sd * tl.dot(qq, s0, input_precision="tf32") * eg[:, None], axis=1)
    ds0 = tl.dot(tl.trans(qq * eg[:, None]), sd, input_precision="tf32")

    nvalid = tl.minimum(BC, T - ic * BC)
    glast = tl.sum(tl.where(r == nvalid - 1, sg, 0.0), axis=0)
    ef = tl.exp(glast - sg)
    z = sk * ef[:, None]
    ddelta += tl.dot(z, state_bar, input_precision="tf32")
    dz = tl.dot(delta, tl.trans(state_bar), input_precision="tf32")
    dk_part += dz * ef[:, None]
    zprod = tl.sum(dz * z, axis=1)
    dgc -= zprod
    dlast = tl.sum(zprod, axis=0) + tl.sum(tl.sum(state_bar * s0, axis=1), axis=0) * tl.exp(glast)
    dgc += tl.where(r == nvalid - 1, dlast, 0.0)
    ds0 += state_bar * tl.exp(glast)

    dx = tl.dot(tl.trans(ma), ddelta, input_precision="tf32")
    ddv = dx * sb[:, None]
    db = tl.sum(dx * (sv.to(tl.float32) - pred), axis=1)
    dpred = -dx * sb[:, None]
    dk_part += tl.dot(dpred, tl.trans(s0), input_precision="tf32") * eg[:, None]
    ds0 += tl.dot(tl.trans(sk * eg[:, None]), dpred, input_precision="tf32")
    dgc += tl.sum(dpred * pred, axis=1)

    dl = -tl.dot(dx, tl.trans(delta), input_precision="tf32")
    strict = (r[:, None] > r[None, :]) & mt[:, None] & mt[None, :]
    dl = tl.where(strict, dl, 0.0)
    gram = tl.dot(sk, tl.trans(sk), input_precision="tf32")
    le = tl.where(strict, tl.exp(sg[:, None] - sg[None, :]), 0.0)
    db += tl.sum(dl * le * gram, axis=1)
    cmat = dl * le * sb[:, None]
    dk_part += tl.dot(cmat + tl.trans(cmat), sk, input_precision="tf32")
    lg = cmat * gram
    dgc += tl.sum(lg, axis=1) - tl.sum(lg, axis=0)

    dgc = tl.cumsum(dgc, axis=0, reverse=True)
    po = ((((ib * T + t[:, None]) * HV + ihv) * nv + iv) * K + x[None, :])
    tl.store(pdq + po, dqq * SCALE, mask=mt[:, None] & (x[None, :] < K))
    tl.store(pdk + po, dk_part, mask=mt[:, None] & (x[None, :] < K))
    tl.store(dv + vo, ddv, mask=mt[:, None] & (y[None, :] < V))
    so = (((ib * T + t) * HV + ihv) * nv + iv)
    tl.store(pdg + so, dgc, mask=mt)
    tl.store(pdb + so, db, mask=mt)
    state_bar = ds0

@triton.jit
def _reduce_qk(pdq, pdk, dq, dk, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
               BK: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    ih = p % H
    it = (p // H) % T
    ib = p // (H * T)
    x = tl.arange(0, BK)
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    groups = HV // H
    for j in range(groups):
        ihv = ih * groups + j
        for iv in range(NV):
            off = ((((ib * T + it) * HV + ihv) * NV + iv) * K + x)
            aq += tl.load(pdq + off, mask=x < K, other=0.0)
            ak += tl.load(pdk + off, mask=x < K, other=0.0)
    dst = ((ib * T + it) * H + ih) * K + x
    tl.store(dq + dst, aq, mask=x < K)
    tl.store(dk + dst, ak, mask=x < K)


@triton.jit
def _reduce_scalars(pdg, pdb, dg, db, N: tl.constexpr, NV: tl.constexpr,
                    BN: tl.constexpr):
    z = tl.program_id(0).to(tl.int64) * BN + tl.arange(0, BN)
    ag = tl.zeros((BN,), tl.float32)
    ab = tl.zeros((BN,), tl.float32)
    for iv in range(NV):
        ag += tl.load(pdg + z * NV + iv, mask=z < N, other=0.0)
        ab += tl.load(pdb + z * NV + iv, mask=z < N, other=0.0)
    tl.store(dg + z, ag, mask=z < N)
    tl.store(db + z, ab, mask=z < N)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        nc = triton.cdiv(T, C)
        nv = triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        gc = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
        inv = torch.empty(B, nc, HV, C, C, device=q.device, dtype=torch.float32)
        checkpoints = torch.empty(B, HV, nc, K, V, device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _make_inverse[(nc, B * HV)](k, g, beta, gc, inv, T=T, H=H, HV=HV,
                                      K=K, NC=nc, BK=bk, BC=C, num_warps=4)
        _forward_scan[(B * HV * nv,)](k, v, gc, beta, inv, checkpoints,
                                      T=T, H=H, HV=HV, K=K, V=V, NC=nc,
                                      BK=bk, BC=C, VB=BV, SCALE=K ** -0.5,
                                      num_warps=8)
        _emit_chunks[(B * HV * nc * nv,)](q, k, v, gc, beta, inv, checkpoints, out,
                                          T=T, H=H, HV=HV, K=K, V=V, NC=nc,
                                          BK=bk, BC=C, VB=BV, SCALE=K ** -0.5,
                                          num_warps=8)
        ctx.save_for_backward(q, k, v, gc, beta, inv, checkpoints)
        ctx.dims = (B, T, H, HV, K, V, nc, nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, gc, beta, inv, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, HV, K, V, nc, nv, bk = ctx.dims
        shape = (B, T, HV, nv, K)
        pdq = torch.empty(shape, device=q.device, dtype=torch.float32)
        pdk = torch.empty(shape, device=q.device, dtype=torch.float32)
        pdg = torch.empty(B, T, HV, nv, device=q.device, dtype=torch.float32)
        pdb = torch.empty_like(pdg, dtype=torch.float32)
        adjoints = torch.empty_like(checkpoints, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
        db = torch.empty_like(beta)
        _adjoint_scan[(B * HV * nv,)](q, k, gc, beta, inv, dout, adjoints,
                                      T=T, H=H, HV=HV, K=K, V=V, NC=nc,
                                      BK=bk, BC=C, VB=BV, SCALE=K ** -0.5,
                                      num_warps=8)
        _reverse_chunks[(B * HV * nc * nv,)](q, k, v, gc, beta, inv, checkpoints,
                                             adjoints, dout, pdq, pdk, dv, pdg, pdb,
                                             T=T, H=H, HV=HV, K=K, V=V, NC=nc,
                                             BK=bk, BC=C, VB=BV, SCALE=K ** -0.5,
                                             num_warps=8)
        _reduce_qk[(B * T * H,)](pdq, pdk, dq, dk, T=T, H=H, HV=HV,
                                 K=K, NV=nv, BK=bk, num_warps=4)
        _reduce_scalars[(triton.cdiv(B * T * HV, 256),)](
            pdg, pdb, dg, db, N=B * T * HV, NV=nv, BN=256, num_warps=4)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
