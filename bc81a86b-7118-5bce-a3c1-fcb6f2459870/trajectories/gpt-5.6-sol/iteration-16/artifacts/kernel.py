import math

import torch
import triton
import triton.language as tl

CHUNK = 16
C = tl.constexpr(16)
BV = 16


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(0.0, x)))


@triton.jit
def _build_inverse(k, g, beta, ainv, T: tl.constexpr, H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0)
    ph = tl.program_id(1)
    b = ph // HV
    vh = ph - b * HV
    kh = vh // (HV // H)
    r = tl.arange(0, C)
    x = tl.arange(0, BK)
    tt = ic * C + r
    valid = tt < T
    gi = tl.load(g + (b * T + tt) * HV + vh, mask=valid, other=0.0).to(tl.float32)
    hs = tl.cumsum(gi, axis=0)
    be = tl.load(beta + (b * T + tt) * HV + vh, mask=valid, other=0.0).to(tl.float32)
    kk = tl.load(k + ((b * T + tt[:, None]) * H + kh) * K + x[None, :],
                 mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    rr = r[:, None]
    cc = r[None, :]
    chronological = (rr > cc) & valid[:, None] & valid[None, :]
    e = _bounded_exp(hs[:, None] - hs[None, :])
    mat = tl.where(chronological, be[:, None] * e * gram, 0.0)
    inv = tl.where(rr == cc, 1.0, 0.0)
    for i in tl.static_range(0, C):
        coeff = tl.sum(tl.where(rr == i, mat, 0.0), axis=0)
        coeff = tl.where((r < i) & valid, coeff, 0.0)
        row = -tl.sum(coeff[:, None] * inv, axis=0)
        row += tl.where(r == i, 1.0, 0.0)
        inv = tl.where(rr == i, row[None, :], inv)
    base = ((ph * tl.cdiv(T, C) + ic) * C + rr) * C + cc
    tl.store(ainv + base, inv, mask=valid[:, None] & valid[None, :])


@triton.jit
def _state_scan(k, v, g, beta, ainv, states, delta,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                BK: tl.constexpr, BV_: tl.constexpr):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV_)
    ph = p // nv
    iv = p - ph * nv
    b = ph // HV
    vh = ph - b * HV
    kh = vh // (HV // H)
    x = tl.arange(0, BK)
    y = iv * BV_ + tl.arange(0, BV_)
    r = tl.arange(0, C)
    state = tl.zeros((BK, BV_), tl.float32)
    for ic in range(NC):
        tt = ic * C + r
        tm = tt < T
        tl.store(states + (((ph * NC + ic) * K + x[:, None]) * V + y[None, :]), state,
                 mask=(x[:, None] < K) & (y[None, :] < V))
        gg = tl.load(g + (b * T + tt) * HV + vh, mask=tm, other=0.0).to(tl.float32)
        hs = tl.cumsum(gg, axis=0)
        bb = tl.load(beta + (b * T + tt) * HV + vh, mask=tm, other=0.0).to(tl.float32)
        kb = tl.load(k + ((b * T + tt[:, None]) * H + kh) * K + x[None, :],
                     mask=tm[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        vv = tl.load(v + ((b * T + tt[:, None]) * HV + vh) * V + y[None, :],
                     mask=tm[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        pred = tl.dot(kb, state, input_precision="tf32")
        rhs = bb[:, None] * (vv - _bounded_exp(hs)[:, None] * pred)
        rr = r[:, None]
        cc = r[None, :]
        aa = tl.load(ainv + ((ph * NC + ic) * C + rr) * C + cc,
                     mask=tm[:, None] & tm[None, :], other=0.0)
        dd = tl.dot(aa, rhs, input_precision="tf32")
        tl.store(delta + ((ph * T + tt[:, None]) * V + y[None, :]), dd,
                 mask=tm[:, None] & (y[None, :] < V))
        he = tl.sum(gg, axis=0)
        weights = _bounded_exp(he - hs)
        state = _bounded_exp(he) * state + tl.dot(tl.trans(kb), weights[:, None] * dd,
                                                   input_precision="tf32")


@triton.jit
def _emit(q, k, g, states, delta, out,
          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
          K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
          BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV_)
    iv = p % nv
    z = p // nv
    ic = z % NC
    ph = z // NC
    b = ph // HV
    vh = ph - b * HV
    kh = vh // (HV // H)
    r = tl.arange(0, C)
    x = tl.arange(0, BK)
    y = iv * BV_ + tl.arange(0, BV_)
    tt = ic * C + r
    tm = tt < T
    gg = tl.load(g + (b * T + tt) * HV + vh, mask=tm, other=0.0).to(tl.float32)
    hs = tl.cumsum(gg, axis=0)
    qb = tl.load(q + ((b * T + tt[:, None]) * H + kh) * K + x[None, :],
                 mask=tm[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    kb = tl.load(k + ((b * T + tt[:, None]) * H + kh) * K + x[None, :],
                 mask=tm[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    st = tl.load(states + (((ph * NC + ic) * K + x[:, None]) * V + y[None, :]),
                 mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    dd = tl.load(delta + ((ph * T + tt[:, None]) * V + y[None, :]),
                 mask=tm[:, None] & (y[None, :] < V), other=0.0)
    score = tl.dot(qb, tl.trans(kb), input_precision="tf32") * SCALE
    rr = r[:, None]
    cc = r[None, :]
    causal = (rr >= cc) & tm[:, None] & tm[None, :]
    decay = _bounded_exp(hs[:, None] - hs[None, :])
    local = tl.where(causal, score * decay, 0.0)
    oo = _bounded_exp(hs)[:, None] * tl.dot(qb, st, input_precision="tf32") * SCALE
    oo += tl.dot(local, dd, input_precision="tf32")
    tl.store(out + ((b * T + tt[:, None]) * HV + vh) * V + y[None, :], oo,
             mask=tm[:, None] & (y[None, :] < V))


@triton.jit
def _reverse_scan(q, k, v, g, beta, states, delta, dout,
                  pdq, pdk, dv, pdg, pdb,
                  T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                  K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                  BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV_)
    ph = p // nv
    iv = p - ph * nv
    b = ph // HV
    vh = ph - b * HV
    kh = vh // (HV // H)
    x = tl.arange(0, BK)
    y = iv * BV_ + tl.arange(0, BV_)
    r = tl.arange(0, C)
    adj = tl.zeros((BK, BV_), tl.float32)
    for z in range(NC):
        ic = NC - 1 - z
        tt = ic * C + r
        tm = tt < T
        gg = tl.load(g + (b * T + tt) * HV + vh, mask=tm, other=0.0).to(tl.float32)
        hs = tl.cumsum(gg, axis=0)
        kb = tl.load(k + ((b * T + tt[:, None]) * H + kh) * K + x[None, :],
                     mask=tm[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        dd = tl.load(delta + ((ph * T + tt[:, None]) * V + y[None, :]),
                     mask=tm[:, None] & (y[None, :] < V), other=0.0)
        start = tl.load(states + (((ph * NC + ic) * K + x[:, None]) * V + y[None, :]),
                        mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
        for zz in tl.static_range(0, C):
            i = C - 1 - zz
            ti = ic * C + i
            ok = ti < T
            qi = tl.load(q + ((b * T + ti) * H + kh) * K + x,
                         mask=ok & (x < K), other=0.0).to(tl.float32)
            pick = r == i
            ki = tl.sum(tl.where(pick[:, None], kb, 0.0), axis=0)
            ddi_row = tl.sum(tl.where(pick[:, None], dd, 0.0), axis=0)
            hsi = tl.sum(tl.where(pick, hs, 0.0), axis=0)
            ggi = tl.sum(tl.where(pick, gg, 0.0), axis=0)
            vi = tl.load(v + ((b * T + ti) * HV + vh) * V + y,
                         mask=ok & (y < V), other=0.0).to(tl.float32)
            doi = tl.load(dout + ((b * T + ti) * HV + vh) * V + y,
                          mask=ok & (y < V), other=0.0).to(tl.float32)
            bi = tl.load(beta + (b * T + ti) * HV + vh, mask=ok, other=0.0).to(tl.float32)
            causal = (r <= i) & tm
            weights = tl.where(causal, _bounded_exp(hsi - hs), 0.0)
            after = _bounded_exp(hsi) * start + tl.dot(tl.trans(kb), weights[:, None] * dd,
                                                          input_precision="tf32")
            before_write = after - ki[:, None] * ddi_row[None, :]
            dqi = tl.sum(after * doi[None, :], axis=1) * SCALE
            full_adj = adj + qi[:, None] * (doi[None, :] * SCALE)
            ddi = tl.sum(ki[:, None] * full_adj, axis=0)
            pred = tl.sum(ki[:, None] * before_write, axis=0)
            residual = vi - pred
            dbi = tl.sum(ddi * residual, axis=0)
            dvi = bi * ddi
            dp = -bi * ddi
            dki = tl.sum(full_adj * ddi_row[None, :], axis=1)
            dki += tl.sum(before_write * dp[None, :], axis=1)
            dbefore = full_adj + ki[:, None] * dp[None, :]
            dgi = tl.sum(tl.sum(dbefore * before_write, axis=0), axis=0)
            decay = _bounded_exp(ggi)
            adj = decay * dbefore
            qoff = (((ph * nv + iv) * T + ti) * K + x).to(tl.int64)
            soff = ((ph * nv + iv) * T + ti).to(tl.int64)
            tl.store(pdq + qoff, dqi, mask=ok & (x < K))
            tl.store(pdk + qoff, dki, mask=ok & (x < K))
            tl.store(dv + ((b * T + ti) * HV + vh) * V + y, dvi,
                     mask=ok & (y < V))
            tl.store(pdg + soff, dgi, mask=ok)
            tl.store(pdb + soff, dbi, mask=ok)


@triton.jit
def _reduce_qk(pdq, pdk, dq, dk, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
               BK: tl.constexpr):
    p = tl.program_id(0)
    t = p % T
    z = p // T
    h = z % H
    b = z // H
    x = tl.arange(0, BK)
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    groups = HV // H
    for j in range(groups):
        vh = h * groups + j
        ph = b * HV + vh
        for iv in range(NV):
            off = (((ph * NV + iv) * T + t) * K + x).to(tl.int64)
            aq += tl.load(pdq + off, mask=x < K, other=0.0).to(tl.float32)
            ak += tl.load(pdk + off, mask=x < K, other=0.0).to(tl.float32)
    outoff = ((b * T + t) * H + h) * K + x
    tl.store(dq + outoff, aq, mask=x < K)
    tl.store(dk + outoff, ak, mask=x < K)


@triton.jit
def _reduce_scalars(pdg, pdb, dg, db, T: tl.constexpr, HV: tl.constexpr,
                    NV: tl.constexpr):
    p = tl.program_id(0)
    t = p % T
    ph = p // T
    b = ph // HV
    vh = ph - b * HV
    ag = 0.0
    ab = 0.0
    for iv in range(NV):
        off = ((ph * NV + iv) * T + t).to(tl.int64)
        ag += tl.load(pdg + off)
        ab += tl.load(pdb + off)
    tl.store(dg + (b * T + t) * HV + vh, ag)
    tl.store(db + (b * T + t) * HV + vh, ab)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        if HV % H:
            raise ValueError("value heads must be divisible by key heads")
        nc = triton.cdiv(T, CHUNK)
        nv = triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        ainv = torch.empty((B, HV, nc, CHUNK, CHUNK), device=q.device, dtype=torch.float32)
        states = torch.empty((B, HV, nc, K, V), device=q.device, dtype=torch.float32)
        delta = torch.empty((B, HV, T, V), device=q.device, dtype=torch.float32)
        out = torch.empty((B, T, HV, V), device=q.device, dtype=v.dtype)
        _build_inverse[(nc, B * HV)](k, g, beta, ainv, T=T, H=H, HV=HV, K=K,
                                     BK=bk, num_warps=4)
        _state_scan[(B * HV * nv,)](k, v, g, beta, ainv, states, delta,
                                    T=T, H=H, HV=HV, K=K, V=V, NC=nc,
                                    BK=bk, BV_=BV, num_warps=8)
        _emit[(B * HV * nc * nv,)](q, k, g, states, delta, out,
                                   T=T, H=H, HV=HV, K=K, V=V, NC=nc,
                                   BK=bk, BV_=BV, SCALE=K ** -0.5, num_warps=8)
        ctx.save_for_backward(q, k, v, g, beta, states, delta)
        ctx.dims = (B, T, H, HV, K, V, nc, nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        q, k, v, g, beta, states, delta = ctx.saved_tensors
        B, T, H, HV, K, V, nc, nv, bk = ctx.dims
        pdq = torch.empty((B, HV, nv, T, K), device=q.device, dtype=q.dtype)
        pdk = torch.empty((B, HV, nv, T, K), device=q.device, dtype=k.dtype)
        pdg = torch.empty((B, HV, nv, T), device=q.device, dtype=torch.float32)
        pdb = torch.empty((B, HV, nv, T), device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        _reverse_scan[(B * HV * nv,)](q, k, v, g, beta, states, delta, dout,
                                      pdq, pdk, dv, pdg, pdb, T=T, H=H, HV=HV,
                                      K=K, V=V, NC=nc, BK=bk, BV_=BV,
                                      SCALE=K ** -0.5, num_warps=8)
        _reduce_qk[(B * H * T,)](pdq, pdk, dq, dk, T=T, H=H, HV=HV, K=K,
                                 NV=nv, BK=bk, num_warps=4)
        _reduce_scalars[(B * HV * T,)](pdg, pdb, dg, db, T=T, HV=HV, NV=nv)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
