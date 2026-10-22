import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 32


@triton.jit
def _prepare_chunks(k, g, beta, inverse, T: tl.constexpr, HQ: tl.constexpr,
                    HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
                    BK: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    ic, z = pid % NC, pid // NC
    ih, ib = z % HV, z // HV
    iqh = ih // (HV // HQ)
    rt = ic * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    rows = tl.arange(0, BT)[:, None]
    cols = tl.arange(0, BT)[None, :]
    mt = rt < T
    pk = (((ib * T + rt[:, None]) * HQ + iqh) * K + rk[None, :]).to(tl.int64)
    pg = ((ib * T + rt) * HV + ih).to(tl.int64)
    kk = tl.load(k + pk, mask=mt[:, None] & (rk[None, :] < K), other=0.).to(tl.float32)
    gg = tl.load(g + pg, mask=mt, other=0.).to(tl.float32)
    bb = tl.load(beta + pg, mask=mt, other=0.).to(tl.float32)
    cg = tl.maximum(tl.cumsum(gg, axis=0), -60.0)
    left = kk * tl.exp(cg)[:, None]
    right = kk / tl.exp(cg)[:, None]
    lower = tl.dot(left, tl.trans(right), input_precision='ieee') * bb[:, None]
    lower = tl.where(rows > cols, lower, 0.)
    inv = tl.where(rows == cols, 1., 0.)
    for i in range(BT):
        source = tl.sum(tl.where(rows == i, lower, 0.), axis=0)
        solved = -tl.sum(source[:, None] * inv, axis=0)
        inv = tl.where(rows == i, tl.where(cols < i, solved[None, :], inv), inv)
    pi = (((((ib * HV + ih) * NC + ic) * BT + rows) * BT) + cols).to(tl.int64)
    tl.store(inverse + pi, inv)


@triton.jit
def _state_fwd(k, v, g, beta, inverse, states, updates, T: tl.constexpr, HQ: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
               BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, z = pid % nv, pid // nv
    ih, ib = z % HV, z // HV
    rk, rv = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    st = tl.zeros((BK, BV), tl.float32)
    rr, cc = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    for ic in range(NC):
        rt = ic * BT + tl.arange(0, BT)
        mt = rt < T
        mk = mt[:, None] & (rk[None, :] < K)
        mv = mt[:, None] & (rv[None, :] < V)
        ps = (((((ib * HV + ih) * NC + ic) * K + rk[:, None]) * V) + rv[None, :]).to(tl.int64)
        tl.store(states + ps, st, mask=(rk[:, None] < K) & (rv[None, :] < V))
        pk = (((ib * T + rt[:, None]) * HQ + ih // (HV // HQ)) * K + rk[None, :]).to(tl.int64)
        pv = (((ib * T + rt[:, None]) * HV + ih) * V + rv[None, :]).to(tl.int64)
        pg = ((ib * T + rt) * HV + ih).to(tl.int64)
        kk = tl.load(k + pk, mask=mk, other=0.).to(tl.float32)
        vv = tl.load(v + pv, mask=mv, other=0.).to(tl.float32)
        gg = tl.load(g + pg, mask=mt, other=0.).to(tl.float32)
        bb = tl.load(beta + pg, mask=mt, other=0.).to(tl.float32)
        cg = tl.maximum(tl.cumsum(gg, axis=0), -60.0)
        ee = tl.exp(cg)
        yy, xx = kk * ee[:, None], kk / ee[:, None]
        raw = bb[:, None] * (vv - tl.dot(yy, st, input_precision='ieee'))
        pi = (((((ib * HV + ih) * NC + ic) * BT + rr) * BT) + cc).to(tl.int64)
        inv = tl.load(inverse + pi)
        uu = tl.dot(inv, raw, input_precision='ieee')
        tl.store(updates + pv, uu, mask=mv)
        ef = tl.sum(tl.where(tl.arange(0, BT) == BT - 1, ee, 0.), axis=0)
        st = ef * (st + tl.dot(tl.trans(xx), uu, input_precision='ieee'))


@triton.jit
def _chunk_output(q, k, g, states, updates, out, T: tl.constexpr, HQ: tl.constexpr,
                  HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                  NV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    iv, z = pid % NV, pid // NV
    ic, z = z % NC, z // NC
    ih, ib = z % HV, z // HV
    rt = ic * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mt = rt < T
    pk = (((ib * T + rt[:, None]) * HQ + ih // (HV // HQ)) * K + rk[None, :]).to(tl.int64)
    pv = (((ib * T + rt[:, None]) * HV + ih) * V + rv[None, :]).to(tl.int64)
    pg = ((ib * T + rt) * HV + ih).to(tl.int64)
    ps = (((((ib * HV + ih) * NC + ic) * K + rk[:, None]) * V) + rv[None, :]).to(tl.int64)
    kk = tl.load(k + pk, mask=mt[:, None] & (rk[None, :] < K), other=0.).to(tl.float32)
    qq = tl.load(q + pk, mask=mt[:, None] & (rk[None, :] < K), other=0.).to(tl.float32) * (K ** -0.5)
    gg = tl.load(g + pg, mask=mt, other=0.).to(tl.float32)
    st = tl.load(states + ps, mask=(rk[:, None] < K) & (rv[None, :] < V), other=0.)
    uu = tl.load(updates + pv, mask=mt[:, None] & (rv[None, :] < V), other=0.)
    cg = tl.maximum(tl.cumsum(gg, axis=0), -60.0)
    ee = tl.exp(cg)
    xx = kk / ee[:, None]
    qb = qq * ee[:, None]
    rows = tl.arange(0, BT)[:, None]
    cols = tl.arange(0, BT)[None, :]
    causal = tl.where(rows >= cols, tl.dot(qb, tl.trans(xx), input_precision='ieee'), 0.)
    oo = tl.dot(qb, st, input_precision='ieee') + tl.dot(causal, uu, input_precision='ieee')
    tl.store(out + pv, oo, mask=mt[:, None] & (rv[None, :] < V))


@triton.jit
def _chunk_bwd(q, k, v, g, beta, do, inverse, states, updates, dvout, pq, pkout, pgout, pbout,
               T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
               V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
               BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    iv, z = pid % NV, pid // NV
    ih, ib = z % HV, z // HV
    iqh = ih // (HV // HQ)
    rk, rv = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    rr, cc = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    dst = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    for ir in range(NC):
        ic = NC - 1 - ir
        rt = ic * BT + tl.arange(0, BT)
        mt = rt < T
        mk = mt[:, None] & (rk[None, :] < K)
        mv = mt[:, None] & (rv[None, :] < V)
        pki = (((ib * T + rt[:, None]) * HQ + iqh) * K + rk[None, :]).to(tl.int64)
        pvi = (((ib * T + rt[:, None]) * HV + ih) * V + rv[None, :]).to(tl.int64)
        pgi = ((ib * T + rt) * HV + ih).to(tl.int64)
        psi = (((((ib * HV + ih) * NC + ic) * K + rk[:, None]) * V) + rv[None, :]).to(tl.int64)
        kk = tl.load(k + pki, mask=mk, other=0.).to(tl.float32)
        qq = tl.load(q + pki, mask=mk, other=0.).to(tl.float32) * scale
        vv = tl.load(v + pvi, mask=mv, other=0.).to(tl.float32)
        dd = tl.load(do + pvi, mask=mv, other=0.).to(tl.float32)
        gg = tl.load(g + pgi, mask=mt, other=0.).to(tl.float32)
        bb = tl.load(beta + pgi, mask=mt, other=0.).to(tl.float32)
        st = tl.load(states + psi, mask=(rk[:, None] < K) & (rv[None, :] < V), other=0.)
        cg = tl.maximum(tl.cumsum(gg, axis=0), -60.0)
        ee = tl.exp(cg)
        yy, xx, qb = kk * ee[:, None], kk / ee[:, None], qq * ee[:, None]
        res = vv - tl.dot(yy, st, input_precision='ieee')
        pi = (((((ib * HV + ih) * NC + ic) * BT + rr) * BT) + cc).to(tl.int64)
        inv = tl.load(inverse + pi)
        uu = tl.load(updates + pvi, mask=mv, other=0.)
        gram = tl.dot(yy, tl.trans(xx), input_precision='ieee')
        mm = tl.where(rr >= cc, tl.dot(qb, tl.trans(xx), input_precision='ieee'), 0.)
        ef = tl.sum(tl.where(tl.arange(0, BT) == BT - 1, ee, 0.), axis=0)
        ds = ef * dst + tl.dot(tl.trans(qb), dd, input_precision='ieee')
        dx = ef * tl.dot(uu, tl.trans(dst), input_precision='ieee')
        du = ef * tl.dot(xx, dst, input_precision='ieee') + tl.dot(tl.trans(mm), dd, input_precision='ieee')
        dm = tl.where(rr >= cc, tl.dot(dd, tl.trans(uu), input_precision='ieee'), 0.)
        dr = tl.dot(tl.trans(inv), du, input_precision='ieee')
        dl = tl.where(rr > cc, -tl.dot(dr, tl.trans(uu), input_precision='ieee'), 0.)
        base = bb[:, None] * dr
        db = tl.sum(dr * res, axis=1) + tl.sum(dl * gram, axis=1)
        dy = -tl.dot(base, tl.trans(st), input_precision='ieee')
        ds += -tl.dot(tl.trans(yy), base, input_precision='ieee')
        da = dl * bb[:, None]
        dy += tl.dot(da, xx, input_precision='ieee')
        dx += tl.dot(tl.trans(da), yy, input_precision='ieee')
        dqb = tl.dot(dm, xx, input_precision='ieee') + tl.dot(dd, tl.trans(st), input_precision='ieee')
        dx += tl.dot(tl.trans(dm), qb, input_precision='ieee')
        dqi = dqb * ee[:, None] * scale
        dki = dy * ee[:, None] + dx / ee[:, None]
        dc = tl.sum(dqb * qb + dy * yy - dx * xx, axis=1)
        fin = st + tl.dot(tl.trans(xx), uu, input_precision='ieee')
        dc += tl.where(tl.arange(0, BT) == BT - 1, ef * tl.sum(dst * fin), 0.)
        dgi = tl.cumsum(dc, axis=0, reverse=True)
        tl.store(dvout + pvi, base, mask=mv)
        pp = (((((ib * HV + ih) * NV + iv) * T + rt[:, None]) * K) + rk[None, :]).to(tl.int64)
        tl.store(pq + pp, dqi, mask=mk)
        tl.store(pkout + pp, dki, mask=mk)
        ps = ((((ib * HV + ih) * NV + iv) * T) + rt).to(tl.int64)
        tl.store(pgout + ps, dgi, mask=mt)
        tl.store(pbout + ps, db, mask=mt)
        dst = ds


@triton.jit
def _reduce_qk(part, out, T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    it, z = pid % T, pid // T
    ihq, ib = z % HQ, z // HQ
    rk = tl.arange(0, BK)
    acc = tl.zeros((BK,), tl.float32)
    G: tl.constexpr = HV // HQ
    for ig in range(G):
        ih = ihq * G + ig
        for iv in range(NV):
            p = (((((ib * HV + ih) * NV + iv) * T + it) * K) + rk).to(tl.int64)
            acc += tl.load(part + p, mask=rk < K, other=0.)
    po = ((((ib * T + it) * HQ + ihq) * K) + rk).to(tl.int64)
    tl.store(out + po, acc, mask=rk < K)


@triton.jit
def _reduce_s(part, out, N: tl.constexpr, T: tl.constexpr, HV: tl.constexpr,
              NV: tl.constexpr, BS: tl.constexpr):
    o = tl.program_id(0).to(tl.int64) * BS + tl.arange(0, BS)
    m = o < N
    ib, rem = o // (T * HV), o % (T * HV)
    it, ih = rem // HV, rem % HV
    acc = tl.zeros((BS,), tl.float32)
    for iv in range(NV):
        p = ((((ib * HV + ih) * NV + iv) * T) + it).to(tl.int64)
        acc += tl.load(part + p, mask=m, other=0.)
    tl.store(out + o, acc, mask=m)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, HQ, K = q.shape
        HV, V = v.shape[2:]
        if HV % HQ:
            raise ValueError('value heads must be a multiple of key heads')
        BK = triton.next_power_of_2(K)
        NC, NV = triton.cdiv(T, CHUNK), triton.cdiv(V, VALUE_BLOCK)
        out = torch.empty_like(v)
        states = torch.empty((B, HV, NC, K, V), device=q.device, dtype=torch.float32)
        updates = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        inverse = torch.empty((B, HV, NC, CHUNK, CHUNK), device=q.device, dtype=torch.float32)
        _prepare_chunks[(B * HV * NC,)](k, g, beta, inverse, T=T, HQ=HQ, HV=HV,
            K=K, NC=NC, BK=BK, BT=CHUNK, num_warps=4)
        _state_fwd[(B * HV * NV,)](k, v, g, beta, inverse, states, updates,
            T=T, HQ=HQ, HV=HV, K=K, V=V, NC=NC, BK=BK,
            BV=VALUE_BLOCK, BT=CHUNK, num_warps=8)
        _chunk_output[(B * HV * NC * NV,)](q, k, g, states, updates, out,
            T=T, HQ=HQ, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK,
            BV=VALUE_BLOCK, BT=CHUNK, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, inverse, states, updates)
        ctx.dims = B, T, HQ, HV, K, V, BK, NC, NV
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, inverse, states, updates = ctx.saved_tensors
        B, T, HQ, HV, K, V, BK, NC, NV = ctx.dims
        do = do.contiguous()
        dv = torch.empty_like(v)
        pq = torch.empty((B, HV, NV, T, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq, dtype=torch.float32)
        pg = torch.empty((B, HV, NV, T), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg, dtype=torch.float32)
        _chunk_bwd[(B * HV * NV,)](q, k, v, g, beta, do, inverse, states, updates, dv, pq, pk, pg, pb,
            T=T, HQ=HQ, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK,
            BV=VALUE_BLOCK, BT=CHUNK, num_warps=8)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        grid = (B * HQ * T,)
        _reduce_qk[grid](pq, dq, T=T, HQ=HQ, HV=HV, K=K, NV=NV, BK=BK)
        _reduce_qk[grid](pk, dk, T=T, HQ=HQ, HV=HV, K=K, NV=NV, BK=BK)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        n, bs = B*T*HV, 256
        _reduce_s[(triton.cdiv(n, bs),)](pg, dg, N=n, T=T, HV=HV, NV=NV, BS=bs)
        _reduce_s[(triton.cdiv(n, bs),)](pb, db, N=n, T=T, HV=HV, NV=NV, BS=bs)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
