import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 32


@triton.jit
def _chunk_fwd(q, k, v, g, beta, out, states, T: tl.constexpr, HQ: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
               BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, z = pid % nv, pid // nv
    ih, ib = z % HV, z // HV
    iqh = ih // (HV // HQ)
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
        pk = (((ib * T + rt[:, None]) * HQ + iqh) * K + rk[None, :]).to(tl.int64)
        pv = (((ib * T + rt[:, None]) * HV + ih) * V + rv[None, :]).to(tl.int64)
        pg = ((ib * T + rt) * HV + ih).to(tl.int64)
        kk = tl.load(k + pk, mask=mk, other=0.).to(tl.float32)
        qq = tl.load(q + pk, mask=mk, other=0.).to(tl.float32) * (K ** -0.5)
        vv = tl.load(v + pv, mask=mv, other=0.).to(tl.float32)
        gg = tl.load(g + pg, mask=mt, other=0.).to(tl.float32)
        bb = tl.load(beta + pg, mask=mt, other=0.).to(tl.float32)
        cg = tl.cumsum(gg, axis=0)
        ee = tl.exp(cg)
        yy, xx, qb = kk * ee[:, None], kk / ee[:, None], qq * ee[:, None]
        uu = bb[:, None] * (vv - tl.dot(yy, st, input_precision='ieee'))
        gram = tl.dot(yy, tl.trans(xx), input_precision='ieee')
        aa = tl.where(rr > cc, gram * bb[:, None], 0.)
        for i in range(BT):
            old = tl.sum(tl.where(rr == i, uu, 0.), axis=0)
            arow = tl.sum(tl.where(rr == i, aa, 0.), axis=0)
            ui = old - tl.sum(arow[:, None] * uu, axis=0)
            uu = tl.where(rr == i, ui[None, :], uu)
        mm = tl.where(rr >= cc, tl.dot(qb, tl.trans(xx), input_precision='ieee'), 0.)
        oo = tl.dot(qb, st, input_precision='ieee') + tl.dot(mm, uu, input_precision='ieee')
        tl.store(out + pv, oo, mask=mv)
        ef = tl.sum(tl.where(tl.arange(0, BT) == BT - 1, ee, 0.), axis=0)
        st = ef * (st + tl.dot(tl.trans(xx), uu, input_precision='ieee'))


@triton.jit
def _chunk_bwd(q, k, v, g, beta, do, states, dvout, pq, pkout, pgout, pbout,
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
        cg, ee = tl.cumsum(gg, axis=0), tl.exp(tl.cumsum(gg, axis=0))
        yy, xx, qb = kk * ee[:, None], kk / ee[:, None], qq * ee[:, None]
        res = vv - tl.dot(yy, st, input_precision='ieee')
        uu = bb[:, None] * res
        gram = tl.dot(yy, tl.trans(xx), input_precision='ieee')
        aa = tl.where(rr > cc, gram * bb[:, None], 0.)
        for i in range(BT):
            old = tl.sum(tl.where(rr == i, uu, 0.), axis=0)
            arow = tl.sum(tl.where(rr == i, aa, 0.), axis=0)
            ui = old - tl.sum(arow[:, None] * uu, axis=0)
            uu = tl.where(rr == i, ui[None, :], uu)
        mm = tl.where(rr >= cc, tl.dot(qb, tl.trans(xx), input_precision='ieee'), 0.)
        ef = tl.sum(tl.where(tl.arange(0, BT) == BT - 1, ee, 0.), axis=0)
        ds = ef * dst + tl.dot(tl.trans(qb), dd, input_precision='ieee')
        dx = ef * tl.dot(uu, tl.trans(dst), input_precision='ieee')
        du = ef * tl.dot(xx, dst, input_precision='ieee') + tl.dot(tl.trans(mm), dd, input_precision='ieee')
        dm = tl.where(rr >= cc, tl.dot(dd, tl.trans(uu), input_precision='ieee'), 0.)
        dr = du
        for j in range(BT):
            i = BT - 1 - j
            old = tl.sum(tl.where(rr == i, du, 0.), axis=0)
            acol = tl.sum(tl.where(cc == i, aa, 0.), axis=1)
            ui = old - tl.sum(acol[:, None] * dr, axis=0)
            dr = tl.where(rr == i, ui[None, :], dr)
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
def _reduce_qk(part, out, N: tl.constexpr, T: tl.constexpr, HQ: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    it, z = pid % T, pid // T
    ihq, ib = z % HQ, z // HQ
    rk = tl.arange(0, BK)
    acc = tl.zeros((BK,), tl.float32)
    G = HV // HQ
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
        BK, NC, NV = triton.next_power_of_2(K), triton.cdiv(T, CHUNK), triton.cdiv(V, VALUE_BLOCK)
        out = torch.empty_like(v)
        states = torch.empty((B, HV, NC, K, V), device=q.device, dtype=torch.float32)
        _chunk_fwd[(B * HV * NV,)](q, k, v, g, beta, out, states, T=T, HQ=HQ, HV=HV,
            K=K, V=V, NC=NC, BK=BK, BV=VALUE_BLOCK, BT=CHUNK, num_warps=8)
        ctx.save_for_backward(q, k, v, g, beta, states)
        ctx.dims = B, T, HQ, HV, K, V, BK, NC, NV
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, states = ctx.saved_tensors
        B, T, HQ, HV, K, V, BK, NC, NV = ctx.dims
        do = do.contiguous()
        dv = torch.empty_like(v)
        pq = torch.empty((B, HV, NV, T, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq, dtype=torch.float32)
        pg = torch.empty((B, HV, NV, T), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg, dtype=torch.float32)
        _chunk_bwd[(B * HV * NV,)](q, k, v, g, beta, do, states, dv, pq, pk, pg, pb,
            T=T, HQ=HQ, HV=HV, K=K, V=V, NC=NC, NV=NV, BK=BK,
            BV=VALUE_BLOCK, BT=CHUNK, num_warps=8)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        grid = (B * HQ * T,)
        _reduce_qk[grid](pq, dq, N=B*HQ*T, T=T, HQ=HQ, HV=HV, K=K, NV=NV, BK=BK)
        _reduce_qk[grid](pk, dk, N=B*HQ*T, T=T, HQ=HQ, HV=HV, K=K, NV=NV, BK=BK)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        n, bs = B*T*HV, 256
        _reduce_s[(triton.cdiv(n, bs),)](pg, dg, N=n, T=T, HV=HV, NV=NV, BS=bs)
        _reduce_s[(triton.cdiv(n, bs),)](pb, db, N=n, T=T, HV=HV, NV=NV, BS=bs)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
