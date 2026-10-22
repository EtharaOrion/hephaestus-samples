import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 16


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(80.0, x)))


@triton.jit
def _wy_build(k, g, beta, a, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
              C: tl.constexpr, BK: tl.constexpr):
    ic = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // HV
    ihv = ibh - ib * HV
    ih = ihv // (HV // H)
    r = tl.arange(0, C)
    x = tl.arange(0, BK)
    t = ic * C + r
    mt = t < T
    kk = tl.load(k + ((ib * T + t[:, None]) * H + ih) * K + x[None, :], mask=mt[:, None] & (x[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    h = tl.cumsum(gg, axis=0)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    lower = r[:, None] > r[None, :]
    mat = tl.where(lower & mt[:, None] & mt[None, :], gram * _bounded_exp(h[:, None] - h[None, :]) * bb[:, None], 0.0)
    inv = -mat
    for i in range(1, C):
        row = tl.sum(tl.where((r == i)[:, None], -mat, 0.0), axis=0)
        row = tl.where(r < i, row, 0.0)
        row += tl.sum(row[:, None] * inv, axis=0)
        inv = tl.where((r == i)[:, None], row, inv)
    inv += tl.where(r[:, None] == r[None, :], 1.0, 0.0)
    base = ((ib * NC + ic) * HV + ihv) * C * C
    tl.store(a + base + r[:, None] * C + r[None, :], inv, mask=mt[:, None] & mt[None, :])


@triton.jit
def _state_scan(k, v, g, beta, a, states, updates, T: tl.constexpr, H: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                NV: tl.constexpr, C: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    p = tl.program_id(0)
    ibhv = p // NV
    iv = p - ibhv * NV
    ib = ibhv // HV
    ihv = ibhv - ib * HV
    ih = ihv // (HV // H)
    x = tl.arange(0, BK)
    z = iv * BV + tl.arange(0, BV)
    r = tl.arange(0, C)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        t = ic * C + r
        mt = t < T
        sm = ((((ib * NC + ic) * HV + ihv) * K + x[:, None]) * V + z[None, :]).to(tl.int64)
        tl.store(states + sm, state, mask=(x[:, None] < K) & (z[None, :] < V))
        kk = tl.load(k + ((ib * T + t[:, None]) * H + ih) * K + x[None, :], mask=mt[:, None] & (x[None, :] < K), other=0.0)
        vv = tl.load(v + ((ib * T + t[:, None]) * HV + ihv) * V + z[None, :], mask=mt[:, None] & (z[None, :] < V), other=0.0).to(tl.float32)
        gg = tl.load(g + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
        bb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
        hh = tl.cumsum(gg, axis=0)
        aa = tl.load(a + ((ib * NC + ic) * HV + ihv) * C * C + r[:, None] * C + r[None, :], mask=mt[:, None] & mt[None, :], other=0.0)
        initial = tl.dot(kk * (bb * _bounded_exp(hh))[:, None], state, input_precision="tf32")
        delta = tl.dot(aa, bb[:, None] * vv - initial, input_precision="tf32")
        um = (((ib * T + t[:, None]) * HV + ihv) * V + z[None, :]).to(tl.int64)
        tl.store(updates + um, delta, mask=mt[:, None] & (z[None, :] < V))
        total = tl.sum(gg, axis=0)
        state = state * _bounded_exp(total) + tl.dot(tl.trans(kk * _bounded_exp(total - hh)[:, None]), delta, input_precision="tf32")


@triton.jit
def _output_chunks(q, k, g, states, updates, out, T: tl.constexpr, H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                   NV: tl.constexpr, C: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                   SCALE: tl.constexpr):
    p = tl.program_id(0)
    iv = p % NV
    p //= NV
    ic = p % NC
    ibhv = p // NC
    ib = ibhv // HV
    ihv = ibhv - ib * HV
    ih = ihv // (HV // H)
    r = tl.arange(0, C)
    x = tl.arange(0, BK)
    z = iv * BV + tl.arange(0, BV)
    t = ic * C + r
    mt = t < T
    qq = tl.load(q + ((ib * T + t[:, None]) * H + ih) * K + x[None, :], mask=mt[:, None] & (x[None, :] < K), other=0.0)
    kk = tl.load(k + ((ib * T + t[:, None]) * H + ih) * K + x[None, :], mask=mt[:, None] & (x[None, :] < K), other=0.0)
    gg = tl.load(g + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
    hh = tl.cumsum(gg, axis=0)
    sm = ((((ib * NC + ic) * HV + ihv) * K + x[:, None]) * V + z[None, :]).to(tl.int64)
    state = tl.load(states + sm, mask=(x[:, None] < K) & (z[None, :] < V), other=0.0)
    um = (((ib * T + t[:, None]) * HV + ihv) * V + z[None, :]).to(tl.int64)
    delta = tl.load(updates + um, mask=mt[:, None] & (z[None, :] < V), other=0.0)
    cross = tl.dot(qq * (SCALE * _bounded_exp(hh))[:, None], state, input_precision="tf32")
    scores = tl.dot(qq, tl.trans(kk), input_precision="tf32") * SCALE
    scores *= tl.where((r[:, None] >= r[None, :]) & mt[:, None] & mt[None, :], _bounded_exp(hh[:, None] - hh[None, :]), 0.0)
    result = cross + tl.dot(scores, delta, input_precision="tf32")
    om = (((ib * T + t[:, None]) * HV + ihv) * V + z[None, :]).to(tl.int64)
    tl.store(out + om, result, mask=mt[:, None] & (z[None, :] < V))


@triton.jit
def _reverse_chunks(q, k, v, g, beta, dout, states, updates, pq, pk, pv, pg, pb,
                    T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr, C: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0)
    ibhv = p // NV
    iv = p - ibhv * NV
    ib = ibhv // HV
    ihv = ibhv - ib * HV
    ih = ihv // (HV // H)
    x = tl.arange(0, BK)
    z = iv * BV + tl.arange(0, BV)
    r = tl.arange(0, C)
    adj = tl.zeros((BK, BV), tl.float32)
    for ci in range(NC):
        ic = NC - 1 - ci
        t = ic * C + r
        mt = t < T
        kk = tl.load(k + ((ib * T + t[:, None]) * H + ih) * K + x[None, :], mask=mt[:, None] & (x[None, :] < K), other=0.0)
        qq = tl.load(q + ((ib * T + t[:, None]) * H + ih) * K + x[None, :], mask=mt[:, None] & (x[None, :] < K), other=0.0)
        gg = tl.load(g + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
        bb = tl.load(beta + (ib * T + t) * HV + ihv, mask=mt, other=0.0).to(tl.float32)
        vv = tl.load(v + ((ib * T + t[:, None]) * HV + ihv) * V + z[None, :], mask=mt[:, None] & (z[None, :] < V), other=0.0).to(tl.float32)
        dd = tl.load(dout + ((ib * T + t[:, None]) * HV + ihv) * V + z[None, :], mask=mt[:, None] & (z[None, :] < V), other=0.0).to(tl.float32)
        um = (((ib * T + t[:, None]) * HV + ihv) * V + z[None, :]).to(tl.int64)
        uu = tl.load(updates + um, mask=mt[:, None] & (z[None, :] < V), other=0.0)
        sm = ((((ib * NC + ic) * HV + ihv) * K + x[:, None]) * V + z[None, :]).to(tl.int64)
        start = tl.load(states + sm, mask=(x[:, None] < K) & (z[None, :] < V), other=0.0)
        hh = tl.cumsum(gg, axis=0)
        last = tl.minimum(C - 1, T - ic * C - 1)
        h_last = tl.sum(tl.where(r == last, hh, 0.0), axis=0)
        coeff = tl.where((r < last) & mt, _bounded_exp(h_last - hh), 0.0)
        sp = start * _bounded_exp(h_last) + tl.dot(tl.trans(kk * coeff[:, None]), uu, input_precision="tf32")
        for jj in range(C):
            i = C - 1 - jj
            valid = (ic * C + i) < T
            ki = tl.sum(tl.where((r == i)[:, None], kk, 0.0), axis=0)
            qi = tl.sum(tl.where((r == i)[:, None], qq, 0.0), axis=0) * SCALE
            ui = tl.sum(tl.where((r == i)[:, None], uu, 0.0), axis=0)
            vi = tl.sum(tl.where((r == i)[:, None], vv, 0.0), axis=0)
            doi = tl.sum(tl.where((r == i)[:, None], dd, 0.0), axis=0)
            sp -= tl.where(i < last, ki[:, None] * ui[None, :], 0.0)
            if i % 4 == 3:
                h_i = tl.sum(tl.where(r == i, hh, 0.0), axis=0)
                local = tl.where((r < i) & mt, _bounded_exp(h_i - hh), 0.0)
                exact = start * _bounded_exp(h_i) + tl.dot(tl.trans(kk * local[:, None]), uu, input_precision="tf32")
                sp = tl.where(i < last, exact, sp)
            state_i = sp + ki[:, None] * ui[None, :]
            dq_i = tl.sum(state_i * doi[None, :], axis=1) * SCALE
            adj += qi[:, None] * doi[None, :]
            ddelta = tl.sum(adj * ki[:, None], axis=0)
            pred = tl.sum(sp * ki[:, None], axis=0)
            b_i = tl.sum(tl.where(r == i, bb, 0.0), axis=0)
            dv_i = b_i * ddelta
            dp = -b_i * ddelta
            dk_i = tl.sum(adj * ui[None, :] + sp * dp[None, :], axis=1)
            adj_sp = adj + ki[:, None] * dp[None, :]
            dg_i = tl.sum(adj_sp * sp)
            ti = ic * C + i
            baseq = (((((ib * T + ti) * HV + ihv) * NV + iv) * K) + x).to(tl.int64)
            bases = ((((ib * T + ti) * HV + ihv) * NV + iv).to(tl.int64))
            vm = ((((ib * T + ti) * HV + ihv) * V + z).to(tl.int64))
            tl.store(pq + baseq, dq_i, mask=valid & (x < K))
            tl.store(pk + baseq, dk_i, mask=valid & (x < K))
            tl.store(pv + vm, dv_i, mask=valid & (z < V))
            tl.store(pg + bases, dg_i, mask=valid)
            tl.store(pb + bases, tl.sum(ddelta * (vi - pred)), mask=valid)
            g_i = tl.sum(tl.where(r == i, gg, 0.0), axis=0)
            adj = adj_sp * _bounded_exp(g_i)
            inverse_decay = tl.exp(tl.maximum(-80.0, tl.minimum(2.0, -g_i)))
            sp = tl.where(valid, sp * inverse_decay, sp)


@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr):
    p = tl.program_id(0)
    ih = p % H
    bt = p // H
    ib = bt // T
    it = bt - ib * T
    x = tl.arange(0, BK)
    sq = tl.zeros((BK,), tl.float32)
    sk = tl.zeros((BK,), tl.float32)
    for ir in range(HV // H):
        ihv = ih * (HV // H) + ir
        for iv in range(NV):
            off = (((((ib * T + it) * HV + ihv) * NV + iv) * K) + x).to(tl.int64)
            sq += tl.load(pq + off, mask=x < K, other=0.0)
            sk += tl.load(pk + off, mask=x < K, other=0.0)
    dst = ((ib * T + it) * H + ih) * K + x
    tl.store(dq + dst, sq, mask=x < K)
    tl.store(dk + dst, sk, mask=x < K)


@triton.jit
def _reduce_scalars(pg, pb, dg, db, T: tl.constexpr, HV: tl.constexpr,
                    NV: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0)
    ihv = p % HV
    bt = p // HV
    ib = bt // T
    it = bt - ib * T
    n = tl.arange(0, BN)
    off = ((((ib * T + it) * HV + ihv) * NV + n).to(tl.int64))
    mask = n < NV
    dst = (ib * T + it) * HV + ihv
    tl.store(dg + dst, tl.sum(tl.load(pg + off, mask=mask, other=0.0)))
    tl.store(db + dst, tl.sum(tl.load(pb + off, mask=mask, other=0.0)))


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        if HV % H:
            raise ValueError("value heads must be divisible by key heads")
        nc, nv = triton.cdiv(T, CHUNK), triton.cdiv(V, VALUE_BLOCK)
        bk = triton.next_power_of_2(K)
        a = torch.empty((B, nc, HV, CHUNK, CHUNK), device=q.device, dtype=torch.float32)
        states = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        updates = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _wy_build[(nc, B * HV)](k, g, beta, a, T=T, H=H, HV=HV, K=K, NC=nc, C=CHUNK, BK=bk, num_warps=4)
        _state_scan[(B * HV * nv,)](k, v, g, beta, a, states, updates, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, C=CHUNK, BK=bk, BV=VALUE_BLOCK, num_warps=8)
        _output_chunks[(B * HV * nc * nv,)](q, k, g, states, updates, out, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, C=CHUNK, BK=bk, BV=VALUE_BLOCK, SCALE=K ** -0.5, num_warps=8)
        ctx.save_for_backward(q, k, v, g, beta, states, updates)
        ctx.dims = (B, T, H, HV, K, V, nc, nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, states, updates = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, HV, K, V, nc, nv, bk = ctx.dims
        pq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        pg = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        _reverse_chunks[(B * HV * nv,)](q, k, v, g, beta, dout, states, updates, pq, pk, dv, pg, pb, T=T, H=H, HV=HV, K=K, V=V, NC=nc, NV=nv, C=CHUNK, BK=bk, BV=VALUE_BLOCK, SCALE=K ** -0.5, num_warps=8)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T=T, H=H, HV=HV, K=K, NV=nv, BK=bk, num_warps=4)
        _reduce_scalars[(B * T * HV,)](pg, pb, dg, db, T=T, HV=HV, NV=nv, BN=triton.next_power_of_2(nv), num_warps=4)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
