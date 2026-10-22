import torch
import triton
import triton.language as tl

BT = 16
BV = 16
BWD_BV = 16

@triton.jit
def _checkpoint_kernel(k, v, a, b, g, checkpoints, T, H: tl.constexpr,
                       K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                       BK: tl.constexpr, BV_: tl.constexpr, BT_: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV_), tl.float32)
    base_k, base_v = (ib * T * H + ih) * K, (ib * T * H + ih) * V
    for ic in range(NC):
        cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(checkpoints + cp, state, mask=mk[:, None] & mv[None, :])
        for j in range(BT_):
            t = ic * BT_ + j
            pk, pv = base_k + t * H * K + ok, base_v + t * H * V + ov
            aa = tl.load(a + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
            bb = tl.load(b + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
            kk = tl.load(k + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
            gg = tl.load(g + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
            vv = tl.load(v + pv, mask=(t < T) & mv, other=0.).to(tl.float32)
            read = tl.sum(tl.dot(aa[None, :], state, input_precision="ieee"), axis=0)
            nxt = tl.exp(gg)[:, None] * state + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
            state = tl.where(t < T, nxt, state)

@triton.jit
def _chunk_read_kernel(k, v, a, b, g, checkpoints, reads, T, H: tl.constexpr,
                       K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                       BK: tl.constexpr, BV_: tl.constexpr, BT_: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    ok, ov, ot = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_), tl.arange(0, BT_)
    t, mk, mv = ic * BT_ + ot, ok < K, ov < V
    mt = t < T
    pk = (ib * T * H + ih) * K + t[:, None] * H * K + ok[None, :]
    pv = (ib * T * H + ih) * V + t[:, None] * H * V + ov[None, :]
    aa = tl.load(a + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    bb = tl.load(b + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    kk = tl.load(k + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    gg = tl.load(g + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    vv = tl.load(v + pv, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
    cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
    state = tl.load(checkpoints + cp, mask=mk[:, None] & mv[None, :], other=0.)
    G = tl.cumsum(gg, axis=0)
    Gp = G - gg
    first = tl.sum(tl.where((ot == 0)[:, None], Gp, 0.), axis=0)
    last = tl.sum(tl.where((ot == BT_ - 1)[:, None], G, 0.), axis=0)
    center = 0.5 * (first + last)
    ag = aa * tl.exp(tl.maximum(-40., tl.minimum(40., Gp - center[None, :])))
    bg = bb * tl.exp(tl.maximum(-40., tl.minimum(40., center[None, :] - G)))
    kg = kk * tl.exp(tl.maximum(-40., tl.minimum(40., center[None, :] - G)))
    causal = ot[:, None] > ot[None, :]
    lab = tl.where(causal, tl.dot(ag, tl.trans(bg), input_precision="ieee"), 0.)
    lak = tl.where(causal, tl.dot(ag, tl.trans(kg), input_precision="ieee"), 0.)
    rr = tl.dot(aa * tl.exp(Gp), state, input_precision="ieee")
    rr += tl.dot(lak, vv, input_precision="ieee")
    for j in range(1, BT_):
        row, prior = ot == j, ot < j
        x = tl.sum(tl.where(row[:, None], rr, 0.), axis=0)
        coeff = tl.sum(tl.where(row[:, None], lab, 0.), axis=0)
        x += tl.sum(tl.where(prior[:, None], coeff[:, None] * rr, 0.), axis=0)
        rr = tl.where(row[:, None], x[None, :], rr)
    tl.store(reads + pv, rr, mask=mt[:, None] & mv[None, :])

@triton.jit
def _chunk_output_kernel(q, k, v, b, g, checkpoints, reads, out,
                         T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                         NC: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr,
                         SCALE: tl.constexpr, BT_: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    ot, ok, ov = tl.arange(0, BT_), tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_)
    t = ic * BT_ + ot
    mt, mk, mv = t < T, ok < K, ov < V
    pk = (ib * T * H + ih) * K + t[:, None] * H * K + ok[None, :]
    pv = (ib * T * H + ih) * V + t[:, None] * H * V + ov[None, :]
    qq = tl.load(q + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32) * SCALE
    kk = tl.load(k + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    bb = tl.load(b + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    gg = tl.load(g + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    vv = tl.load(v + pv, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
    rr = tl.load(reads + pv, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
    cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
    ss = tl.load(checkpoints + cp, mask=mk[:, None] & mv[None, :], other=0.)
    G = tl.cumsum(gg, axis=0)
    y = tl.dot(qq * tl.exp(G), ss, input_precision="ieee")
    first = tl.sum(tl.where((ot == 0)[:, None], G, 0.), axis=0)
    last = tl.sum(tl.where((ot == BT_ - 1)[:, None], G, 0.), axis=0)
    center = 0.5 * (first + last)
    qg = qq * tl.exp(tl.maximum(-40., tl.minimum(40., G - center[None, :])))
    kg = kk * tl.exp(tl.maximum(-40., tl.minimum(40., center[None, :] - G)))
    bg = bb * tl.exp(tl.maximum(-40., tl.minimum(40., center[None, :] - G)))
    causal = ot[:, None] >= ot[None, :]
    aqk = tl.dot(qg, tl.trans(kg), input_precision="ieee")
    aqb = tl.dot(qg, tl.trans(bg), input_precision="ieee")
    aqk = tl.where(causal & mt[:, None] & mt[None, :], aqk, 0.)
    aqb = tl.where(causal & mt[:, None] & mt[None, :], aqb, 0.)
    y += tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, rr, input_precision="ieee")
    tl.store(out + pv, y, mask=mt[:, None] & mv[None, :])

@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NC: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr,
                          BT_: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    ok, ov, ot = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_), tl.arange(0, BT_)
    mk, mv = ok < K, ov < V
    cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
    state = tl.load(checkpoints + cp, mask=mk[:, None] & mv[None, :], other=0.)
    base_k, base_v = (ib * T * H + ih) * K, (ib * T * H + ih) * V
    for j in range(BT_):
        t = ic * BT_ + j
        pk, pv = base_k + t * H * K + ok, base_v + t * H * V + ov
        aa = tl.load(a + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
        bb = tl.load(b + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
        kk = tl.load(k + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
        gg = tl.load(g + pk, mask=(t < T) & mk, other=0.).to(tl.float32)
        vv = tl.load(v + pv, mask=(t < T) & mv, other=0.).to(tl.float32)
        read = tl.sum(tl.dot(aa[None, :], state, input_precision="ieee"), axis=0)
        state = tl.exp(gg)[:, None] * state + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        ps = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(states + ps, state, mask=(t < T) & mk[:, None] & mv[None, :])

@triton.jit
def _reverse_kernel(q, k, v, a, b, g, dout, states, dv, partial,
                    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BV_), tl.float32)
    base_k, base_v = (ib * T * H + ih) * K, (ib * T * H + ih) * V
    for rev in range(T):
        t = T - 1 - rev
        pk, pv = base_k + t * H * K + ok, base_v + t * H * V + ov
        qq = tl.load(q + pk, mask=mk, other=0.).to(tl.float32) * SCALE
        kk = tl.load(k + pk, mask=mk, other=0.).to(tl.float32)
        aa = tl.load(a + pk, mask=mk, other=0.).to(tl.float32)
        bb = tl.load(b + pk, mask=mk, other=0.).to(tl.float32)
        gg = tl.load(g + pk, mask=mk, other=0.).to(tl.float32)
        vv = tl.load(v + pv, mask=mv, other=0.).to(tl.float32)
        do = tl.load(dout + pv, mask=mv, other=0.).to(tl.float32)
        ps = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        pp = ((((ib * T + t - 1) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        st = tl.load(states + ps, mask=mk[:, None] & mv[None, :], other=0.)
        prev = tl.load(states + pp, mask=(t > 0) & mk[:, None] & mv[None, :], other=0.)
        ds += qq[:, None] * do[None, :]
        dqv = tl.sum(tl.dot(st, do[:, None], input_precision="ieee"), axis=1) * SCALE
        dkv = tl.sum(tl.dot(ds, vv[:, None], input_precision="ieee"), axis=1)
        dvv = tl.sum(tl.dot(kk[None, :], ds, input_precision="ieee"), axis=0)
        read = tl.sum(tl.dot(aa[None, :], prev, input_precision="ieee"), axis=0)
        dbv = tl.sum(tl.dot(ds, read[:, None], input_precision="ieee"), axis=1)
        z = tl.sum(tl.dot(bb[None, :], ds, input_precision="ieee"), axis=0)
        dav = tl.sum(tl.dot(prev, z[:, None], input_precision="ieee"), axis=1)
        dgv = tl.sum(ds * (tl.exp(gg)[:, None] * prev), axis=1)
        tl.store(dv + pv, dvv, mask=mv)
        root = iv * tl.num_programs(1) * T * 5 * K + ((ib * T + t) * H + ih) * 5 * K + ok
        tl.store(partial + root, dqv, mask=mk)
        tl.store(partial + root + K, dkv, mask=mk)
        tl.store(partial + root + 2 * K, dav, mask=mk)
        tl.store(partial + root + 3 * K, dbv, mask=mk)
        tl.store(partial + root + 4 * K, dgv, mask=mk)
        ds = tl.exp(gg)[:, None] * ds + aa[:, None] * z[None, :]

@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr,
                   NV: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m, kk, token = x < N, x % K, x // K
    s0 = tl.zeros((BLOCK,), tl.float32)
    s1 = tl.zeros((BLOCK,), tl.float32)
    s2 = tl.zeros((BLOCK,), tl.float32)
    s3 = tl.zeros((BLOCK,), tl.float32)
    s4 = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        root = iv * N * 5 + token * 5 * K + kk
        s0 += tl.load(partial + root, mask=m, other=0.)
        s1 += tl.load(partial + root + K, mask=m, other=0.)
        s2 += tl.load(partial + root + 2 * K, mask=m, other=0.)
        s3 += tl.load(partial + root + 3 * K, mask=m, other=0.)
        s4 += tl.load(partial + root + 4 * K, mask=m, other=0.)
    tl.store(dq + x, s0, mask=m)
    tl.store(dk + x, s1, mask=m)
    tl.store(da + x, s2, mask=m)
    tl.store(db + x, s3, mask=m)
    tl.store(dg + x, s4, mask=m)

class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V, nc, nv = v.shape[-1], triton.cdiv(T, BT), triton.cdiv(v.shape[-1], BV)
        bk = max(16, triton.next_power_of_2(K))
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        reads = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _checkpoint_kernel[(nv, B * H)](k, v, alpha, beta, gk, checkpoints,
            T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV, BT_=BT, num_warps=4, num_stages=2)
        _chunk_read_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, reads,
            T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV, BT_=BT, num_warps=4, num_stages=2)
        _chunk_output_kernel[(nv, nc, B * H)](q, k, v, beta, gk, checkpoints, reads, out,
            T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV, SCALE=K ** -0.5, BT_=BT,
            num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V, nv, bk = v.shape[-1], triton.cdiv(v.shape[-1], BWD_BV), max(16, triton.next_power_of_2(K))
        nc = triton.cdiv(T, BT)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](
            k, v, alpha, beta, gk, checkpoints, states, T, H=H, K=K, V=V,
            NC=nc, BK=bk, BV_=BWD_BV, BT_=BT, num_warps=4, num_stages=2)
        dv = torch.empty_like(v)
        partial = torch.zeros((nv, B * H, T, 5, K), device=q.device, dtype=torch.float32)
        _reverse_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, dout, states, dv, partial,
            T, H=H, K=K, V=V, BK=bk, BV_=BWD_BV, SCALE=K ** -0.5,
            num_warps=4, num_stages=2)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg,
            N=n, NV=nv, K=K, BLOCK=256, num_warps=8)
        return dq, dk, dv, da, db, dg

def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
