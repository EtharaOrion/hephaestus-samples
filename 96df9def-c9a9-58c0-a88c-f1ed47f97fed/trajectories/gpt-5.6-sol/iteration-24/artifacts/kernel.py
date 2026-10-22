import torch
import triton
import triton.language as tl

_BT = 16
_BV = 16


@triton.jit
def _inverse_lower(l):
    eye = tl.arange(0, 16)[:, None] == tl.arange(0, 16)[None, :]
    p = l
    inv = eye.to(tl.float32) + p
    p = tl.dot(p, p, input_precision="ieee")
    inv += tl.dot(inv, p, input_precision="ieee")
    p = tl.dot(p, p, input_precision="ieee")
    inv += tl.dot(inv, p, input_precision="ieee")
    p = tl.dot(p, p, input_precision="ieee")
    inv += tl.dot(inv, p, input_precision="ieee")
    return inv


@triton.jit
def _chunk_factors(q, k, a, b, g, valid, scale: tl.constexpr):
    rows = tl.arange(0, 16)
    gi = tl.cumsum(g, axis=0)
    gp = gi - g
    ag = a * tl.exp(tl.maximum(gp, -60.0))
    kg = k * tl.exp(tl.minimum(-gi, 60.0))
    bg = b * tl.exp(tl.minimum(-gi, 60.0))
    lower = rows[:, None] > rows[None, :]
    inclusive = rows[:, None] >= rows[None, :]
    lab = tl.dot(ag, tl.trans(bg), input_precision="ieee")
    lak = tl.dot(ag, tl.trans(kg), input_precision="ieee")
    inv = _inverse_lower(tl.where(lower, lab, 0.0))
    aak = tl.dot(inv, tl.where(lower, lak, 0.0), input_precision="ieee")
    w = tl.dot(inv, ag, input_precision="ieee")
    qg = q * tl.exp(tl.maximum(gi, -60.0)) * scale
    aqk = tl.dot(qg, tl.trans(kg), input_precision="ieee")
    aqb = tl.dot(qg, tl.trans(bg), input_precision="ieee")
    pair_mask = inclusive & valid[:, None] & valid[None, :]
    return gi, w, aak, qg, tl.where(pair_mask, aqk, 0.0), tl.where(pair_mask, aqb, 0.0)


@triton.jit
def _forward_kernel(qp, kp, vp, ap, bp, gp, op, hp,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    SCALE: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * 16 + tl.arange(0, 16)
    state = tl.zeros([BK, 16], tl.float32)
    ot = tl.arange(0, 16)
    for c in range(NC):
        ts = c * 16 + ot
        valid = ts < T
        xk = ((ib * T + ts[:, None]) * H + ih) * K + ok[None, :]
        xv = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        mk = valid[:, None] & (ok[None, :] < K)
        mv = valid[:, None] & (ov[None, :] < V)
        q = tl.load(qp + xk, mask=mk, other=0.0).to(tl.float32)
        k = tl.load(kp + xk, mask=mk, other=0.0).to(tl.float32)
        a = tl.load(ap + xk, mask=mk, other=0.0).to(tl.float32)
        b = tl.load(bp + xk, mask=mk, other=0.0).to(tl.float32)
        g = tl.load(gp + xk, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(vp + xv, mask=mv, other=0.0).to(tl.float32)
        hoff = ((((ib * NC + c) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(hp + hoff, state, mask=(ok[:, None] < K) & (ov[None, :] < V))
        gi, w, aak, qg, aqk, aqb = _chunk_factors(q, k, a, b, g, valid, SCALE)
        r = tl.dot(w, state, input_precision="ieee") + tl.dot(aak, vv, input_precision="ieee")
        out = tl.dot(qg, state, input_precision="ieee")
        out += tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, r, input_precision="ieee")
        tl.store(op + xv, out, mask=mv)
        last = tl.minimum(T - c * 16, 16) - 1
        ge = tl.sum(tl.where((ot == last)[:, None], gi, 0.0), axis=0)
        endk = tl.where(valid[:, None], k * tl.exp(tl.maximum(ge[None, :] - gi, -60.0)), 0.0)
        endb = tl.where(valid[:, None], b * tl.exp(tl.maximum(ge[None, :] - gi, -60.0)), 0.0)
        state = state * tl.exp(tl.maximum(ge, -60.0))[:, None]
        state += tl.dot(tl.trans(endk), vv, input_precision="ieee")
        state += tl.dot(tl.trans(endb), r, input_precision="ieee")


@triton.jit
def _boundary_backward_kernel(qp, kp, ap, bp, gp, dop, dep,
                              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                              V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                              SCALE: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * 16 + tl.arange(0, 16)
    ot = tl.arange(0, 16)
    ds = tl.zeros([BK, 16], tl.float32)
    for rc in range(NC):
        c = NC - 1 - rc
        ts = c * 16 + ot
        valid = ts < T
        xk = ((ib * T + ts[:, None]) * H + ih) * K + ok[None, :]
        xv = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        mk = valid[:, None] & (ok[None, :] < K)
        mv = valid[:, None] & (ov[None, :] < V)
        q = tl.load(qp + xk, mask=mk, other=0.0).to(tl.float32)
        k = tl.load(kp + xk, mask=mk, other=0.0).to(tl.float32)
        a = tl.load(ap + xk, mask=mk, other=0.0).to(tl.float32)
        b = tl.load(bp + xk, mask=mk, other=0.0).to(tl.float32)
        g = tl.load(gp + xk, mask=mk, other=0.0).to(tl.float32)
        do = tl.load(dop + xv, mask=mv, other=0.0).to(tl.float32)
        deoff = ((((ib * NC + c) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(dep + deoff, ds, mask=(ok[:, None] < K) & (ov[None, :] < V))
        gi, w, aak, qg, aqk, aqb = _chunk_factors(q, k, a, b, g, valid, SCALE)
        last = tl.minimum(T - c * 16, 16) - 1
        ge = tl.sum(tl.where((ot == last)[:, None], gi, 0.0), axis=0)
        endb = tl.where(valid[:, None], b * tl.exp(tl.maximum(ge[None, :] - gi, -60.0)), 0.0)
        read_adj = tl.dot(endb, ds, input_precision="ieee")
        read_adj += tl.dot(tl.trans(aqb), do, input_precision="ieee")
        ds = ds * tl.exp(tl.maximum(ge, -60.0))[:, None]
        ds += tl.dot(tl.trans(qg), do, input_precision="ieee")
        ds += tl.dot(tl.trans(w), read_adj, input_precision="ieee")




@triton.jit
def _expand_chunk_states_kernel(kp, vp, ap, bp, gp, hp, sp,
                                T: tl.constexpr, H: tl.constexpr,
                                K: tl.constexpr, V: tl.constexpr,
                                NC: tl.constexpr, BK: tl.constexpr):
    iv = tl.program_id(0)
    c = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * 16 + tl.arange(0, 16)
    state_mask = (ok[:, None] < K) & (ov[None, :] < V)
    hoff = ((((ib * NC + c) * H + ih) * K + ok[:, None]) * V + ov[None, :])
    state = tl.load(hp + hoff, mask=state_mask, other=0.0)
    for j in tl.static_range(0, 16):
        t = c * 16 + j
        active = t < T
        soff = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(sp + soff, state, mask=active & state_mask)
        xk = ((ib * T + t) * H + ih) * K + ok
        xv = ((ib * T + t) * H + ih) * V + ov
        km = active & (ok < K)
        vm = active & (ov < V)
        k = tl.load(kp + xk, mask=km, other=0.0).to(tl.float32)
        a = tl.load(ap + xk, mask=km, other=0.0).to(tl.float32)
        b = tl.load(bp + xk, mask=km, other=0.0).to(tl.float32)
        g = tl.load(gp + xk, mask=km, other=0.0).to(tl.float32)
        v = tl.load(vp + xv, mask=vm, other=0.0).to(tl.float32)
        read = tl.sum(state * a[:, None], axis=0)
        state = state * tl.exp(g)[:, None] + k[:, None] * v[None, :] + b[:, None] * read[None, :]


@triton.jit
def _expand_chunk_adjoints_kernel(qp, ap, bp, gp, dop, dep, dp,
                                  T: tl.constexpr, H: tl.constexpr,
                                  K: tl.constexpr, V: tl.constexpr,
                                  NC: tl.constexpr, BK: tl.constexpr,
                                  SCALE: tl.constexpr):
    iv = tl.program_id(0)
    c = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * 16 + tl.arange(0, 16)
    mask = (ok[:, None] < K) & (ov[None, :] < V)
    eoff = ((((ib * NC + c) * H + ih) * K + ok[:, None]) * V + ov[None, :])
    ds = tl.load(dep + eoff, mask=mask, other=0.0)
    for jj in tl.static_range(0, 16):
        j = 15 - jj
        t = c * 16 + j
        active = t < T
        xk = ((ib * T + t) * H + ih) * K + ok
        xv = ((ib * T + t) * H + ih) * V + ov
        km = active & (ok < K)
        vm = active & (ov < V)
        q = tl.load(qp + xk, mask=km, other=0.0).to(tl.float32)
        a = tl.load(ap + xk, mask=km, other=0.0).to(tl.float32)
        b = tl.load(bp + xk, mask=km, other=0.0).to(tl.float32)
        g = tl.load(gp + xk, mask=km, other=0.0).to(tl.float32)
        do = tl.load(dop + xv, mask=vm, other=0.0).to(tl.float32)
        ds += q[:, None] * (do[None, :] * SCALE)
        doff = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(dp + doff, ds, mask=active & mask)
        dr = tl.sum(ds * b[:, None], axis=0)
        ds = ds * tl.exp(g)[:, None] + a[:, None] * dr[None, :]


@triton.jit
def _timestep_gradient_kernel(qp, kp, vp, ap, bp, gp, dop, sp, dp,
                              pp, dvp, T: tl.constexpr, H: tl.constexpr,
                              K: tl.constexpr, V: tl.constexpr,
                              BK: tl.constexpr, SCALE: tl.constexpr):
    iv = tl.program_id(0)
    t = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * 16 + tl.arange(0, 16)
    km = ok < K
    vm = ov < V
    xk = ((ib * T + t) * H + ih) * K + ok
    xv = ((ib * T + t) * H + ih) * V + ov
    q = tl.load(qp + xk, mask=km, other=0.0).to(tl.float32)
    k = tl.load(kp + xk, mask=km, other=0.0).to(tl.float32)
    a = tl.load(ap + xk, mask=km, other=0.0).to(tl.float32)
    b = tl.load(bp + xk, mask=km, other=0.0).to(tl.float32)
    g = tl.load(gp + xk, mask=km, other=0.0).to(tl.float32)
    v = tl.load(vp + xv, mask=vm, other=0.0).to(tl.float32)
    do = tl.load(dop + xv, mask=vm, other=0.0).to(tl.float32)
    soff = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
    sm = km[:, None] & vm[None, :]
    state = tl.load(sp + soff, mask=sm, other=0.0)
    ds = tl.load(dp + soff, mask=sm, other=0.0)
    read = tl.sum(state * a[:, None], axis=0)
    current = state * tl.exp(g)[:, None] + k[:, None] * v[None, :] + b[:, None] * read[None, :]
    dq = tl.sum(current * do[None, :], axis=1) * SCALE
    dk = tl.sum(ds * v[None, :], axis=1)
    dv = tl.sum(ds * k[:, None], axis=0)
    db = tl.sum(ds * read[None, :], axis=1)
    dr = tl.sum(ds * b[:, None], axis=0)
    da = tl.sum(state * dr[None, :], axis=1)
    dg = tl.sum(ds * (state * tl.exp(g)[:, None]), axis=1)
    pbase = (((iv * (T * H) + (ib * T + t) * H + ih) * 5) * K + ok)
    tl.store(pp + pbase + 0 * K, dq, mask=km)
    tl.store(pp + pbase + 1 * K, dk, mask=km)
    tl.store(pp + pbase + 2 * K, da, mask=km)
    tl.store(pp + pbase + 3 * K, db, mask=km)
    tl.store(pp + pbase + 4 * K, dg, mask=km)
    tl.store(dvp + xv, dv, mask=vm)
@triton.jit
def _reduce_kernel(pp, dqp, dkp, dap, dbp, dgp,
                   N: tl.constexpr, K: tl.constexpr,
                   NV: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    bth = idx // K
    kk = idx - bth * K
    acc0 = tl.zeros([BLOCK], tl.float32)
    acc1 = tl.zeros([BLOCK], tl.float32)
    acc2 = tl.zeros([BLOCK], tl.float32)
    acc3 = tl.zeros([BLOCK], tl.float32)
    acc4 = tl.zeros([BLOCK], tl.float32)
    for iv in range(NV):
        base = (iv * (N // K) + bth) * 5 * K + kk
        acc0 += tl.load(pp + base + 0 * K, mask=mask, other=0.0)
        acc1 += tl.load(pp + base + 1 * K, mask=mask, other=0.0)
        acc2 += tl.load(pp + base + 2 * K, mask=mask, other=0.0)
        acc3 += tl.load(pp + base + 3 * K, mask=mask, other=0.0)
        acc4 += tl.load(pp + base + 4 * K, mask=mask, other=0.0)
    tl.store(dqp + idx, acc0, mask=mask)
    tl.store(dkp + idx, acc1, mask=mask)
    tl.store(dap + idx, acc2, mask=mask)
    tl.store(dbp + idx, acc3, mask=mask)
    tl.store(dgp + idx, acc4, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc, nv = triton.cdiv(T, _BT), triton.cdiv(V, _BV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc, nv = triton.cdiv(T, _BT), triton.cdiv(V, _BV)
        bk = max(16, triton.next_power_of_2(K))
        dend = torch.empty_like(checkpoints)
        _boundary_backward_kernel[(nv, B * H)](
            q, k, alpha, beta, gk, dout, dend,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_chunk_states_kernel[(nv, nc, B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk,
            num_warps=8, num_stages=2,
        )
        adjoints = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_chunk_adjoints_kernel[(nv, nc, B * H)](
            q, alpha, beta, gk, dout, dend, adjoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        partial = torch.empty((nv, B, T, H, 5, K), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _timestep_gradient_kernel[(nv, T, B * H)](
            q, k, v, alpha, beta, gk, dout, states, adjoints, partial, dv,
            T=T, H=H, K=K, V=V, BK=bk, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(n, 256),)](
            partial, dq, dk, da, db, dg, N=n, K=K, NV=nv, BLOCK=256,
            num_warps=4,
        )
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
