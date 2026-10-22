import torch
import triton
import triton.language as tl

BT = 16
BV = 32
BWD_BV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, o, h, T: tl.constexpr, H: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                    BK: tl.constexpr, BV_: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV_ + tl.arange(0, BV_)
    rt = tl.arange(0, 16)
    state = tl.zeros((BK, BV_), tl.float32)
    scale: tl.constexpr = K ** -0.5
    eye = rt[:, None] == rt[None, :]
    lower = rt[:, None] > rt[None, :]
    causal = rt[:, None] >= rt[None, :]
    for ic in range(NC):
        hs = ((bh * NC + ic) * K + rk[:, None]) * V + rv[None, :]
        tl.store(h + hs, state, mask=(rk[:, None] < K) & (rv[None, :] < V))
        tt = ic * 16 + rt
        base = (ib * T * H + tt * H + ih) * K
        m = (tt[:, None] < T) & (rk[None, :] < K)
        qq = tl.load(q + base[:, None] + rk[None, :], mask=m, other=0.).to(tl.float32) * scale
        kk = tl.load(k + base[:, None] + rk[None, :], mask=m, other=0.).to(tl.float32)
        aa = tl.load(a + base[:, None] + rk[None, :], mask=m, other=0.).to(tl.float32)
        bb = tl.load(b + base[:, None] + rk[None, :], mask=m, other=0.).to(tl.float32)
        gg = tl.load(g + base[:, None] + rk[None, :], mask=m, other=0.).to(tl.float32)
        cg = tl.cumsum(gg, axis=0)
        last = tl.minimum(15, T - 1 - ic * 16)
        mid = last // 2
        center = tl.sum(tl.where(rt[:, None] == mid, cg, 0.), axis=0)
        q_exp = tl.maximum(-40., tl.minimum(40., cg - center[None, :]))
        k_exp = tl.maximum(-40., tl.minimum(40., center[None, :] - cg))
        a_exp = tl.maximum(-40., tl.minimum(40., cg - gg - center[None, :]))
        qg = qq * tl.exp(q_exp)
        kg = kk * tl.exp(k_exp)
        bg = bb * tl.exp(k_exp)
        ag = aa * tl.exp(a_exp)
        Aqk = tl.where(causal, tl.dot(qg, tl.trans(kg), input_precision='ieee'), 0.)
        Aqb = tl.where(causal, tl.dot(qg, tl.trans(bg), input_precision='ieee'), 0.)
        Aak = tl.where(lower, tl.dot(ag, tl.trans(kg), input_precision='ieee'), 0.)
        Lab = tl.where(lower, tl.dot(ag, tl.trans(bg), input_precision='ieee'), 0.)
        inv = eye.to(tl.float32) + Lab
        power = Lab
        for _ in tl.static_range(0, 4):
            power = tl.dot(power, power, input_precision='ieee')
            inv = tl.dot(inv, eye.to(tl.float32) + power, input_precision='ieee')
        Aak = tl.dot(inv, Aak, input_precision='ieee')
        w = tl.dot(inv, ag, input_precision='ieee')
        vb = (ib * T * H + tt * H + ih) * V
        vm = (tt[:, None] < T) & (rv[None, :] < V)
        vv = tl.load(v + vb[:, None] + rv[None, :], mask=vm, other=0.).to(tl.float32)
        u = tl.dot(Aak, vv, input_precision='ieee')
        centered_state = state * tl.exp(center)[:, None]
        vn = u + tl.dot(w, centered_state, input_precision='ieee')
        out = tl.dot(qg, centered_state, input_precision='ieee') + tl.dot(Aqk, vv, input_precision='ieee') + tl.dot(Aqb, vn, input_precision='ieee')
        tl.store(o + vb[:, None] + rv[None, :], out, mask=vm)
        glast = tl.sum(tl.where(rt[:, None] == last, cg, 0.), axis=0)
        decay = tl.exp(glast[None, :] - cg)
        state = state * tl.exp(glast)[:, None]
        state += tl.dot(tl.trans(kk * decay), vv, input_precision='ieee')
        state += tl.dot(tl.trans(bb * decay), vn, input_precision='ieee')


@triton.jit
def _expand_states_kernel(k, v, a, b, g, h, states, T: tl.constexpr, H: tl.constexpr,
                          K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                          BK: tl.constexpr, BV_: tl.constexpr):
    iv = tl.program_id(0)
    ic_bh = tl.program_id(1)
    ic = ic_bh % NC
    bh = ic_bh // NC
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV_ + tl.arange(0, BV_)
    hs = ((bh * NC + ic) * K + rk[:, None]) * V + rv[None, :]
    state = tl.load(h + hs, mask=(rk[:, None] < K) & (rv[None, :] < V), other=0.)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        kb = (ib * T * H + t * H + ih) * K
        vb = (ib * T * H + t * H + ih) * V
        km = valid & (rk < K)
        vm = valid & (rv < V)
        kk = tl.load(k + kb + rk, mask=km, other=0.).to(tl.float32)
        aa = tl.load(a + kb + rk, mask=km, other=0.).to(tl.float32)
        bb = tl.load(b + kb + rk, mask=km, other=0.).to(tl.float32)
        gg = tl.load(g + kb + rk, mask=km, other=0.).to(tl.float32)
        vv = tl.load(v + vb + rv, mask=vm, other=0.).to(tl.float32)
        read = tl.sum(state * aa[:, None], axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        ss = ((bh * T + t) * K + rk[:, None]) * V + rv[None, :]
        tl.store(states + ss, state, mask=valid & (rk[:, None] < K) & (rv[None, :] < V))


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, h, states, dv, partial,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                     BK: tl.constexpr, BV_: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV_ + tl.arange(0, BV_)
    dstate = tl.zeros((BK, BV_), tl.float32)
    scale: tl.constexpr = K ** -0.5
    for iz in range(NC):
        ic = NC - 1 - iz
        for j in tl.static_range(0, 16):
            t = ic * 16 + 15 - j
            valid = t < T
            kb = (ib * T * H + t * H + ih) * K
            vb = (ib * T * H + t * H + ih) * V
            km = valid & (rk < K)
            vm = valid & (rv < V)
            qq = tl.load(q + kb + rk, mask=km, other=0.).to(tl.float32) * scale
            kk = tl.load(k + kb + rk, mask=km, other=0.).to(tl.float32)
            aa = tl.load(a + kb + rk, mask=km, other=0.).to(tl.float32)
            bb = tl.load(b + kb + rk, mask=km, other=0.).to(tl.float32)
            gg = tl.load(g + kb + rk, mask=km, other=0.).to(tl.float32)
            vv = tl.load(v + vb + rv, mask=vm, other=0.).to(tl.float32)
            dd = tl.load(do + vb + rv, mask=vm, other=0.).to(tl.float32)
            ss = ((bh * T + t) * K + rk[:, None]) * V + rv[None, :]
            state = tl.load(states + ss, mask=km[:, None] & vm[None, :], other=0.)
            prev_t = tl.maximum(t - 1, 0)
            prev_ss = ((bh * T + prev_t) * K + rk[:, None]) * V + rv[None, :]
            hs = ((bh * NC + ic) * K + rk[:, None]) * V + rv[None, :]
            prev = tl.where((t % 16) == 0,
                            tl.load(h + hs, mask=km[:, None] & vm[None, :], other=0.),
                            tl.load(states + prev_ss, mask=km[:, None] & vm[None, :], other=0.))
            dstate += qq[:, None] * dd[None, :]
            dq = tl.sum(state * dd[None, :], axis=1) * scale
            dk = tl.sum(dstate * vv[None, :], axis=1)
            dvv = tl.sum(dstate * kk[:, None], axis=0)
            read = tl.sum(prev * aa[:, None], axis=0)
            db = tl.sum(dstate * read[None, :], axis=1)
            dread = tl.sum(dstate * bb[:, None], axis=0)
            da = tl.sum(prev * dread[None, :], axis=1)
            eg = tl.exp(gg)
            dg = tl.sum(dstate * prev * eg[:, None], axis=1)
            row = (ib * T + t) * H + ih
            po = (row * NV + iv) * 5 * K
            tl.store(partial + po + 0 * K + rk, dq, mask=km)
            tl.store(partial + po + 1 * K + rk, dk, mask=km)
            tl.store(partial + po + 2 * K + rk, da, mask=km)
            tl.store(partial + po + 3 * K + rk, db, mask=km)
            tl.store(partial + po + 4 * K + rk, dg, mask=km)
            tl.store(dv + vb + rv, dvv, mask=vm)
            dstate = dstate * tl.where(km, tl.exp(gg), 1.)[:, None] + aa[:, None] * dread[None, :]


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr,
                   K: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    row = i // K
    rk = i - row * K
    accq = tl.zeros((BLOCK,), tl.float32)
    acck = tl.zeros((BLOCK,), tl.float32)
    acca = tl.zeros((BLOCK,), tl.float32)
    accb = tl.zeros((BLOCK,), tl.float32)
    accg = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        p = (row * NV + iv) * 5 * K + rk
        accq += tl.load(partial + p + 0 * K, mask=m, other=0.)
        acck += tl.load(partial + p + 1 * K, mask=m, other=0.)
        acca += tl.load(partial + p + 2 * K, mask=m, other=0.)
        accb += tl.load(partial + p + 3 * K, mask=m, other=0.)
        accg += tl.load(partial + p + 4 * K, mask=m, other=0.)
    tl.store(dq + i, accq, mask=m)
    tl.store(dk + i, acck, mask=m)
    tl.store(da + i, acca, mask=m)
    tl.store(db + i, accb, mask=m)
    tl.store(dg + i, accg, mask=m)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, BT)
        nv = triton.cdiv(V, BV)
        bwd_nv = triton.cdiv(V, BWD_BV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((B * H, nc, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        ctx.dims = (B, T, H, K, V, nc, bwd_nv, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K, V, nc, nv, bk = ctx.dims
        dout = dout.contiguous()
        states = torch.empty((B * H, T, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc * B * H)](k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BWD_BV, num_warps=4, num_stages=2)
        dv = torch.empty_like(v)
        partial = torch.zeros((B * H * T, nv, 5, K), device=q.device, dtype=torch.float32)
        _backward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, dout, checkpoints, states, dv, partial,
            T=T, H=H, K=K, V=V, NC=nc, NV=nv, BK=bk, BV_=BWD_BV, num_warps=8, num_stages=2)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg,
            N=n, K=K, NV=nv, BLOCK=256, num_warps=8)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
