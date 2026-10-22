import torch
import triton
import triton.language as tl

_BT = 16
_BV = 32
_BWD_BV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    rows = tl.arange(0, 16)
    causal = rows[:, None] >= rows[None, :]
    strict = rows[:, None] > rows[None, :]
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    nc: tl.constexpr = (T + 15) // 16

    for ic in range(nc):
        t = ic * 16 + rows
        mt = t < T
        mk = mt[:, None] & (ok[None, :] < K)
        mv = mt[:, None] & (ov[None, :] < V)
        pk = base_k + t[:, None] * H * K + ok[None, :]
        pv = base_v + t[:, None] * H * V + ov[None, :]
        q0 = tl.load(q + pk, mask=mk, other=0.0).to(tl.float32)
        k0 = tl.load(k + pk, mask=mk, other=0.0).to(tl.float32)
        a0 = tl.load(a + pk, mask=mk, other=0.0).to(tl.float32)
        b0 = tl.load(b + pk, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + pk, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)

        hp = (((ib * nc + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(checkpoints + hp, state, mask=(ok[:, None] < K) & (ov[None, :] < V))

        gi = tl.cumsum(gg, axis=0)
        gp = gi - gg
        eg = tl.exp(tl.maximum(tl.minimum(gi, 60.0), -60.0))
        eig = tl.exp(tl.maximum(tl.minimum(-gi, 60.0), -60.0))
        ep = tl.exp(tl.maximum(tl.minimum(gp, 60.0), -60.0))
        qg = q0 * eg * (1.0 / (K ** 0.5))
        kg = k0 * eig
        bg = b0 * eig
        ag = a0 * ep

        lab = tl.where(strict, tl.dot(ag, tl.trans(bg), input_precision="ieee"), 0.0)
        lak = tl.where(strict, tl.dot(ag, tl.trans(kg), input_precision="ieee"), 0.0)
        eye = (rows[:, None] == rows[None, :]).to(tl.float32)
        inv = eye
        power = lab
        for _ in tl.static_range(0, 4):
            inv += tl.dot(power, inv, input_precision="ieee")
            power = tl.dot(power, power, input_precision="ieee")
        rhs = tl.dot(ag, state, input_precision="ieee") + tl.dot(lak, vv, input_precision="ieee")
        u = tl.dot(inv, rhs, input_precision="ieee")

        aqk = tl.where(causal, tl.dot(qg, tl.trans(kg), input_precision="ieee"), 0.0)
        aqb = tl.where(causal, tl.dot(qg, tl.trans(bg), input_precision="ieee"), 0.0)
        oo = tl.dot(qg, state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee")
        oo += tl.dot(aqb, u, input_precision="ieee")
        tl.store(out + pv, oo, mask=mv)

        last = tl.sum(tl.where((rows == 15)[:, None], gi, 0.0), axis=0)
        dend = tl.exp(tl.maximum(tl.minimum(last, 60.0), -60.0))
        state += tl.dot(tl.trans(kg), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bg), u, input_precision="ieee")
        state *= dend[:, None]


@triton.jit
def _expand_states_kernel(k, v, a, b, g, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    for t in range(T):
        pk = base_k + t * H * K + ok
        pv = base_v + t * H * V + ov
        ps = ((ib * T * H + t * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(states + ps, state, mask=(ok[:, None] < K) & (ov[None, :] < V))
        kk = tl.load(k + pk, mask=ok < K, other=0.0).to(tl.float32)
        aa = tl.load(a + pk, mask=ok < K, other=0.0).to(tl.float32)
        bb = tl.load(b + pk, mask=ok < K, other=0.0).to(tl.float32)
        gg = tl.load(g + pk, mask=ok < K, other=0.0).to(tl.float32)
        vv = tl.load(v + pv, mask=ov < V, other=0.0).to(tl.float32)
        read = tl.sum(state.to(tl.float64) * aa[:, None].to(tl.float64), axis=0).to(tl.float32)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, dout, states, dv, partial,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, B: tl.constexpr, BK: tl.constexpr,
                     BV: tl.constexpr, NV: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk = ok < K
    mv = ov < V
    ds = tl.zeros((BK, BV), tl.float32)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    n: tl.constexpr = B * T * H * K

    for it in range(T):
        t = T - 1 - it
        pk = base_k + t * H * K + ok
        pv = base_v + t * H * V + ov
        ps = ((ib * T * H + t * H + ih) * K + ok[:, None]) * V + ov[None, :]
        s0 = tl.load(states + ps, mask=mk[:, None] & mv[None, :], other=0.0)
        qq = tl.load(q + pk, mask=mk, other=0.0).to(tl.float32)
        kk = tl.load(k + pk, mask=mk, other=0.0).to(tl.float32)
        aa = tl.load(a + pk, mask=mk, other=0.0).to(tl.float32)
        bb = tl.load(b + pk, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + pk, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        do = tl.load(dout + pv, mask=mv, other=0.0).to(tl.float32)
        scale = 1.0 / (K ** 0.5)
        read = tl.sum(s0 * aa[:, None], axis=0)
        sn = s0 * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        dq0 = tl.sum(sn * do[None, :], axis=1) * scale
        ds += qq[:, None] * (do * scale)[None, :]
        dr = tl.sum(ds * bb[:, None], axis=0)
        dk0 = tl.sum(ds * vv[None, :], axis=1)
        da0 = tl.sum(s0 * dr[None, :], axis=1)
        db0 = tl.sum(ds * read[None, :], axis=1)
        dg0 = tl.sum(ds * (s0 * tl.exp(gg)[:, None]), axis=1)
        dv0 = tl.sum(ds * kk[:, None], axis=0)
        tl.store(dv + pv, dv0, mask=mv)
        idx = ib * T * H * K + t * H * K + ih * K + ok
        tl.store(partial + iv * 5 * n + 0 * n + idx, dq0, mask=mk)
        tl.store(partial + iv * 5 * n + 1 * n + idx, dk0, mask=mk)
        tl.store(partial + iv * 5 * n + 2 * n + idx, da0, mask=mk)
        tl.store(partial + iv * 5 * n + 3 * n + idx, db0, mask=mk)
        tl.store(partial + iv * 5 * n + 4 * n + idx, dg0, mask=mk)
        ds = ds * tl.exp(gg)[:, None] + aa[:, None] * dr[None, :]


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr, NV: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = i < N
    xq = tl.zeros((256,), tl.float32)
    xk = tl.zeros((256,), tl.float32)
    xa = tl.zeros((256,), tl.float32)
    xb = tl.zeros((256,), tl.float32)
    xg = tl.zeros((256,), tl.float32)
    for iv in tl.static_range(0, NV):
        p = partial + iv * 5 * N + i
        xq += tl.load(p + 0 * N, mask=m, other=0.0)
        xk += tl.load(p + 1 * N, mask=m, other=0.0)
        xa += tl.load(p + 2 * N, mask=m, other=0.0)
        xb += tl.load(p + 3 * N, mask=m, other=0.0)
        xg += tl.load(p + 4 * N, mask=m, other=0.0)
    tl.store(dq + i, xq, mask=m)
    tl.store(dk + i, xk, mask=m)
    tl.store(da + i, xa, mask=m)
    tl.store(db + i, xb, mask=m)
    tl.store(dg + i, xg, mask=m)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BV)
        nc = triton.cdiv(T, _BT)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BV,
            num_warps=8, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        bv = _BWD_BV
        nv = triton.cdiv(V, bv)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, B * H)](
            k, v, alpha, beta, gk, states,
            T=T, H=H, K=K, V=V, BK=bk, BV=bv,
            num_warps=4, num_stages=2,
        )
        dv = torch.empty_like(v)
        partial = torch.zeros((nv, 5, B, T, H, K), device=q.device, dtype=torch.float32)
        _backward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout.contiguous(), states, dv, partial,
            T=T, H=H, K=K, V=V, B=B, BK=bk, BV=bv, NV=nv,
            num_warps=4, num_stages=2,
        )
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gk)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(n, 256),)](
            partial, dq, dk, da, db, dg, N=n, NV=nv, num_warps=4,
        )
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
