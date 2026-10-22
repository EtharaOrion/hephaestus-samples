import torch
import triton
import triton.language as tl

BT = 16
BV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, o, hbuf, T: tl.constexpr, H: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                    scale: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ih = ibh % H
    ib = ibh // H
    rk = tl.arange(0, BK)
    rv = iv * 16 + tl.arange(0, 16)
    rt = tl.arange(0, 16)
    mk, mv = rk < K, rv < V
    state = tl.zeros((BK, 16), tl.float32)
    nc = tl.cdiv(T, 16)
    for ic in range(nc):
        ts = ic * 16 + rt
        mt = ts < T
        base = (ib * T + ts[:, None]) * H + ih
        xoff = base * K + rk[None, :]
        voff = base * V + rv[None, :]
        q0 = tl.load(q + xoff, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32) * scale
        k0 = tl.load(k + xoff, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        a0 = tl.load(a + xoff, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        b0 = tl.load(b + xoff, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        gg = tl.load(g + xoff, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        vv = tl.load(v + voff, mask=mt[:, None] & mv[None, :], other=0.0).to(tl.float32)
        cg = tl.cumsum(gg, axis=0)
        qg = q0 * tl.exp(cg)
        kg = k0 * tl.exp(-cg)
        bg = b0 * tl.exp(-cg)
        ag = a0 * tl.exp(cg - gg)
        hab = tl.dot(ag, tl.trans(bg), input_precision="ieee")
        hak = tl.dot(ag, tl.trans(kg), input_precision="ieee")
        strict = rt[:, None] > rt[None, :]
        hab = tl.where(strict, hab, 0.0)
        hak = tl.where(strict, hak, 0.0)
        z = tl.dot(ag, state, input_precision="ieee") + tl.dot(hak, vv, input_precision="ieee")
        for j in tl.static_range(0, 16):
            row = tl.sum(tl.where(rt[:, None] == j, hab, 0.0), axis=0)
            add = tl.sum(tl.where(rt[:, None] < j, row[:, None] * z, 0.0), axis=0)
            z = tl.where(rt[:, None] == j, z + add[None, :], z)
        causal = rt[:, None] >= rt[None, :]
        aqb = tl.where(causal, tl.dot(qg, tl.trans(bg), input_precision="ieee"), 0.0)
        aqk = tl.where(causal, tl.dot(qg, tl.trans(kg), input_precision="ieee"), 0.0)
        out = tl.dot(qg, state, input_precision="ieee")
        out += tl.dot(aqb, z, input_precision="ieee") + tl.dot(aqk, vv, input_precision="ieee")
        tl.store(o + voff, out, mask=mt[:, None] & mv[None, :])
        hoff = ((ibh * nc + ic) * K + rk[:, None]) * V + rv[None, :]
        tl.store(hbuf + hoff, state, mask=mk[:, None] & mv[None, :])
        for j in tl.static_range(0, 16):
            pick = rt[:, None] == j
            kj = tl.sum(tl.where(pick, k0, 0.0), axis=0)
            aj = tl.sum(tl.where(pick, a0, 0.0), axis=0)
            bj = tl.sum(tl.where(pick, b0, 0.0), axis=0)
            gj = tl.sum(tl.where(pick, gg, 0.0), axis=0)
            vj = tl.sum(tl.where(pick, vv, 0.0), axis=0)
            read = tl.sum(state * aj[:, None], axis=0)
            state = state * tl.exp(gj)[:, None] + kj[:, None] * vj[None, :] + bj[:, None] * read[None, :]


@triton.jit
def _expand_kernel(k, v, a, b, g, hbuf, states, T: tl.constexpr, H: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ih, ib = ibh % H, ibh // H
    rk = tl.arange(0, BK)
    rv = iv * 16 + tl.arange(0, 16)
    mk, mv = rk < K, rv < V
    nc = tl.cdiv(T, 16)
    hoff = ((ibh * nc + ic) * K + rk[:, None]) * V + rv[None, :]
    state = tl.load(hbuf + hoff, mask=mk[:, None] & mv[None, :], other=0.0)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        mt = t < T
        base = (ib * T + t) * H + ih
        xo, vo = base * K + rk, base * V + rv
        so = ((ibh * T + t) * K + rk[:, None]) * V + rv[None, :]
        tl.store(states + so, state, mask=mt & mk[:, None] & mv[None, :])
        kk = tl.load(k + xo, mask=mt & mk, other=0.0).to(tl.float32)
        aa = tl.load(a + xo, mask=mt & mk, other=0.0).to(tl.float32)
        bb = tl.load(b + xo, mask=mt & mk, other=0.0).to(tl.float32)
        gg = tl.load(g + xo, mask=mt & mk, other=0.0).to(tl.float32)
        vv = tl.load(v + vo, mask=mt & mv, other=0.0).to(tl.float32)
        read = tl.sum(state * aa[:, None], axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, dv, partial,
                     B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, BK: tl.constexpr, scale: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ih, ib = ibh % H, ibh // H
    rk = tl.arange(0, BK)
    rv = iv * 16 + tl.arange(0, 16)
    mk, mv = rk < K, rv < V
    ds = tl.zeros((BK, 16), tl.float32)
    stride = 5 * K
    for ii in range(T):
        t = T - 1 - ii
        base = (ib * T + t) * H + ih
        xo, vo = base * K + rk, base * V + rv
        so = ((ibh * T + t) * K + rk[:, None]) * V + rv[None, :]
        old = tl.load(states + so, mask=mk[:, None] & mv[None, :], other=0.0)
        qq = tl.load(q + xo, mask=mk, other=0.0).to(tl.float32) * scale
        kk = tl.load(k + xo, mask=mk, other=0.0).to(tl.float32)
        aa = tl.load(a + xo, mask=mk, other=0.0).to(tl.float32)
        bb = tl.load(b + xo, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + xo, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + vo, mask=mv, other=0.0).to(tl.float32)
        dd = tl.load(do + vo, mask=mv, other=0.0).to(tl.float32)
        decay = tl.exp(gg)
        read = tl.sum(old * aa[:, None], axis=0)
        new = old * decay[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        total = ds + qq[:, None] * dd[None, :]
        dq0 = tl.sum(new * dd[None, :], axis=1) * scale
        dk0 = tl.sum(total * vv[None, :], axis=1)
        db0 = tl.sum(total * read[None, :], axis=1)
        dread = tl.sum(total * bb[:, None], axis=0)
        da0 = tl.sum(old * dread[None, :], axis=1)
        dg0 = tl.sum(total * old, axis=1) * decay
        tl.store(dv + vo, tl.sum(total * kk[:, None], axis=0), mask=mv)
        po = ((iv * B + ib) * T * H + t * H + ih) * stride + rk
        tl.store(partial + po + 0 * K, dq0, mask=mk)
        tl.store(partial + po + 1 * K, dk0, mask=mk)
        tl.store(partial + po + 2 * K, da0, mask=mk)
        tl.store(partial + po + 3 * K, db0, mask=mk)
        tl.store(partial + po + 4 * K, dg0, mask=mk)
        ds = total * decay[:, None] + aa[:, None] * dread[None, :]


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr,
                   K: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < N
    x, row = p % K, p // K
    stride = 5 * K
    aq = tl.zeros((BLOCK,), tl.float32)
    ak = tl.zeros((BLOCK,), tl.float32)
    aa = tl.zeros((BLOCK,), tl.float32)
    ab = tl.zeros((BLOCK,), tl.float32)
    ag = tl.zeros((BLOCK,), tl.float32)
    for iv in tl.static_range(0, NV):
        off = (iv * (N // K) + row) * stride + x
        aq += tl.load(partial + off + 0 * K, mask=mask, other=0.0)
        ak += tl.load(partial + off + 1 * K, mask=mask, other=0.0)
        aa += tl.load(partial + off + 2 * K, mask=mask, other=0.0)
        ab += tl.load(partial + off + 3 * K, mask=mask, other=0.0)
        ag += tl.load(partial + off + 4 * K, mask=mask, other=0.0)
    tl.store(dq + p, aq, mask=mask)
    tl.store(dk + p, ak, mask=mask)
    tl.store(da + p, aa, mask=mask)
    tl.store(db + p, ab, mask=mask)
    tl.store(dg + p, ag, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nv, nc = triton.cdiv(V, BV), triton.cdiv(T, BT)
        out = torch.empty_like(v)
        hbuf = torch.empty((B * H, nc, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, hbuf,
            T=T, H=H, K=K, V=V, BK=bk, scale=K ** -0.5, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, hbuf)
        ctx.shape = B, T, H, K, V, bk, nv, nc
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk, hbuf = ctx.saved_tensors
        B, T, H, K, V, bk, nv, nc = ctx.shape
        do = do.contiguous()
        states = torch.empty((B * H, T, K, V), device=q.device, dtype=torch.float32)
        _expand_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, hbuf, states,
            T=T, H=H, K=K, V=V, BK=bk, num_warps=4, num_stages=2)
        dv = torch.empty_like(v)
        partial = torch.zeros((nv, B, T, H, 5 * K), device=q.device, dtype=torch.float32)
        _backward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, do, states, dv, partial,
            B=B, T=T, H=H, K=K, V=V, BK=bk, scale=K ** -0.5, num_warps=4, num_stages=2)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg,
            N=n, K=K, NV=nv, BLOCK=256, num_warps=8)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
