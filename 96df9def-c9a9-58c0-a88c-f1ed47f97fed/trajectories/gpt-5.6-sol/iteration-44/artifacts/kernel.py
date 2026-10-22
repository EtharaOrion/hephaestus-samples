import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 8


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(tl.minimum(x, 40.0), -40.0))


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    rt = tl.arange(0, 16)
    state = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(K * 1.0)

    for ic in range(NC):
        t = ic * 16 + rt
        mt = t < T
        ok = ((ib * T + t[:, None]) * H + ih) * K + rk[None, :]
        ov = ((ib * T + t[:, None]) * H + ih) * V + rv[None, :]
        mk = mt[:, None] & (rk[None, :] < K)
        mv = mt[:, None] & (rv[None, :] < V)
        hs = ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(checkpoints + hs, state,
                 mask=(rk[:, None] < K) & (rv[None, :] < V))

        gg = tl.load(g + ok, mask=mk, other=0.0).to(tl.float32)
        z = tl.cumsum(gg, axis=0)
        e = z - gg
        qq = tl.load(q + ok, mask=mk, other=0.0).to(tl.float32)
        kk = tl.load(k + ok, mask=mk, other=0.0).to(tl.float32)
        aa = tl.load(a + ok, mask=mk, other=0.0).to(tl.float32)
        bb = tl.load(b + ok, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + ov, mask=mv, other=0.0).to(tl.float32)

        qg = qq * _bounded_exp(z) * scale
        ag = aa * _bounded_exp(e)
        bg = bb * _bounded_exp(-z)
        kg = kk * _bounded_exp(-z)
        rows = rt[:, None]
        cols = rt[None, :]
        strict = rows > cols
        causal = rows >= cols
        lab = tl.dot(ag, tl.trans(bg), input_precision="ieee")
        lak = tl.dot(ag, tl.trans(kg), input_precision="ieee")
        aqk = tl.dot(qg, tl.trans(kg), input_precision="ieee")
        aqb = tl.dot(qg, tl.trans(bg), input_precision="ieee")
        lab = tl.where(strict & mt[:, None] & mt[None, :], lab, 0.0)
        lak = tl.where(strict & mt[:, None] & mt[None, :], lak, 0.0)
        aqk = tl.where(causal & mt[:, None] & mt[None, :], aqk, 0.0)
        aqb = tl.where(causal & mt[:, None] & mt[None, :], aqb, 0.0)

        rhs = tl.dot(ag, state, input_precision="ieee")
        rhs += tl.dot(lak, vv, input_precision="ieee")
        u = tl.zeros((16, BV), tl.float32)
        for j in tl.static_range(0, 16):
            rj = tl.sum(tl.where(rt[:, None] == j, rhs, 0.0), axis=0)
            lj = tl.sum(tl.where(rt[:, None] == j, lab, 0.0), axis=0)
            uj = rj + tl.sum(lj[:, None] * u, axis=0)
            u = tl.where(rt[:, None] == j, uj[None, :], u)

        oo = tl.dot(qg, state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee")
        oo += tl.dot(aqb, u, input_precision="ieee")
        tl.store(out + ov, oo, mask=mv)

        state += tl.dot(tl.trans(kg), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bg), u, input_precision="ieee")
        last = tl.maximum(tl.minimum(T - ic * 16, 16), 1) - 1
        zl = tl.sum(tl.where(rt[:, None] == last, z, 0.0), axis=0)
        state *= _bounded_exp(zl)[:, None]


@triton.jit
def _exact_checkpoints_kernel(k, v, a, b, g, checkpoints,
                              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                              V: tl.constexpr, NC: tl.constexpr,
                              BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mask = (rk[:, None] < K) & (rv[None, :] < V)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        hs = ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(checkpoints + hs, state, mask=mask)
        for j in tl.static_range(0, 16):
            t = ic * 16 + j
            valid = t < T
            ok = ((ib * T + t) * H + ih) * K + rk
            ov = ((ib * T + t) * H + ih) * V + rv
            kk = tl.load(k + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
            aa = tl.load(a + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
            bb = tl.load(b + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
            gg = tl.load(g + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
            vv = tl.load(v + ov, mask=valid & (rv < V), other=0.0).to(tl.float32)
            read = tl.sum(aa[:, None] * state, axis=0)
            state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]



@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, NC: tl.constexpr,
                          BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    hs = ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
    mask = (rk[:, None] < K) & (rv[None, :] < V)
    state = tl.load(checkpoints + hs, mask=mask, other=0.0)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        ok = ((ib * T + t) * H + ih) * K + rk
        ov = ((ib * T + t) * H + ih) * V + rv
        ss = ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(states + ss, state, mask=valid & mask)
        kk = tl.load(k + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        aa = tl.load(a + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        bb = tl.load(b + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        gg = tl.load(g + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        vv = tl.load(v + ov, mask=valid & (rv < V), other=0.0).to(tl.float32)
        read = tl.sum(aa[:, None] * state, axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, boundaries,
                     dq, dv, pk, pa, pb, pg,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, NC: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mask = (rk[:, None] < K) & (rv[None, :] < V)
    hb = ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
    ds = tl.load(boundaries + hb, mask=mask, other=0.0)
    scale = 1.0 / tl.sqrt(K * 1.0)
    for jr in tl.static_range(0, 16):
        j = 15 - jr
        t = ic * 16 + j
        valid = t < T
        ok = ((ib * T + t) * H + ih) * K + rk
        ov = ((ib * T + t) * H + ih) * V + rv
        ss = ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        old = tl.load(states + ss, mask=valid & mask, other=0.0)
        qq = tl.load(q + ok, mask=valid & (rk < K), other=0.0).to(tl.float32) * scale
        kk = tl.load(k + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        aa = tl.load(a + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        bb = tl.load(b + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        gg = tl.load(g + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
        vv = tl.load(v + ov, mask=valid & (rv < V), other=0.0).to(tl.float32)
        dd = tl.load(do + ov, mask=valid & (rv < V), other=0.0).to(tl.float32)
        eg = tl.exp(gg)
        read = tl.sum(aa[:, None] * old, axis=0)
        new = old * eg[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        dq0 = tl.sum(new * dd[None, :], axis=1) * scale
        ds += qq[:, None] * dd[None, :]
        dv0 = tl.sum(kk[:, None] * ds, axis=0)
        dk0 = tl.sum(ds * vv[None, :], axis=1)
        db0 = tl.sum(ds * read[None, :], axis=1)
        dy = tl.sum(bb[:, None] * ds, axis=0)
        da0 = tl.sum(old * dy[None, :], axis=1)
        dg0 = tl.sum(ds * old * eg[:, None], axis=1)
        po = iv * (T * H * K * tl.num_programs(2) // H) + ((ib * T + t) * H + ih) * K + rk
        tl.store(dq + po, dq0, mask=valid & (rk < K))
        tl.store(pk + po, dk0, mask=valid & (rk < K))
        tl.store(pa + po, da0, mask=valid & (rk < K))
        tl.store(pb + po, db0, mask=valid & (rk < K))
        tl.store(pg + po, dg0, mask=valid & (rk < K))
        tl.store(dv + ov, dv0, mask=valid & (rv < V))
        ds = ds * eg[:, None] + aa[:, None] * dy[None, :]


@triton.jit
def _reduce_kernel(parts, dst, N: tl.constexpr, NV: tl.constexpr,
                   BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        val += tl.load(parts + iv * N + x, mask=x < N, other=0.0)
    tl.store(dst + x, val, mask=x < N)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        nv = triton.cdiv(V, _FBV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        h = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, h,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV,
            num_warps=8, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, h)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk, h = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bv = _BBV
        nv = triton.cdiv(V, bv)
        bk = max(16, triton.next_power_of_2(K))
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        bounds = torch.empty_like(h)
        do = do.contiguous()
        _exact_checkpoints_kernel[(nv, B * H)](
            k, v, alpha, beta, gk, h,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv,
            num_warps=4, num_stages=2,
        )
        _expand_states_kernel[(nv, nc, B * H)](
            k, v, alpha, beta, gk, h, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv,
            num_warps=4, num_stages=2,
        )
        _boundary_exact_kernel[(nv, B * H)](
            q, alpha, beta, gk, do, bounds,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv,
            num_warps=4, num_stages=2,
        )
        n = B * T * H * K
        parts = [torch.zeros((nv, n), device=q.device, dtype=torch.float32) for _ in range(5)]
        dq, dk, da, db, dg = [torch.empty_like(q) for _ in range(5)]
        dv = torch.empty_like(v)
        _backward_kernel[(nv, nc, B * H)](
            q, k, v, alpha, beta, gk, do, states, bounds,
            parts[0], dv, parts[1], parts[2], parts[3], parts[4],
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv,
            num_warps=4, num_stages=2,
        )
        for src, dst in zip(parts, (dq, dk, da, db, dg)):
            _reduce_kernel[(triton.cdiv(n, 256),)](src, dst, N=n, NV=nv, BLOCK=256)
        return dq, dk, dv, da, db, dg


@triton.jit
def _boundary_exact_kernel(q, a, b, g, do, boundaries,
                           T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                           V: tl.constexpr, NC: tl.constexpr,
                           BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mask = (rk[:, None] < K) & (rv[None, :] < V)
    ds = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(K * 1.0)
    for ir in range(NC):
        ic = NC - 1 - ir
        hb = ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(boundaries + hb, ds, mask=mask)
        for jr in tl.static_range(0, 16):
            t = ic * 16 + 15 - jr
            valid = t < T
            ok = ((ib * T + t) * H + ih) * K + rk
            ov = ((ib * T + t) * H + ih) * V + rv
            qq = tl.load(q + ok, mask=valid & (rk < K), other=0.0).to(tl.float32) * scale
            aa = tl.load(a + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
            bb = tl.load(b + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
            gg = tl.load(g + ok, mask=valid & (rk < K), other=0.0).to(tl.float32)
            dd = tl.load(do + ov, mask=valid & (rv < V), other=0.0).to(tl.float32)
            ds += qq[:, None] * dd[None, :]
            dy = tl.sum(bb[:, None] * ds, axis=0)
            ds = ds * tl.exp(gg)[:, None] + aa[:, None] * dy[None, :]


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
