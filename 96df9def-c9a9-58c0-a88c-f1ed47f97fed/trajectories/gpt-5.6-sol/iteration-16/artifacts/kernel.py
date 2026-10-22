import torch
import triton
import triton.language as tl

_BT = 16
_BV = 32


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(-60.0, tl.minimum(60.0, x)))


@triton.jit
def _forward_kernel(
    q, k, v, a, b, g, out, states,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    tk = tl.arange(0, BK)
    tv = iv * BV + tl.arange(0, BV)
    mtv = tv < V
    state = tl.zeros((BK, BV), tl.float32)
    scale = SCALE
    rows = tl.arange(0, 16)
    causal = rows[:, None] >= rows[None, :]
    strict = rows[:, None] > rows[None, :]
    eye = rows[:, None] == rows[None, :]

    for ic in range(NC):
        ts = ic * 16 + rows
        valid = ts < T
        xoff = ((ib * T + ts[:, None]) * H + ih) * K + tk[None, :]
        voff = ((ib * T + ts[:, None]) * H + ih) * V + tv[None, :]
        mk = valid[:, None] & (tk[None, :] < K)
        mv = valid[:, None] & mtv[None, :]
        qq = tl.load(q + xoff, mask=mk, other=0.0).to(tl.float32) * scale
        kk = tl.load(k + xoff, mask=mk, other=0.0).to(tl.float32)
        aa = tl.load(a + xoff, mask=mk, other=0.0).to(tl.float32)
        bb = tl.load(b + xoff, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + xoff, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + voff, mask=mv, other=0.0).to(tl.float32)

        gi = tl.cumsum(gg, axis=0)
        qg = qq * _bounded_exp(gi)
        kg = kk * _bounded_exp(-gi)
        bg = bb * _bounded_exp(-gi)
        ag = aa * _bounded_exp(gi - gg)

        lab = tl.where(strict, tl.dot(ag, tl.trans(bg), input_precision="ieee"), 0.0)
        lak = tl.where(strict, tl.dot(ag, tl.trans(kg), input_precision="ieee"), 0.0)
        for i in tl.static_range(1, 16):
            row = tl.sum(tl.where(rows[:, None] == i, lab, 0.0), axis=0)
            row += tl.sum(row[:, None] * lab, axis=0) * (rows < i)
            lab = tl.where(rows[:, None] == i, row[None, :], lab)
        inv = lab + eye
        w = tl.dot(inv, ag, input_precision="ieee")
        ucoef = tl.dot(inv, lak, input_precision="ieee")
        read = tl.dot(w, state, input_precision="ieee")
        read += tl.dot(ucoef, vv, input_precision="ieee")
        aqk = tl.where(causal, tl.dot(qg, tl.trans(kg), input_precision="ieee"), 0.0)
        aqb = tl.where(causal, tl.dot(qg, tl.trans(bg), input_precision="ieee"), 0.0)
        oo = tl.dot(qg, state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee")
        oo += tl.dot(aqb, read, input_precision="ieee")
        tl.store(out + voff, oo, mask=mv)

        replay = state
        for j in tl.static_range(16):
            pick = rows == j
            gj = tl.sum(tl.where(pick[:, None], gg, 0.0), axis=0)
            kj = tl.sum(tl.where(pick[:, None], kk, 0.0), axis=0)
            aj = tl.sum(tl.where(pick[:, None], aa, 0.0), axis=0)
            bj = tl.sum(tl.where(pick[:, None], bb, 0.0), axis=0)
            vj = tl.sum(tl.where(pick[:, None], vv, 0.0), axis=0)
            z = tl.sum(replay * aj[:, None], axis=0)
            replay = replay * _bounded_exp(gj)[:, None] + kj[:, None] * vj[None, :] + bj[:, None] * z[None, :]
            t = ic * 16 + j
            soff = ((((ib * T + t) * H + ih) * K + tk[:, None]) * V + tv[None, :]).to(tl.int64)
            tl.store(states + soff, replay, mask=(t < T) & (tk[:, None] < K) & mtv[None, :])

        last = tl.minimum((ic + 1) * 16, T) - ic * 16 - 1
        glast = tl.sum(tl.where(rows[:, None] == last, gi, 0.0), axis=0)
        state += tl.dot(tl.trans(kg), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bg), read, input_precision="ieee")
        state *= _bounded_exp(glast)[:, None]


@triton.jit
def _backward_kernel(
    q, k, v, a, b, g, do, states, dv, partial,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr, SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    tk = tl.arange(0, BK)
    tv = iv * BV + tl.arange(0, BV)
    mk = tk < K
    mv = tv < V
    dstate = tl.zeros((BK, BV), tl.float32)
    scale = SCALE

    for rc in range(NC):
        ic = NC - 1 - rc
        for rj in tl.static_range(16):
            j = 15 - rj
            t = ic * 16 + j
            xoff = ((ib * T + t) * H + ih) * K + tk
            voff = ((ib * T + t) * H + ih) * V + tv
            valid = t < T
            qx = tl.load(q + xoff, mask=valid & mk, other=0.0).to(tl.float32)
            kx = tl.load(k + xoff, mask=valid & mk, other=0.0).to(tl.float32)
            ax = tl.load(a + xoff, mask=valid & mk, other=0.0).to(tl.float32)
            bx = tl.load(b + xoff, mask=valid & mk, other=0.0).to(tl.float32)
            gx = tl.load(g + xoff, mask=valid & mk, other=0.0).to(tl.float32)
            vx = tl.load(v + voff, mask=valid & mv, other=0.0).to(tl.float32)
            dox = tl.load(do + voff, mask=valid & mv, other=0.0).to(tl.float32)
            soff = ((((ib * T + t) * H + ih) * K + tk[:, None]) * V + tv[None, :]).to(tl.int64)
            st = tl.load(states + soff, mask=valid & mk[:, None] & mv[None, :], other=0.0)
            pt = t - 1
            poff = ((((ib * T + pt) * H + ih) * K + tk[:, None]) * V + tv[None, :]).to(tl.int64)
            sp = tl.load(states + poff, mask=(pt >= 0) & mk[:, None] & mv[None, :], other=0.0)

            dq = tl.sum(st * dox[None, :], axis=1) * scale
            ds = dstate + qx[:, None] * (dox[None, :] * scale)
            z = tl.sum(sp * ax[:, None], axis=0)
            dk = tl.sum(ds * vx[None, :], axis=1)
            db = tl.sum(ds * z[None, :], axis=1)
            dz = tl.sum(ds * bx[:, None], axis=0)
            da = tl.sum(sp * dz[None, :], axis=1)
            dg = tl.sum(ds * (sp * _bounded_exp(gx)[:, None]), axis=1)
            dvx = tl.sum(ds * kx[:, None], axis=0)
            tl.store(dv + voff, dvx, mask=valid & mv)

            n = (ib * T + t) * H + ih
            pbase = ((iv * (T * H * (tl.num_programs(1) // H)) + n) * 5 * K).to(tl.int64)
            tl.store(partial + pbase + 0 * K + tk, dq, mask=valid & mk)
            tl.store(partial + pbase + 1 * K + tk, dk, mask=valid & mk)
            tl.store(partial + pbase + 2 * K + tk, da, mask=valid & mk)
            tl.store(partial + pbase + 3 * K + tk, db, mask=valid & mk)
            tl.store(partial + pbase + 4 * K + tk, dg, mask=valid & mk)
            dstate = ds * _bounded_exp(gx)[:, None] + ax[:, None] * dz[None, :]
            dstate = tl.where(valid, dstate, dstate)


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr, K: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N * K
    n = off // K
    x = off - n * K
    r0 = tl.zeros((BLOCK,), tl.float32)
    r1 = tl.zeros((BLOCK,), tl.float32)
    r2 = tl.zeros((BLOCK,), tl.float32)
    r3 = tl.zeros((BLOCK,), tl.float32)
    r4 = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        base = ((iv * N + n) * 5 * K + x).to(tl.int64)
        r0 += tl.load(partial + base + 0 * K, mask=mask, other=0.0)
        r1 += tl.load(partial + base + 1 * K, mask=mask, other=0.0)
        r2 += tl.load(partial + base + 2 * K, mask=mask, other=0.0)
        r3 += tl.load(partial + base + 3 * K, mask=mask, other=0.0)
        r4 += tl.load(partial + base + 4 * K, mask=mask, other=0.0)
    tl.store(dq + off, r0, mask=mask)
    tl.store(dk + off, r1, mask=mask)
    tl.store(da + off, r2, mask=mask)
    tl.store(db + off, r3, mask=mask)
    tl.store(dg + off, r4, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BV)
        nc = triton.cdiv(T, _BT)
        out = torch.empty_like(v)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, states,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BV, NC=nc, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, states)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, states = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BV)
        nc = triton.cdiv(T, _BT)
        n = B * T * H
        dv = torch.empty_like(v)
        partial = torch.empty((nv, n, 5, K), device=q.device, dtype=torch.float32)
        _backward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout.contiguous(), states, dv, partial,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BV, NC=nc, NV=nv, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gk)
        _reduce_kernel[(triton.cdiv(n * K, 256),)](
            partial, dq, dk, da, db, dg, N=n, K=K, NV=nv, BLOCK=256,
            num_warps=4,
        )
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
