
import torch
import triton
import triton.language as tl

_BT = 16
_BV = 32


@triton.jit
def _chunk_forward_kernel(
    q, k, v, a, b, g, out, checkpoints,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    rt = tl.arange(0, 16)
    mk = rk < K
    mv = rv < V
    state = tl.zeros((BK, BV), tl.float32)

    for ic in range(NC):
        cp = checkpoints + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(cp, state, mask=mk[:, None] & mv[None, :])

        ts = ic * 16 + rt
        mt = ts < T
        xbase = (ib * T * H + ts * H + ih) * K
        vbase = (ib * T * H + ts * H + ih) * V
        mask_tk = mt[:, None] & mk[None, :]
        mask_tv = mt[:, None] & mv[None, :]
        xq = tl.load(q + xbase[:, None] + rk[None, :], mask=mask_tk, other=0.0).to(tl.float32) * SCALE
        xk = tl.load(k + xbase[:, None] + rk[None, :], mask=mask_tk, other=0.0).to(tl.float32)
        xa = tl.load(a + xbase[:, None] + rk[None, :], mask=mask_tk, other=0.0).to(tl.float32)
        xb = tl.load(b + xbase[:, None] + rk[None, :], mask=mask_tk, other=0.0).to(tl.float32)
        xg = tl.load(g + xbase[:, None] + rk[None, :], mask=mask_tk, other=0.0).to(tl.float32)
        xv = tl.load(v + vbase[:, None] + rv[None, :], mask=mask_tv, other=0.0).to(tl.float32)
        G = tl.cumsum(xg, axis=0)
        Gprev = G - xg

        Aak = tl.zeros((16, 16), tl.float32)
        Aab = tl.zeros((16, 16), tl.float32)
        Aqk = tl.zeros((16, 16), tl.float32)
        Aqb = tl.zeros((16, 16), tl.float32)
        for j in range(16):
            pick = rt == j
            kj = tl.sum(tl.where(pick[:, None], xk, 0.0), axis=0)
            bj = tl.sum(tl.where(pick[:, None], xb, 0.0), axis=0)
            Gj = tl.sum(tl.where(pick[:, None], G, 0.0), axis=0)
            before = rt > j
            through = rt >= j
            e_read = tl.exp(tl.minimum(Gprev - Gj[None, :], 0.0))
            e_out = tl.exp(tl.minimum(G - Gj[None, :], 0.0))
            col_ak = tl.sum(xa * kj[None, :] * e_read, axis=1) * before
            col_ab = tl.sum(xa * bj[None, :] * e_read, axis=1) * before
            col_qk = tl.sum(xq * kj[None, :] * e_out, axis=1) * through
            col_qb = tl.sum(xq * bj[None, :] * e_out, axis=1) * through
            Aak += col_ak[:, None] * pick[None, :]
            Aab += col_ab[:, None] * pick[None, :]
            Aqk += col_qk[:, None] * pick[None, :]
            Aqb += col_qb[:, None] * pick[None, :]

        ag = xa * tl.exp(Gprev)
        reads = tl.dot(ag, state, input_precision="ieee")
        reads += tl.dot(Aak, xv, input_precision="ieee")
        for j in range(1, 16):
            pick = rt == j
            row = tl.sum(tl.where(pick[:, None], Aab, 0.0), axis=0)
            corr = tl.sum(row[:, None] * reads, axis=0)
            reads += pick[:, None] * corr[None, :]

        y = tl.dot(xq * tl.exp(G), state, input_precision="ieee")
        y += tl.dot(Aqk, xv, input_precision="ieee")
        y += tl.dot(Aqb, reads, input_precision="ieee")
        tl.store(out + vbase[:, None] + rv[None, :], y, mask=mask_tv)

        valid_count = tl.minimum(16, T - ic * 16)
        last = valid_count - 1
        last_pick = rt == last
        Gend = tl.sum(tl.where(last_pick[:, None], G, 0.0), axis=0)
        kg = xk * tl.exp(tl.minimum(Gend[None, :] - G, 0.0))
        bg = xb * tl.exp(tl.minimum(Gend[None, :] - G, 0.0))
        state *= tl.exp(Gend)[:, None]
        state += tl.dot(tl.trans(kg), xv, input_precision="ieee")
        state += tl.dot(tl.trans(bg), reads, input_precision="ieee")


@triton.jit
def _expand_states_kernel(
    k, v, a, b, g, checkpoints, states,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    iv = tl.program_id(0)
    icbh = tl.program_id(1)
    ibh = icbh % (tl.num_programs(1) // NC)
    ic = icbh // (tl.num_programs(1) // NC)
    ib = ibh // H
    ih = ibh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mk = rk < K
    mv = rv < V
    cp = checkpoints + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
    state = tl.load(cp, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    for j in range(16):
        t = ic * 16 + j
        valid = t < T
        xb = (ib * T * H + t * H + ih) * K
        vb = (ib * T * H + t * H + ih) * V
        xk = tl.load(k + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
        xa = tl.load(a + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
        xbeta = tl.load(b + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
        xg = tl.load(g + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
        xv = tl.load(v + vb + rv, mask=valid & mv, other=0.0).to(tl.float32)
        read = tl.sum(xa[:, None] * state, axis=0)
        state = state * tl.exp(xg)[:, None]
        state += xk[:, None] * xv[None, :] + xbeta[:, None] * read[None, :]
        sp = states + ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(sp, state, mask=valid & mk[:, None] & mv[None, :])


@triton.jit
def _reverse_chunks_kernel(
    q, k, v, a, b, g, dout, checkpoints, states,
    partial, dv,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mk = rk < K
    mv = rv < V
    dstate = tl.zeros((BK, BV), tl.float32)

    for revc in range(NC):
        ic = NC - 1 - revc
        for revj in range(16):
            j = 16 - 1 - revj
            t = ic * 16 + j
            valid = t < T
            xb = (ib * T * H + t * H + ih) * K
            vb = (ib * T * H + t * H + ih) * V
            xq = tl.load(q + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
            xk = tl.load(k + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
            xa = tl.load(a + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
            xbeta = tl.load(b + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
            xg = tl.load(g + xb + rk, mask=valid & mk, other=0.0).to(tl.float32)
            xv = tl.load(v + vb + rv, mask=valid & mv, other=0.0).to(tl.float32)
            do = tl.load(dout + vb + rv, mask=valid & mv, other=0.0).to(tl.float32)
            snp = states + ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
            snew = tl.load(snp, mask=valid & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
            if j == 0:
                sop = checkpoints + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
            else:
                sop = states + ((((ib * T + t - 1) * H + ih) * K + rk[:, None]) * V + rv[None, :])
            sold = tl.load(sop, mask=valid & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)

            dq = tl.sum(snew * do[None, :], axis=1) * SCALE
            dcur = dstate + xq[:, None] * (do * SCALE)[None, :]
            read = tl.sum(xa[:, None] * sold, axis=0)
            dk = tl.sum(dcur * xv[None, :], axis=1)
            dbeta = tl.sum(dcur * read[None, :], axis=1)
            dr = tl.sum(xbeta[:, None] * dcur, axis=0)
            dalpha = tl.sum(sold * dr[None, :], axis=1)
            decay = tl.exp(xg)
            dg = tl.sum(dcur * (sold * decay[:, None]), axis=1)
            dvi = tl.sum(xk[:, None] * dcur, axis=0)
            dstate = dcur * decay[:, None] + xa[:, None] * dr[None, :]

            n = tl.num_programs(1) * T * K
            elem = ((ib * T + t) * H + ih) * K + rk
            base = iv * 5 * n
            tl.store(partial + base + 0 * n + elem, dq, mask=valid & mk)
            tl.store(partial + base + 1 * n + elem, dk, mask=valid & mk)
            tl.store(partial + base + 2 * n + elem, dalpha, mask=valid & mk)
            tl.store(partial + base + 3 * n + elem, dbeta, mask=valid & mk)
            tl.store(partial + base + 4 * n + elem, dg, mask=valid & mk)
            tl.store(dv + vb + rv, dvi, mask=valid & mv)


@triton.jit
def _reduce_partials_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc0 = tl.zeros((BLOCK,), tl.float32)
    acc1 = tl.zeros((BLOCK,), tl.float32)
    acc2 = tl.zeros((BLOCK,), tl.float32)
    acc3 = tl.zeros((BLOCK,), tl.float32)
    acc4 = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        base = iv * 5 * N
        acc0 += tl.load(partial + base + 0 * N + offs, mask=mask, other=0.0)
        acc1 += tl.load(partial + base + 1 * N + offs, mask=mask, other=0.0)
        acc2 += tl.load(partial + base + 2 * N + offs, mask=mask, other=0.0)
        acc3 += tl.load(partial + base + 3 * N + offs, mask=mask, other=0.0)
        acc4 += tl.load(partial + base + 4 * N + offs, mask=mask, other=0.0)
    tl.store(dq + offs, acc0, mask=mask)
    tl.store(dk + offs, acc1, mask=mask)
    tl.store(da + offs, acc2, mask=mask)
    tl.store(db + offs, acc3, mask=mask)
    tl.store(dg + offs, acc4, mask=mask)


def _launch_forward(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    nc = triton.cdiv(T, _BT)
    bk = max(16, triton.next_power_of_2(K))
    nv = triton.cdiv(V, _BV)
    out = torch.empty_like(v)
    checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
    _chunk_forward_kernel[(nv, B * H)](
        q, k, v, alpha, beta, gk, out, checkpoints,
        T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV,
        SCALE=K ** -0.5, num_warps=8, num_stages=2,
    )
    return out, checkpoints


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        out, checkpoints = _launch_forward(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BV)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc * B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV,
            num_warps=8, num_stages=2,
        )
        n = B * T * H * K
        partial = torch.empty((nv, 5, B, T, H, K), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _reverse_chunks_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout.contiguous(), checkpoints, states,
            partial, dv, T=T, H=H, K=K, V=V, NC=nc, NV=nv,
            BK=bk, BV=_BV, SCALE=K ** -0.5, num_warps=8, num_stages=2,
        )
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gk)
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](
            partial, dq, dk, da, db, dg, N=n, NV=nv, BLOCK=256,
            num_warps=8,
        )
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
