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
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    nc = tl.cdiv(T, 16)
    ok, ov, ot = tl.arange(0, BK), iv * BV + tl.arange(0, BV), tl.arange(0, 16)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(K * 1.0)
    for ic in range(nc):
        hp = checkpoints + ((ib * nc + ic) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
        tl.store(hp, state, mask=mk[:, None] & mv[None, :])
        tt = ic * 16 + ot
        mt = tt < T
        xk = ((ib * T + tt[:, None]) * H + ih) * K + ok[None, :]
        xv = ((ib * T + tt[:, None]) * H + ih) * V + ov[None, :]
        mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
        qq = tl.load(q + xk, mask=mtk, other=0.0).to(tl.float32) * scale
        kk = tl.load(k + xk, mask=mtk, other=0.0).to(tl.float32)
        aa = tl.load(a + xk, mask=mtk, other=0.0).to(tl.float32)
        bb = tl.load(b + xk, mask=mtk, other=0.0).to(tl.float32)
        gg = tl.load(g + xk, mask=mtk, other=0.0).to(tl.float32)
        vv = tl.load(v + xv, mask=mtv, other=0.0).to(tl.float32)
        gc = tl.cumsum(gg, axis=0)
        gp = gc - gg
        last = tl.minimum(T - ic * 16, 16) - 1
        gend = tl.sum(tl.where(ot[:, None] == last, gc, 0.0), axis=0)
        qg = qq * tl.exp(tl.minimum(gc, 0.0))
        ag = aa * tl.exp(tl.minimum(gp, 0.0))
        kg = kk * tl.exp(tl.minimum(gend[None, :] - gc, 0.0))
        bg = bb * tl.exp(tl.minimum(gend[None, :] - gc, 0.0))
        row, col = ot[:, None], ot[None, :]
        causal, strict = row >= col, row > col
        eq = tl.where(causal[:, :, None], tl.exp(tl.minimum(gc[:, None, :] - gc[None, :, :], 0.0)), 0.0)
        ea = tl.where(strict[:, :, None], tl.exp(tl.minimum(gp[:, None, :] - gc[None, :, :], 0.0)), 0.0)
        aqk = tl.sum(qq[:, None, :] * kk[None, :, :] * eq, axis=2)
        aqb = tl.sum(qq[:, None, :] * bb[None, :, :] * eq, axis=2)
        aak = tl.sum(aa[:, None, :] * kk[None, :, :] * ea, axis=2)
        lab = tl.sum(aa[:, None, :] * bb[None, :, :] * ea, axis=2)
        valid = mt[:, None] & mt[None, :]
        aqk, aqb = tl.where(causal & valid, aqk, 0.0), tl.where(causal & valid, aqb, 0.0)
        aak, lab = tl.where(strict & valid, aak, 0.0), tl.where(strict & valid, lab, 0.0)
        rhs = tl.dot(ag, state, input_precision="ieee") + tl.dot(aak, vv, input_precision="ieee")
        xx = rhs
        for j in tl.static_range(1, 16):
            pick = ot == j
            lr = tl.sum(tl.where(pick[:, None], lab, 0.0), axis=0)
            rr = tl.sum(tl.where(pick[:, None], rhs, 0.0), axis=0) + tl.sum(lr[:, None] * xx, axis=0)
            xx = tl.where(pick[:, None], rr[None, :], xx)
        oo = tl.dot(qg, state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, xx, input_precision="ieee")
        op = out + ((ib * T + tt[:, None]) * H + ih) * V + ov[None, :]
        tl.store(op, oo, mask=mtv)
        state *= tl.exp(tl.minimum(gend, 0.0))[:, None]
        state += tl.dot(tl.trans(kg), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bg), xx, input_precision="ieee")


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    nc = tl.cdiv(T, 16)
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    hp = checkpoints + ((ib * nc + ic) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
    state = tl.load(hp, mask=mk[:, None] & mv[None, :], other=0.0)
    for j in tl.static_range(0, 16):
        tt = ic * 16 + j
        valid = tt < T
        sp = states + ((ib * T + tt) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
        tl.store(sp, state, mask=valid & mk[:, None] & mv[None, :])
        xk = ((ib * T + tt) * H + ih) * K + ok
        xv = ((ib * T + tt) * H + ih) * V + ov
        kk = tl.load(k + xk, mask=valid & mk, other=0.0).to(tl.float32)
        aa = tl.load(a + xk, mask=valid & mk, other=0.0).to(tl.float32)
        bb = tl.load(b + xk, mask=valid & mk, other=0.0).to(tl.float32)
        gg = tl.load(g + xk, mask=valid & mk, other=0.0).to(tl.float32)
        vv = tl.load(v + xv, mask=valid & mv, other=0.0).to(tl.float32)
        read = tl.sum(aa[:, None] * state, axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, partial, dv,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    nbh = tl.num_programs(1)
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(K * 1.0)
    n = T * nbh * K
    nc = tl.cdiv(T, 16)
    for rc in range(nc):
        ic = nc - 1 - rc
        for j in tl.static_range(0, 16):
            tt = ic * 16 + 15 - j
            valid = tt < T
            xk = ((ib * T + tt) * H + ih) * K + ok
            xv = ((ib * T + tt) * H + ih) * V + ov
            sp = states + ((ib * T + tt) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
            sold = tl.load(sp, mask=valid & mk[:, None] & mv[None, :], other=0.0)
            qq = tl.load(q + xk, mask=valid & mk, other=0.0).to(tl.float32) * scale
            kk = tl.load(k + xk, mask=valid & mk, other=0.0).to(tl.float32)
            aa = tl.load(a + xk, mask=valid & mk, other=0.0).to(tl.float32)
            bb = tl.load(b + xk, mask=valid & mk, other=0.0).to(tl.float32)
            decay = tl.exp(tl.load(g + xk, mask=valid & mk, other=0.0).to(tl.float32))
            vv = tl.load(v + xv, mask=valid & mv, other=0.0).to(tl.float32)
            doval = tl.load(do + xv, mask=valid & mv, other=0.0).to(tl.float32)
            read = tl.sum(aa[:, None] * sold, axis=0)
            snew = sold * decay[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
            pdq = tl.sum(snew * doval[None, :], axis=1) * scale
            dsc = ds + qq[:, None] * doval[None, :]
            pdk = tl.sum(dsc * vv[None, :], axis=1)
            pdv = tl.sum(dsc * kk[:, None], axis=0)
            pdb = tl.sum(dsc * read[None, :], axis=1)
            dr = tl.sum(dsc * bb[:, None], axis=0)
            pda = tl.sum(sold * dr[None, :], axis=1)
            pdg = tl.sum(dsc * sold, axis=1) * decay
            pos = (tt * nbh + ibh) * K + ok
            tl.store(partial + iv * 5 * n + pos, pdq, mask=valid & mk)
            tl.store(partial + iv * 5 * n + n + pos, pdk, mask=valid & mk)
            tl.store(partial + iv * 5 * n + 2 * n + pos, pda, mask=valid & mk)
            tl.store(partial + iv * 5 * n + 3 * n + pos, pdb, mask=valid & mk)
            tl.store(partial + iv * 5 * n + 4 * n + pos, pdg, mask=valid & mk)
            tl.store(dv + xv, pdv, mask=valid & mv)
            ds = tl.where(valid, dsc * decay[:, None] + aa[:, None] * dr[None, :], ds)


@triton.jit
def _reduce_kernel(partial, dst, N: tl.constexpr, NV: tl.constexpr):
    offs = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offs < N
    total = tl.zeros((256,), tl.float32)
    for iv in tl.static_range(0, NV):
        total += tl.load(partial + iv * 5 * N + offs, mask=mask, other=0.0)
    tl.store(dst + offs, total, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V, bk = v.shape[-1], max(16, triton.next_power_of_2(K))
        nc, nv = triton.cdiv(T, _BT), triton.cdiv(V, _BV)
        out = torch.empty((B, T, H, V), device=q.device, dtype=v.dtype)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BV, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V, bk = v.shape[-1], max(16, triton.next_power_of_2(K))
        nc, nv = triton.cdiv(T, _BT), triton.cdiv(V, _BWD_BV)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BWD_BV, num_warps=4, num_stages=2)
        n = B * T * H * K
        partial = torch.empty((nv, 5, n), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _backward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, dout, states, partial, dv,
            T=T, H=H, K=K, V=V, BK=bk, BV=_BWD_BV, num_warps=8, num_stages=2)
        grads = [torch.empty_like(q), torch.empty_like(k), torch.empty_like(alpha),
                 torch.empty_like(beta), torch.empty_like(gk)]
        for i, dst in enumerate(grads):
            _reduce_kernel[(triton.cdiv(n, 256),)](partial[:, i], dst, N=n, NV=nv, num_warps=8)
        return grads[0], grads[1], dv, grads[2], grads[3], grads[4]


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
