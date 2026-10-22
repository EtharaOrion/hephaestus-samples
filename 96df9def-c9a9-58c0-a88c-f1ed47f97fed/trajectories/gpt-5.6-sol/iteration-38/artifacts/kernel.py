import torch
import triton
import triton.language as tl


_BT = 16
_FBV = 32
_BBV = 16


@triton.jit
def _fwd_kernel(q, k, v, a, b, g, out, checkpoints,
                T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                SCALE: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    vk = iv * BV + tl.arange(0, BV)
    kk = tl.arange(0, BK)
    tt = tl.arange(0, 16)
    mk = kk < K
    mv = vk < V
    state = tl.zeros((BK, BV), tl.float32)
    scale = SCALE
    for ic in range(NC):
        t = ic * 16 + tt
        mt = t < T
        base_k = (ib * T * H + t * H + ih) * K
        base_v = (ib * T * H + t * H + ih) * V
        q0 = tl.load(q + base_k[:, None] + kk[None, :], mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32) * scale
        k0 = tl.load(k + base_k[:, None] + kk[None, :], mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        a0 = tl.load(a + base_k[:, None] + kk[None, :], mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        b0 = tl.load(b + base_k[:, None] + kk[None, :], mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        g0 = tl.load(g + base_k[:, None] + kk[None, :], mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        v0 = tl.load(v + base_v[:, None] + vk[None, :], mask=mt[:, None] & mv[None, :], other=0.0).to(tl.float32)

        cp = tl.cumsum(g0, axis=0)
        cp_end = tl.sum(g0, axis=0)
        qg = q0 * tl.exp(cp)
        ag = a0 * tl.exp(cp - g0)
        ki = k0 * tl.exp(-cp)
        bi = b0 * tl.exp(-cp)
        kg = ki * tl.exp(cp_end)
        bg = bi * tl.exp(cp_end)

        lower = tt[:, None] > tt[None, :]
        causal = tt[:, None] >= tt[None, :]
        lab = tl.where(lower, tl.dot(ag, tl.trans(bi), input_precision="ieee"), 0.0)
        lak = tl.where(lower, tl.dot(ag, tl.trans(ki), input_precision="ieee"), 0.0)
        aqk = tl.where(causal, tl.dot(qg, tl.trans(ki), input_precision="ieee"), 0.0)
        aqb = tl.where(causal, tl.dot(qg, tl.trans(bi), input_precision="ieee"), 0.0)
        hp = checkpoints + ((((ib * NC + ic) * H + ih) * K + kk[:, None]) * V + vk[None, :])
        tl.store(hp, state, mask=mk[:, None] & mv[None, :])
        read = tl.dot(ag, state, input_precision="ieee")
        x = read + tl.dot(lak, v0, input_precision="ieee")
        for row in tl.static_range(1, 16):
            row_mask = tt == row
            xr = tl.sum(tl.where(row_mask[:, None], x, 0.0), axis=0)
            coeff = tl.sum(tl.where(row_mask[:, None], lab, 0.0), axis=0)
            xr += tl.sum(coeff[:, None] * x, axis=0)
            x = tl.where(row_mask[:, None], xr[None, :], x)
        y = tl.dot(qg, state, input_precision="ieee")
        y += tl.dot(aqk, v0, input_precision="ieee")
        y += tl.dot(aqb, x, input_precision="ieee")
        tl.store(out + base_v[:, None] + vk[None, :], y, mask=mt[:, None] & mv[None, :])
        state *= tl.exp(cp_end)[:, None]
        state += tl.dot(tl.trans(kg), v0, input_precision="ieee")
        state += tl.dot(tl.trans(bg), x, input_precision="ieee")

@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib = ibh // H
    ih = ibh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mk, mv = kk < K, vv < V
    hp = checkpoints + ((((ib * NC + ic) * H + ih) * K + kk[:, None]) * V + vv[None, :])
    state = tl.load(hp, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        bk = (ib * T * H + t * H + ih) * K
        bv = (ib * T * H + t * H + ih) * V
        sp = states + ((((ib * T + t) * H + ih) * K + kk[:, None]) * V + vv[None, :])
        tl.store(sp, state, mask=valid & mk[:, None] & mv[None, :])
        kt = tl.load(k + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        at = tl.load(a + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        bt = tl.load(b + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        gt = tl.load(g + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        vt = tl.load(v + bv + vv, mask=valid & mv, other=0.0).to(tl.float32)
        read = tl.sum(state * at[:, None], axis=0)
        state = state * tl.exp(gt)[:, None] + kt[:, None] * vt[None, :] + bt[:, None] * read[None, :]


@triton.jit
def _boundary_adjoint_kernel(q, a, b, g, do, states, boundaries,
                             T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                             NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                             SCALE: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mk, mv = kk < K, vv < V
    ds = tl.zeros((BK, BV), tl.float32)
    scale = SCALE
    for z in range(NC):
        ic = NC - 1 - z
        dp = boundaries + ((((ib * NC + ic) * H + ih) * K + kk[:, None]) * V + vv[None, :])
        tl.store(dp, ds, mask=mk[:, None] & mv[None, :])
        for jj in tl.static_range(0, 16):
            j = 15 - jj
            t = ic * 16 + j
            valid = t < T
            bk = (ib * T * H + t * H + ih) * K
            bv = (ib * T * H + t * H + ih) * V
            qt = tl.load(q + bk + kk, mask=valid & mk, other=0.0).to(tl.float32) * scale
            at = tl.load(a + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
            bt = tl.load(b + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
            gt = tl.load(g + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
            dot = tl.load(do + bv + vv, mask=valid & mv, other=0.0).to(tl.float32)
            ds += qt[:, None] * dot[None, :]
            dr = tl.sum(bt[:, None] * ds, axis=0)
            ds = ds * tl.exp(gt)[:, None] + at[:, None] * dr[None, :]


@triton.jit
def _backward_chunks_kernel(q, k, v, a, b, g, do, states, boundaries,
                            pq, pk, pa, pb, pg, dv,
                            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                            NC: tl.constexpr, NV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                            SCALE: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib = ibh // H
    ih = ibh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mk, mv = kk < K, vv < V
    dp = boundaries + ((((ib * NC + ic) * H + ih) * K + kk[:, None]) * V + vv[None, :])
    ds = tl.load(dp, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    scale = SCALE
    for jj in tl.static_range(0, 16):
        j = 15 - jj
        t = ic * 16 + j
        valid = t < T
        bk = (ib * T * H + t * H + ih) * K
        bv = (ib * T * H + t * H + ih) * V
        sp = states + ((((ib * T + t) * H + ih) * K + kk[:, None]) * V + vv[None, :])
        s0 = tl.load(sp, mask=valid & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
        qt = tl.load(q + bk + kk, mask=valid & mk, other=0.0).to(tl.float32) * scale
        kt = tl.load(k + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        at = tl.load(a + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        bt = tl.load(b + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        gt = tl.load(g + bk + kk, mask=valid & mk, other=0.0).to(tl.float32)
        vt = tl.load(v + bv + vv, mask=valid & mv, other=0.0).to(tl.float32)
        dot = tl.load(do + bv + vv, mask=valid & mv, other=0.0).to(tl.float32)
        decay = tl.exp(gt)
        read = tl.sum(s0 * at[:, None], axis=0)
        st = s0 * decay[:, None] + kt[:, None] * vt[None, :] + bt[:, None] * read[None, :]
        dq = tl.sum(st * dot[None, :], axis=1) * scale
        ds += qt[:, None] * dot[None, :]
        dk = tl.sum(ds * vt[None, :], axis=1)
        dvt = tl.sum(ds * kt[:, None], axis=0)
        db = tl.sum(ds * read[None, :], axis=1)
        dr = tl.sum(ds * bt[:, None], axis=0)
        da = tl.sum(s0 * dr[None, :], axis=1)
        dg = tl.sum(ds * (s0 * decay[:, None]), axis=1)
        po = (((iv * T + t) * H + ih) * K + kk)
        tl.store(pq + po, dq, mask=valid & mk)
        tl.store(pk + po, dk, mask=valid & mk)
        tl.store(pa + po, da, mask=valid & mk)
        tl.store(pb + po, db, mask=valid & mk)
        tl.store(pg + po, dg, mask=valid & mk)
        tl.store(dv + bv + vv, dvt, mask=valid & mv)
        ds = ds * decay[:, None] + at[:, None] * dr[None, :]


@triton.jit
def _reduce_kernel(src, dst, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        acc += tl.load(src + iv * N + off, mask=mask, other=0.0)
    tl.store(dst + off, acc, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _FBV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _fwd_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
                                  T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV,
                                  SCALE=K**-0.5, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        do = do.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        bv = _BBV
        nv = triton.cdiv(V, bv)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        boundaries = torch.empty_like(checkpoints)
        _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states,
                                                T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv,
                                                num_warps=4, num_stages=2)
        _boundary_adjoint_kernel[(nv, B * H)](q, alpha, beta, gk, do, states, boundaries,
                                               T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv,
                                               SCALE=K**-0.5, num_warps=4, num_stages=2)
        shape = (nv, B, T, H, K)
        pq = torch.zeros(shape, device=q.device, dtype=torch.float32)
        pk = torch.zeros_like(pq)
        pa = torch.zeros_like(pq)
        pb = torch.zeros_like(pq)
        pg = torch.zeros_like(pq)
        dv = torch.empty_like(v)
        _backward_chunks_kernel[(nv, nc, B * H)](q, k, v, alpha, beta, gk, do, states, boundaries,
                                                  pq, pk, pa, pb, pg, dv,
                                                  T=T, H=H, K=K, V=V, NC=nc, NV=nv, BK=bk, BV=bv,
                                                  SCALE=K**-0.5, num_warps=4, num_stages=2)
        n = B * T * H * K
        grads = []
        for src, like in ((pq, q), (pk, k), (pa, alpha), (pb, beta), (pg, gk)):
            dst = torch.empty_like(like)
            _reduce_kernel[(triton.cdiv(n, 256),)](src, dst, N=n, NV=nv, BLOCK=256,
                                                   num_warps=4, num_stages=2)
            grads.append(dst)
        dq, dk, da, db, dg = grads
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
