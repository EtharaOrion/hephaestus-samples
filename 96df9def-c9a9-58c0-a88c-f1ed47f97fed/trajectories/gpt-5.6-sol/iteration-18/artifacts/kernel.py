import torch
import triton
import triton.language as tl

_BT = 16
_BV = 16
_BWD_BV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr, SCALE: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ot, ok = tl.arange(0, 16), tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    for c in range(NC):
        cp = checkpoints + ((bh * NC + c) * K + ok[:, None]) * V + ov[None, :]
        tl.store(cp, state, mask=mk[:, None] & mv[None, :])
        tt = c * 16 + ot
        mt = tt < T
        qoff = (ib * T * H + tt[:, None] * H + ih) * K + ok[None, :]
        voff = (ib * T * H + tt[:, None] * H + ih) * V + ov[None, :]
        mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
        qq = tl.load(q + qoff, mask=mtk, other=0.).to(tl.float32)
        kk = tl.load(k + qoff, mask=mtk, other=0.).to(tl.float32)
        aa = tl.load(a + qoff, mask=mtk, other=0.).to(tl.float32)
        bb = tl.load(b + qoff, mask=mtk, other=0.).to(tl.float32)
        gg = tl.load(g + qoff, mask=mtk, other=0.).to(tl.float32)
        vv = tl.load(v + voff, mask=mtv, other=0.).to(tl.float32)
        raw_gi = tl.cumsum(gg, axis=0)
        gi = tl.maximum(raw_gi, -60.0)
        qg = qq * tl.exp(gi) * SCALE
        ag = aa * tl.exp(tl.maximum(raw_gi - gg, -60.0))
        inv_gate = tl.exp(tl.minimum(-raw_gi, 60.0))
        ki, bi = kk * inv_gate, bb * inv_gate
        strict = ot[:, None] > ot[None, :]
        causal = ot[:, None] >= ot[None, :]
        lab = tl.where(strict, tl.dot(ag, tl.trans(bi), input_precision="ieee"), 0.)
        lak = tl.where(strict, tl.dot(ag, tl.trans(ki), input_precision="ieee"), 0.)
        aqk = tl.where(causal, tl.dot(qg, tl.trans(ki), input_precision="ieee"), 0.)
        aqb = tl.where(causal, tl.dot(qg, tl.trans(bi), input_precision="ieee"), 0.)
        read = tl.dot(ag, state, input_precision="ieee") + tl.dot(lak, vv, input_precision="ieee")
        inv = tl.where(ot[:, None] == ot[None, :], 1.0, lab)
        power = lab
        for _ in tl.static_range(0, 3):
            power = tl.dot(power, power, input_precision="ieee")
            inv += tl.dot(power, inv, input_precision="ieee")
        solved = tl.dot(inv, read, input_precision="ieee")
        oo = tl.dot(qg, state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee")
        oo += tl.dot(aqb, solved, input_precision="ieee")
        tl.store(out + voff, oo, mask=mtv)
        last = tl.sum(tl.where(ot[:, None] == 15, gi, 0.), axis=0)
        state *= tl.exp(last)[:, None]
        state += tl.dot(tl.trans(ki * tl.exp(last)[None, :]), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bi * tl.exp(last)[None, :]), solved, input_precision="ieee")



@triton.jit
def _exact_checkpoints_kernel(k, v, a, b, g, checkpoints,
                              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                              V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                              BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    for c in range(NC):
        cp = checkpoints + ((bh * NC + c) * K + ok[:, None]) * V + ov[None, :]
        tl.store(cp, state, mask=mk[:, None] & mv[None, :])
        for j in tl.static_range(0, 16):
            t, valid = c * 16 + j, c * 16 + j < T
            x = ib * T * H + t * H + ih
            kk = tl.load(k + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            aa = tl.load(a + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            bb = tl.load(b + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            gg = tl.load(g + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            vv = tl.load(v + x * V + ov, mask=valid & mv, other=0.).to(tl.float32)
            read = tl.sum(aa[:, None] * state, axis=0)
            state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                          BV: tl.constexpr):
    iv, c, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    cp = checkpoints + ((bh * NC + c) * K + ok[:, None]) * V + ov[None, :]
    state = tl.load(cp, mask=mk[:, None] & mv[None, :], other=0.)
    for j in tl.static_range(0, 16):
        t, valid = c * 16 + j, c * 16 + j < T
        sp = states + ((bh * T + t) * K + ok[:, None]) * V + ov[None, :]
        tl.store(sp, state, mask=valid & mk[:, None] & mv[None, :])
        x = ib * T * H + t * H + ih
        kk = tl.load(k + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
        aa = tl.load(a + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
        bb = tl.load(b + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
        gg = tl.load(g + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
        vv = tl.load(v + x * V + ov, mask=valid & mv, other=0.).to(tl.float32)
        read = tl.sum(aa[:, None] * state, axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _dstate_boundaries_kernel(q, a, b, g, do, boundaries,
                              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                              V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                              BV: tl.constexpr, SCALE: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BV), tl.float32)
    for zc in range(NC):
        c = NC - 1 - zc
        bp = boundaries + ((bh * NC + c) * K + ok[:, None]) * V + ov[None, :]
        tl.store(bp, ds, mask=mk[:, None] & mv[None, :])
        for zj in tl.static_range(0, 16):
            t, valid = c * 16 + 15 - zj, c * 16 + 15 - zj < T
            x = ib * T * H + t * H + ih
            qq = tl.load(q + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            aa = tl.load(a + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            bb = tl.load(b + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            gg = tl.load(g + x * K + ok, mask=valid & mk, other=0.).to(tl.float32)
            dov = tl.load(do + x * V + ov, mask=valid & mv, other=0.).to(tl.float32)
            ds += qq[:, None] * (dov[None, :] * SCALE)
            br = tl.sum(bb[:, None] * ds, axis=0)
            ds = ds * tl.exp(gg)[:, None] + aa[:, None] * br[None, :]


@triton.jit
def _backward_chunks_kernel(q, k, v, a, b, g, do, states, boundaries,
                            dq_part, dk_part, da_part, db_part, dg_part, dv,
                            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                            V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                            BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    iv, c, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    bp = boundaries + ((bh * NC + c) * K + ok[:, None]) * V + ov[None, :]
    ds = tl.load(bp, mask=mk[:, None] & mv[None, :], other=0.)
    for zj in tl.static_range(0, 16):
        t, valid = c * 16 + 15 - zj, c * 16 + 15 - zj < T
        x = ib * T * H + t * H + ih
        qoff, voff = x * K + ok, x * V + ov
        qq = tl.load(q + qoff, mask=valid & mk, other=0.).to(tl.float32)
        kk = tl.load(k + qoff, mask=valid & mk, other=0.).to(tl.float32)
        aa = tl.load(a + qoff, mask=valid & mk, other=0.).to(tl.float32)
        bb = tl.load(b + qoff, mask=valid & mk, other=0.).to(tl.float32)
        gg = tl.load(g + qoff, mask=valid & mk, other=0.).to(tl.float32)
        vv = tl.load(v + voff, mask=valid & mv, other=0.).to(tl.float32)
        dov = tl.load(do + voff, mask=valid & mv, other=0.).to(tl.float32)
        sp = states + ((bh * T + t) * K + ok[:, None]) * V + ov[None, :]
        prev = tl.load(sp, mask=valid & mk[:, None] & mv[None, :], other=0.)
        read = tl.sum(aa[:, None] * prev, axis=0)
        now = prev * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        dqv = tl.sum(now * dov[None, :], axis=1) * SCALE
        ds += qq[:, None] * (dov[None, :] * SCALE)
        br = tl.sum(bb[:, None] * ds, axis=0)
        dkv = tl.sum(ds * vv[None, :], axis=1)
        dbv = tl.sum(ds * read[None, :], axis=1)
        dav = tl.sum(prev * br[None, :], axis=1)
        dgv = tl.sum(ds * prev, axis=1) * tl.exp(gg)
        dvv = tl.sum(ds * kk[:, None], axis=0)
        po = (iv * T * H + x) * K + ok
        tl.store(dq_part + po, dqv, mask=valid & mk)
        tl.store(dk_part + po, dkv, mask=valid & mk)
        tl.store(da_part + po, dav, mask=valid & mk)
        tl.store(db_part + po, dbv, mask=valid & mk)
        tl.store(dg_part + po, dgv, mask=valid & mk)
        tl.store(dv + voff, dvv, mask=valid & mv)
        ds = ds * tl.exp(gg)[:, None] + aa[:, None] * br[None, :]


@triton.jit
def _reduce_partials_kernel(src, dst, N: tl.constexpr, NV: tl.constexpr,
                            BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        acc += tl.load(src + iv * N + off, mask=mask, other=0.)
    tl.store(dst + off, acc, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc, bk, nv = triton.cdiv(T, _BT), max(16, triton.next_power_of_2(K)), triton.cdiv(V, _BV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, H, nc, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
                                     T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV,
                                     SCALE=K ** -0.5, num_warps=8, num_stages=2)
        _exact_checkpoints_kernel[(nv, B * H)](k, v, alpha, beta, gk, checkpoints,
                                               T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV,
                                               num_warps=8, num_stages=2)
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
        nv = triton.cdiv(V, _BWD_BV)
        states = torch.empty((B, H, T, K, V), device=q.device, dtype=torch.float32)
        boundaries = torch.empty((B, H, nc, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states,
                                               T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV,
                                               num_warps=8, num_stages=2)
        _dstate_boundaries_kernel[(nv, B * H)](q, alpha, beta, gk, do, boundaries,
                                               T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV,
                                               SCALE=K ** -0.5, num_warps=8, num_stages=2)
        n = B * T * H * K
        parts = [torch.zeros((nv, B, T, H, K), device=q.device, dtype=torch.float32) for _ in range(5)]
        dv = torch.empty_like(v)
        _backward_chunks_kernel[(nv, nc, B * H)](q, k, v, alpha, beta, gk, do, states, boundaries,
                                                 *parts, dv, T=T, H=H, K=K, V=V, NC=nc, NV=nv,
                                                 BK=bk, BV=_BWD_BV, SCALE=K ** -0.5,
                                                 num_warps=8, num_stages=2)
        grads = [torch.empty_like(q) for _ in range(5)]
        for src, dst in zip(parts, grads):
            _reduce_partials_kernel[(triton.cdiv(n, 256),)](src, dst, N=n, NV=nv, BLOCK=256,
                                                            num_warps=4)
        return grads[0], grads[1], dv, grads[2], grads[3], grads[4]


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
