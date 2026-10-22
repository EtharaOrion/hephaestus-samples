import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BT = 16
_BV = 32
_BWD_BV = 32


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                    SCALE: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    ot = tl.arange(0, 16)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    for ic in range(NC):
        ts = ic * 16 + ot
        mt = ts < T
        mtk = mt[:, None] & mk[None, :]
        mtv = mt[:, None] & mv[None, :]
        ptk = base_k + ts[:, None] * H * K + ok[None, :]
        ptv = base_v + ts[:, None] * H * V + ov[None, :]
        cp = (((ib * NC + ic) * H + ih) * K) * V
        tl.store(checkpoints + cp + ok[:, None] * V + ov[None, :], state,
                 mask=mk[:, None] & mv[None, :])
        qq = tl.load(q + ptk, mask=mtk, other=0.0).to(tl.float32)
        kk = tl.load(k + ptk, mask=mtk, other=0.0).to(tl.float32)
        aa = tl.load(a + ptk, mask=mtk, other=0.0).to(tl.float32)
        bb = tl.load(b + ptk, mask=mtk, other=0.0).to(tl.float32)
        gg = tl.load(g + ptk, mask=mtk, other=0.0).to(tl.float32)
        vv = tl.load(v + ptv, mask=mtv, other=0.0).to(tl.float32)
        gi = tl.cumsum(gg, axis=0)
        ge = gi - gg
        valid_n = tl.minimum(T - ic * 16, 16)
        glast = tl.sum(tl.where((ot == valid_n - 1)[:, None], gi, 0.0), axis=0)
        center = tl.sum(tl.where((ot == valid_n // 2)[:, None], gi, 0.0), axis=0)
        qop = qq * libdevice.exp(gi - center[None, :]) * SCALE
        kop = kk * libdevice.exp(center[None, :] - gi)
        aop = aa * libdevice.exp(ge - center[None, :])
        bop = bb * libdevice.exp(center[None, :] - gi)
        valid2 = mt[:, None] & mt[None, :]
        lower_eq = ot[:, None] >= ot[None, :]
        lower = ot[:, None] > ot[None, :]
        aqk = tl.where(lower_eq & valid2, tl.dot(qop, tl.trans(kop), input_precision="ieee"), 0.0)
        aqb = tl.where(lower_eq & valid2, tl.dot(qop, tl.trans(bop), input_precision="ieee"), 0.0)
        lab = tl.where(lower & valid2, tl.dot(aop, tl.trans(bop), input_precision="ieee"), 0.0)
        lak = tl.where(lower & valid2, tl.dot(aop, tl.trans(kop), input_precision="ieee"), 0.0)
        eye = (ot[:, None] == ot[None, :]) & valid2
        inv = tl.where(eye, 1.0, 0.0) + lab
        power = tl.dot(lab, lab, input_precision="ieee")
        inv += tl.dot(power, inv, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv += tl.dot(power, inv, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv += tl.dot(power, inv, input_precision="ieee")
        u = tl.dot(inv, tl.dot(lak, vv, input_precision="ieee"), input_precision="ieee")
        w = tl.dot(inv, aop * libdevice.exp(center[None, :]), input_precision="ieee")
        vnew = u + tl.dot(w, state, input_precision="ieee")
        oo = tl.dot(qq * libdevice.exp(gi) * SCALE, state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, vnew, input_precision="ieee")
        tl.store(out + ptv, oo, mask=mtv)
        kg = kk * libdevice.exp(glast[None, :] - gi)
        bg = bb * libdevice.exp(glast[None, :] - gi)
        state = state * libdevice.exp(glast[:, None])
        state += tl.dot(tl.trans(kg), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bg), vnew, input_precision="ieee")


@triton.jit
def _checkpoint_kernel(k, v, a, b, g, checkpoints,
                       T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    for ic in range(NC):
        cp = (((ib * NC + ic) * H + ih) * K) * V
        tl.store(checkpoints + cp + ok[:, None] * V + ov[None, :], state,
                 mask=mk[:, None] & mv[None, :])
        for j in tl.static_range(0, 16):
            t = ic * 16 + j
            valid = t < T
            pk = base_k + t * H * K + ok
            pv = base_v + t * H * V + ov
            kk = tl.load(k + pk, mask=valid & mk, other=0.0).to(tl.float32)
            aa = tl.load(a + pk, mask=valid & mk, other=0.0).to(tl.float32)
            bb = tl.load(b + pk, mask=valid & mk, other=0.0).to(tl.float32)
            gg = tl.load(g + pk, mask=valid & mk, other=0.0).to(tl.float32)
            vv = tl.load(v + pv, mask=valid & mv, other=0.0).to(tl.float32)
            prod = state * aa[:, None]
            read = tl.sum(tl.where(ok[:, None] < 32, prod, 0.0), axis=0)
            read += tl.sum(tl.where((ok[:, None] >= 32) & (ok[:, None] < 64), prod, 0.0), axis=0)
            read += tl.sum(tl.where((ok[:, None] >= 64) & (ok[:, None] < 96), prod, 0.0), axis=0)
            read += tl.sum(tl.where(ok[:, None] >= 96, prod, 0.0), axis=0)
            updated = state * libdevice.exp(gg[:, None]) + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
            state = tl.where(valid, updated, state)



@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    cp = (((ib * NC + ic) * H + ih) * K) * V
    state = tl.load(checkpoints + cp + ok[:, None] * V + ov[None, :],
                    mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        pk = base_k + t * H * K + ok
        pv = base_v + t * H * V + ov
        kk = tl.load(k + pk, mask=valid & mk, other=0.0).to(tl.float32)
        aa = tl.load(a + pk, mask=valid & mk, other=0.0).to(tl.float32)
        bb = tl.load(b + pk, mask=valid & mk, other=0.0).to(tl.float32)
        gg = tl.load(g + pk, mask=valid & mk, other=0.0).to(tl.float32)
        vv = tl.load(v + pv, mask=valid & mv, other=0.0).to(tl.float32)
        prod = state * aa[:, None]
        read = tl.sum(tl.where(ok[:, None] < 32, prod, 0.0), axis=0)
        read += tl.sum(tl.where((ok[:, None] >= 32) & (ok[:, None] < 64), prod, 0.0), axis=0)
        read += tl.sum(tl.where((ok[:, None] >= 64) & (ok[:, None] < 96), prod, 0.0), axis=0)
        read += tl.sum(tl.where(ok[:, None] >= 96, prod, 0.0), axis=0)
        updated = state * libdevice.exp(gg[:, None]) + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        state = tl.where(valid, updated, state)
        ps = (((ib * T + t) * H + ih) * K) * V
        tl.store(states + ps + ok[:, None] * V + ov[None, :], state,
                 mask=valid & mk[:, None] & mv[None, :])


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states,
                     dq_part, dk_part, da_part, db_part, dg_part, dv,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                     SCALE: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    batches = tl.num_programs(1) // H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BV), tl.float32)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    for rc in range(NC):
        ic = NC - 1 - rc
        for rj in tl.static_range(0, 16):
            t = ic * 16 + 15 - rj
            valid = t < T
            pk = base_k + t * H * K + ok
            pv = base_v + t * H * V + ov
            qq = tl.load(q + pk, mask=valid & mk, other=0.0).to(tl.float32)
            kk = tl.load(k + pk, mask=valid & mk, other=0.0).to(tl.float32)
            aa = tl.load(a + pk, mask=valid & mk, other=0.0).to(tl.float32)
            bb = tl.load(b + pk, mask=valid & mk, other=0.0).to(tl.float32)
            gg = tl.load(g + pk, mask=valid & mk, other=0.0).to(tl.float32)
            vv = tl.load(v + pv, mask=valid & mv, other=0.0).to(tl.float32)
            dd = tl.load(do + pv, mask=valid & mv, other=0.0).to(tl.float32)
            ps = (((ib * T + t) * H + ih) * K) * V
            st = tl.load(states + ps + ok[:, None] * V + ov[None, :],
                         mask=valid & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
            pprev = (((ib * T + t - 1) * H + ih) * K) * V
            sprev = tl.load(states + pprev + ok[:, None] * V + ov[None, :],
                            mask=valid & (t > 0) & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
            ds_now = ds + qq[:, None] * (dd[None, :] * SCALE)
            read_prod = sprev * aa[:, None]
            read = tl.sum(tl.where(ok[:, None] < 32, read_prod, 0.0), axis=0)
            read += tl.sum(tl.where(ok[:, None] >= 32, read_prod, 0.0), axis=0)
            dread_prod = ds_now * bb[:, None]
            dread = tl.sum(tl.where(ok[:, None] < 32, dread_prod, 0.0), axis=0)
            dread += tl.sum(tl.where(ok[:, None] >= 32, dread_prod, 0.0), axis=0)
            gdq = tl.sum(st * dd[None, :], axis=1) * SCALE
            gdk = tl.sum(ds_now * vv[None, :], axis=1)
            gda = tl.sum(sprev * dread[None, :], axis=1)
            gdb = tl.sum(ds_now * read[None, :], axis=1)
            gdg = tl.sum(ds_now * sprev * libdevice.exp(gg[:, None]), axis=1)
            gdv = tl.sum(ds_now * kk[:, None], axis=0)
            part = ((((iv * batches + ib) * T + t) * H + ih) * K)
            tl.store(dq_part + part + ok, gdq, mask=valid & mk)
            tl.store(dk_part + part + ok, gdk, mask=valid & mk)
            tl.store(da_part + part + ok, gda, mask=valid & mk)
            tl.store(db_part + part + ok, gdb, mask=valid & mk)
            tl.store(dg_part + part + ok, gdg, mask=valid & mk)
            tl.store(dv + pv, gdv, mask=valid & mv)
            ds_prev = ds_now * libdevice.exp(gg[:, None]) + aa[:, None] * dread[None, :]
            ds = tl.where(valid, ds_prev, ds)


@triton.jit
def _reduce_kernel(src, dst, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        acc += tl.load(src + iv * N + offs, mask=mask, other=0.0)
    tl.store(dst + offs, acc, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
                                      T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV,
                                      SCALE=K ** -0.5, num_warps=8, num_stages=2)
        _checkpoint_kernel[(nv, B * H)](k, v, alpha, beta, gk, checkpoints,
                                        T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV,
                                        num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        ctx.dims = (B, T, H, K, V, nc, bk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K, V, nc, bk = ctx.dims
        nv = triton.cdiv(V, _BWD_BV)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states,
                                                T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV,
                                                num_warps=4, num_stages=2)
        shape = (nv, B, T, H, K)
        parts = [torch.empty(shape, device=q.device, dtype=torch.float32) for _ in range(5)]
        dv = torch.empty_like(v)
        _backward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, dout.contiguous(), states,
                                       *parts, dv, T=T, H=H, K=K, V=V, NC=nc, BK=bk,
                                       BV=_BWD_BV, SCALE=K ** -0.5, num_warps=4, num_stages=2)
        n = B * T * H * K
        likes = (q, k, alpha, beta, gk)
        grads = []
        for src, like in zip(parts, likes):
            dst = torch.empty_like(like)
            _reduce_kernel[(triton.cdiv(n, 256),)](src, dst, N=n, NV=nv, BLOCK=256,
                                                   num_warps=4, num_stages=2)
            grads.append(dst)
        dq, dk, da, db, dg = grads
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
