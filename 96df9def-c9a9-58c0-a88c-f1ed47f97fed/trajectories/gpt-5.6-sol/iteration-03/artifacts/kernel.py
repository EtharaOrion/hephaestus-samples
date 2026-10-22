import torch
import triton
import triton.language as tl

BT = 16
BV = 16
BWD_BV = 64


@triton.jit
def _chunk_fwd_kernel(q, k, v, a, b, g, out, checkpoints, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr, NT, BT_: tl.constexpr, SCALE: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov, rt = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_), tl.arange(0, BT_)
    state = tl.zeros((BK, BV_), tl.float32)
    base_k, base_v = (ib * T * H + ih) * K, (ib * T * H + ih) * V
    cp_base = ibh * (NT + 1) * K * V
    mkv = (ok[:, None] < K) & (ov[None, :] < V)
    tl.store(checkpoints + cp_base + ok[:, None] * V + ov[None, :], state, mask=mkv)
    for ic in range(NT):
        ot = ic * BT_ + rt
        mt = ot < T
        mk, mv = mt[:, None] & (ok[None, :] < K), mt[:, None] & (ov[None, :] < V)
        ik = base_k + ot[:, None] * H * K + ok[None, :]
        ivv = base_v + ot[:, None] * H * V + ov[None, :]
        qx = tl.load(q + ik, mask=mk, other=0.).to(tl.float32) * SCALE
        kx = tl.load(k + ik, mask=mk, other=0.).to(tl.float32)
        ax = tl.load(a + ik, mask=mk, other=0.).to(tl.float32)
        bx = tl.load(b + ik, mask=mk, other=0.).to(tl.float32)
        gx = tl.load(g + ik, mask=mk, other=0.).to(tl.float32)
        vx = tl.load(v + ivv, mask=mv, other=0.).to(tl.float32)
        gates = tl.cumsum(gx, axis=0)
        rel = tl.exp(gates[:, None, :] - gates[None, :, :])
        old_rel = tl.exp(gates[:, None, :] - gx[:, None, :] - gates[None, :, :])
        causal, strict = rt[:, None] >= rt[None, :], rt[:, None] > rt[None, :]
        Aqk = tl.where(causal, tl.sum(qx[:, None, :] * kx[None, :, :] * rel, axis=2), 0.)
        Aqb = tl.where(causal, tl.sum(qx[:, None, :] * bx[None, :, :] * rel, axis=2), 0.)
        Aak = tl.where(strict, tl.sum(ax[:, None, :] * kx[None, :, :] * old_rel, axis=2), 0.)
        Lab = tl.where(strict, tl.sum(ax[:, None, :] * bx[None, :, :] * old_rel, axis=2), 0.)
        inv = tl.where(rt[:, None] == rt[None, :], 1., 0.)
        for row in range(1, BT_):
            product = tl.dot(Lab, inv, input_precision="ieee")
            inv = tl.where(rt[:, None] == row, inv + product, inv)
        u = tl.dot(inv, tl.dot(Aak, vx, input_precision="ieee"), input_precision="ieee")
        w = tl.dot(inv, tl.exp(gates - gx) * ax, input_precision="ieee")
        vnew = u + tl.dot(w, state, input_precision="ieee")
        ox = tl.dot(qx * tl.exp(gates), state, input_precision="ieee")
        ox += tl.dot(Aqk, vx, input_precision="ieee") + tl.dot(Aqb, vnew, input_precision="ieee")
        tl.store(out + ivv, ox, mask=mv)
        next_state = state
        for j in range(BT_):
            active = j < tl.minimum(BT_, T - ic * BT_)
            rowmask = rt[:, None] == j
            aj = tl.sum(tl.where(rowmask, ax, 0.), axis=0)
            bj = tl.sum(tl.where(rowmask, bx, 0.), axis=0)
            kj = tl.sum(tl.where(rowmask, kx, 0.), axis=0)
            gj = tl.sum(tl.where(rowmask, gx, 0.), axis=0)
            vj = tl.sum(tl.where(rt[:, None] == j, vx, 0.), axis=0)
            read = tl.reshape(tl.dot(aj[None, :], next_state, input_precision="ieee"), (BV_,))
            candidate = next_state * tl.exp(gj)[:, None]
            candidate += kj[:, None] * vj[None, :]
            candidate += bj[:, None] * read[None, :]
            next_state = tl.where(active, candidate, next_state)
        state = next_state
        cp = checkpoints + cp_base + (ic + 1) * K * V + ok[:, None] * V + ov[None, :]
        tl.store(cp, state, mask=mkv)


@triton.jit
def _chunk_bwd_kernel(q, k, v, a, b, g, do, checkpoints, pq, pk, pv, pa, pb, pg, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr, NT, NV: tl.constexpr, BT_: tl.constexpr, SCALE: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_)
    base_k, base_v = (ib * T * H + ih) * K, (ib * T * H + ih) * V
    cp_base = ibh * (NT + 1) * K * V
    ds = tl.zeros((BK, BV_), tl.float32)
    for revc in range(NT):
        ic = NT - 1 - revc
        valid_n = tl.minimum(BT_, T - ic * BT_)
        cp0 = checkpoints + cp_base + ic * K * V + ok[:, None] * V + ov[None, :]
        sprev0 = tl.load(cp0, mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.).to(tl.float32)
        for revj in range(BT_):
            j = BT_ - 1 - revj
            active, ot = j < valid_n, ic * BT_ + j
            ik, ivv = base_k + ot * H * K + ok, base_v + ot * H * V + ov
            mk, mv = (ok < K) & active, (ov < V) & active
            qx = tl.load(q + ik, mask=mk, other=0.).to(tl.float32) * SCALE
            kx = tl.load(k + ik, mask=mk, other=0.).to(tl.float32)
            ax = tl.load(a + ik, mask=mk, other=0.).to(tl.float32)
            bx = tl.load(b + ik, mask=mk, other=0.).to(tl.float32)
            gx = tl.load(g + ik, mask=mk, other=0.).to(tl.float32)
            vx = tl.load(v + ivv, mask=mv, other=0.).to(tl.float32)
            dox = tl.load(do + ivv, mask=mv, other=0.).to(tl.float32)
            sprev = sprev0
            for p in range(BT_):
                use, otp = (p < j) & (p < valid_n), ic * BT_ + p
                ikp, ivp = base_k + otp * H * K + ok, base_v + otp * H * V + ov
                kp = tl.load(k + ikp, mask=(ok < K) & use, other=0.).to(tl.float32)
                ap = tl.load(a + ikp, mask=(ok < K) & use, other=0.).to(tl.float32)
                bp = tl.load(b + ikp, mask=(ok < K) & use, other=0.).to(tl.float32)
                gp = tl.load(g + ikp, mask=(ok < K) & use, other=0.).to(tl.float32)
                vp = tl.load(v + ivp, mask=(ov < V) & use, other=0.).to(tl.float32)
                readp = tl.reshape(tl.dot(ap[None, :], sprev, input_precision="ieee"), (BV_,))
                candidate = sprev * tl.exp(gp)[:, None] + kp[:, None] * vp[None, :] + bp[:, None] * readp[None, :]
                sprev = tl.where(use, candidate, sprev)
            decay = tl.exp(gx)
            read = tl.reshape(tl.dot(ax[None, :], sprev, input_precision="ieee"), (BV_,))
            st = sprev * decay[:, None] + kx[:, None] * vx[None, :] + bx[:, None] * read[None, :]
            ds += qx[:, None] * dox[None, :]
            dqv = tl.sum(st * dox[None, :], axis=1) * SCALE
            dkv = tl.sum(ds * vx[None, :], axis=1)
            dvv = tl.sum(kx[:, None] * ds, axis=0)
            dbv = tl.sum(ds * read[None, :], axis=1)
            dread = tl.reshape(tl.dot(bx[None, :], ds, input_precision="ieee"), (BV_,))
            dav = tl.sum(sprev * dread[None, :], axis=1)
            dgv = tl.sum(ds * sprev * decay[:, None], axis=1)
            ds = ds * decay[:, None] + ax[:, None] * dread[None, :]
            part = (iv * T * H + ib * T * H + ot * H + ih) * K
            tl.store(pq + part + ok, dqv, mask=mk)
            tl.store(pk + part + ok, dkv, mask=mk)
            tl.store(pa + part + ok, dav, mask=mk)
            tl.store(pb + part + ok, dbv, mask=mk)
            tl.store(pg + part + ok, dgv, mask=mk)
            tl.store(pv + (ib * T * H + ot * H + ih) * V + ov, dvv, mask=mv)


@triton.jit
def _reduce_parts_kernel(pq, pk, pa, pb, pg, dq, dk, da, db, dg, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    sq = tl.zeros((BLOCK,), tl.float32); sk = tl.zeros((BLOCK,), tl.float32)
    sa = tl.zeros((BLOCK,), tl.float32); sb = tl.zeros((BLOCK,), tl.float32); sg = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        off = iv * N + offs
        sq += tl.load(pq + off, mask=mask, other=0.); sk += tl.load(pk + off, mask=mask, other=0.)
        sa += tl.load(pa + off, mask=mask, other=0.); sb += tl.load(pb + off, mask=mask, other=0.)
        sg += tl.load(pg + off, mask=mask, other=0.)
    tl.store(dq + offs, sq, mask=mask); tl.store(dk + offs, sk, mask=mask)
    tl.store(da + offs, sa, mask=mask); tl.store(db + offs, sb, mask=mask); tl.store(dg + offs, sg, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape; V = v.shape[-1]
        BK, NT, NV = max(16, triton.next_power_of_2(K)), triton.cdiv(T, BT), triton.cdiv(V, BV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, H, NT + 1, K, V), device=q.device, dtype=torch.float32)
        _chunk_fwd_kernel[(NV, B * H)](q, k, v, alpha, beta, gk, out, checkpoints, T=T, H=H, K=K, V=V, BK=BK, BV_=BV, NT=NT, BT_=BT, SCALE=K ** -0.5, num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints); ctx.meta = B, T, H, K, V, BK, NT
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K, V, BK, NT = ctx.meta
        NV = triton.cdiv(V, BWD_BV)
        shape = (NV, B, T, H, K)
        pq, pk, pa, pb, pg = [torch.empty(shape, device=q.device, dtype=torch.float32) for _ in range(5)]
        dv = torch.empty_like(v)
        _chunk_bwd_kernel[(NV, B * H)](q, k, v, alpha, beta, gk, dout, checkpoints, pq, pk, dv, pa, pb, pg, T=T, H=H, K=K, V=V, BK=BK, BV_=BWD_BV, NT=NT, NV=NV, BT_=BT, SCALE=K ** -0.5, num_warps=4, num_stages=1)
        dq, dk, da, db, dg = [torch.empty_like(x) for x in (q, k, alpha, beta, gk)]
        n = B * T * H * K
        _reduce_parts_kernel[(triton.cdiv(n, 256),)](pq, pk, pa, pb, pg, dq, dk, da, db, dg, N=n, NV=NV, BLOCK=256, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
