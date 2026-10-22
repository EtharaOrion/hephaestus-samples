import torch
import triton
import triton.language as tl


BT = 16
BV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BK: tl.constexpr, SCALE: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ov = iv * 16 + tl.arange(0, 16)
    ok = tl.arange(0, BK)
    ot = tl.arange(0, 16)
    mv = ov < V
    mk = ok < K
    state = tl.zeros((BK, 16), tl.float32)
    nc = tl.cdiv(T, 16)

    for ic in range(nc):
        ts = ic * 16 + ot
        mt = ts < T
        ptk = ((ib * T + ts[:, None]) * H + ih) * K + ok[None, :]
        ptv = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        mtk = mt[:, None] & mk[None, :]
        mtv = mt[:, None] & mv[None, :]
        qq = tl.load(q + ptk, mask=mtk, other=0.0).to(tl.float32) * SCALE
        kk = tl.load(k + ptk, mask=mtk, other=0.0).to(tl.float32)
        aa = tl.load(a + ptk, mask=mtk, other=0.0).to(tl.float32)
        bb = tl.load(b + ptk, mask=mtk, other=0.0).to(tl.float32)
        gg = tl.load(g + ptk, mask=mtk, other=0.0).to(tl.float32)
        vv = tl.load(v + ptv, mask=mtv, other=0.0).to(tl.float32)
        cg = tl.cumsum(gg, axis=0)
        pg = cg - gg

        hp = (((ibh * nc + ic) * K + ok[:, None]) * V + ov[None, :])
        tl.store(checkpoints + hp, state, mask=mk[:, None] & mv[None, :])

        Aqk = tl.zeros((16, 16), tl.float32)
        Aqb = tl.zeros((16, 16), tl.float32)
        Aak = tl.zeros((16, 16), tl.float32)
        Aab = tl.zeros((16, 16), tl.float32)
        for j in tl.static_range(0, 16):
            pick = ot == j
            kj = tl.sum(tl.where(pick[:, None], kk, 0.0), axis=0)
            bj = tl.sum(tl.where(pick[:, None], bb, 0.0), axis=0)
            cj = tl.sum(tl.where(pick[:, None], cg, 0.0), axis=0)
            validj = ic * 16 + j < T
            dqi = tl.exp(tl.minimum(cg - cj[None, :], 0.0))
            dai = tl.exp(tl.minimum(pg - cj[None, :], 0.0))
            qi_ok = (ot >= j) & mt & validj
            ai_ok = (ot > j) & mt & validj
            col = tl.arange(0, 16) == j
            xqk = tl.sum(qq * kj[None, :] * dqi, axis=1)
            xqb = tl.sum(qq * bj[None, :] * dqi, axis=1)
            xak = tl.sum(aa * kj[None, :] * dai, axis=1)
            xab = tl.sum(aa * bj[None, :] * dai, axis=1)
            Aqk += tl.where(col[None, :] & qi_ok[:, None], xqk[:, None], 0.0)
            Aqb += tl.where(col[None, :] & qi_ok[:, None], xqb[:, None], 0.0)
            Aak += tl.where(col[None, :] & ai_ok[:, None], xak[:, None], 0.0)
            Aab += tl.where(col[None, :] & ai_ok[:, None], xab[:, None], 0.0)

        z = tl.zeros((16, 16), tl.float32)
        for j in tl.static_range(0, 16):
            row = ot == j
            aj = tl.sum(tl.where(row[:, None], aa, 0.0), axis=0)
            pj = tl.sum(tl.where(row[:, None], pg, 0.0), axis=0)
            zj = tl.sum(aj[:, None] * tl.exp(pj)[:, None] * state, axis=0)
            for l in tl.static_range(0, 16):
                col = ot == l
                cak = tl.sum(tl.where(row[:, None] & col[None, :], Aak, 0.0))
                cab = tl.sum(tl.where(row[:, None] & col[None, :], Aab, 0.0))
                vl = tl.sum(tl.where(col[:, None], vv, 0.0), axis=0)
                zl = tl.sum(tl.where(col[:, None], z, 0.0), axis=0)
                zj += cak * vl + cab * zl
            z = tl.where(row[:, None], zj[None, :], z)

        oo = tl.sum((qq * tl.exp(cg))[:, :, None] * state[None, :, :], axis=1)
        oo += tl.dot(Aqk, vv, input_precision="ieee")
        oo += tl.dot(Aqb, z, input_precision="ieee")
        po = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        tl.store(out + po, oo, mask=mtv)

        last = tl.minimum(T - ic * 16, 16) - 1
        ce = tl.sum(tl.where((ot == last)[:, None], cg, 0.0), axis=0)
        end_decay = tl.exp(ce)
        rel = tl.exp(ce[None, :] - cg)
        kd = kk * rel
        bd = bb * rel
        state = state * end_decay[:, None]
        state += tl.dot(tl.trans(kd), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bd), z, input_precision="ieee")


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          BK: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ov = iv * 16 + tl.arange(0, 16)
    ok = tl.arange(0, BK)
    mv = ov < V
    mk = ok < K
    nc = tl.cdiv(T, 16)
    state = tl.zeros((BK, 16), tl.float32)
    for ic in range(nc):
        for j in tl.static_range(0, 16):
            t = ic * 16 + j
            valid = t < T
            sp = (((ibh * T + t) * K + ok[:, None]) * V + ov[None, :])
            tl.store(states + sp, state, mask=mk[:, None] & mv[None, :] & valid)
            pk = ((ib * T + t) * H + ih) * K + ok
            pv = ((ib * T + t) * H + ih) * V + ov
            kk = tl.load(k + pk, mask=mk & valid, other=0.0).to(tl.float32)
            aa = tl.load(a + pk, mask=mk & valid, other=0.0).to(tl.float32)
            bb = tl.load(b + pk, mask=mk & valid, other=0.0).to(tl.float32)
            gg = tl.load(g + pk, mask=mk & valid, other=0.0).to(tl.float32)
            vv = tl.load(v + pv, mask=mv & valid, other=0.0).to(tl.float32)
            read = tl.sum(state * aa[:, None], axis=0)
            state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states,
                     partial, dv,
                     T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     BK: tl.constexpr, SCALE: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ov = iv * 16 + tl.arange(0, 16)
    ok = tl.arange(0, BK)
    ot = tl.arange(0, 16)
    mv = ov < V
    mk = ok < K
    nc = tl.cdiv(T, 16)
    ds = tl.zeros((BK, 16), tl.float32)

    for ri in range(nc):
        ic = nc - 1 - ri
        ts = ic * 16 + ot
        mt = ts < T
        ptk = ((ib * T + ts[:, None]) * H + ih) * K + ok[None, :]
        ptv = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        mtk = mt[:, None] & mk[None, :]
        mtv = mt[:, None] & mv[None, :]
        qq = tl.load(q + ptk, mask=mtk, other=0.0).to(tl.float32) * SCALE
        kk = tl.load(k + ptk, mask=mtk, other=0.0).to(tl.float32)
        aa = tl.load(a + ptk, mask=mtk, other=0.0).to(tl.float32)
        bb = tl.load(b + ptk, mask=mtk, other=0.0).to(tl.float32)
        gg = tl.load(g + ptk, mask=mtk, other=0.0).to(tl.float32)
        vv = tl.load(v + ptv, mask=mtv, other=0.0).to(tl.float32)
        dd = tl.load(do + ptv, mask=mtv, other=0.0).to(tl.float32)

        for rj in tl.static_range(0, 16):
            j = 15 - rj
            pick = ot == j
            valid = ic * 16 + j < T
            qj = tl.sum(tl.where(pick[:, None], qq, 0.0), axis=0)
            kj = tl.sum(tl.where(pick[:, None], kk, 0.0), axis=0)
            aj = tl.sum(tl.where(pick[:, None], aa, 0.0), axis=0)
            bj = tl.sum(tl.where(pick[:, None], bb, 0.0), axis=0)
            gj = tl.sum(tl.where(pick[:, None], gg, 0.0), axis=0)
            vj = tl.sum(tl.where(pick[:, None], vv, 0.0), axis=0)
            doj = tl.sum(tl.where(pick[:, None], dd, 0.0), axis=0)
            sp = (((ibh * T + (ic * 16 + j)) * K + ok[:, None]) * V + ov[None, :])
            old = tl.load(states + sp, mask=mk[:, None] & mv[None, :] & valid, other=0.0)

            decay = tl.exp(gj)
            read = tl.sum(old * aj[:, None], axis=0)
            new = old * decay[:, None] + kj[:, None] * vj[None, :] + bj[:, None] * read[None, :]
            total = ds + qj[:, None] * doj[None, :]
            dqj = tl.sum(new * doj[None, :], axis=1) * SCALE
            dkj = tl.sum(total * vj[None, :], axis=1)
            dvj = tl.sum(total * kj[:, None], axis=0)
            dbj = tl.sum(total * read[None, :], axis=1)
            dread = tl.sum(total * bj[:, None], axis=0)
            daj = tl.sum(old * dread[None, :], axis=1)
            dgj = tl.sum(total * old, axis=1) * decay
            ds = total * decay[:, None] + aj[:, None] * dread[None, :]

            flat = ((ib * T + (ic * 16 + j)) * H + ih) * K + ok
            n = T * H * K
            fulln = tl.num_programs(1) // H * n
            tl.store(partial + (iv * 5 + 0) * fulln + flat, dqj, mask=mk & valid)
            tl.store(partial + (iv * 5 + 1) * fulln + flat, dkj, mask=mk & valid)
            tl.store(partial + (iv * 5 + 2) * fulln + flat, daj, mask=mk & valid)
            tl.store(partial + (iv * 5 + 3) * fulln + flat, dbj, mask=mk & valid)
            tl.store(partial + (iv * 5 + 4) * fulln + flat, dgj, mask=mk & valid)
            vflat = ((ib * T + (ic * 16 + j)) * H + ih) * V + ov
            tl.store(dv + vflat, dvj, mask=mv & valid)


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr, NV: tl.constexpr,
                   BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x0 = tl.zeros((BLOCK,), tl.float32)
    x1 = tl.zeros((BLOCK,), tl.float32)
    x2 = tl.zeros((BLOCK,), tl.float32)
    x3 = tl.zeros((BLOCK,), tl.float32)
    x4 = tl.zeros((BLOCK,), tl.float32)
    for iv in tl.static_range(0, NV):
        x0 += tl.load(partial + (iv * 5 + 0) * N + offs, mask=mask, other=0.0)
        x1 += tl.load(partial + (iv * 5 + 1) * N + offs, mask=mask, other=0.0)
        x2 += tl.load(partial + (iv * 5 + 2) * N + offs, mask=mask, other=0.0)
        x3 += tl.load(partial + (iv * 5 + 3) * N + offs, mask=mask, other=0.0)
        x4 += tl.load(partial + (iv * 5 + 4) * N + offs, mask=mask, other=0.0)
    tl.store(dq + offs, x0, mask=mask)
    tl.store(dk + offs, x1, mask=mask)
    tl.store(da + offs, x2, mask=mask)
    tl.store(db + offs, x3, mask=mask)
    tl.store(dg + offs, x4, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, BV)
        nc = triton.cdiv(T, BT)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B * H, nc, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints,
            T, H=H, K=K, V=V, BK=bk, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, BV)
        n = B * T * H * K
        nc = triton.cdiv(T, BT)
        states = torch.empty((B * H, T, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T, H=H, K=K, V=V, BK=bk,
            num_warps=8, num_stages=2,
        )
        partial = torch.zeros((nv, 5, n), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _backward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout, states, partial, dv,
            T, H=H, K=K, V=V, BK=bk, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gk)
        _reduce_kernel[(triton.cdiv(n, 256),)](
            partial, dq, dk, da, db, dg, N=n, NV=nv, BLOCK=256,
            num_warps=4,
        )
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
