import torch
import triton
import triton.language as tl

_BT = 16
_BV = 32
_BWD_BV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr,
                    SCALE: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib, ih = bh // H, bh % H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    rr = tl.arange(0, 16)
    eye = rr[:, None] == rr[None, :]
    lower = rr[:, None] > rr[None, :]
    inclusive = rr[:, None] >= rr[None, :]
    for ic in range(NC):
        tt = ic * 16 + rr
        mt = tt < T
        xoff = (ib * T + tt[:, None]) * H * K + ih * K + kk[None, :]
        mx = mt[:, None] & (kk[None, :] < K)
        qx = tl.load(q + xoff, mask=mx, other=0.).to(tl.float32) * SCALE
        kx = tl.load(k + xoff, mask=mx, other=0.).to(tl.float32)
        ax = tl.load(a + xoff, mask=mx, other=0.).to(tl.float32)
        bx = tl.load(b + xoff, mask=mx, other=0.).to(tl.float32)
        gx = tl.load(g + xoff, mask=mx, other=0.).to(tl.float32)
        gi = tl.cumsum(gx, axis=0)
        Aqk = tl.zeros((16, 16), tl.float32)
        Aqb = tl.zeros((16, 16), tl.float32)
        Aak = tl.zeros((16, 16), tl.float32)
        Aab = tl.zeros((16, 16), tl.float32)
        for i in tl.static_range(0, 16):
            qi = tl.sum(tl.where(rr[:, None] == i, qx, 0.), axis=0)
            ai = tl.sum(tl.where(rr[:, None] == i, ax, 0.), axis=0)
            gii = tl.sum(tl.where(rr[:, None] == i, gi, 0.), axis=0)
            gxi = tl.sum(tl.where(rr[:, None] == i, gx, 0.), axis=0)
            dqg = tl.exp(gii[None, :] - gi)
            dag = tl.exp((gii - gxi)[None, :] - gi)
            rqk = tl.sum(qi[None, :] * kx * dqg, axis=1)
            rqb = tl.sum(qi[None, :] * bx * dqg, axis=1)
            rak = tl.sum(ai[None, :] * kx * dag, axis=1)
            rab = tl.sum(ai[None, :] * bx * dag, axis=1)
            row = rr[:, None] == i
            ok = (i + ic * 16) < T
            Aqk += tl.where(row & (rr[None, :] <= i) & ok, rqk[None, :], 0.)
            Aqb += tl.where(row & (rr[None, :] <= i) & ok, rqb[None, :], 0.)
            Aak += tl.where(row & (rr[None, :] < i) & ok, rak[None, :], 0.)
            Aab += tl.where(row & (rr[None, :] < i) & ok, rab[None, :], 0.)
        inv = Aab
        for i in tl.static_range(1, 16):
            row_mask = rr == i
            row_i = tl.sum(tl.where(row_mask[:, None], inv, 0.), axis=0)
            row_i += tl.sum(row_i[:, None] * inv, axis=0) * (rr < i)
            inv = tl.where(row_mask[:, None], row_i[None, :], inv)
        inv = tl.where(eye, 1., inv)
        inv = tl.where(lower | eye, inv, 0.)
        w = tl.dot(inv, ax * tl.exp(gi - gx), input_precision="ieee")
        voff = (ib * T + tt[:, None]) * H * V + ih * V + vv[None, :]
        mv = mt[:, None] & (vv[None, :] < V)
        vx = tl.load(v + voff, mask=mv, other=0.).to(tl.float32)
        u = tl.dot(inv, tl.dot(Aak, vx, input_precision="ieee"), input_precision="ieee")
        vnew = u + tl.dot(w, state, input_precision="ieee")
        ox = tl.dot(qx * tl.exp(gi), state, input_precision="ieee")
        ox += tl.dot(tl.where(inclusive, Aqk, 0.), vx, input_precision="ieee")
        ox += tl.dot(tl.where(inclusive, Aqb, 0.), vnew, input_precision="ieee")
        tl.store(out + voff, ox, mask=mv)
        hoff = ((bh * NC + ic) * K + kk[:, None]) * V + vv[None, :]
        tl.store(checkpoints + hoff, state, mask=(kk[:, None] < K) & (vv[None, :] < V))
        last = tl.minimum((ic + 1) * 16, T) - ic * 16 - 1
        glast = tl.sum(tl.where(rr[:, None] == last, gi, 0.), axis=0)
        decay = tl.exp(glast[None, :] - gi)
        chunk_update = tl.zeros((BK, BV), tl.float32)
        for j in tl.static_range(0, 16):
            row = rr == j
            valid = (ic * 16 + j) < T
            kj = tl.sum(tl.where(row[:, None], kx * decay, 0.), axis=0)
            bj = tl.sum(tl.where(row[:, None], bx * decay, 0.), axis=0)
            vj = tl.sum(tl.where(row[:, None], vx, 0.), axis=0)
            uj = tl.sum(tl.where(row[:, None], vnew, 0.), axis=0)
            chunk_update += tl.where(valid, kj[:, None] * vj[None, :] + bj[:, None] * uj[None, :], 0.)
        state = state * tl.exp(glast[:, None]) + chunk_update


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, NC: tl.constexpr,
                          BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    hm = (kk[:, None] < K) & (vv[None, :] < V)
    hoff = ((bh * NC + ic) * K + kk[:, None]) * V + vv[None, :]
    state = tl.load(checkpoints + hoff, mask=hm, other=0.).to(tl.float32)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        xoff = (ib * T + t) * H * K + ih * K + kk
        xm = (kk < K) & valid
        kx = tl.load(k + xoff, mask=xm, other=0.).to(tl.float32)
        ax = tl.load(a + xoff, mask=xm, other=0.).to(tl.float32)
        bx = tl.load(b + xoff, mask=xm, other=0.).to(tl.float32)
        gx = tl.load(g + xoff, mask=xm, other=0.).to(tl.float32)
        voff = (ib * T + t) * H * V + ih * V + vv
        vm = (vv < V) & valid
        vx = tl.load(v + voff, mask=vm, other=0.).to(tl.float32)
        read = tl.sum(state * ax[:, None], axis=0)
        state = state * tl.exp(gx[:, None]) + kx[:, None] * vx[None, :] + bx[:, None] * read[None, :]
        soff = ((bh * T + t) * K + kk[:, None]) * V + vv[None, :]
        tl.store(states + soff, state, mask=hm & valid)


@triton.jit
def _backward_kernel(q, k, v, a, b, g, dout, checkpoints, states, dv, partial,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, NC: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr,
                     SCALE: tl.constexpr, B: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    hm = (kk[:, None] < K) & (vv[None, :] < V)
    dstate = tl.zeros((BK, BV), tl.float32)
    N = B * T * H * K
    for ri in range(NC):
        ic = NC - 1 - ri
        hoff = ((bh * NC + ic) * K + kk[:, None]) * V + vv[None, :]
        checkpoint = tl.load(checkpoints + hoff, mask=hm, other=0.).to(tl.float32)
        for rj in tl.static_range(0, 16):
            j = 15 - rj
            t = ic * 16 + j
            valid = t < T
            xoff = (ib * T + t) * H * K + ih * K + kk
            xm = (kk < K) & valid
            qx = tl.load(q + xoff, mask=xm, other=0.).to(tl.float32)
            kx = tl.load(k + xoff, mask=xm, other=0.).to(tl.float32)
            ax = tl.load(a + xoff, mask=xm, other=0.).to(tl.float32)
            bx = tl.load(b + xoff, mask=xm, other=0.).to(tl.float32)
            gx = tl.load(g + xoff, mask=xm, other=0.).to(tl.float32)
            voff = (ib * T + t) * H * V + ih * V + vv
            vm = (vv < V) & valid
            vx = tl.load(v + voff, mask=vm, other=0.).to(tl.float32)
            do = tl.load(dout + voff, mask=vm, other=0.).to(tl.float32)
            if j == 0:
                sold = checkpoint
            else:
                soff0 = ((bh * T + t - 1) * K + kk[:, None]) * V + vv[None, :]
                sold = tl.load(states + soff0, mask=hm & valid, other=0.).to(tl.float32)
            soff1 = ((bh * T + t) * K + kk[:, None]) * V + vv[None, :]
            snew = tl.load(states + soff1, mask=hm & valid, other=0.).to(tl.float32)
            ds = dstate + qx[:, None] * (do[None, :] * SCALE)
            read = tl.sum(sold * ax[:, None], axis=0)
            dread = tl.sum(ds * bx[:, None], axis=0)
            dq = tl.sum(snew * do[None, :], axis=1) * SCALE
            dk = tl.sum(ds * vx[None, :], axis=1)
            da = tl.sum(sold * dread[None, :], axis=1)
            db = tl.sum(ds * read[None, :], axis=1)
            dg = tl.sum(ds * sold, axis=1) * tl.exp(gx)
            tl.store(dv + voff, tl.sum(ds * kx[:, None], axis=0), mask=vm)
            elem = ((ib * T + t) * H + ih) * K + kk
            stride = 5 * N
            pbase = partial + iv * stride
            tl.store(pbase + 0 * N + elem, dq, mask=xm)
            tl.store(pbase + 1 * N + elem, dk, mask=xm)
            tl.store(pbase + 2 * N + elem, da, mask=xm)
            tl.store(pbase + 3 * N + elem, db, mask=xm)
            tl.store(pbase + 4 * N + elem, dg, mask=xm)
            dstate = ds * tl.exp(gx[:, None]) + ax[:, None] * dread[None, :]


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg,
                   N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    x0 = tl.zeros((BLOCK,), tl.float32)
    x1 = tl.zeros((BLOCK,), tl.float32)
    x2 = tl.zeros((BLOCK,), tl.float32)
    x3 = tl.zeros((BLOCK,), tl.float32)
    x4 = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        stride = 5 * N
        x0 += tl.load(partial + iv * stride + 0 * N + off, mask=mask, other=0.)
        x1 += tl.load(partial + iv * stride + 1 * N + off, mask=mask, other=0.)
        x2 += tl.load(partial + iv * stride + 2 * N + off, mask=mask, other=0.)
        x3 += tl.load(partial + iv * stride + 3 * N + off, mask=mask, other=0.)
        x4 += tl.load(partial + iv * stride + 4 * N + off, mask=mask, other=0.)
    tl.store(dq + off, x0, mask=mask)
    tl.store(dk + off, x1, mask=mask)
    tl.store(da + off, x2, mask=mask)
    tl.store(db + off, x3, mask=mask)
    tl.store(dg + off, x4, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, H, nc, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV, SCALE=K ** -0.5, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BWD_BV)
        states = torch.empty((B, H, T, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV, num_warps=4, num_stages=2)
        dv = torch.empty_like(v)
        n = B * T * H * K
        partial = torch.zeros((nv, 5, n), device=q.device, dtype=torch.float32)
        _backward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, dout.contiguous(), checkpoints,
            states, dv, partial, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV,
            SCALE=K ** -0.5, B=B,
            num_warps=4, num_stages=2)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        _reduce_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg,
            N=n, NV=nv, BLOCK=256, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
