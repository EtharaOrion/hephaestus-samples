import torch
import triton
import triton.language as tl

BT = 16

BK_MAT = 32
BV = 32


@triton.jit
def _prep(q, k, a, b, g, qg, kg, ag, bg, mat,
          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
          BK: tl.constexpr):
    ic, ibh = tl.program_id(0), tl.program_id(1)
    ih, ib = ibh % H, ibh // H
    it = tl.arange(0, 16)
    t = ic * 16 + it
    mt = t < T
    C = tl.zeros((16, 16), tl.float32)
    P = tl.zeros((16, 16), tl.float32)
    U = tl.zeros((16, 16), tl.float32)
    Z = tl.zeros((16, 16), tl.float32)
    for ik in range(0, tl.cdiv(K, BK)):
        ok = ik * BK + tl.arange(0, BK)
        m = mt[:, None] & (ok[None, :] < K)
        p = ((ib * T + t[:, None]) * H + ih) * K + ok[None, :]
        xg = tl.load(g + p, mask=m, other=0.).to(tl.float32)
        G = tl.cumsum(xg, axis=0)
        xq = tl.load(q + p, mask=m, other=0.).to(tl.float32) * tl.exp(G) / tl.sqrt(float(K))
        xk = tl.load(k + p, mask=m, other=0.).to(tl.float32) * tl.exp(-G)
        xa = tl.load(a + p, mask=m, other=0.).to(tl.float32) * tl.exp(G - xg)
        xb = tl.load(b + p, mask=m, other=0.).to(tl.float32) * tl.exp(-G)
        tl.store(qg + p, xq, mask=m)
        tl.store(kg + p, xk, mask=m)
        tl.store(ag + p, xa, mask=m)
        tl.store(bg + p, xb, mask=m)
        C += tl.dot(xa, tl.trans(xb), input_precision='ieee')
        P += tl.dot(xa, tl.trans(xk), input_precision='ieee')
        U += tl.dot(xq, tl.trans(xk), input_precision='ieee')
        Z += tl.dot(xq, tl.trans(xb), input_precision='ieee')
    row, col = it[:, None], it[None, :]
    W = tl.where(row > col, C, 0.)
    for r in range(1, 16):
        pick = it == r
        wr = tl.sum(tl.where(pick[:, None], W, 0.), axis=0)
        wr += tl.sum(wr[:, None] * W, axis=0) * (it < r)
        W = tl.where(pick[:, None], wr[None, :], W)
    W += row == col
    pm = ((ib * T + t[:, None]) * H + ih) * (4 * 16) + it[None, :]
    tl.store(mat + pm, W, mask=mt[:, None])
    tl.store(mat + pm + 16, tl.where(row > col, P, 0.), mask=mt[:, None])
    tl.store(mat + pm + 2 * 16, tl.where(row >= col, U, 0.), mask=mt[:, None])
    tl.store(mat + pm + 3 * 16, tl.where(row >= col, Z, 0.), mask=mt[:, None])


@triton.jit
def _fwd(qg, kg, ag, bg, v, g, mat, states, ys, out,
         T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
         V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
         BV_: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ih, ib = ibh % H, ibh // H
    ov, ok, it = iv * BV_ + tl.arange(0, BV_), tl.arange(0, BK), tl.arange(0, 16)
    mv, mk = ov < V, ok < K
    S = tl.zeros((BK, BV_), tl.float32)
    for ic in range(0, NC):
        t, mt = ic * 16 + it, ic * 16 + it < T
        ps = (((ibh * (NC + 1) + ic) * K + ok[:, None]) * V + ov[None, :])
        tl.store(states + ps, S, mask=mk[:, None] & mv[None, :])
        px = ((ib * T + t[:, None]) * H + ih) * K + ok[None, :]
        pv = ((ib * T + t[:, None]) * H + ih) * V + ov[None, :]
        pm = ((ib * T + t[:, None]) * H + ih) * (4 * 16) + it[None, :]
        mx = mt[:, None] & mk[None, :]
        Q = tl.load(qg + px, mask=mx, other=0.)
        Ke = tl.load(kg + px, mask=mx, other=0.)
        A = tl.load(ag + px, mask=mx, other=0.)
        B = tl.load(bg + px, mask=mx, other=0.)
        vv = tl.load(v + pv, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
        C = tl.load(mat + pm, mask=mt[:, None], other=0.)
        P = tl.load(mat + pm + 16, mask=mt[:, None], other=0.)
        U = tl.load(mat + pm + 2 * 16, mask=mt[:, None], other=0.)
        Z = tl.load(mat + pm + 3 * 16, mask=mt[:, None], other=0.)
        F = tl.dot(A, S, input_precision='ieee') + tl.dot(P, vv, input_precision='ieee')
        Y = tl.dot(C, F, input_precision='ieee')
        O = tl.dot(Q, S, input_precision='ieee')
        O += tl.dot(U, vv, input_precision='ieee') + tl.dot(Z, Y, input_precision='ieee')
        tl.store(ys + pv, Y, mask=mt[:, None] & mv[None, :])
        tl.store(out + pv, O, mask=mt[:, None] & mv[None, :])
        R = S + tl.dot(tl.trans(Ke), vv, input_precision='ieee')
        R += tl.dot(tl.trans(B), Y, input_precision='ieee')
        xg = tl.load(g + px, mask=mx, other=0.).to(tl.float32)
        S = R * tl.exp(tl.sum(xg, axis=0))[:, None]
    ps = (((ibh * (NC + 1) + NC) * K + ok[:, None]) * V + ov[None, :])
    tl.store(states + ps, S, mask=mk[:, None] & mv[None, :])


@triton.jit
def _bwd_boundaries(qg, ag, bg, g, mat, do, dstates,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    BV_: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ih, ib = ibh % H, ibh // H
    ov, ok, it = iv * BV_ + tl.arange(0, BV_), tl.arange(0, BK), tl.arange(0, 16)
    mv, mk = ov < V, ok < K
    dS = tl.zeros((BK, BV_), tl.float32)
    pe = (((ibh * (NC + 1) + NC) * K + ok[:, None]) * V + ov[None, :])
    tl.store(dstates + pe, dS, mask=mk[:, None] & mv[None, :])
    for rev in range(0, NC):
        ic = NC - 1 - rev
        t, mt = ic * 16 + it, ic * 16 + it < T
        px = ((ib * T + t[:, None]) * H + ih) * K + ok[None, :]
        pv = ((ib * T + t[:, None]) * H + ih) * V + ov[None, :]
        pm = ((ib * T + t[:, None]) * H + ih) * (4 * 16) + it[None, :]
        mx = mt[:, None] & mk[None, :]
        Q = tl.load(qg + px, mask=mx, other=0.)
        A = tl.load(ag + px, mask=mx, other=0.)
        B = tl.load(bg + px, mask=mx, other=0.)
        C = tl.load(mat + pm, mask=mt[:, None], other=0.)
        Z = tl.load(mat + pm + 3 * 16, mask=mt[:, None], other=0.)
        DO = tl.load(do + pv, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
        xg = tl.load(g + px, mask=mx, other=0.).to(tl.float32)
        dR = dS * tl.exp(tl.sum(xg, axis=0))[:, None]
        dY = tl.dot(B, dR, input_precision='ieee') + tl.dot(tl.trans(Z), DO, input_precision='ieee')
        dF = tl.dot(tl.trans(C), dY, input_precision='ieee')
        dS = dR + tl.dot(tl.trans(Q), DO, input_precision='ieee')
        dS += tl.dot(tl.trans(A), dF, input_precision='ieee')
        ps = (((ibh * (NC + 1) + ic) * K + ok[:, None]) * V + ov[None, :])
        tl.store(dstates + ps, dS, mask=mk[:, None] & mv[None, :])


@triton.jit
def _expand(states, k, v, a, b, g, full,
            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
            V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
            BV_: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ih, ib = ibh % H, ibh // H
    ov, ok = iv * BV_ + tl.arange(0, BV_), tl.arange(0, BK)
    mv, mk = ov < V, ok < K
    ps = (((ibh * (NC + 1) + ic) * K + ok[:, None]) * V + ov[None, :])
    S = tl.load(states + ps, mask=mk[:, None] & mv[None, :], other=0.)
    for j in range(0, 16):
        t = ic * 16 + j
        valid = t < T
        px = ((ib * T + t) * H + ih) * K + ok
        pv = ((ib * T + t) * H + ih) * V + ov
        xk = tl.load(k + px, mask=valid & mk, other=0.).to(tl.float32)
        xa = tl.load(a + px, mask=valid & mk, other=0.).to(tl.float32)
        xb = tl.load(b + px, mask=valid & mk, other=0.).to(tl.float32)
        xg = tl.load(g + px, mask=valid & mk, other=0.).to(tl.float32)
        xv = tl.load(v + pv, mask=valid & mv, other=0.).to(tl.float32)
        read = tl.sum(S * xa[:, None], axis=0)
        S = S * tl.exp(xg)[:, None] + xk[:, None] * xv[None, :] + xb[:, None] * read[None, :]
        pf = ((ib * T + t) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
        tl.store(full + pf, S, mask=valid & mk[:, None] & mv[None, :])


@triton.jit
def _reverse_exact(states, full, q, k, v, a, b, g, do, dstates,
                   partial, dv, T: tl.constexpr, H: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                   NV: tl.constexpr, B_CONST: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ih, ib = ibh % H, ibh // H
    ov, ok = iv * BV_ + tl.arange(0, BV_), tl.arange(0, BK)
    mv, mk = ov < V, ok < K
    pd = (((ibh * (NC + 1) + ic + 1) * K + ok[:, None]) * V + ov[None, :])
    dS = tl.load(dstates + pd, mask=mk[:, None] & mv[None, :], other=0.)
    for jj in range(0, 16):
        j = 15 - jj
        t = ic * 16 + j
        valid = t < T
        px = ((ib * T + t) * H + ih) * K + ok
        pv = ((ib * T + t) * H + ih) * V + ov
        xq = tl.load(q + px, mask=valid & mk, other=0.).to(tl.float32) / tl.sqrt(float(K))
        xk = tl.load(k + px, mask=valid & mk, other=0.).to(tl.float32)
        xa = tl.load(a + px, mask=valid & mk, other=0.).to(tl.float32)
        xb = tl.load(b + px, mask=valid & mk, other=0.).to(tl.float32)
        xg = tl.load(g + px, mask=valid & mk, other=0.).to(tl.float32)
        xv = tl.load(v + pv, mask=valid & mv, other=0.).to(tl.float32)
        dO = tl.load(do + pv, mask=valid & mv, other=0.).to(tl.float32)
        pf = ((ib * T + t) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
        St = tl.load(full + pf, mask=valid & mk[:, None] & mv[None, :], other=0.)
        if j == 0:
            pp = (((ibh * (NC + 1) + ic) * K + ok[:, None]) * V + ov[None, :])
            Sp = tl.load(states + pp, mask=mk[:, None] & mv[None, :], other=0.)
        else:
            pp = ((ib * T + t - 1) * H + ih) * K * V + ok[:, None] * V + ov[None, :]
            Sp = tl.load(full + pp, mask=valid & mk[:, None] & mv[None, :], other=0.)
        dSt = dS + xq[:, None] * dO[None, :]
        read = tl.sum(Sp * xa[:, None], axis=0)
        rank_adj = tl.sum(dSt * xb[:, None], axis=0)
        dqv = tl.sum(St * dO[None, :], axis=1) / tl.sqrt(float(K))
        dkv = tl.sum(dSt * xv[None, :], axis=1)
        dbv = tl.sum(dSt * read[None, :], axis=1)
        dav = tl.sum(Sp * rank_adj[None, :], axis=1)
        dgv = tl.sum(dSt * (Sp * tl.exp(xg)[:, None]), axis=1)
        dvv = tl.sum(dSt * xk[:, None], axis=0)
        base = ((((iv * B_CONST + ib) * T + t) * H + ih) * (5 * K) + ok)
        tl.store(partial + base + 0 * K, dqv, mask=valid & mk)
        tl.store(partial + base + 1 * K, dkv, mask=valid & mk)
        tl.store(partial + base + 2 * K, dav, mask=valid & mk)
        tl.store(partial + base + 3 * K, dbv, mask=valid & mk)
        tl.store(partial + base + 4 * K, dgv, mask=valid & mk)
        tl.store(dv + pv, dvv, mask=valid & mv)
        dS = dSt * tl.exp(xg)[:, None] + xa[:, None] * rank_adj[None, :]


@triton.jit
def _reduce_exact(partial, dq, dk, da, db, dg,
                  N: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
                  BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < N
    channel = idx % K
    zq = tl.zeros((BLOCK,), tl.float32)
    zk = tl.zeros((BLOCK,), tl.float32)
    za = tl.zeros((BLOCK,), tl.float32)
    zb = tl.zeros((BLOCK,), tl.float32)
    zg = tl.zeros((BLOCK,), tl.float32)
    for iv in range(0, NV):
        base = iv * N * 5 + (idx // K) * (5 * K) + channel
        zq += tl.load(partial + base + 0 * K, mask=m, other=0.)
        zk += tl.load(partial + base + 1 * K, mask=m, other=0.)
        za += tl.load(partial + base + 2 * K, mask=m, other=0.)
        zb += tl.load(partial + base + 3 * K, mask=m, other=0.)
        zg += tl.load(partial + base + 4 * K, mask=m, other=0.)
    tl.store(dq + idx, zq, mask=m)
    tl.store(dk + idx, zk, mask=m)
    tl.store(da + idx, za, mask=m)
    tl.store(db + idx, zb, mask=m)
    tl.store(dg + idx, zg, mask=m)

class _DPLR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, BV)
        f32 = lambda *s: torch.empty(*s, device=q.device, dtype=torch.float32)
        qg, kg, ag, bg = (f32(B, T, H, K) for _ in range(4))
        mat = f32(B, T, H, 4 * BT)
        _prep[(nc, B * H)](q, k, alpha, beta, gk, qg, kg, ag, bg, mat,
                           T=T, H=H, K=K, BK=BK_MAT, num_warps=4)
        states = f32(B, H, nc + 1, K, V)
        ys = f32(B, T, H, V)
        out = torch.empty_like(v)
        _fwd[(nv, B * H)](qg, kg, ag, bg, v, gk, mat, states, ys, out,
                          T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV,
                          num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, qg, kg, ag, bg, mat, states, ys)
        ctx.dims = B, T, H, K, V, nc, bk, nv
        return out

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v, a, b, g, qg, kg, ag, bg, mat, states, ys = ctx.saved_tensors
        B, T, H, K, V, nc, bk, nv = ctx.dims
        f32 = lambda *s: torch.empty(*s, device=q.device, dtype=torch.float32)
        dstates = torch.zeros(B, H, nc + 1, K, V, device=q.device, dtype=torch.float32)
        _bwd_boundaries[(nv, B * H)](qg, ag, bg, g, mat, do, dstates,
                                    T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV,
                                    num_warps=4, num_stages=2)
        full = f32(B, T, H, K, V)
        _expand[(nv, nc, B * H)](states, k, v, a, b, g, full, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV, num_warps=4)
        partial = f32(nv, B, T, H, 5 * K)
        dv = torch.empty_like(v)
        _reverse_exact[(nv, nc, B * H)](states, full, q, k, v, a, b, g, do, dstates, partial, dv, T=T, H=H, K=K, V=V, NC=nc, NV=nv, B_CONST=B, BK=bk, BV_=BV, num_warps=4)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(a), torch.empty_like(b), torch.empty_like(g)
        n = B * T * H * K
        _reduce_exact[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg, N=n, K=K, NV=nv, BLOCK=256, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _DPLR.apply(q, k, v, alpha, beta, gk)
