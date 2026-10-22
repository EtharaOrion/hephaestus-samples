import torch
import triton
import triton.language as tl

_BT = 16
_FWD_BV = 32
_BWD_BV = 16


@triton.jit
def _chunk_forward_kernel(
    q, k, v, a, b, g, out, checkpoints,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    ot = tl.arange(0, 16)
    state = tl.zeros((BK, BV), tl.float32)
    scale = SCALE

    for ic in range(NC):
        ck = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(checkpoints + ck, state, mask=(ok[:, None] < K) & (ov[None, :] < V))

        ts = ic * 16 + ot
        mt = ts < T
        base_k = (ib * T * H + ts[:, None] * H + ih) * K + ok[None, :]
        base_v = (ib * T * H + ts[:, None] * H + ih) * V + ov[None, :]
        mmk = mt[:, None] & (ok[None, :] < K)
        mmv = mt[:, None] & (ov[None, :] < V)
        q0 = tl.load(q + base_k, mask=mmk, other=0.0).to(tl.float32) * scale
        k0 = tl.load(k + base_k, mask=mmk, other=0.0).to(tl.float32)
        a0 = tl.load(a + base_k, mask=mmk, other=0.0).to(tl.float32)
        b0 = tl.load(b + base_k, mask=mmk, other=0.0).to(tl.float32)
        gg = tl.load(g + base_k, mask=mmk, other=0.0).to(tl.float32)
        vv = tl.load(v + base_v, mask=mmv, other=0.0).to(tl.float32)
        gi = tl.cumsum(gg, axis=0)
        gp = gi - gg

        aqk = tl.zeros((16, 16), tl.float32)
        aqb = tl.zeros((16, 16), tl.float32)
        aak = tl.zeros((16, 16), tl.float32)
        aab = tl.zeros((16, 16), tl.float32)
        for i in tl.static_range(0, 16):
            ri = tl.arange(0, 16) == i
            qi = tl.sum(tl.where(ri[:, None], q0, 0.0), axis=0)
            ai = tl.sum(tl.where(ri[:, None], a0, 0.0), axis=0)
            gie = tl.sum(tl.where(ri[:, None], gi, 0.0), axis=0)
            gpe = tl.sum(tl.where(ri[:, None], gp, 0.0), axis=0)
            fq = tl.exp(gie[None, :] - gi)
            fa = tl.exp(gpe[None, :] - gi)
            xqk = tl.sum(k0 * qi[None, :] * fq, axis=1)
            xqb = tl.sum(b0 * qi[None, :] * fq, axis=1)
            xak = tl.sum(k0 * ai[None, :] * fa, axis=1)
            xab = tl.sum(b0 * ai[None, :] * fa, axis=1)
            cols = tl.arange(0, 16)
            aqk = tl.where(ri[:, None] & (cols[None, :] <= i), xqk[None, :], aqk)
            aqb = tl.where(ri[:, None] & (cols[None, :] <= i), xqb[None, :], aqb)
            aak = tl.where(ri[:, None] & (cols[None, :] < i), xak[None, :], aak)
            aab = tl.where(ri[:, None] & (cols[None, :] < i), xab[None, :], aab)

        read = tl.dot(a0 * tl.exp(gp), state, input_precision="ieee")
        rhs = read + tl.dot(aak, vv, input_precision="ieee")
        rr = tl.zeros((16, BV), tl.float32)
        for i in tl.static_range(0, 16):
            ri = tl.arange(0, 16) == i
            row = tl.sum(tl.where(ri[:, None], rhs, 0.0), axis=0)
            coeff = tl.sum(tl.where(ri[:, None], aab, 0.0), axis=0)
            row += tl.sum(coeff[:, None] * rr, axis=0)
            rr = tl.where(ri[:, None], row[None, :], rr)

        oo = tl.dot(q0 * tl.exp(gi), state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee")
        oo += tl.dot(aqb, rr, input_precision="ieee")
        out_idx = (ib * T * H + ts[:, None] * H + ih) * V + ov[None, :]
        tl.store(out + out_idx, oo, mask=mmv)

        gl = tl.sum(tl.where((ot == 15)[:, None], gi, 0.0), axis=0)
        kg = k0 * tl.exp(gl[None, :] - gi)
        bg = b0 * tl.exp(gl[None, :] - gi)
        state *= tl.exp(gl)[:, None]
        state += tl.dot(tl.trans(kg), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bg), rr, input_precision="ieee")



@triton.jit
def _exact_checkpoint_kernel(
    k, v, a, b, g, checkpoints,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mask = (ok[:, None] < K) & (ov[None, :] < V)
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(checkpoints + cp, state, mask=mask)
        for j in tl.static_range(0, 16):
            t = ic * 16 + j
            valid = t < T
            ik = ((ib * T + t) * H + ih) * K + ok
            ivv = ((ib * T + t) * H + ih) * V + ov
            kk = tl.load(k + ik, mask=valid & (ok < K), other=0.0).to(tl.float32)
            aa = tl.load(a + ik, mask=valid & (ok < K), other=0.0).to(tl.float32)
            bb = tl.load(b + ik, mask=valid & (ok < K), other=0.0).to(tl.float32)
            gg = tl.load(g + ik, mask=valid & (ok < K), other=0.0).to(tl.float32)
            vv = tl.load(v + ivv, mask=valid & (ov < V), other=0.0).to(tl.float32)
            read = tl.sum(tl.dot(aa[None, :], state, input_precision="ieee"), axis=0)
            nxt = state * tl.exp(gg)[:, None]
            nxt = nxt + kk[:, None] * vv[None, :]
            nxt = nxt + bb[:, None] * read[None, :]
            state = tl.where(valid, nxt, state)


@triton.jit
def _expand_states_kernel(
    k, v, a, b, g, checkpoints, states,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    ibh = tl.program_id(2)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    ck = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
    state = tl.load(checkpoints + ck, mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.0)

    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        ik = ((ib * T + t) * H + ih) * K + ok
        ivv = ((ib * T + t) * H + ih) * V + ov
        mk = valid & (ok < K)
        mv = valid & (ov < V)
        kk = tl.load(k + ik, mask=mk, other=0.0).to(tl.float32)
        aa = tl.load(a + ik, mask=mk, other=0.0).to(tl.float32)
        bb = tl.load(b + ik, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + ik, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + ivv, mask=mv, other=0.0).to(tl.float32)
        read = tl.sum(tl.dot(aa[None, :], state, input_precision="ieee"), axis=0)
        state = state * tl.exp(gg)[:, None]
        state = state + kk[:, None] * vv[None, :]
        state = state + bb[:, None] * read[None, :]
        si = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(states + si, state, mask=valid & (ok[:, None] < K) & (ov[None, :] < V))


@triton.jit
def _backward_scan_kernel(
    q, k, v, a, b, g, do, checkpoints, states,
    partial_q, partial_k, dv, partial_a, partial_b, partial_g,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mkv = (ok[:, None] < K) & (ov[None, :] < V)
    dstate = tl.zeros((BK, BV), tl.float32)
    scale = SCALE

    for z in range(T):
        t = T - 1 - z
        ik = ((ib * T + t) * H + ih) * K + ok
        ivv = ((ib * T + t) * H + ih) * V + ov
        qq = tl.load(q + ik, mask=ok < K, other=0.0).to(tl.float32) * scale
        kk = tl.load(k + ik, mask=ok < K, other=0.0).to(tl.float32)
        aa = tl.load(a + ik, mask=ok < K, other=0.0).to(tl.float32)
        bb = tl.load(b + ik, mask=ok < K, other=0.0).to(tl.float32)
        gg = tl.load(g + ik, mask=ok < K, other=0.0).to(tl.float32)
        vv = tl.load(v + ivv, mask=ov < V, other=0.0).to(tl.float32)
        dd = tl.load(do + ivv, mask=ov < V, other=0.0).to(tl.float32)
        scur_i = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        scur = tl.load(states + scur_i, mask=mkv, other=0.0)
        ic = t // 16
        at_boundary = (t & 15) == 0
        sprev_t = t - 1
        sprev_i = ((((ib * T + sprev_t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        ck = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        sprev = tl.load(states + sprev_i, mask=(sprev_t >= 0) & mkv, other=0.0)
        sprev = tl.where(at_boundary, tl.load(checkpoints + ck, mask=mkv, other=0.0), sprev)

        dstate += qq[:, None] * dd[None, :]
        read = tl.sum(tl.dot(aa[None, :], sprev, input_precision="ieee"), axis=0)
        dr = tl.sum(tl.dot(bb[None, :], dstate, input_precision="ieee"), axis=0)
        dq0 = tl.sum(scur * dd[None, :], axis=1) * scale
        dk0 = tl.sum(dstate * vv[None, :], axis=1)
        da0 = tl.sum(sprev * dr[None, :], axis=1)
        db0 = tl.sum(tl.dot(dstate, read[:, None], input_precision="ieee"), axis=1)
        dg0 = tl.sum(dstate * (tl.exp(gg)[:, None] * sprev), axis=1)
        dv0 = tl.sum(tl.dot(kk[None, :], dstate, input_precision="ieee"), axis=0)

        pbase = ((((iv * B + ib) * T + t) * H + ih) * K + ok)
        tl.store(partial_q + pbase, dq0, mask=ok < K)
        tl.store(partial_k + pbase, dk0, mask=ok < K)
        tl.store(partial_a + pbase, da0, mask=ok < K)
        tl.store(partial_b + pbase, db0, mask=ok < K)
        tl.store(partial_g + pbase, dg0, mask=ok < K)
        tl.store(dv + ivv, dv0, mask=ov < V)
        dstate = dstate * tl.exp(gg)[:, None] + aa[:, None] * dr[None, :]


@triton.jit
def _reduce_partials_kernel(x, y, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.float32)
    for iv in tl.static_range(0, NV):
        total += tl.load(x + iv * N + offs, mask=offs < N, other=0.0)
    tl.store(y + offs, total, mask=offs < N)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v, alpha, beta, gk = [x.contiguous() for x in (q, k, v, alpha, beta, gk)]
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        nv = triton.cdiv(V, _FWD_BV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _chunk_forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FWD_BV, SCALE=K ** -0.5,
            num_warps=8, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        nv = triton.cdiv(V, _BWD_BV)
        bk = max(16, triton.next_power_of_2(K))
        exact_checkpoints = torch.empty_like(checkpoints)
        _exact_checkpoint_kernel[(nv, B * H)](
            k, v, alpha, beta, gk, exact_checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV,
            num_warps=4, num_stages=2,
        )
        checkpoints = exact_checkpoints
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV,
            num_warps=4, num_stages=2,
        )
        shape = (nv, B, T, H, K)
        pq = torch.zeros(shape, device=q.device, dtype=torch.float32)
        pk = torch.zeros(shape, device=q.device, dtype=torch.float32)
        pa = torch.zeros(shape, device=q.device, dtype=torch.float32)
        pb = torch.zeros(shape, device=q.device, dtype=torch.float32)
        pg = torch.zeros(shape, device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _backward_scan_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout.contiguous(), checkpoints, states,
            pq, pk, dv, pa, pb, pg,
            B=B, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BWD_BV, SCALE=K ** -0.5,
            num_warps=4, num_stages=2,
        )
        n = B * T * H * K
        grads = []
        for src, like in ((pq, q), (pk, k), (pa, alpha), (pb, beta), (pg, gk)):
            dst = torch.empty_like(like)
            _reduce_partials_kernel[(triton.cdiv(n, 256),)](
                src, dst, N=n, NV=nv, BLOCK=256, num_warps=4,
            )
            grads.append(dst)
        dq, dk, da, db, dg = grads
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
