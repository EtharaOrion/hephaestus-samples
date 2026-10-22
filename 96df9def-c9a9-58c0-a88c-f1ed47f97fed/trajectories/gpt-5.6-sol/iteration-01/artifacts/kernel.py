"""Triton implementation of the DPLR generalized delta recurrence."""

import math

import torch
import triton
import triton.language as tl

CHUNK = 16
_FWD_BV = 16
_BWD_BV = 64


@triton.jit
def _dplr_fwd_kernel(
    q, k, v, alpha, beta, gk, out, states, T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
    SCALE: tl.constexpr,
):
    chunk = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_v = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh - i_b * H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]
    if chunk == 0:
        state = tl.zeros((BK, BV), tl.float32)
    else:
        prev_t = chunk * BT - 1
        state_off = ((((i_b * T + prev_t) * H + i_h) * K + o_k[:, None]) * V
                     + o_v[None, :]).to(tl.int64)
        state = tl.load(states + state_off, mask=m_kv, other=0.0)

    times = chunk * BT + o_t
    mt = times < T
    pk = (((i_b * T + times[:, None]) * H + i_h) * K + o_k[None, :]).to(tl.int64)
    pv = (((i_b * T + times[:, None]) * H + i_h) * V + o_v[None, :]).to(tl.int64)
    mtk = mt[:, None] & m_k[None, :]
    mtv = mt[:, None] & m_v[None, :]
    q0 = tl.load(q + pk, mask=mtk, other=0.0).to(tl.float32)
    k0 = tl.load(k + pk, mask=mtk, other=0.0).to(tl.float32)
    a0 = tl.load(alpha + pk, mask=mtk, other=0.0).to(tl.float32)
    b0 = tl.load(beta + pk, mask=mtk, other=0.0).to(tl.float32)
    g0 = tl.load(gk + pk, mask=mtk, other=0.0).to(tl.float32)
    vv = tl.load(v + pv, mask=mtv, other=0.0).to(tl.float32)

    # Cumulative gates rescale every K channel independently inside a tile.
    gi = tl.cumsum(g0, axis=0)
    safe_gi = tl.maximum(gi, -60.0)
    safe_prev_gi = tl.maximum(gi - g0, -60.0)
    qg = q0 * tl.exp(safe_gi) * SCALE
    ag = a0 * tl.exp(safe_prev_gi)
    bg = b0 * tl.exp(-safe_gi)
    kg = k0 * tl.exp(-safe_gi)
    rows = o_t[:, None]
    cols = o_t[None, :]
    strict = rows > cols
    causal = rows >= cols

    A_ab = tl.dot(ag, tl.trans(bg), input_precision="ieee")
    A_ak = tl.dot(ag, tl.trans(kg), input_precision="ieee")
    A_qb = tl.dot(qg, tl.trans(bg), input_precision="ieee")
    A_qk = tl.dot(qg, tl.trans(kg), input_precision="ieee")
    A_ab = tl.where(strict, A_ab, 0.0)

    # Forward substitution materializes (I - lower(A_ab))^-1.
    inv = A_ab
    for row in range(1, BT):
        select = o_t == row
        r = tl.sum(tl.where(select[:, None], inv, 0.0), axis=0)
        r += tl.sum(r[:, None] * inv, axis=0) * (o_t < row)
        inv = tl.where(select[:, None], r, inv)
    inv += rows == cols

    rhs = tl.dot(ag, state, input_precision="ieee")
    rhs += tl.dot(tl.where(strict, A_ak, 0.0), vv, input_precision="ieee")
    reads = tl.dot(inv, rhs, input_precision="ieee")
    result = tl.dot(qg, state, input_precision="ieee")
    result += tl.dot(tl.where(causal, A_qb, 0.0), reads, input_precision="ieee")
    result += tl.dot(tl.where(causal, A_qk, 0.0), vv, input_precision="ieee")
    tl.store(out + pv, result, mask=mtv)



@triton.jit
def _checkpoint_kernel(
    k, v, alpha, beta, gk, states, T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh - i_b * H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]
    state = tl.zeros((BK, BV), tl.float32)
    chunk = 0
    while chunk < NT:
        for j in range(0, BT):
            t = chunk * BT + j
            valid = t < T
            pk = ((i_b * T + t) * H + i_h) * K + o_k
            pv = ((i_b * T + t) * H + i_h) * V + o_v
            a = tl.load(alpha + pk, mask=m_k & valid, other=0.0).to(tl.float32)
            b = tl.load(beta + pk, mask=m_k & valid, other=0.0).to(tl.float32)
            key = tl.load(k + pk, mask=m_k & valid, other=0.0).to(tl.float32)
            gate = tl.load(gk + pk, mask=m_k & valid, other=0.0).to(tl.float32)
            value = tl.load(v + pv, mask=m_v & valid, other=0.0).to(tl.float32)
            read = tl.sum(state * a[:, None], axis=0)
            state = tl.exp(gate)[:, None] * state
            state += key[:, None] * value[None, :] + b[:, None] * read[None, :]
            so = ((((i_b * T + t) * H + i_h) * K + o_k[:, None]) * V
                  + o_v[None, :]).to(tl.int64)
            tl.store(states + so, state, mask=m_kv & valid)
        chunk += 1


@triton.jit
def _dplr_bwd_kernel(
    q, k, v, alpha, beta, gk, dout, states, partial, dv, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr,
    SCALE: tl.constexpr, N: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh - i_b * H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]
    ds = tl.zeros((BK, BV), tl.float32)

    it = 0
    while it < T:
        t = T - 1 - it
        base_k = ((i_b * T + t) * H + i_h) * K + o_k
        base_v = ((i_b * T + t) * H + i_h) * V + o_v
        query = tl.load(q + base_k, mask=m_k, other=0.0).to(tl.float32)
        key = tl.load(k + base_k, mask=m_k, other=0.0).to(tl.float32)
        a = tl.load(alpha + base_k, mask=m_k, other=0.0).to(tl.float32)
        b = tl.load(beta + base_k, mask=m_k, other=0.0).to(tl.float32)
        gate = tl.load(gk + base_k, mask=m_k, other=0.0).to(tl.float32)
        value = tl.load(v + base_v, mask=m_v, other=0.0).to(tl.float32)
        do = tl.load(dout + base_v, mask=m_v, other=0.0).to(tl.float32)

        cur_off = ((((i_b * T + t) * H + i_h) * K + o_k[:, None]) * V
                   + o_v[None, :]).to(tl.int64)
        cur = tl.load(states + cur_off, mask=m_kv, other=0.0)
        if t == 0:
            prev = tl.zeros((BK, BV), tl.float32)
        else:
            prev_off = ((((i_b * T + t - 1) * H + i_h) * K + o_k[:, None]) * V
                        + o_v[None, :]).to(tl.int64)
            prev = tl.load(states + prev_off, mask=m_kv, other=0.0)

        ds += (query * SCALE)[:, None] * do[None, :]
        read = tl.sum(prev * a[:, None], axis=0)
        dread = tl.sum(ds * b[:, None], axis=0)
        dq_part = tl.sum(cur * do[None, :], axis=1) * SCALE
        dk_part = tl.sum(ds * value[None, :], axis=1)
        da_part = tl.sum(prev * dread[None, :], axis=1)
        db_part = tl.sum(ds * read[None, :], axis=1)
        dg_part = tl.sum(ds * (tl.exp(gate)[:, None] * prev), axis=1)
        dv_value = tl.sum(ds * key[:, None], axis=0)

        linear = ((i_b * T + t) * H + i_h) * K + o_k
        part_base = (i_v * N + linear).to(tl.int64)
        tl.store(partial + part_base, dq_part, mask=m_k)
        tl.store(partial + NV * N + part_base, dk_part, mask=m_k)
        tl.store(partial + 2 * NV * N + part_base, da_part, mask=m_k)
        tl.store(partial + 3 * NV * N + part_base, db_part, mask=m_k)
        tl.store(partial + 4 * NV * N + part_base, dg_part, mask=m_k)
        tl.store(dv + base_v, dv_value, mask=m_v)

        ds = tl.exp(gate)[:, None] * ds + a[:, None] * dread[None, :]
        it += 1


@triton.jit
def _reduce_partials_kernel(
    partial, dq, dk, da, db, dg,
    N: tl.constexpr, BLOCK: tl.constexpr, NV: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    sq = tl.zeros((BLOCK,), tl.float32)
    sk = tl.zeros((BLOCK,), tl.float32)
    sa = tl.zeros((BLOCK,), tl.float32)
    sb = tl.zeros((BLOCK,), tl.float32)
    sg = tl.zeros((BLOCK,), tl.float32)
    for j in tl.static_range(0, NV):
        p = j * N + offsets
        sq += tl.load(partial + p, mask=mask, other=0.0)
        sk += tl.load(partial + NV * N + p, mask=mask, other=0.0)
        sa += tl.load(partial + 2 * NV * N + p, mask=mask, other=0.0)
        sb += tl.load(partial + 3 * NV * N + p, mask=mask, other=0.0)
        sg += tl.load(partial + 4 * NV * N + p, mask=mask, other=0.0)
    tl.store(dq + offsets, sq, mask=mask)
    tl.store(dk + offsets, sk, mask=mask)
    tl.store(da + offsets, sa, mask=mask)
    tl.store(db + offsets, sb, mask=mask)
    tl.store(dg + offsets, sg, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v, alpha, beta, gk = [x.contiguous() for x in (q, k, v, alpha, beta, gk)]
        B, T, H, K = q.shape
        V = v.shape[-1]
        BK = triton.next_power_of_2(K)
        fwd_bv = min(_FWD_BV, triton.next_power_of_2(V))
        fwd_nv = triton.cdiv(V, fwd_bv)
        out = torch.empty_like(v)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        checkpoint_bv = min(64, triton.next_power_of_2(V))
        chunks = triton.cdiv(T, CHUNK)
        _checkpoint_kernel[(B * H, triton.cdiv(V, checkpoint_bv))](
            k, v, alpha, beta, gk, states, T, chunks,
            H=H, K=K, V=V, BK=BK, BV=checkpoint_bv, BT=CHUNK, num_warps=4,
        )
        _dplr_fwd_kernel[(chunks, B * H, fwd_nv)](
            q, k, v, alpha, beta, gk, out, states, T, chunks,
            H=H, K=K, V=V, BK=BK, BV=fwd_bv, BT=CHUNK,
            SCALE=1.0 / math.sqrt(K), num_warps=4,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, states)
        ctx.shape = (B, T, H, K, V, BK)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, states = ctx.saved_tensors
        B, T, H, K, V, BK = ctx.shape
        BV = min(_BWD_BV, triton.next_power_of_2(V))
        NV = triton.cdiv(V, BV)
        dout = dout.contiguous()
        N = B * T * H * K
        partial = torch.empty((5, NV, N), device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gk)
        _dplr_bwd_kernel[(B * H, NV)](
            q, k, v, alpha, beta, gk, dout, states, partial, dv, T,
            H=H, K=K, V=V, BK=BK, BV=BV, NV=NV,
            SCALE=1.0 / math.sqrt(K), N=N, num_warps=4,
        )
        block = 256
        _reduce_partials_kernel[(triton.cdiv(N, block),)](
            partial, dq, dk, da, db, dg, N=N, BLOCK=block, NV=NV,
            num_warps=4,
        )
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
) -> torch.Tensor:
    return _Dplr.apply(q, k, v, alpha, beta, gk)
