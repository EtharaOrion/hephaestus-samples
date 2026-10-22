import torch
import triton
import triton.language as tl

BC = tl.constexpr(16)
BV = tl.constexpr(32)


@triton.jit
def _build_transform(k, g, beta, transform, gc_out, T: tl.constexpr,
                     H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                     BK: tl.constexpr, IP: tl.constexpr):
    chunk = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    ti = chunk * BC + tl.arange(0, BC)
    kk = tl.arange(0, BK)
    valid = ti < T
    base_k = ((b * T + ti[:, None]) * H + kh) * K + kk[None, :]
    keys = tl.load(k + base_k, mask=valid[:, None] & (kk[None, :] < K), other=0.0)
    gate = tl.load(g + (b * T + ti) * HV + vh, mask=valid, other=0.0).to(tl.float32)
    strength = tl.load(beta + (b * T + ti) * HV + vh, mask=valid, other=0.0).to(tl.float32)
    gc = tl.cumsum(gate, axis=0)
    lanes = tl.arange(0, BC)
    rows = lanes[:, None]
    cols = lanes[None, :]
    gram = tl.dot(keys, tl.trans(keys), input_precision=IP)
    lower = rows > cols
    mat = tl.where(lower & valid[:, None] & valid[None, :],
                   gram * strength[:, None] * tl.exp(gc[:, None] - gc[None, :]), 0.0)
    inv = tl.where(rows == cols, 1.0, 0.0)
    for i in range(BC):
        for j in range(i):
            coeff = tl.sum(tl.where((rows == i) & (cols == j), mat, 0.0))
            source = tl.sum(tl.where(rows == j, inv, 0.0), axis=0)
            inv = tl.where(rows == i, inv - coeff * source[None, :], inv)
    ao = tl.arange(0, BC)
    aoff = (((b * ((T + BC - 1) // BC) + chunk) * HV + vh) * BC + rows) * BC + cols
    tl.store(transform + aoff, inv, mask=valid[:, None] & (ao[None, :] < BC))
    tl.store(gc_out + (b * T + ti) * HV + vh, gc, mask=valid)


@triton.jit
def _scan_emit(q, k, v, beta, transform, gc, states, out,
               T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
               BK: tl.constexpr, IP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    vb = pid % nv
    bh = pid // nv
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    ko = tl.arange(0, BK)
    vo = vb * BV + tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(K * 1.0)
    rr = tl.arange(0, BC)[:, None]
    cc = tl.arange(0, BC)[None, :]
    for chunk in range(NC):
        ti = chunk * BC + tl.arange(0, BC)
        valid = ti < T
        state_off = (((b * NC + chunk) * HV + vh) * K + ko[:, None]) * V + vo[None, :]
        tl.store(states + state_off, state, mask=(ko[:, None] < K) & (vo[None, :] < V))
        koff = ((b * T + ti[:, None]) * H + kh) * K + ko[None, :]
        qoff = koff
        xoff = ((b * T + ti[:, None]) * HV + vh) * V + vo[None, :]
        keys = tl.load(k + koff, mask=valid[:, None] & (ko[None, :] < K), other=0.0)
        queries = tl.load(q + qoff, mask=valid[:, None] & (ko[None, :] < K), other=0.0)
        values = tl.load(v + xoff, mask=valid[:, None] & (vo[None, :] < V), other=0.0)
        gates = tl.load(gc + (b * T + ti) * HV + vh, mask=valid, other=0.0)
        strength = tl.load(beta + (b * T + ti) * HV + vh, mask=valid, other=0.0)
        aoff = (((b * NC + chunk) * HV + vh) * BC + rr) * BC + cc
        inv = tl.load(transform + aoff, mask=valid[:, None] & valid[None, :], other=0.0)
        prediction = tl.dot(keys, state.to(keys.dtype), input_precision=IP)
        residual = strength[:, None] * (values - tl.exp(gates)[:, None] * prediction)
        delta = tl.dot(inv, residual, input_precision=IP)
        qk = tl.dot(queries, tl.trans(keys), input_precision=IP) * scale
        causal = rr >= cc
        mixing = tl.where(causal & valid[:, None] & valid[None, :],
                          qk * tl.exp(gates[:, None] - gates[None, :]), 0.0)
        result = tl.exp(gates)[:, None] * tl.dot(queries, state.to(queries.dtype), input_precision=IP) * scale
        result += tl.dot(mixing, delta, input_precision=IP)
        tl.store(out + xoff, result, mask=valid[:, None] & (vo[None, :] < V))
        last = tl.sum(tl.where(valid, gates, -float("inf")), axis=0)
        # Only the final valid lane contributes after replacing invalid lanes by zero.
        nvalid = tl.minimum(BC, T - chunk * BC)
        last = tl.sum(tl.where(tl.arange(0, BC) == nvalid - 1, gates, 0.0), axis=0)
        weighted = delta * tl.exp(last - gates)[:, None]
        state = tl.exp(last) * state + tl.dot(tl.trans(keys), weighted.to(keys.dtype), input_precision=IP)


@triton.jit
def _chunk_backward(q, k, v, g, beta, transform, gc, states, dout,
                    pq, pk, dv, pg, pb, T: tl.constexpr, H: tl.constexpr,
                    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    NC: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = (V + BV - 1) // BV
    vb = pid % nv
    bh = pid // nv
    b = bh // HV
    vh = bh - b * HV
    kh = vh // (HV // H)
    ko = tl.arange(0, BK)
    vo = vb * BV + tl.arange(0, BV)
    rr = tl.arange(0, BC)[:, None]
    cc = tl.arange(0, BC)[None, :]
    dstate = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(K * 1.0)
    for rev in range(NC):
        chunk = NC - 1 - rev
        ti = chunk * BC + tl.arange(0, BC)
        valid = ti < T
        kmask = valid[:, None] & (ko[None, :] < K)
        vmask = valid[:, None] & (vo[None, :] < V)
        koff = ((b * T + ti[:, None]) * H + kh) * K + ko[None, :]
        xoff = ((b * T + ti[:, None]) * HV + vh) * V + vo[None, :]
        keys = tl.load(k + koff, mask=kmask, other=0.0)
        queries = tl.load(q + koff, mask=kmask, other=0.0)
        values = tl.load(v + xoff, mask=vmask, other=0.0)
        dy = tl.load(dout + xoff, mask=vmask, other=0.0)
        gates = tl.load(gc + (b * T + ti) * HV + vh, mask=valid, other=0.0)
        strength = tl.load(beta + (b * T + ti) * HV + vh, mask=valid, other=0.0)
        soff = (((b * NC + chunk) * HV + vh) * K + ko[:, None]) * V + vo[None, :]
        state = tl.load(states + soff, mask=(ko[:, None] < K) & (vo[None, :] < V), other=0.0)
        aoff = (((b * NC + chunk) * HV + vh) * BC + rr) * BC + cc
        inv = tl.load(transform + aoff, mask=valid[:, None] & valid[None, :], other=0.0)
        expg = tl.exp(gates)
        pred = tl.dot(keys, state.to(keys.dtype), input_precision=IP)
        residual = strength[:, None] * (values - expg[:, None] * pred)
        delta = tl.dot(inv, residual, input_precision=IP)
        qk = tl.dot(queries, tl.trans(keys), input_precision=IP) * scale
        causal = (rr >= cc) & valid[:, None] & valid[None, :]
        factor = tl.where(causal, tl.exp(gates[:, None] - gates[None, :]), 0.0)
        mixing = qk * factor
        nvalid = tl.minimum(BC, T - chunk * BC)
        last = tl.sum(tl.where(tl.arange(0, BC) == nvalid - 1, gates, 0.0), axis=0)
        rdec = tl.exp(last - gates)

        state_output = expg[:, None] * tl.dot(queries, state.to(queries.dtype), input_precision=IP) * scale
        dquery = expg[:, None] * tl.dot(dy, tl.trans(state.to(dy.dtype)), input_precision=IP) * scale
        dstate0 = tl.dot(tl.trans(queries), (dy * expg[:, None]).to(queries.dtype), input_precision=IP) * scale
        dmix = tl.dot(dy, tl.trans(delta.to(dy.dtype)), input_precision=IP)
        ddelta = tl.dot(tl.trans(mixing.to(dy.dtype)), dy, input_precision=IP)
        weighted = delta * rdec[:, None]
        kdstate = tl.dot(keys, dstate.to(keys.dtype), input_precision=IP)
        dkey = tl.dot(weighted, tl.trans(dstate), input_precision=IP)
        ddelta += kdstate * rdec[:, None]
        dgc = tl.sum(dy * state_output, axis=1) - tl.sum(kdstate * weighted, axis=1)
        dlast = tl.sum(kdstate * weighted) + tl.sum(dstate * state) * tl.exp(last)
        dgc += tl.where(tl.arange(0, BC) == nvalid - 1, dlast, 0.0)
        dstate0 += dstate * tl.exp(last)

        core = dmix * factor
        dquery += tl.dot(core.to(keys.dtype), keys, input_precision=IP) * scale
        dkey += tl.dot(tl.trans(core.to(queries.dtype)), queries, input_precision=IP) * scale
        dgc += tl.sum(dmix * mixing, axis=1) - tl.sum(dmix * mixing, axis=0)

        dinv = tl.dot(ddelta, tl.trans(residual.to(ddelta.dtype)), input_precision=IP)
        dres = tl.dot(tl.trans(inv), ddelta, input_precision=IP)
        dlower = -tl.dot(tl.trans(inv), tl.dot(dinv, tl.trans(inv), input_precision=IP), input_precision=IP)
        strict = (rr > cc) & valid[:, None] & valid[None, :]
        gram = tl.dot(keys, tl.trans(keys), input_precision=IP)
        lf = tl.where(strict, tl.exp(gates[:, None] - gates[None, :]), 0.0)
        dlbase = dlower * lf
        scaled_dl = (dlbase * strength[:, None]).to(keys.dtype)
        dkey += tl.dot(scaled_dl, keys, input_precision=IP)
        dkey += tl.dot(tl.trans(scaled_dl), keys, input_precision=IP)
        dbeta = tl.sum(dlbase * gram, axis=1)
        dgc += tl.sum(dlower * (gram * strength[:, None] * lf), axis=1)
        dgc -= tl.sum(dlower * (gram * strength[:, None] * lf), axis=0)

        raw = values - expg[:, None] * pred
        dbeta += tl.sum(dres * raw, axis=1)
        dvalue = dres * strength[:, None]
        dpred = -dres * (strength * expg)[:, None]
        dgc += tl.sum(dres * (-strength * expg)[:, None] * pred, axis=1)
        dkey += tl.dot(dpred, tl.trans(state.to(dpred.dtype)), input_precision=IP)
        dstate0 += tl.dot(tl.trans(keys), dpred.to(keys.dtype), input_precision=IP)
        dstate = dstate0
        # Reverse the inclusive cumulative sum within this chunk.
        dgates = tl.sum(tl.where(tl.arange(0, BC)[:, None] <= tl.arange(0, BC)[None, :],
                                 dgc[None, :], 0.0), axis=1)
        poff = ((((b * T + ti[:, None]) * HV + vh) * nv + vb) * K + ko[None, :])
        tl.store(pq + poff, dquery, mask=kmask)
        tl.store(pk + poff, dkey, mask=kmask)
        tl.store(dv + xoff, dvalue, mask=vmask)
        soff2 = (((b * T + ti) * HV + vh) * nv + vb)
        tl.store(pg + soff2, dgates, mask=valid)
        tl.store(pb + soff2, dbeta, mask=valid)


@triton.jit
def _reduce_vector(part, out, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)
    ko = tl.arange(0, BK)
    t = pid % T
    z = pid // T
    h = z % H
    b = z // H
    nv = (V + BV - 1) // BV
    groups = HV // H
    acc = tl.zeros((BK,), tl.float32)
    for group in range(groups):
        vh = h * groups + group
        for vb in range(nv):
            off = ((((b * T + t) * HV + vh) * nv + vb) * K + ko)
            acc += tl.load(part + which * (2 * 0) + off, mask=ko < K, other=0.0)
    tl.store(out + ((b * T + t) * H + h) * K + ko, acc, mask=ko < K)


@triton.jit
def _reduce_qk(pq, pk, dq, dk, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
               K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    ko = tl.arange(0, BK)
    t = pid % T
    z = pid // T
    h = z % H
    b = z // H
    nv = (V + BV - 1) // BV
    groups = HV // H
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    for group in range(groups):
        vh = h * groups + group
        for vb in range(nv):
            off = ((((b * T + t) * HV + vh) * nv + vb) * K + ko)
            aq += tl.load(pq + off, mask=ko < K, other=0.0)
            ak += tl.load(pk + off, mask=ko < K, other=0.0)
    outoff = ((b * T + t) * H + h) * K + ko
    tl.store(dq + outoff, aq, mask=ko < K)
    tl.store(dk + outoff, ak, mask=ko < K)


@triton.jit
def _reduce_scalar(pg, pb, dg, db, N: tl.constexpr, NV: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    ag = 0.0
    ab = 0.0
    for vb in range(NV):
        ag += tl.load(pg + pid * NV + vb)
        ab += tl.load(pb + pid * NV + vb)
    tl.store(dg + pid, ag)
    tl.store(db + pid, ab)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        nc = triton.cdiv(T, BC)
        nv = triton.cdiv(V, BV)
        bk = triton.next_power_of_2(K)
        ip = "ieee" if q.dtype == torch.float32 else "tf32"
        transform = torch.empty((B, nc, HV, BC, BC), device=q.device, dtype=torch.float32)
        gc = torch.empty_like(g, dtype=torch.float32)
        states = torch.empty((B, nc, HV, K, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_transform[(nc, B * HV)](k, g, beta, transform, gc, T=T, H=H, HV=HV,
                                        K=K, BK=bk, IP=ip, num_warps=4)
        _scan_emit[(B * HV * nv,)](q, k, v, beta, transform, gc, states, out,
                                    T=T, H=H, HV=HV, K=K, V=V, NC=nc, BK=bk,
                                    IP=ip, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, transform, gc, states)
        ctx.shape = (B, T, H, HV, K, V, nc, nv, bk, ip)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, transform, gc, states = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, HV, K, V, nc, nv, bk, ip = ctx.shape
        pq = torch.empty((B, T, HV, nv, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        pg = torch.empty((B, T, HV, nv), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg)
        dv = torch.empty_like(v)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        _chunk_backward[(B * HV * nv,)](q, k, v, g, beta, transform, gc, states,
                                         dout, pq, pk, dv, pg, pb, T=T, H=H, HV=HV,
                                         K=K, V=V, NC=nc, BK=bk, IP=ip, num_warps=4)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T=T, H=H, HV=HV, K=K, V=V,
                                  BK=bk, num_warps=4)
        _reduce_scalar[(B * T * HV,)](pg, pb, dg, db, N=B*T*HV, NV=nv)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
