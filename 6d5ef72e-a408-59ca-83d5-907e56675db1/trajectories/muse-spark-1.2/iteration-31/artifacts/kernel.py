import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 128, 32
EBLOCK = 2048
GBLOCK = 1024


@triton.jit
def _mm_kernel(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _silu_mul_fwd_kernel(G, U, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    tl.store(OUT + o, g * s * u, mask=m)


@triton.jit
def _silu_mul_bwd_kernel(G, U, DO, DG, DU, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = o < n
    g = tl.load(G + o, mask=m, other=0.0)
    u = tl.load(U + o, mask=m, other=0.0)
    do = tl.load(DO + o, mask=m, other=0.0)
    s = tl.sigmoid(g)
    silu = g * s
    tl.store(DG + o, do * u * (s + silu * (1.0 - s)), mask=m)
    tl.store(DU + o, do * silu, mask=m)


@triton.jit
def _gather_kernel(X, Tok, Y, D, sx0, sx1, sy0, sy1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    tok = tl.load(Tok + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    for base in range(0, D, BLOCK):
        cols = base + offs
        mask = cols < D
        v = tl.load(X + tok * sx0 + cols * sx1, mask=mask, other=0.0)
        tl.store(Y + pid * sy0 + cols * sy1, v, mask=mask)


@triton.jit
def _combine_fwd_kernel(Y, FE, W, Inv, T, A, D, sy0, sy1, sfe0, sfe1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK).to(tl.int64)
    for base in range(0, D, BLOCK):
        cols = base + offs
        mask = cols < D
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for a in range(0, A):
            pos = tl.load(Inv + pid * A + a).to(tl.int64)
            w = tl.load(W + pos)
            fe = tl.load(FE + pos * sfe0 + cols * sfe1, mask=mask, other=0.0)
            acc += w * fe
        tl.store(Y + pid * sy0 + cols * sy1, acc, mask=mask)


@triton.jit
def _combine_bwd_kernel(DY, FE, W, Inv, DFE, DW, T, A, D, sdy0, sdy1, sfe0, sfe1, sdfe0, sdfe1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK).to(tl.int64)
    for a in range(0, A):
        pos = tl.load(Inv + pid * A + a).to(tl.int64)
        acc = 0.0
        for base in range(0, D, BLOCK):
            cols = base + offs
            mask = cols < D
            dy = tl.load(DY + pid * sdy0 + cols * sdy1, mask=mask, other=0.0)
            fe = tl.load(FE + pos * sfe0 + cols * sfe1, mask=mask, other=0.0)
            acc += tl.sum(tl.where(mask, dy * fe, 0.0), axis=0)
            w = tl.load(W + pos)
            tl.store(DFE + pos * sdfe0 + cols * sdfe1, dy * w, mask=mask)
        tl.store(DW + pid * A + a, acc)


@triton.jit
def _dx_expert_kernel(DH, Inv, DX, T, A, D, sdh0, sdh1, sdx0, sdx1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK).to(tl.int64)
    for base in range(0, D, BLOCK):
        cols = base + offs
        mask = cols < D
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for a in range(0, A):
            pos = tl.load(Inv + pid * A + a).to(tl.int64)
            dh = tl.load(DH + pos * sdh0 + cols * sdh1, mask=mask, other=0.0)
            acc += dh
        tl.store(DX + pid * sdx0 + cols * sdx1, acc, mask=mask)


@triton.jit
def _norm_bwd_kernel(Ssel, DW, Sum, DS, T, A: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    s_sum = tl.load(Sum + pid)
    dot = 0.0
    for a in range(0, A):
        dw = tl.load(DW + pid * A + a)
        s = tl.load(Ssel + pid * A + a)
        dot += dw * s
    inv = 1.0 / s_sum
    inv2 = inv * inv
    for a in range(0, A):
        dw = tl.load(DW + pid * A + a)
        tl.store(DS + pid * A + a, dw * inv - dot * inv2)


@triton.jit
def _dlogits_kernel(DSel, Sel, S, DLogits, T, A: tl.constexpr, E: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    for a in range(0, A):
        e = tl.load(Sel + pid * A + a).to(tl.int64)
        ds = tl.load(DSel + pid * A + a)
        s = tl.load(S + pid * E + e)
        tl.store(DLogits + pid * E + e, ds * s * (1.0 - s))


@triton.jit
def _grouped_fwd_kernel(A_ptr, B_ptr, C_ptr, Offs, TileExp, TileM, TileN, K, N, s_a0, s_a1, s_b0, s_b1, s_b2, s_c0, s_c1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(TileExp + pid).to(tl.int64)
    tile_m = tl.load(TileM + pid).to(tl.int64)
    tile_n = tl.load(TileN + pid).to(tl.int64)
    row_base = tl.load(Offs + e).to(tl.int64)
    n_e = tl.load(Offs + e + 1).to(tl.int64) - row_base
    rm = row_base + tile_m + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = tile_n + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(A_ptr + rm[:, None] * s_a0 + rk[None, :] * s_a1, mask=(rm[:, None] < row_base + n_e) & (rk[None, :] < K), other=0.0)
        b = tl.load(B_ptr + e * s_b0 + rk[:, None] * s_b1 + rn[None, :] * s_b2, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C_ptr + rm[:, None] * s_c0 + rn[None, :] * s_c1, acc, mask=(rm[:, None] < row_base + n_e) & (rn[None, :] < N))


@triton.jit
def _grouped_bwd_a_kernel(A_ptr, B_ptr, C_ptr, Offs, TileExp, TileM, TileN, K, N, s_a0, s_a1, s_b0, s_b1, s_b2, s_c0, s_c1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(TileExp + pid).to(tl.int64)
    tile_m = tl.load(TileM + pid).to(tl.int64)
    tile_n = tl.load(TileN + pid).to(tl.int64)
    row_base = tl.load(Offs + e).to(tl.int64)
    n_e = tl.load(Offs + e + 1).to(tl.int64) - row_base
    rm = row_base + tile_m + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = tile_n + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(A_ptr + rm[:, None] * s_a0 + rk[None, :] * s_a1, mask=(rm[:, None] < row_base + n_e) & (rk[None, :] < K), other=0.0)
        b = tl.load(B_ptr + e * s_b0 + rk[:, None] * s_b2 + rn[None, :] * s_b1, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C_ptr + rm[:, None] * s_c0 + rn[None, :] * s_c1, acc, mask=(rm[:, None] < row_base + n_e) & (rn[None, :] < N))


def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK, num_warps=4, num_stages=3)
    return c


class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm_launch(a, b)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm_launch(dc, b.t()), _mm_launch(a.t(), dc)


class _SiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        out = torch.empty_like(g)
        n = g.numel()
        if n > 0:
            _silu_mul_fwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, out, n, BLOCK=EBLOCK)
        return out

    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dout = dout.contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        if n > 0:
            _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, dout, dg, du, n, BLOCK=EBLOCK)
        return dg, du


def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    shift = (E - 1).bit_length()
    if shift == 0:
        shift = 1
    keys = (u << shift) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices


def _build_grouped_tiles(offs: torch.Tensor, N: int, BM_, BN_, device):
    E = offs.numel() - 1
    exps = []
    ms = []
    ns = []
    offs_cpu = offs.cpu().tolist()
    for e in range(E):
        n_e = int(offs_cpu[e + 1] - offs_cpu[e])
        if n_e == 0:
            continue
        for m in range(0, n_e, BM_):
            for n in range(0, N, BN_):
                exps.append(e)
                ms.append(m)
                ns.append(n)
    if not exps:
        return None, None, None, 0
    return (torch.tensor(exps, dtype=torch.int32, device=device), torch.tensor(ms, dtype=torch.int32, device=device), torch.tensor(ns, dtype=torch.int32, device=device), len(exps))


def _grouped_forward(a_sorted, w, offs):
    TA, K = a_sorted.shape
    E, K2, N = w.shape
    assert K == K2
    tile_exp, tile_m, tile_n, num_tiles = _build_grouped_tiles(offs, N, BM, BN, a_sorted.device)
    if num_tiles == 0:
        return torch.empty(TA, N, device=a_sorted.device, dtype=torch.float32)
    c = torch.empty(TA, N, device=a_sorted.device, dtype=torch.float32)
    _grouped_fwd_kernel[(num_tiles,)](a_sorted, w, c, offs, tile_exp, tile_m, tile_n, K, N, a_sorted.stride(0), a_sorted.stride(1), w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=3)
    return c


def _grouped_bwd_a(a_sorted, w, offs, N_out):
    TA, K = a_sorted.shape
    tile_exp, tile_m, tile_n, num_tiles = _build_grouped_tiles(offs, N_out, BM, BN, a_sorted.device)
    if num_tiles == 0:
        return torch.empty(TA, N_out, device=a_sorted.device, dtype=torch.float32)
    c = torch.empty(TA, N_out, device=a_sorted.device, dtype=torch.float32)
    _grouped_bwd_a_kernel[(num_tiles,)](a_sorted, w, c, offs, tile_exp, tile_m, tile_n, K, N_out, a_sorted.stride(0), a_sorted.stride(1), w.stride(0), w.stride(1), w.stride(2), c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=3)
    return c


class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        xf = x.detach().float()
        rwf = router_w.detach().float()
        w1f = w1.detach().float()
        w2f = w2.detach().float()
        T, D = xf.shape
        E = rwf.shape[1]
        F = w1f.shape[2] // 2
        logits = _mm_launch(xf, rwf)
        s = torch.sigmoid(logits)
        with torch.no_grad():
            sel = _select_topA(s, router_b.float(), A)
        s_sel = torch.gather(s, 1, sel)
        s_sum = s_sel.sum(dim=-1)
        w = s_sel / s_sum.unsqueeze(1)
        flat = sel.reshape(-1)
        TA = T * A
        order = torch.argsort(flat, stable=True).to(torch.int32)
        tok_sorted = (order.to(torch.int64) // A).to(torch.int32)
        cnts = torch.bincount(flat, minlength=E)
        offs = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offs[0] = 0
        offs[1:] = torch.cumsum(cnts, 0)
        inv = torch.empty(TA, dtype=torch.int32, device=x.device)
        inv[order.to(torch.int64)] = torch.arange(TA, device=x.device, dtype=torch.int32)
        x_sorted = torch.empty(TA, D, device=x.device, dtype=torch.float32)
        if TA > 0:
            _gather_kernel[(TA,)](xf, tok_sorted, x_sorted, D, xf.stride(0), xf.stride(1), x_sorted.stride(0), x_sorted.stride(1), BLOCK=GBLOCK)
        gu = _grouped_forward(x_sorted, w1f, offs)
        gu_g = gu[:, :F].contiguous() if TA > 0 else gu
        gu_u = gu[:, F:].contiguous() if TA > 0 else gu
        act = torch.empty(TA, F, device=x.device, dtype=torch.float32)
        if TA * F > 0:
            _silu_mul_fwd_kernel[(triton.cdiv(TA * F, EBLOCK),)](gu_g, gu_u, act, TA * F, BLOCK=EBLOCK)
        fe = _grouped_forward(act, w2f, offs)
        w_sorted = w.reshape(-1)[order.to(torch.int64)]
        y = torch.zeros(T, D, device=x.device, dtype=torch.float32)
        if T > 0 and TA > 0:
            _combine_fwd_kernel[(T,)](y, fe, w_sorted, inv, T, A, D, y.stride(0), y.stride(1), fe.stride(0), fe.stride(1), BLOCK=GBLOCK)
        ctx.save_for_backward(xf, rwf, w1f, w2f, s, sel, w, s_sum, offs, order, inv, x_sorted, gu, act, fe, w_sorted, s_sel)
        ctx.D = D
        ctx.F = F
        ctx.E = E
        ctx.T = T
        ctx.A = A
        ctx.TA = TA
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, s, sel, w, s_sum, offs, order, inv, x_sorted, gu, act, fe, w_sorted, s_sel = ctx.saved_tensors
        D = ctx.D; F = ctx.F; E = ctx.E; T = ctx.T; A = ctx.A; TA = ctx.TA
        dyf = dy.float().contiguous()
        d_fe = torch.empty(TA, D, device=dy.device, dtype=torch.float32)
        dw_token = torch.empty(T, A, device=dy.device, dtype=torch.float32)
        if T > 0 and TA > 0:
            _combine_bwd_kernel[(T,)](dyf, fe, w_sorted, inv, d_fe, dw_token, T, A, D, dyf.stride(0), dyf.stride(1), fe.stride(0), fe.stride(1), d_fe.stride(0), d_fe.stride(1), BLOCK=GBLOCK)
        elif TA > 0:
            d_fe.zero_()
        ds_sel = torch.empty(T, A, device=dy.device, dtype=torch.float32)
        if T > 0:
            _norm_bwd_kernel[(T,)](s_sel, dw_token, s_sum, ds_sel, T, A)
        dlogits = torch.zeros(T, E, device=dy.device, dtype=torch.float32)
        if T > 0:
            _dlogits_kernel[(T,)](ds_sel, sel, s, dlogits, T, A, E)
        dx_router = _mm_launch(dlogits, rwf.t())
        drw = _mm_launch(xf.t(), dlogits)
        d_act = _grouped_bwd_a(d_fe, w2f, offs, F)
        n_act = TA * F
        dg = torch.empty(TA, F, device=dy.device, dtype=torch.float32)
        du = torch.empty(TA, F, device=dy.device, dtype=torch.float32)
        if n_act > 0:
            gu_g = gu[:, :F].contiguous()
            gu_u = gu[:, F:].contiguous()
            _silu_mul_bwd_kernel[(triton.cdiv(n_act, EBLOCK),)](gu_g, gu_u, d_act, dg, du, n_act, BLOCK=EBLOCK)
        d_gu = torch.cat([dg, du], dim=1) if n_act > 0 else torch.empty(TA, 2 * F, device=dy.device, dtype=torch.float32)
        d_h = _grouped_bwd_a(d_gu, w1f, offs, D)
        dx_expert = torch.empty(T, D, device=dy.device, dtype=torch.float32)
        if T > 0 and TA > 0:
            _dx_expert_kernel[(T,)](d_h, inv, dx_expert, T, A, D, d_h.stride(0), d_h.stride(1), dx_expert.stride(0), dx_expert.stride(1), BLOCK=GBLOCK)
        elif T > 0:
            dx_expert.zero_()
        dx = dx_router + dx_expert if T > 0 else dx_router
        dw1 = torch.zeros_like(w1f)
        dw2 = torch.zeros_like(w2f)
        offs_cpu = offs.cpu().tolist()
        for e in range(E):
            s0 = int(offs_cpu[e]); e1 = int(offs_cpu[e+1]); n_e = e1 - s0
            if n_e == 0:
                continue
            xs = x_sorted[s0:e1]
            dg_e = d_gu[s0:e1]
            dw1[e] = _mm_launch(xs.t(), dg_e)
            act_e = act[s0:e1]
            dfe_e = d_fe[s0:e1]
            dw2[e] = _mm_launch(act_e.t(), dfe_e)
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (dx.to(dx_t), drw.to(drw_t), None, dw1.to(dw1_t), dw2.to(dw2_t), None)


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
