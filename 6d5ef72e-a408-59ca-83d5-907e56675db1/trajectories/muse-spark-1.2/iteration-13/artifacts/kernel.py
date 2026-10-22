import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048
GBM, GBN, GBK = 64, 64, 32

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
def _grouped_fwd_kernel(X_ptr, W_ptr, Out_ptr, TileExp_ptr, TileRow_ptr, TileNRows_ptr, K, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    expert = tl.load(TileExp_ptr + pid_m).to(tl.int64)
    row0 = tl.load(TileRow_ptr + pid_m).to(tl.int64)
    nrows = tl.load(TileNRows_ptr + pid_m).to(tl.int64)
    col0 = pid_n * BLOCK_N
    rm = row0 + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = col0 + tl.arange(0, BLOCK_N).to(tl.int64)
    row_mask = tl.arange(0, BLOCK_M) < nrows
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cur_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = cur_k < K
        a = tl.load(X_ptr + rm[:, None] * K + cur_k[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(W_ptr + expert * K * N + cur_k[:, None] * N + rn[None, :], mask=k_mask[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(Out_ptr + rm[:, None] * N + rn[None, :], acc, mask=row_mask[:, None] & (rn[None, :] < N))

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
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
        _silu_mul_fwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, out, n, BLOCK=EBLOCK)
        return out
    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dout = dout.contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        _silu_mul_bwd_kernel[(triton.cdiv(n, EBLOCK),)](g, u, dout, dg, du, n, BLOCK=EBLOCK)
        return dg, du

def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 10) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

def _build_tiles(counts, offsets, block_m):
    E = counts.numel()
    te, tr, tn = [], [], []
    for e in range(E):
        s = int(offsets[e].item())
        e1 = int(offsets[e + 1].item())
        for cur in range(s, e1, block_m):
            te.append(e)
            tr.append(cur)
            tn.append(min(block_m, e1 - cur))
    if not te:
        dev = counts.device
        return torch.empty(0, dtype=torch.int32, device=dev), torch.empty(0, dtype=torch.int32, device=dev), torch.empty(0, dtype=torch.int32, device=dev)
    dev = counts.device
    return torch.tensor(te, dtype=torch.int32, device=dev), torch.tensor(tr, dtype=torch.int32, device=dev), torch.tensor(tn, dtype=torch.int32, device=dev)

def _grouped_forward(X_sorted, W, tile_exp, tile_row, tile_nrows):
    TA, K = X_sorted.shape
    _, _, N = W.shape
    out = torch.empty(TA, N, device=X_sorted.device, dtype=torch.float32)
    if TA == 0 or tile_exp.numel() == 0:
        return out
    ntiles_n = triton.cdiv(N, GBN)
    _grouped_fwd_kernel[(tile_exp.numel(), ntiles_n)](X_sorted, W, out, tile_exp, tile_row, tile_nrows, K, N, BLOCK_M=GBM, BLOCK_N=GBN, BLOCK_K=GBK)
    return out

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_sorted, W, tile_exp, tile_row, tile_nrows, offsets):
        ctx.save_for_backward(X_sorted, W, tile_exp, tile_row, tile_nrows, offsets)
        return _grouped_forward(X_sorted, W, tile_exp, tile_row, tile_nrows)
    @staticmethod
    def backward(ctx, grad_out):
        X_sorted, W, tile_exp, tile_row, tile_nrows, offsets = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        Wt = W.transpose(1, 2).contiguous()
        grad_X = _grouped_forward(grad_out, Wt, tile_exp, tile_row, tile_nrows)
        E = W.shape[0]
        grad_W = torch.zeros_like(W)
        for e in range(E):
            s = int(offsets[e].item())
            e1 = int(offsets[e + 1].item())
            if e1 == s:
                continue
            xg = X_sorted[s:e1]
            do = grad_out[s:e1]
            grad_W[e] = _mm_launch(xg.t(), do)
        return grad_X, grad_W, None, None, None, None

class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        with torch.enable_grad():
            xf = x.detach().float().requires_grad_(True)
            rwf = router_w.detach().float().requires_grad_(True)
            w1f = w1.detach().float().requires_grad_(True)
            w2f = w2.detach().float().requires_grad_(True)
            T, D = xf.shape
            E = rwf.shape[1]
            F = w1f.shape[2] // 2
            s = torch.sigmoid(xf @ rwf)
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            sel_flat = sel.reshape(-1)
            order = torch.argsort(sel_flat, stable=True)
            token_sorted = order // A
            counts = torch.bincount(sel_flat[order], minlength=E)
            offsets = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
            if E > 0:
                offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
            tile_exp, tile_row, tile_nrows = _build_tiles(counts, offsets, GBM)
            X_sorted = xf[token_sorted]
            TA = sel_flat.numel()
            if TA == 0:
                y = xf.new_zeros(T, D)
            else:
                gu_sorted = _GroupedMM.apply(X_sorted, w1f, tile_exp, tile_row, tile_nrows, offsets)
                act_sorted = _SiLUMul.apply(gu_sorted[:, :F].contiguous(), gu_sorted[:, F:].contiguous())
                out_sorted = _GroupedMM.apply(act_sorted, w2f, tile_exp, tile_row, tile_nrows, offsets)
                inv = torch.argsort(order)
                out_unsorted = out_sorted[inv]
                out_view = out_unsorted.view(T, A, D)
                w_view = w.view(T, A, 1)
                y = (out_view * w_view).sum(dim=1)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None

def fused_moe(x: torch.Tensor, router_w: torch.Tensor, router_b: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, A: int) -> torch.Tensor:
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
