import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048
GBM = 32
GBN = 32
G_BM = 64
G_BN = 64
G_BK = 32

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
def _gather_kernel(X, ROWS, OUT, N, D, s0: tl.constexpr, s1: tl.constexpr, os0: tl.constexpr, os1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_m = rm < N
    mask_n = rn < D
    rows = tl.load(ROWS + rm, mask=mask_m, other=0).to(tl.int64)
    mask = mask_m[:, None] & mask_n[None, :]
    vals = tl.load(X + rows[:, None] * s0 + rn[None, :] * s1, mask=mask, other=0.0)
    tl.store(OUT + rm[:, None] * os0 + rn[None, :] * os1, vals, mask=mask)

@triton.jit
def _scatter_kernel(Y, ROWS, SRC, N, D, ys0: tl.constexpr, ys1: tl.constexpr, ss0: tl.constexpr, ss1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_m = rm < N
    mask_n = rn < D
    mask = mask_m[:, None] & mask_n[None, :]
    rows = tl.load(ROWS + rm, mask=mask_m, other=0).to(tl.int64)
    src = tl.load(SRC + rm[:, None] * ss0 + rn[None, :] * ss1, mask=mask, other=0.0)
    y_old = tl.load(Y + rows[:, None] * ys0 + rn[None, :] * ys1, mask=mask, other=0.0)
    tl.store(Y + rows[:, None] * ys0 + rn[None, :] * ys1, y_old + src, mask=mask)

@triton.jit
def _grouped_mm_kernel(X, W, Out, Expert, Row, Col, Limit, N_total, K, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(Expert + pid).to(tl.int64)
    row = tl.load(Row + pid).to(tl.int64)
    col = tl.load(Col + pid).to(tl.int64)
    lim = tl.load(Limit + pid).to(tl.int64)
    rm = row + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = col + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        x = tl.load(X + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < lim) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + e * K * N + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, w, input_precision="ieee")
    tl.store(Out + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < lim) & (rn[None, :] < N))

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    _mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK)
    return c

def _grouped_launch(xg: torch.Tensor, w: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    N_total, K = xg.shape
    E, K2, N = w.shape
    out = torch.empty(N_total, N, device=xg.device, dtype=torch.float32)
    cnt_list = cnt.tolist()
    starts = []
    s = 0
    for c in cnt_list:
        starts.append(s)
        s += c
    expert_arr = []
    row_arr = []
    col_arr = []
    lim_arr = []
    for e, (st, ne) in enumerate(zip(starts, cnt_list)):
        ne = int(ne)
        m_tiles = (ne + G_BM - 1) // G_BM
        n_tiles = (N + G_BN - 1) // G_BN
        lim = st + ne
        for tm in range(m_tiles):
            row = st + tm * G_BM
            for tn in range(n_tiles):
                col = tn * G_BN
                expert_arr.append(e)
                row_arr.append(row)
                col_arr.append(col)
                lim_arr.append(lim)
    total = len(expert_arr)
    w_c = w.contiguous()
    _grouped_mm_kernel[(total,)](xg, w_c, out, torch.tensor(expert_arr, dtype=torch.int32, device=xg.device), torch.tensor(row_arr, dtype=torch.int32, device=xg.device), torch.tensor(col_arr, dtype=torch.int32, device=xg.device), torch.tensor(lim_arr, dtype=torch.int32, device=xg.device), N_total, K, N, BLOCK_M=G_BM, BLOCK_N=G_BN, BLOCK_K=G_BK)
    return out

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

class _GatherTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows, cnt):
        ctx.save_for_backward(rows, cnt)
        ctx.T = x.shape[0]
        ctx.D = x.shape[1]
        N = rows.numel()
        D = x.shape[1]
        out = torch.empty(N, D, device=x.device, dtype=torch.float32)
        _gather_kernel[(triton.cdiv(N if N else 1, GBM), triton.cdiv(D, GBN))](x, rows, out, N, D, x.stride(0), x.stride(1), out.stride(0), out.stride(1), BLOCK_M=GBM, BLOCK_N=GBN)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        rows, cnt = ctx.saved_tensors
        T = ctx.T
        D = ctx.D
        N = rows.numel()
        grad_x = torch.zeros(T, D, device=grad_out.device, dtype=torch.float32)
        grad_out_c = grad_out.contiguous()
        cnt_list = cnt.tolist()
        starts = []
        s = 0
        for c in cnt_list:
            starts.append(s)
            s += c
        for e in range(cnt.numel()):
            st = starts[e]
            ne = int(cnt_list[e])
            en = st + ne
            r = rows[st:en]
            g = grad_out_c[st:en]
            _scatter_kernel[(triton.cdiv(r.numel() if r.numel() else 1, GBM), triton.cdiv(D, GBN))](grad_x, r, g, r.numel(), D, grad_x.stride(0), grad_x.stride(1), g.stride(0), g.stride(1), BLOCK_M=GBM, BLOCK_N=GBN)
        return grad_x, None, None

class _ScatterTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, rows, src):
        ctx.save_for_backward(rows)
        out = y.clone()
        N = rows.numel()
        D = y.shape[1]
        src_c = src.contiguous()
        _scatter_kernel[(triton.cdiv(N if N else 1, GBM), triton.cdiv(D, GBN))](out, rows, src_c, N, D, out.stride(0), out.stride(1), src_c.stride(0), src_c.stride(1), BLOCK_M=GBM, BLOCK_N=GBN)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        rows, = ctx.saved_tensors
        N = rows.numel()
        D = grad_out.shape[1]
        grad_src = torch.empty(N, D, device=grad_out.device, dtype=torch.float32)
        _gather_kernel[(triton.cdiv(N if N else 1, GBM), triton.cdiv(D, GBN))](grad_out, rows, grad_src, N, D, grad_out.stride(0), grad_out.stride(1), grad_src.stride(0), grad_src.stride(1), BLOCK_M=GBM, BLOCK_N=GBN)
        return grad_out, None, grad_src

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xg, w, cnt):
        ctx.save_for_backward(xg, w, cnt)
        return _grouped_launch(xg, w, cnt)
    @staticmethod
    def backward(ctx, grad_out):
        xg, w, cnt = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        E = w.shape[0]
        cnt_list = cnt.tolist()
        starts = []
        s = 0
        for c in cnt_list:
            starts.append(s)
            s += c
        grad_xg = torch.empty_like(xg)
        grad_w = torch.zeros_like(w)
        for e in range(E):
            st = starts[e]
            ne = int(cnt_list[e])
            en = st + ne
            g_slice = grad_out_c[st:en]
            x_slice = xg[st:en]
            grad_xg[st:en] = _mm_launch(g_slice, w[e].t())
            grad_w[e] = _mm_launch(x_slice.t(), g_slice)
        return grad_xg, grad_w, None

def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 11) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

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
            s = torch.sigmoid(_MM.apply(xf, rwf))
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w_norm = s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat_sel = sel.reshape(-1)
            flat_w = w_norm.reshape(-1)
            tok_idx = torch.arange(T, device=x.device, dtype=torch.int64).repeat_interleave(A)
            order = torch.argsort(flat_sel)
            sorted_tok = tok_idx[order]
            sorted_w = flat_w[order]
            cnt = torch.bincount(flat_sel, minlength=E)
            xg = _GatherTriton.apply(xf, sorted_tok, cnt)
            gu = _GroupedMM.apply(xg, w1f, cnt)
            act = _SiLUMul.apply(gu[:, :F].contiguous(), gu[:, F:].contiguous())
            y_exp = _GroupedMM.apply(act, w2f, cnt)
            y = xf.new_zeros(T, D)
            cnt_list = cnt.tolist()
            starts = []
            s_acc = 0
            for c in cnt_list:
                starts.append(s_acc)
                s_acc += c
            for e in range(E):
                st = starts[e]
                ne = int(cnt_list[e])
                en = st + ne
                rows = sorted_tok[st:en]
                src = y_exp[st:en]
                w_slice = sorted_w[st:en]
                fe_w = src * w_slice[:, None]
                y = _ScatterTriton.apply(y, rows, fe_w)
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        dx_t, drw_t, dw1_t, dw2_t = ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None)

def fused_moe(x: torch.Tensor, router_w: torch.Tensor, router_b: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, A: int) -> torch.Tensor:
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
