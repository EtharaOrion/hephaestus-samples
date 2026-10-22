import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 128, 32
EBLOCK = 1024

@triton.jit
def _fast_mm_kernel(A, B, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
        a = tl.load(A + rm[:, None] * sam + rk[None, :] * sak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        a_hi = a.to(tl.bfloat16)
        a_hi_f32 = a_hi.to(tl.float32)
        a_lo = (a - a_hi_f32).to(tl.bfloat16)
        b_hi = b.to(tl.bfloat16)
        b_hi_f32 = b_hi.to(tl.float32)
        b_lo = (b - b_hi_f32).to(tl.bfloat16)
        acc += tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))

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

def _fast_mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0 or K == 0:
        return c
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fast_mm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c

class _FastMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _fast_mm_launch(a, b)
    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _fast_mm_launch(dc, b.t()), _fast_mm_launch(a.t(), dc)

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
            s = torch.sigmoid(_FastMM.apply(xf, rwf))
            w1s = w1f.unbind(0)
            w2s = w2f.unbind(0)
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat_sel = sel.reshape(-1)
            flat_w = w.reshape(-1)
            N = flat_sel.numel()
            perm = torch.argsort(flat_sel, stable=True)
            sorted_w = flat_w[perm]
            sorted_tok = torch.div(perm, A, rounding_mode='floor')
            counts = torch.bincount(flat_sel, minlength=E)
            counts_cpu = counts.detach().cpu().tolist()
            row_starts = []
            cur = 0
            for c in counts_cpu:
                row_starts.append(cur)
                cur += c
            tok_slices = []
            w_slices = []
            for e in range(E):
                n = counts_cpu[e]
                if n == 0:
                    tok_slices.append(None)
                    w_slices.append(None)
                else:
                    rs = row_starts[e]
                    tok_slices.append(sorted_tok[rs:rs+n])
                    w_slices.append(sorted_w[rs:rs+n])
            outs1 = []
            for e in range(E):
                ts = tok_slices[e]
                if ts is None:
                    continue
                a_slice = xf.index_select(0, ts)
                out = _FastMM.apply(a_slice, w1s[e])
                outs1.append(out)
            if outs1:
                h_sorted = torch.cat(outs1, dim=0)
            else:
                h_sorted = torch.empty(N, 2*F, device=xf.device, dtype=torch.float32)
            gate = h_sorted[:, :F].contiguous()
            up = h_sorted[:, F:].contiguous()
            act_sorted = _SiLUMul.apply(gate, up)
            outs2 = []
            for e in range(E):
                if tok_slices[e] is None:
                    continue
                rs = row_starts[e]
                n = counts_cpu[e]
                a_slice = act_sorted[rs:rs+n]
                out = _FastMM.apply(a_slice, w2s[e])
                outs2.append(out)
            if outs2:
                fe_sorted = torch.cat(outs2, dim=0)
            else:
                fe_sorted = torch.empty(N, D, device=xf.device, dtype=torch.float32)
            y = xf.new_zeros(T, D)
            for e in range(E):
                ts = tok_slices[e]
                if ts is None:
                    continue
                rs = row_starts[e]
                ws = w_slices[e]
                fe_slice = fe_sorted[rs:rs+ts.numel()] * ws.unsqueeze(1)
                y = y.index_add(0, ts, fe_slice)
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
