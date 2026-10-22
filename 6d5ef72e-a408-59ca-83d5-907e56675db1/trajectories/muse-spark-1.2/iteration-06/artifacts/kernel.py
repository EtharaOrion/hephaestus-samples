import torch
import triton
import triton.language as tl

torch.use_deterministic_algorithms(True)

EBLOCK = 2048


@triton.jit
def _mm_kernel(A, B, C, M, N, K, sAm, sAk, sBk, sBn, sCm, sCn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * sAm + rk[None, :] * sAk,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + rk[:, None] * sBk + rn[None, :] * sBn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rm[:, None] * sCm + rn[None, :] * sCn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return c
    if M >= 128 and N >= 128:
        BM, BN, BK, W, S = 64, 64, 32, 4, 3
    elif M >= 64 or N >= 64:
        BM, BN, BK, W, S = 32, 64, 32, 4, 3
    else:
        BM, BN, BK, W, S = 32, 32, 32, 4, 2
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm_kernel[grid](a, b, c, M, N, K,
                     a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1),
                     BM=BM, BN=BN, BK=BK, num_warps=W, num_stages=S)
    return c


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





class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm(a, b)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.contiguous()
        return _mm(dc, b.t().contiguous()), _mm(a.t().contiguous(), dc)


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


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx)
        ctx.src_shape = src.shape
        return src.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        grad_src = torch.zeros(ctx.src_shape, device=grad_out.device, dtype=grad_out.dtype)
        grad_src.index_add_(0, idx, grad_out.contiguous())
        return grad_src, None


@triton.jit
def _gather_kernel(SRC, IDX, DST, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.load(IDX + pid)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    vals = tl.load(SRC + idx * 2048 + offs, mask=mask)
    tl.store(DST + pid * 2048 + offs, vals, mask=mask)


@triton.jit
def _scale_kernel(X, W, OUT, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.load(W + pid)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + pid * 2048 + offs, mask=mask)
    tl.store(OUT + pid * 2048 + offs, x * w, mask=mask)


@triton.jit
def _scatter_kernel(SRC, IDX, DST, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.load(IDX + pid)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    vals = tl.load(SRC + pid * 2048 + offs, mask=mask)
    cur = tl.load(DST + idx * 2048 + offs, mask=mask)
    tl.store(DST + idx * 2048 + offs, cur + vals, mask=mask)


class _Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, w):
        ctx.save_for_backward(src, w)
        return src * w.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_out):
        src, w = ctx.saved_tensors
        grad_src = grad_out * w.unsqueeze(1)
        grad_w = (grad_out * src).sum(dim=1)
        return grad_src, grad_w








class _ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, out_shape):
        ctx.save_for_backward(idx)
        ctx.out_shape = out_shape
        T, D = out_shape
        dst = torch.zeros(T, D, device=src.device, dtype=src.dtype)
        dst.index_add_(0, idx, src)
        ctx.N = idx.shape[0]
        return dst

    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        grad_src = grad_out.index_select(0, idx)
        return grad_src, None, None


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
            s = torch.sigmoid(_MM.apply(xf, rwf))
            with torch.no_grad():
                sel = _select_topA(s.detach(), router_b.float(), A)
            s_sel = torch.gather(s, 1, sel)
            w = s_sel / s_sel.sum(dim=-1, keepdim=True)
            sel_flat = sel.reshape(-1)
            w_flat = w.reshape(-1)
            N = T * A
            tok_rep = torch.arange(T, device=x.device).repeat_interleave(A)
            sort_key = sel_flat.to(torch.int64) * (T + 1) + tok_rep.to(torch.int64)
            order = torch.argsort(sort_key)
            tok_s = tok_rep[order]
            w_s = w_flat[order]
            x_s = _Gather.apply(xf, tok_s)
            counts = torch.bincount(sel_flat, minlength=E)
            w1s = w1f.unbind(0)
            w2s = w2f.unbind(0)
            h = torch.empty(N, 2 * F, device=x.device, dtype=torch.float32)
            off = 0
            for e in range(E):
                cnt = int(counts[e].item())
                if cnt == 0:
                    continue
                h[off:off + cnt] = _MM.apply(x_s[off:off + cnt], w1s[e])
                off += cnt
            act = _SiLUMul.apply(h[:, :F].contiguous(), h[:, F:].contiguous())
            y_s = torch.empty(N, D, device=x.device, dtype=torch.float32)
            off = 0
            for e in range(E):
                cnt = int(counts[e].item())
                if cnt == 0:
                    continue
                y_s[off:off + cnt] = _MM.apply(act[off:off + cnt], w2s[e])
                off += cnt
            y_s_scaled = _Scale.apply(y_s, w_s)
            y = _ScatterAdd.apply(y_s_scaled, tok_s, (T, D))
        ctx.save_for_backward(xf, rwf, w1f, w2f, y)
        ctx.dtypes = (x.dtype, router_w.dtype, w1.dtype, w2.dtype)
        return y.detach().to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xf, rwf, w1f, w2f, y = ctx.saved_tensors
        gx, grw, gw1, gw2 = torch.autograd.grad(y, (xf, rwf, w1f, w2f), dy.float())
        a, b, c, d = ctx.dtypes
        return gx.to(a), grw.to(b), None, gw1.to(c), gw2.to(d), None


def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
