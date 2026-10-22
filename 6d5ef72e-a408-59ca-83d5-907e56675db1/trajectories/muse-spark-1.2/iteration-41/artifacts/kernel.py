import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 2048

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
    tl.store(OUT + o, tl.load(G + o, mask=m, other=0.0) * tl.sigmoid(tl.load(G + o, mask=m, other=0.0)) * tl.load(U + o, mask=m, other=0.0), mask=m)

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
def _gather_fwd_kernel(X, IDX, OUT, N, D, s_x0, s_x1, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    idx = tl.load(IDX + rm, mask=rm < N, other=0)
    mask = (rm[:, None] < N) & (rn[None, :] < D)
    tl.store(OUT + rm[:, None] * D + rn[None, :], tl.load(X + idx[:, None] * s_x0 + rn[None, :] * s_x1, mask=mask, other=0.0), mask=mask)

@triton.jit
def _gather_bwd_kernel(GOUT, INV, GIN, T, D, A: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    rn = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for a in range(A):
        p = tl.load(INV + rt * A + a, mask=rt < T, other=0)
        acc += tl.load(GOUT + p[:, None] * D + rn[None, :], mask=(rt[:, None] < T) & (rn[None, :] < D), other=0.0)
    tl.store(GIN + rt[:, None] * D + rn[None, :], acc, mask=(rt[:, None] < T) & (rn[None, :] < D))

@triton.jit
def _combine_fwd_kernel(YP, WS, INV, OUT, T, D, A: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    rn = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for a in range(A):
        p = tl.load(INV + rt * A + a, mask=rt < T, other=0)
        acc += tl.load(WS + p, mask=rt < T, other=0.0)[:, None] * tl.load(YP + p[:, None] * D + rn[None, :], mask=(rt[:, None] < T) & (rn[None, :] < D), other=0.0)
    tl.store(OUT + rt[:, None] * D + rn[None, :], acc, mask=(rt[:, None] < T) & (rn[None, :] < D))

@triton.jit
def _combine_bwd_yp_kernel(GY, WS, IDX, GYP, N, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rn = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
    idx = tl.load(IDX + rm, mask=rm < N, other=0)
    tl.store(GYP + rm[:, None] * D + rn[None, :], tl.load(WS + rm, mask=rm < N, other=0.0)[:, None] * tl.load(GY + idx[:, None] * D + rn[None, :], mask=(rm[:, None] < N) & (rn[None, :] < D), other=0.0), mask=(rm[:, None] < N) & (rn[None, :] < D))

@triton.jit
def _combine_bwd_ws_kernel(GY, YP, IDX, GWS, N, D: tl.constexpr, BLOCK: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    idx = tl.load(IDX + o, mask=o < N, other=0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for d0 in range(0, D, BD):
        rn = d0 + tl.arange(0, BD).to(tl.int64)
        acc += tl.sum(tl.load(GY + idx[:, None] * D + rn[None, :], mask=(o[:, None] < N) & (rn[None, :] < D), other=0.0) * tl.load(YP + o[:, None] * D + rn[None, :], mask=(o[:, None] < N) & (rn[None, :] < D), other=0.0), axis=1)
    tl.store(GWS + o, acc, mask=o < N)

@triton.jit
def _grouped_fwd_kernel(X, W, OUT, OFFS, TILES, N, K, Nout, s_x0, s_x1, s_w0, s_w1, s_w2, s_o0, s_o1, BM_: tl.constexpr, BN_: tl.constexpr, BK_: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    e = tl.load(TILES + pid * 3 + 0).to(tl.int64)
    tm = tl.load(TILES + pid * 3 + 1).to(tl.int64)
    tn = tl.load(TILES + pid * 3 + 2).to(tl.int64)
    off = tl.load(OFFS + e - 1, mask=e > 0, other=0).to(tl.int64)
    off_e = tl.load(OFFS + e).to(tl.int64)
    rm = off + tm * BM_ + tl.arange(0, BM_).to(tl.int64)
    rn = tn * BN_ + tl.arange(0, BN_).to(tl.int64)
    acc = tl.zeros((BM_, BN_), dtype=tl.float32)
    for k0 in range(0, K, BK_):
        rk = k0 + tl.arange(0, BK_).to(tl.int64)
        a = tl.load(X + rm[:, None] * s_x0 + rk[None, :] * s_x1, mask=(rm[:, None] < off_e) & (rk[None, :] < K), other=0.0)
        b = tl.load(W + e * s_w0 + rk[:, None] * s_w1 + rn[None, :] * s_w2, mask=(rk[:, None] < K) & (rn[None, :] < Nout), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(OUT + rm[:, None] * s_o0 + rn[None, :] * s_o1, acc, mask=(rm[:, None] < off_e) & (rn[None, :] < Nout))

def _mm_launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    gm = triton.cdiv(M, BM) if M > 0 else 1
    gn = triton.cdiv(N, BN) if N > 0 else 1
    _mm_kernel[(gm, gn)](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLK_M=BM, BLK_N=BN, BLK_K=BK, num_warps=4, num_stages=3)
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
        _silu_mul_fwd_kernel[(triton.cdiv(g.numel(), EBLOCK),)](g, u, out, g.numel(), BLOCK=EBLOCK)
        return out
    @staticmethod
    def backward(ctx, dout):
        g, u = ctx.saved_tensors
        dg, du = torch.empty_like(g), torch.empty_like(u)
        _silu_mul_bwd_kernel[(triton.cdiv(g.numel(), EBLOCK),)](g, u, dout.contiguous(), dg, du, g.numel(), BLOCK=EBLOCK)
        return dg, du

class _GatherSorted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, inv, A):
        ctx.save_for_backward(idx, inv); ctx.A=int(A); ctx.T=x.shape[0]
        out = torch.empty(idx.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        _gather_fwd_kernel[(triton.cdiv(idx.shape[0], 32), triton.cdiv(x.shape[1], 32))](x, idx, out, idx.shape[0], x.shape[1], x.stride(0), x.stride(1), BLOCK_M=32, BLOCK_D=32)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        idx, inv = ctx.saved_tensors
        grad_x = torch.empty(ctx.T, grad_out.shape[1], device=grad_out.device, dtype=grad_out.dtype)
        _gather_bwd_kernel[(triton.cdiv(ctx.T, 32), triton.cdiv(grad_out.shape[1], 32))](grad_out.contiguous(), inv, grad_x, ctx.T, grad_out.shape[1], ctx.A, BLOCK_T=32, BLOCK_D=32)
        return grad_x, None, None, None

class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, yp, ws, inv, tok, T, D, A):
        ctx.save_for_backward(yp, ws, tok, inv); ctx.T=int(T); ctx.D=int(D); ctx.A=int(A)
        out = torch.empty(T, D, device=yp.device, dtype=yp.dtype)
        _combine_fwd_kernel[(triton.cdiv(T, 32), triton.cdiv(D, 32))](yp, ws, inv, out, T, D, A, BLOCK_T=32, BLOCK_D=32)
        return out
    @staticmethod
    def backward(ctx, grad_out):
        yp, ws, tok, inv = ctx.saved_tensors
        N = yp.shape[0]
        grad_yp = torch.empty_like(yp)
        grad_ws = torch.empty_like(ws)
        _combine_bwd_yp_kernel[(triton.cdiv(N, 32), triton.cdiv(grad_out.shape[1], 32))](grad_out.contiguous(), ws, tok, grad_yp, N, grad_out.shape[1], BLOCK_M=32, BLOCK_D=32)
        _combine_bwd_ws_kernel[(triton.cdiv(N, 1024),)](grad_out.contiguous(), yp, tok, grad_ws, N, grad_out.shape[1], BLOCK=1024, BD=32)
        return grad_yp, grad_ws, None, None, None, None, None

def _select_topA(s: torch.Tensor, router_b: torch.Tensor, A: int) -> torch.Tensor:
    E = s.shape[-1]
    b = s + router_b
    b = torch.where(b == 0.0, torch.zeros_like(b), b).contiguous()
    ib = b.view(torch.int32).to(torch.int64)
    u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx = torch.arange(E, device=s.device, dtype=torch.int64)
    keys = (u << 11) | (E - 1 - e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

def _make_tiles(offs_list, Nout, BM_, BN_):
    tiles=[]
    for e,(s,e_off) in enumerate(zip([0]+offs_list[:-1], offs_list)):
        me=e_off-s
        ntm = (me + BM_ - 1)//BM_ if me > 0 else 1
        for tm in range(ntm):
            for tn in range((Nout+BN_-1)//BN_):
                tiles.append([e,tm,tn])
    return torch.tensor(tiles, dtype=torch.int64, device="cuda")

def _grouped_forward(x, w, offs, tiles, N, K, Nout):
    out = torch.empty(N, Nout, device=x.device, dtype=torch.float32)
    _grouped_fwd_kernel[(tiles.shape[0],)](x, w, out, offs, tiles, N, K, Nout, x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2), out.stride(0), out.stride(1), BM_=64, BN_=64, BK_=32, num_warps=4, num_stages=3)
    return out

class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offs, N, K, Nout, tiles):
        ctx.save_for_backward(x, w, offs, tiles); ctx.N=N; ctx.K=K; ctx.Nout=Nout
        return _grouped_forward(x,w,offs,tiles,N,K,Nout)
    @staticmethod
    def backward(ctx, grad_out):
        x,w,offs,tiles = ctx.saved_tensors
        offs_list = offs.tolist()
        grad_x = torch.empty_like(x)
        prev=0
        w_list = w.unbind(0)
        gw_list=[]
        for e,cur in enumerate(offs_list):
            xs = x[prev:cur]
            gs = grad_out[prev:cur]
            grad_x[prev:cur]=_MM.apply(gs, w_list[e].t())
            gw_list.append(_MM.apply(xs.t(), gs))
            prev=cur
        grad_w = torch.stack(gw_list, dim=0)
        return grad_x, grad_w, None, None, None, None, None

class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A=int(A)
        with torch.enable_grad():
            xf=x.detach().float().requires_grad_(True)
            rwf=router_w.detach().float().requires_grad_(True)
            w1f=w1.detach().float().requires_grad_(True)
            w2f=w2.detach().float().requires_grad_(True)
            T,D=xf.shape; E=rwf.shape[1]; F=w1f.shape[2]//2
            s=torch.sigmoid(_MM.apply(xf, rwf))
            with torch.no_grad():
                sel=_select_topA(s.detach(), router_b.float(), A)
            s_sel=torch.gather(s,1,sel)
            w=s_sel / s_sel.sum(dim=-1, keepdim=True)
            flat=sel.reshape(-1)
            order=torch.argsort(flat)
            tok=torch.div(order, A, rounding_mode="floor").to(torch.int64)
            inv=torch.argsort(order).to(torch.int64)
            ws=w.reshape(-1)[order]
            offs=torch.bincount(flat, minlength=E).cumsum(0)
            offs_list=offs.tolist()
            offs_i64=offs.to(torch.int64)
            N=T*A
            xg=_GatherSorted.apply(xf,tok,inv,A)
            tiles_w1=_make_tiles(offs_list,2*F,64,64)
            gu=_GroupedMM.apply(xg,w1f,offs_i64,N,D,2*F,tiles_w1)
            act=_SiLUMul.apply(gu[:,:F].contiguous(), gu[:,F:].contiguous())
            tiles_w2=_make_tiles(offs_list,D,64,64)
            yp=_GroupedMM.apply(act,w2f,offs_i64,N,F,D,tiles_w2)
            y=_Combine.apply(yp,ws,inv,tok,T,D,A)
        ctx.save_for_backward(xf,rwf,w1f,w2f,y)
        ctx.dtypes=(x.dtype,router_w.dtype,w1.dtype,w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx, dy):
        xf,rwf,w1f,w2f,y=ctx.saved_tensors
        gx,grw,gw1,gw2=torch.autograd.grad(y,(xf,rwf,w1f,w2f), dy.float())
        dx_t,drw_t,dw1_t,dw2_t=ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None)

def fused_moe(x, router_w, router_b, w1, w2, A):
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
