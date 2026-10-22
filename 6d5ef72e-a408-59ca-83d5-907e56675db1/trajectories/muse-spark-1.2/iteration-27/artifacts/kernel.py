import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32
EBLOCK = 1024
GBM, GBN = 32, 32

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
def _gather_kernel(Inp, Out, Idx, N, D, s_in0, s_in1, s_out0, s_out1, BLK_M: tl.constexpr, BLK_N: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    mask = (rm[:, None] < N) & (rn[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < N, other=0).to(tl.int64)
    vals = tl.load(Inp + idx[:, None] * s_in0 + rn[None, :] * s_in1, mask=mask, other=0.0)
    tl.store(Out + rm[:, None] * s_out0 + rn[None, :] * s_out1, vals, mask=mask)

@triton.jit
def _gather_col_kernel(Wfull, Out, Idx, Eidx, N, s_w0, s_w1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    idx = tl.load(Idx + offs, mask=mask, other=0).to(tl.int64)
    v = tl.load(Wfull + idx * s_w0 + Eidx * s_w1, mask=mask, other=0.0)
    tl.store(Out + offs, v, mask=mask)

@triton.jit
def _scatter_kernel(Y, Fe, Idx, N, D, s_y0, s_y1, s_fe0, s_fe1, BLK_M: tl.constexpr, BLK_N: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    mask = (rm[:, None] < N) & (rn[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < N, other=0).to(tl.int64)
    fe = tl.load(Fe + rm[:, None] * s_fe0 + rn[None, :] * s_fe1, mask=mask, other=0.0)
    y_ptr = Y + idx[:, None] * s_y0 + rn[None, :] * s_y1
    y_old = tl.load(y_ptr, mask=mask, other=0.0)
    tl.store(y_ptr, y_old + fe, mask=mask)

@triton.jit
def _scatter_weighted_kernel(Y, Fe, Idx, Wfull, Eidx, N, D, s_y0, s_y1, s_fe0, s_fe1, s_w0, s_w1, BLK_M: tl.constexpr, BLK_N: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    mask = (rm[:, None] < N) & (rn[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < N, other=0).to(tl.int64)
    w = tl.load(Wfull + idx * s_w0 + Eidx * s_w1, mask=rm < N, other=0.0).to(tl.float32)
    fe = tl.load(Fe + rm[:, None] * s_fe0 + rn[None, :] * s_fe1, mask=mask, other=0.0)
    contrib = fe * w[:, None]
    y_ptr = Y + idx[:, None] * s_y0 + rn[None, :] * s_y1
    y_old = tl.load(y_ptr, mask=mask, other=0.0)
    tl.store(y_ptr, y_old + contrib, mask=mask)

@triton.jit
def _scatter_grad_kernel(GradInp, GradOut, Idx, N, D, s_gin0, s_gin1, s_gout0, s_gout1, BLK_M: tl.constexpr, BLK_N: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    mask = (rm[:, None] < N) & (rn[None, :] < D)
    idx = tl.load(Idx + rm, mask=rm < N, other=0).to(tl.int64)
    g = tl.load(GradOut + rm[:, None] * s_gout0 + rn[None, :] * s_gout1, mask=mask, other=0.0)
    ptr = GradInp + idx[:, None] * s_gin0 + rn[None, :] * s_gin1
    old = tl.load(ptr, mask=mask, other=0.0)
    tl.store(ptr, old + g, mask=mask)

@triton.jit
def _scatter_col_grad_kernel(GradWfull, GradCol, Idx, Eidx, N, s_w0, s_w1, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    idx = tl.load(Idx + offs, mask=mask, other=0).to(tl.int64)
    g = tl.load(GradCol + offs, mask=mask, other=0.0)
    ptr = GradWfull + idx * s_w0 + Eidx * s_w1
    old = tl.load(ptr, mask=mask, other=0.0)
    tl.store(ptr, old + g, mask=mask)

@triton.jit
def _scale_kernel(Inp, Out, W, N, D, s_in0, s_in1, s_out0, s_out1, BLK_M: tl.constexpr, BLK_N: tl.constexpr):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLK_M + tl.arange(0, BLK_M).to(tl.int64)
    rn = pid_n * BLK_N + tl.arange(0, BLK_N).to(tl.int64)
    mask = (rm[:, None] < N) & (rn[None, :] < D)
    w = tl.load(W + rm, mask=rm < N, other=0.0).to(tl.float32)
    vals = tl.load(Inp + rm[:, None] * s_in0 + rn[None, :] * s_in1, mask=mask, other=0.0)
    tl.store(Out + rm[:, None] * s_out0 + rn[None, :] * s_out1, vals * w[:, None], mask=mask)

def _mm_launch(a,b):
    M,K=a.shape; _,N=b.shape
    c=torch.empty(M,N,device=a.device,dtype=torch.float32)
    if M==0 or N==0: return c
    _mm_kernel[(triton.cdiv(M,BM),triton.cdiv(N,BN))](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),BLK_M=BM,BLK_N=BN,BLK_K=BK)
    return c

class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b):
        ctx.save_for_backward(a,b)
        return _mm_launch(a,b)
    @staticmethod
    def backward(ctx,dc):
        a,b=ctx.saved_tensors
        dc=dc.contiguous()
        return _mm_launch(dc,b.t()), _mm_launch(a.t(),dc)

class _SiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,u):
        ctx.save_for_backward(g,u)
        out=torch.empty_like(g)
        n=g.numel()
        _silu_mul_fwd_kernel[(triton.cdiv(n,EBLOCK),)](g,u,out,n,BLOCK=EBLOCK)
        return out
    @staticmethod
    def backward(ctx,dout):
        g,u=ctx.saved_tensors
        dout=dout.contiguous()
        dg,du=torch.empty_like(g),torch.empty_like(u)
        n=g.numel()
        _silu_mul_bwd_kernel[(triton.cdiv(n,EBLOCK),)](g,u,dout,dg,du,n,BLOCK=EBLOCK)
        return dg,du

class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inp,idx):
        ctx.save_for_backward(idx)
        ctx.inp_shape=inp.shape
        N=idx.shape[0]; D=inp.shape[1]
        out=torch.empty(N,D,device=inp.device,dtype=inp.dtype)
        if N!=0 and D!=0:
            _gather_kernel[(triton.cdiv(N,GBM),triton.cdiv(D,GBN))](inp,out,idx,N,D,inp.stride(0),inp.stride(1),out.stride(0),out.stride(1),BLK_M=GBM,BLK_N=GBN)
        return out
    @staticmethod
    def backward(ctx,grad_out):
        idx,=ctx.saved_tensors
        T,D=ctx.inp_shape
        grad_inp=torch.zeros(T,D,device=grad_out.device,dtype=grad_out.dtype)
        N=idx.shape[0]
        if N!=0 and D!=0:
            go=grad_out.contiguous()
            _scatter_grad_kernel[(triton.cdiv(N,GBM),triton.cdiv(D,GBN))](grad_inp,go,idx,N,D,grad_inp.stride(0),grad_inp.stride(1),go.stride(0),go.stride(1),BLK_M=GBM,BLK_N=GBN)
        return grad_inp,None

class _GatherCol(torch.autograd.Function):
    @staticmethod
    def forward(ctx,wfull,idx,eidx):
        ctx.save_for_backward(idx)
        ctx.eidx=eidx; ctx.w_shape=wfull.shape
        N=idx.shape[0]
        out=torch.empty(N, device=wfull.device, dtype=wfull.dtype)
        if N!=0:
            _gather_col_kernel[(triton.cdiv(N,1024),)](wfull,out,idx,eidx,N,wfull.stride(0),wfull.stride(1),BLOCK=1024)
        return out
    @staticmethod
    def backward(ctx,grad_out):
        idx,=ctx.saved_tensors
        eidx=ctx.eidx; w_shape=ctx.w_shape
        grad_wfull=torch.zeros(w_shape, device=grad_out.device, dtype=grad_out.dtype)
        N=idx.shape[0]
        if N!=0:
            go=grad_out.contiguous()
            _scatter_col_grad_kernel[(triton.cdiv(N,1024),)](grad_wfull,go,idx,eidx,N,grad_wfull.stride(0),grad_wfull.stride(1),BLOCK=1024)
        return grad_wfull,None,None

class _Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inp,w):
        ctx.save_for_backward(inp,w)
        N,D=inp.shape
        out=torch.empty_like(inp)
        if N!=0 and D!=0:
            _scale_kernel[(triton.cdiv(N,GBM),triton.cdiv(D,GBN))](inp,out,w,N,D,inp.stride(0),inp.stride(1),out.stride(0),out.stride(1),BLK_M=GBM,BLK_N=GBN)
        return out
    @staticmethod
    def backward(ctx,grad_out):
        inp,w=ctx.saved_tensors
        N,D=inp.shape
        go=grad_out.contiguous()
        grad_inp=torch.empty_like(inp)
        if N!=0 and D!=0:
            _scale_kernel[(triton.cdiv(N,GBM),triton.cdiv(D,GBN))](go,grad_inp,w,N,D,go.stride(0),go.stride(1),grad_inp.stride(0),grad_inp.stride(1),BLK_M=GBM,BLK_N=GBN)
            grad_w=(go * inp).sum(dim=1)
        else:
            grad_w=torch.empty(0,device=inp.device,dtype=inp.dtype)
        return grad_inp, grad_w

class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx,y,fe,idx):
        ctx.save_for_backward(idx)
        out=y.clone()
        N=idx.shape[0]; D=y.shape[1]
        if N!=0 and D!=0:
            fec=fe.contiguous()
            _scatter_kernel[(triton.cdiv(N,GBM),triton.cdiv(D,GBN))](out,fec,idx,N,D,out.stride(0),out.stride(1),fec.stride(0),fec.stride(1),BLK_M=GBM,BLK_N=GBN)
        return out
    @staticmethod
    def backward(ctx,grad_out):
        idx,=ctx.saved_tensors
        N=idx.shape[0]
        go=grad_out.contiguous()
        gathered=torch.empty(N, go.shape[1], device=go.device, dtype=go.dtype)
        if N!=0 and go.shape[1]!=0:
            _gather_kernel[(triton.cdiv(N,GBM),triton.cdiv(go.shape[1],GBN))](go,gathered,idx,N,go.shape[1],go.stride(0),go.stride(1),gathered.stride(0),gathered.stride(1),BLK_M=GBM,BLK_N=GBN)
        return grad_out, gathered, None

class _ScatterWeighted(torch.autograd.Function):
    @staticmethod
    def forward(ctx,y,fe,idx,wfull,eidx):
        ctx.save_for_backward(idx,wfull,fe)
        ctx.eidx=eidx
        out=y.clone()
        N=idx.shape[0]; D=y.shape[1]
        if N!=0 and D!=0:
            fec=fe.contiguous()
            _scatter_weighted_kernel[(triton.cdiv(N,GBM),triton.cdiv(D,GBN))](out,fec,idx,wfull,eidx,N,D,out.stride(0),out.stride(1),fec.stride(0),fec.stride(1),wfull.stride(0),wfull.stride(1),BLK_M=GBM,BLK_N=GBN)
        return out
    @staticmethod
    def backward(ctx,grad_out):
        idx,wfull,fe=ctx.saved_tensors
        eidx=ctx.eidx
        N=idx.shape[0]; D=grad_out.shape[1]
        go=grad_out.contiguous()
        gathered=torch.empty(N,D, device=go.device, dtype=go.dtype)
        if N!=0 and D!=0:
            _gather_kernel[(triton.cdiv(N,GBM),triton.cdiv(D,GBN))](go,gathered,idx,N,D,go.stride(0),go.stride(1),gathered.stride(0),gathered.stride(1),BLK_M=GBM,BLK_N=GBN)
        else:
            gathered=torch.empty(0,D,device=go.device,dtype=go.dtype)
        # grad for fe = gathered * w
        if N!=0:
            w=torch.empty(N, device=wfull.device, dtype=wfull.dtype)
            _gather_col_kernel[(triton.cdiv(N,1024),)](wfull,w,idx,eidx,N,wfull.stride(0),wfull.stride(1),BLOCK=1024)
            grad_fe=gathered * w[:,None].to(gathered.dtype)
            fec=fe.contiguous()
            grad_wcol=(gathered * fec.to(gathered.dtype)).sum(dim=1)
            grad_wfull=torch.zeros(wfull.shape, device=wfull.device, dtype=wfull.dtype)
            _scatter_col_grad_kernel[(triton.cdiv(N,1024),)](grad_wfull,grad_wcol,idx,eidx,N,grad_wfull.stride(0),grad_wfull.stride(1),BLOCK=1024)
        else:
            grad_fe=gathered
            grad_wfull=torch.zeros(wfull.shape, device=go.device, dtype=go.dtype)
        return grad_out, grad_fe, None, grad_wfull, None

def _select_topA(s, router_b, A):
    E=s.shape[-1]
    b=s+router_b
    b=torch.where(b==0.0, torch.zeros_like(b), b).contiguous()
    ib=b.view(torch.int32).to(torch.int64)
    u=torch.where(ib>=0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    e_idx=torch.arange(E, device=s.device, dtype=torch.int64)
    shift=max(1,(E-1).bit_length())
    keys=(u << shift) | (E-1-e_idx)
    return torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,router_w,router_b,w1,w2,A):
        A=int(A)
        with torch.enable_grad():
            xf=x.detach().float().requires_grad_(True)
            rwf=router_w.detach().float().requires_grad_(True)
            w1f=w1.detach().float().requires_grad_(True)
            w2f=w2.detach().float().requires_grad_(True)
            T,D=xf.shape; E=rwf.shape[1]; F=w1f.shape[2]//2
            s=torch.sigmoid(_MM.apply(xf,rwf))
            w1s=w1f.unbind(0); w2s=w2f.unbind(0)
            with torch.no_grad():
                sel=_select_topA(s.detach(), router_b.float(), A)
            s_sel=torch.gather(s,1,sel)
            w=s_sel / s_sel.sum(dim=-1, keepdim=True)
            w_full=s.new_zeros(T,E).scatter(1,sel,w)
            member=torch.zeros(T,E,dtype=torch.bool,device=x.device).scatter(1,sel,True)
            y=xf.new_zeros(T,D)
            for e in range(E):
                rows=member[:,e].nonzero(as_tuple=True)[0]
                h=_Gather.apply(xf,rows)
                gu=_MM.apply(h,w1s[e])
                act=_SiLUMul.apply(gu[:,:F].contiguous(), gu[:,F:].contiguous())
                fe=_MM.apply(act,w2s[e])
                y=_ScatterWeighted.apply(y,fe,rows,w_full,e)
        ctx.save_for_backward(xf,rwf,w1f,w2f,y)
        ctx.dtypes=(x.dtype,router_w.dtype,w1.dtype,w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx,dy):
        xf,rwf,w1f,w2f,y=ctx.saved_tensors
        gx,grw,gw1,gw2=torch.autograd.grad(y,(xf,rwf,w1f,w2f),dy.float())
        dx_t,drw_t,dw1_t,dw2_t=ctx.dtypes
        return (gx.to(dx_t), grw.to(drw_t), None, gw1.to(dw1_t), gw2.to(dw2_t), None)

def fused_moe(x,router_w,router_b,w1,w2,A):
    return _FusedMoe.apply(x,router_w,router_b,w1,w2,A)
