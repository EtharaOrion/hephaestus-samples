
import torch, triton, triton.language as tl
BM,BN,BK=64,64,32
EBLOCK=2048
@triton.jit
def _mm_kernel(A,B,C,M,N,K,sam,sak,sbk,sbn,scm,scn,BLK_M:tl.constexpr,BLK_N:tl.constexpr,BLK_K:tl.constexpr):
    pid_m=tl.program_id(0).to(tl.int64);pid_n=tl.program_id(1).to(tl.int64)
    rm=pid_m*BLK_M+tl.arange(0,BLK_M).to(tl.int64);rn=pid_n*BLK_N+tl.arange(0,BLK_N).to(tl.int64)
    acc=tl.zeros((BLK_M,BLK_N),dtype=tl.float32)
    for k0 in range(0,K,BLK_K):
        rk=k0+tl.arange(0,BLK_K).to(tl.int64)
        a=tl.load(A+rm[:,None]*sam+rk[None,:]*sak,mask=(rm[:,None]<M)&(rk[None,:]<K),other=0.0)
        b=tl.load(B+rk[:,None]*sbk+rn[None,:]*sbn,mask=(rk[:,None]<K)&(rn[None,:]<N),other=0.0)
        acc+=tl.dot(a,b,input_precision="ieee")
    tl.store(C+rm[:,None]*scm+rn[None,:]*scn,acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
@triton.jit
def _silu_mul_fwd_kernel(G,U,OUT,n,BLOCK:tl.constexpr):
    pid=tl.program_id(0).to(tl.int64);o=pid*BLOCK+tl.arange(0,BLOCK).to(tl.int64);m=o<n;g=tl.load(G+o,mask=m,other=0.0);u=tl.load(U+o,mask=m,other=0.0);s=tl.sigmoid(g);tl.store(OUT+o,g*s*u,mask=m)
@triton.jit
def _silu_mul_bwd_kernel(G,U,DO,DG,DU,n,BLOCK:tl.constexpr):
    pid=tl.program_id(0).to(tl.int64);o=pid*BLOCK+tl.arange(0,BLOCK).to(tl.int64);m=o<n;g=tl.load(G+o,mask=m,other=0.0);u=tl.load(U+o,mask=m,other=0.0);do=tl.load(DO+o,mask=m,other=0.0);s=tl.sigmoid(g);silu=g*s;tl.store(DG+o,do*u*(s+silu*(1.0-s)),mask=m);tl.store(DU+o,do*silu,mask=m)
@triton.jit
def _gather_kernel(SRC,IDX,DST,D,N,s_src0,s_src1,s_dst0,s_dst1,BLOCK_D:tl.constexpr,BLOCK_N:tl.constexpr):
    pid_n=tl.program_id(0);pid_d=tl.program_id(1);off_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N);off_d=pid_d*BLOCK_D+tl.arange(0,BLOCK_D);mask_n=off_n<N;mask_d=off_d<D;idx=tl.load(IDX+off_n,mask=mask_n,other=0);mask=mask_n[:,None]&mask_d[None,:];vals=tl.load(SRC+idx[:,None]*s_src0+off_d[None,:]*s_src1,mask=mask,other=0.0);tl.store(DST+off_n[:,None]*s_dst0+off_d[None,:]*s_dst1,vals,mask=mask)
@triton.jit
def _scatter_kernel(DST,IDX,SRC,W,D,N,s_dst0,s_dst1,s_src0,s_src1,BLOCK_D:tl.constexpr,BLOCK_N:tl.constexpr):
    pid_n=tl.program_id(0);pid_d=tl.program_id(1);off_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N);off_d=pid_d*BLOCK_D+tl.arange(0,BLOCK_D);mask_n=off_n<N;mask_d=off_d<D;idx=tl.load(IDX+off_n,mask=mask_n,other=0);w=tl.load(W+off_n,mask=mask_n,other=0.0);mask=mask_n[:,None]&mask_d[None,:];src_vals=tl.load(SRC+off_n[:,None]*s_src0+off_d[None,:]*s_src1,mask=mask,other=0.0);scaled=src_vals*w[:,None];cur=tl.load(DST+idx[:,None]*s_dst0+off_d[None,:]*s_dst1,mask=mask,other=0.0);tl.store(DST+idx[:,None]*s_dst0+off_d[None,:]*s_dst1,cur+scaled,mask=mask)
def _mm_launch(a,b):
    M,K=a.shape;_,N=b.shape;c=torch.empty(M,N,device=a.device,dtype=torch.float32)
    if M==0 or N==0: return c
    _mm_kernel[(triton.cdiv(M,BM),triton.cdiv(N,BN))](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),BLK_M=BM,BLK_N=BN,BLK_K=BK)
    return c
def _gather_launch(src,idx):
    N=int(idx.numel());D=int(src.shape[1])
    dst=torch.empty(N,D,device=src.device,dtype=src.dtype)
    if N>0:
        idx_c=idx.contiguous()
        _gather_kernel[(triton.cdiv(N,32),triton.cdiv(D,32))](src,idx_c,dst,D,N,src.stride(0),src.stride(1),dst.stride(0),dst.stride(1),BLOCK_D=32,BLOCK_N=32)
    return dst

@triton.jit
def _gather_bwd_kernel(GOUT, IDX, GIN, D, N, s_gout0, s_gout1, s_gin0, s_gin1, BLOCK_D:tl.constexpr, BLOCK_N:tl.constexpr):
    pid_n=tl.program_id(0); pid_d=tl.program_id(1)
    off_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N); off_d=pid_d*BLOCK_D+tl.arange(0,BLOCK_D)
    mask_n=off_n<N; mask_d=off_d<D; idx=tl.load(IDX+off_n, mask=mask_n, other=0)
    mask=mask_n[:,None]&mask_d[None,:]
    vals=tl.load(GOUT+off_n[:,None]*s_gout0+off_d[None,:]*s_gout1, mask=mask, other=0.0)
    cur=tl.load(GIN+idx[:,None]*s_gin0+off_d[None,:]*s_gin1, mask=mask, other=0.0)
    tl.store(GIN+idx[:,None]*s_gin0+off_d[None,:]*s_gin1, cur+vals, mask=mask)
@triton.jit
def _weighted_scatter_bwd_kernel(GOUT, IDX, W, GSRC, D, N, s_gout0, s_gout1, s_gsrc0, s_gsrc1, BLOCK_D:tl.constexpr, BLOCK_N:tl.constexpr):
    pid_n=tl.program_id(0); pid_d=tl.program_id(1)
    off_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N); off_d=pid_d*BLOCK_D+tl.arange(0,BLOCK_D)
    mask_n=off_n<N; mask_d=off_d<D; idx=tl.load(IDX+off_n, mask=mask_n, other=0); w=tl.load(W+off_n, mask=mask_n, other=0.0)
    mask=mask_n[:,None]&mask_d[None,:]
    g=tl.load(GOUT+idx[:,None]*s_gout0+off_d[None,:]*s_gout1, mask=mask, other=0.0)
    tl.store(GSRC+off_n[:,None]*s_gsrc0+off_d[None,:]*s_gsrc1, g*w[:,None], mask=mask)
def _gather_bwd_launch(gout, idx, gin):
    N=int(idx.numel())
    if N==0:
        return
    D=int(gin.shape[1])
    idx_c=idx.contiguous()
    _gather_bwd_kernel[(triton.cdiv(N,32), triton.cdiv(D,32))](gout, idx_c, gin, D, N, gout.stride(0), gout.stride(1), gin.stride(0), gin.stride(1), BLOCK_D=32, BLOCK_N=32)
def _weighted_scatter_bwd_launch(gout, idx, w, gsrc):
    N=int(idx.numel())
    if N==0:
        return
    D=int(gsrc.shape[1])
    idx_c=idx.contiguous(); w_c=w.contiguous()
    _weighted_scatter_bwd_kernel[(triton.cdiv(N,32), triton.cdiv(D,32))](gout, idx_c, w_c, gsrc, D, N, gout.stride(0), gout.stride(1), gsrc.stride(0), gsrc.stride(1), BLOCK_D=32, BLOCK_N=32)

def _scatter_launch(dst,idx,src,w):
    N=int(idx.numel())
    if N==0: return
    D=int(dst.shape[1])
    idx_c=idx.contiguous();w_c=w.contiguous()
    _scatter_kernel[(triton.cdiv(N,32),triton.cdiv(D,32))](dst,idx_c,src,w_c,D,N,dst.stride(0),dst.stride(1),src.stride(0),src.stride(1),BLOCK_D=32,BLOCK_N=32)
class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b): ctx.save_for_backward(a,b); return _mm_launch(a,b)
    @staticmethod
    def backward(ctx,dc): a,b=ctx.saved_tensors; dc=dc.contiguous(); return _mm_launch(dc,b.t()),_mm_launch(a.t(),dc)
class _SiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,u): ctx.save_for_backward(g,u); out=torch.empty_like(g); n=g.numel(); _silu_mul_fwd_kernel[(triton.cdiv(n,EBLOCK),)](g,u,out,n,BLOCK=EBLOCK); return out
    @staticmethod
    def backward(ctx,dout): g,u=ctx.saved_tensors; dout=dout.contiguous(); dg,du=torch.empty_like(g),torch.empty_like(u); n=g.numel(); _silu_mul_bwd_kernel[(triton.cdiv(n,EBLOCK),)](g,u,dout,dg,du,n,BLOCK=EBLOCK); return dg,du
class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx); ctx.src_shape=src.shape
        return _gather_launch(src, idx)
    @staticmethod
    def backward(ctx, grad_out):
        idx,=ctx.saved_tensors
        grad_src=torch.zeros(ctx.src_shape, device=grad_out.device, dtype=grad_out.dtype)
        _gather_bwd_launch(grad_out, idx, grad_src)
        return grad_src, None
class _WeightedScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, idx, src, w):
        ctx.save_for_backward(idx, w, src)
        y_new=y.clone()
        _scatter_launch(y_new, idx, src, w)
        return y_new
    @staticmethod
    def backward(ctx, grad_y):
        idx,w,src=ctx.saved_tensors
        grad_y_out=grad_y
        if idx.numel()==0:
            return grad_y_out, None, torch.zeros_like(src), torch.zeros_like(w)
        grad_src=torch.empty_like(src)
        _weighted_scatter_bwd_launch(grad_y, idx, w, grad_src)
        # grad_w via deterministic reduction in triton? use torch sum but deterministic
        grad_w=(grad_y[idx]*src).sum(dim=1) if idx.numel()>0 else torch.zeros_like(w)
        return grad_y_out, None, grad_src, grad_w
def _select_topA(s,router_b,A):
    E=s.shape[-1]; b=s+router_b; b=torch.where(b==0.0,torch.zeros_like(b),b).contiguous(); ib=b.view(torch.int32).to(torch.int64); u=torch.where(ib>=0,ib ^ 0x80000000,(~ib)&0xFFFFFFFF); e_idx=torch.arange(E,device=s.device,dtype=torch.int64); keys=(u << 10) | (E-1-e_idx); return torch.topk(keys,A,dim=-1,largest=True,sorted=True).indices
class _FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,router_w,router_b,w1,w2,A):
        A=int(A)
        with torch.enable_grad():
            xf=x.detach().float().requires_grad_(True); rwf=router_w.detach().float().requires_grad_(True); w1f=w1.detach().float().requires_grad_(True); w2f=w2.detach().float().requires_grad_(True)
            T,D=xf.shape; E=rwf.shape[1]; F=w1f.shape[2]//2
            s=torch.sigmoid(_MM.apply(xf,rwf))
            w1s=w1f.unbind(0); w2s=w2f.unbind(0)
            sel=_select_topA(s.detach(),router_b.float(),A)
            s_sel=torch.gather(s,1,sel); w=s_sel/s_sel.sum(dim=-1,keepdim=True)
            sel_flat=sel.reshape(-1); w_flat=w.reshape(-1)
            order=torch.argsort(sel_flat)
            sorted_tok=torch.div(order,A,rounding_mode="floor"); sorted_w=w_flat[order]
            cnt=torch.bincount(sel_flat,minlength=E)
            y=xf.new_zeros(T,D)
            off=0
            for e in range(E):
                n=int(cnt[e].item())
                rows=sorted_tok[off:off+n]
                h=_Gather.apply(xf, rows)
                gu=_MM.apply(h,w1s[e])
                g=gu[:,:F].contiguous(); u=gu[:,F:].contiguous()
                act=_SiLUMul.apply(g,u)
                fe=_MM.apply(act,w2s[e])
                w_slice=sorted_w[off:off+n]
                y=_WeightedScatter.apply(y, rows, fe, w_slice)
                off+=n
        ctx.save_for_backward(xf,rwf,w1f,w2f,y); ctx.dtypes=(x.dtype,router_w.dtype,w1.dtype,w2.dtype)
        return y.detach().to(x.dtype)
    @staticmethod
    def backward(ctx,dy):
        xf,rwf,w1f,w2f,y=ctx.saved_tensors; gx,grw,gw1,gw2=torch.autograd.grad(y,(xf,rwf,w1f,w2f),dy.float()); dx_t,drw_t,dw1_t,dw2_t=ctx.dtypes; return (gx.to(dx_t),grw.to(drw_t),None,gw1.to(dw1_t),gw2.to(dw2_t),None)
def fused_moe(x,router_w,router_b,w1,w2,A): return _FusedMoe.apply(x,router_w,router_b,w1,w2,A)
