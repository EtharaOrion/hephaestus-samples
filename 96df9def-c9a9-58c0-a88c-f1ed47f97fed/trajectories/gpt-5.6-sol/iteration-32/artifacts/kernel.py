import torch
import triton
import triton.language as tl

_BT = 16
_FWD_BV = 32
_BWD_BV = 16

@triton.jit
def _forward_kernel(q,k,v,a,b,g,out,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv=tl.program_id(0); z=tl.program_id(1); ib=z//H; ih=z%H
    ok=tl.arange(0,BK); ov=iv*BV+tl.arange(0,BV); oi=tl.arange(0,16)
    mk=ok<K; mv=ov<V; s=tl.zeros((BK,BV),tl.float32); nc=tl.cdiv(T,16); scale=K ** -0.5
    for ic in range(nc):
        ts=ic*16+oi; mt=ts<T; mtk=mt[:,None]&mk[None,:]; mtv=mt[:,None]&mv[None,:]
        pk=(ib*T*H+ts[:,None]*H+ih)*K+ok[None,:]; pv=(ib*T*H+ts[:,None]*H+ih)*V+ov[None,:]
        ph=((((ib*nc+ic)*H+ih)*K+ok[:,None])*V+ov[None,:])
        tl.store(h+ph,s,mask=mk[:,None]&mv[None,:])
        q0=tl.load(q+pk,mask=mtk,other=0.).to(tl.float32)*scale; k0=tl.load(k+pk,mask=mtk,other=0.).to(tl.float32)
        a0=tl.load(a+pk,mask=mtk,other=0.).to(tl.float32); b0=tl.load(b+pk,mask=mtk,other=0.).to(tl.float32)
        g0=tl.load(g+pk,mask=mtk,other=0.).to(tl.float32); v0=tl.load(v+pv,mask=mtv,other=0.).to(tl.float32)
        gc=tl.cumsum(g0,axis=0); gp=gc-g0; n=tl.minimum(T-ic*16,16); ge=tl.sum(tl.where((oi==n-1)[:,None],gc,0.),axis=0)
        row=oi[:,None]; col=oi[None,:]; ci=gc[:,None,:]; cj=gc[None,:,:]; pi=gp[:,None,:]
        qk=tl.sum(q0[:,None,:]*k0[None,:,:]*tl.exp(ci-cj),axis=2)
        qb=tl.sum(q0[:,None,:]*b0[None,:,:]*tl.exp(ci-cj),axis=2)
        ak=tl.sum(a0[:,None,:]*k0[None,:,:]*tl.exp(pi-cj),axis=2)
        ab=tl.sum(a0[:,None,:]*b0[None,:,:]*tl.exp(pi-cj),axis=2)
        valid=mt[:,None]&mt[None,:]
        qk=tl.where((row>=col)&valid,qk,0.); qb=tl.where((row>=col)&valid,qb,0.)
        ak=tl.where((row>col)&valid,ak,0.); ab=tl.where((row>col)&valid,ab,0.)
        rhs=tl.dot(a0*tl.exp(gp),s,input_precision='ieee')+tl.dot(ak,v0,input_precision='ieee')
        ident=row==col
        inv=tl.where(ident,1.0,0.0)
        power=ab
        for _ in tl.static_range(0,4):
            inv+=tl.dot(power,inv,input_precision='ieee')
            power=tl.dot(power,power,input_precision='ieee')
        x=tl.dot(inv,rhs,input_precision='ieee')
        y=tl.dot(q0*tl.exp(gc),s,input_precision='ieee')+tl.dot(qk,v0,input_precision='ieee')+tl.dot(qb,x,input_precision='ieee')
        tl.store(out+pv,y,mask=mtv)
        s=s*tl.exp(ge)[:,None]+tl.dot(tl.trans(k0*tl.exp(ge[None,:]-gc)),v0,input_precision='ieee')+tl.dot(tl.trans(b0*tl.exp(ge[None,:]-gc)),x,input_precision='ieee')

@triton.jit
def _expand_kernel(k,v,a,b,g,states,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv=tl.program_id(0); z=tl.program_id(1); ib=z//H; ih=z%H
    ok=tl.arange(0,BK); ov=iv*BV+tl.arange(0,BV); mk=ok<K; mv=ov<V; nc=tl.cdiv(T,16)
    s=tl.zeros((BK,BV),tl.float32)
    for ic in range(nc):
        for j in tl.static_range(0,16):
            t=ic*16+j; valid=t<T; pk=(ib*T*H+t*H+ih)*K+ok; pv=(ib*T*H+t*H+ih)*V+ov
            k0=tl.load(k+pk,mask=valid&mk,other=0.).to(tl.float32); a0=tl.load(a+pk,mask=valid&mk,other=0.).to(tl.float32)
            b0=tl.load(b+pk,mask=valid&mk,other=0.).to(tl.float32); g0=tl.load(g+pk,mask=valid&mk,other=0.).to(tl.float32)
            v0=tl.load(v+pv,mask=valid&mv,other=0.).to(tl.float32); r=tl.sum(tl.dot(a0[None,:],s,input_precision='ieee'),axis=0)
            s=s*tl.exp(g0)[:,None]+k0[:,None]*v0[None,:]+b0[:,None]*r[None,:]
            ps=((((ib*T+t)*H+ih)*K+ok[:,None])*V+ov[None,:]); tl.store(states+ps,s,mask=valid&mk[:,None]&mv[None,:])

@triton.jit
def _backward_kernel(q,k,v,a,b,g,do,states,h,dv,pq,pkout,pa,pb,pg,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,B:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv=tl.program_id(0); z=tl.program_id(1); ib=z//H; ih=z%H
    ok=tl.arange(0,BK); ov=iv*BV+tl.arange(0,BV); mk=ok<K; mv=ov<V; ds=tl.zeros((BK,BV),tl.float32)
    nc=tl.cdiv(T,16); scale=K ** -0.5
    for it in range(T):
        t=T-1-it; px=(ib*T*H+t*H+ih)*K+ok; py=(ib*T*H+t*H+ih)*V+ov
        q0=tl.load(q+px,mask=mk,other=0.).to(tl.float32)*scale; k0=tl.load(k+px,mask=mk,other=0.).to(tl.float32)
        a0=tl.load(a+px,mask=mk,other=0.).to(tl.float32); b0=tl.load(b+px,mask=mk,other=0.).to(tl.float32); g0=tl.load(g+px,mask=mk,other=0.).to(tl.float32)
        v0=tl.load(v+py,mask=mv,other=0.).to(tl.float32); d0=tl.load(do+py,mask=mv,other=0.).to(tl.float32)
        ps=((((ib*T+t)*H+ih)*K+ok[:,None])*V+ov[None,:]); st=tl.load(states+ps,mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32)
        if t % 16 == 0:
            ph=((((ib*nc+t//16)*H+ih)*K+ok[:,None])*V+ov[None,:]); sp=tl.load(h+ph,mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32)
        else:
            pp=((((ib*T+t-1)*H+ih)*K+ok[:,None])*V+ov[None,:]); sp=tl.load(states+pp,mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32)
        ds+=q0[:,None]*d0[None,:]
        r=tl.sum(tl.dot(a0[None,:],sp,input_precision='ieee'),axis=0)
        dr=tl.sum(tl.dot(b0[None,:],ds,input_precision='ieee'),axis=0)
        dq0=tl.sum(tl.dot(st,d0[:,None],input_precision='ieee'),axis=1)
        dk0=tl.sum(tl.dot(ds,v0[:,None],input_precision='ieee'),axis=1)
        da0=tl.sum(tl.dot(sp,dr[:,None],input_precision='ieee'),axis=1)
        db0=tl.sum(tl.dot(ds,r[:,None],input_precision='ieee'),axis=1)
        base=(((iv*B+ib)*T+t)*H+ih)*K+ok
        tl.store(pq+base,dq0*scale,mask=mk); tl.store(pkout+base,dk0,mask=mk)
        tl.store(pa+base,da0,mask=mk); tl.store(pb+base,db0,mask=mk)
        tl.store(pg+base,tl.sum(ds*sp*tl.exp(g0)[:,None],axis=1),mask=mk); tl.store(dv+py,tl.sum(ds*k0[:,None],axis=0),mask=mv)
        ds=ds*tl.exp(g0)[:,None]+a0[:,None]*dr[None,:]

@triton.jit
def _reduce_kernel(pq,pk,pa,pb,pg,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
    o=tl.program_id(0)*BS+tl.arange(0,BS); m=o<N
    xq=tl.zeros((BS,),tl.float32); xk=tl.zeros((BS,),tl.float32); xa=tl.zeros((BS,),tl.float32); xb=tl.zeros((BS,),tl.float32); xg=tl.zeros((BS,),tl.float32)
    for i in tl.static_range(0,NV):
        p=i*N+o; xq+=tl.load(pq+p,mask=m,other=0.); xk+=tl.load(pk+p,mask=m,other=0.); xa+=tl.load(pa+p,mask=m,other=0.); xb+=tl.load(pb+p,mask=m,other=0.); xg+=tl.load(pg+p,mask=m,other=0.)
    tl.store(dq+o,xq,mask=m); tl.store(dk+o,xk,mask=m); tl.store(da+o,xa,mask=m); tl.store(db+o,xb,mask=m); tl.store(dg+o,xg,mask=m)

class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,a,b,g):
        B,T,H,K=q.shape; V=v.shape[-1]; bk=max(16,triton.next_power_of_2(K)); nv=triton.cdiv(V,_FWD_BV); nc=triton.cdiv(T,_BT)
        out=torch.empty((B,T,H,V),device=q.device,dtype=v.dtype); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
        _forward_kernel[(nv,B*H)](q,k,v,a,b,g,out,h,T=T,H=H,K=K,V=V,BK=bk,BV=_FWD_BV,num_warps=8,num_stages=2)
        ctx.save_for_backward(q,k,v,a,b,g,h); return out
    @staticmethod
    def backward(ctx,do):
        q,k,v,a,b,g,h=ctx.saved_tensors; B,T,H,K=q.shape; V=v.shape[-1]; bk=max(16,triton.next_power_of_2(K)); nc=triton.cdiv(T,_BT); bv=_BWD_BV; nv=triton.cdiv(V,bv)
        states=torch.empty((B,T,H,K,V),device=q.device,dtype=torch.float32)
        _expand_kernel[(nv,B*H)](k,v,a,b,g,states,T=T,H=H,K=K,V=V,BK=bk,BV=bv,num_warps=4,num_stages=2)
        shape=(nv,B,T,H,K); parts=[torch.zeros(shape,device=q.device,dtype=torch.float32) for _ in range(5)]; dv=torch.empty_like(v)
        _backward_kernel[(nv,B*H)](q,k,v,a,b,g,do.contiguous(),states,h,dv,*parts,T=T,H=H,K=K,V=V,B=B,BK=bk,BV=bv,num_warps=8,num_stages=2)
        grads=[torch.empty_like(x) for x in (q,k,a,b,g)]; n=B*T*H*K
        _reduce_kernel[(triton.cdiv(n,256),)](*parts,*grads,N=n,NV=nv,BS=256,num_warps=4)
        return grads[0],grads[1],dv,grads[2],grads[3],grads[4]

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
    return _Dplr.apply(q,k,v,alpha,beta,gk)
