import torch
import triton
import triton.language as tl
BT, BV = 16, 32


@triton.jit
def _fwd(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr,SCALE:tl.constexpr):
    iv=tl.program_id(0); z=tl.program_id(1).to(tl.int64); ih=z%H; ib=z//H
    it=tl.arange(0,16); ik=tl.arange(0,BK); ivs=iv*32+tl.arange(0,32)
    mk=ik<K; mv=ivs<V; S=tl.zeros((BK,32),tl.float32)
    rr=it[:,None]; cc=it[None,:]
    for c in range(NC):
        ts=c*16+it; mt=ts<T
        xk=((ib*T+ts[:,None])*H+ih)*K+ik[None,:]
        xv=((ib*T+ts[:,None])*H+ih)*V+ivs[None,:]
        mtk=mt[:,None]&mk[None,:]; mtv=mt[:,None]&mv[None,:]
        Q=tl.load(q+xk,mask=mtk,other=0.).to(tl.float32)*SCALE
        Kt=tl.load(k+xk,mask=mtk,other=0.).to(tl.float32)
        A=tl.load(a+xk,mask=mtk,other=0.).to(tl.float32)
        B=tl.load(b+xk,mask=mtk,other=0.).to(tl.float32)
        G0=tl.load(g+xk,mask=mtk,other=0.).to(tl.float32)
        Val=tl.load(v+xv,mask=mtv,other=0.).to(tl.float32)
        G=tl.cumsum(G0,axis=0); GP=G-G0
        ph=(((z*NC+c)*K+ik[:,None])*V+ivs[None,:])
        tl.store(h+ph,S,mask=mk[:,None]&mv[None,:])
        Aqk=tl.zeros((16,16),tl.float32); Aqb=tl.zeros((16,16),tl.float32)
        Aak=tl.zeros((16,16),tl.float32); Aab=tl.zeros((16,16),tl.float32)
        for j in tl.static_range(0,16):
            pick=it==j
            kj=tl.sum(tl.where(pick[:,None],Kt,0.),axis=0)
            bj=tl.sum(tl.where(pick[:,None],B,0.),axis=0)
            gj=tl.sum(tl.where(pick[:,None],G,0.),axis=0)
            eq=tl.exp(tl.minimum(G-gj[None,:],0.)); ea=tl.exp(tl.minimum(GP-gj[None,:],0.))
            qk=tl.sum(Q*kj[None,:]*eq,axis=1); qb=tl.sum(Q*bj[None,:]*eq,axis=1)
            ak=tl.sum(A*kj[None,:]*ea,axis=1); ab=tl.sum(A*bj[None,:]*ea,axis=1)
            col=cc==j
            Aqk+=tl.where(col&(rr>=j)&mt[:,None],qk[:,None],0.)
            Aqb+=tl.where(col&(rr>=j)&mt[:,None],qb[:,None],0.)
            Aak+=tl.where(col&(rr>j)&mt[:,None],ak[:,None],0.)
            Aab+=tl.where(col&(rr>j)&mt[:,None],ab[:,None],0.)
        X=tl.dot(A*tl.exp(tl.minimum(GP,0.)),S,input_precision='ieee')+tl.dot(Aak,Val,input_precision='ieee')
        for i in tl.static_range(0,16):
            ri=tl.sum(tl.where((it==i)[:,None],X,0.),axis=0)
            coef=tl.sum(tl.where((rr==i)&(cc<i),Aab,0.),axis=0)
            ri+=tl.sum(coef[:,None]*X,axis=0)
            X=tl.where((it==i)[:,None],ri[None,:],X)
        O=tl.dot(Q*tl.exp(tl.minimum(G,0.)),S,input_precision='ieee')
        O+=tl.dot(Aqk,Val,input_precision='ieee')+tl.dot(Aqb,X,input_precision='ieee')
        tl.store(o+xv,O,mask=mtv)
        li=tl.minimum(16-1,T-c*16-1)
        GL=tl.sum(tl.where((it==li)[:,None],G,0.),axis=0)
        EK=Kt*tl.exp(tl.minimum(GL[None,:]-G,0.)); EB=B*tl.exp(tl.minimum(GL[None,:]-G,0.))
        S=S*tl.exp(tl.minimum(GL,0.))[:,None]
        S+=tl.dot(tl.trans(EK),Val,input_precision='ieee')+tl.dot(tl.trans(EB),X,input_precision='ieee')

@triton.jit
def _expand(k,v,a,b,g,h,ss,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr):
    iv=tl.program_id(0); c=tl.program_id(1); z=tl.program_id(2).to(tl.int64); ih=z%H; ib=z//H
    ik=tl.arange(0,BK); ivs=iv*32+tl.arange(0,32); mk=ik<K; mv=ivs<V
    ph=(((z*NC+c)*K+ik[:,None])*V+ivs[None,:])
    S=tl.load(h+ph,mask=mk[:,None]&mv[None,:],other=0.)
    for j in tl.static_range(0,16):
        t=c*16+j; valid=t<T; xk=((ib*T+t)*H+ih)*K+ik; xv=((ib*T+t)*H+ih)*V+ivs
        ps=(((z*T+t)*K+ik[:,None])*V+ivs[None,:])
        tl.store(ss+ps,S,mask=valid&mk[:,None]&mv[None,:])
        Kt=tl.load(k+xk,mask=valid&mk,other=0.).to(tl.float32)
        A=tl.load(a+xk,mask=valid&mk,other=0.).to(tl.float32)
        B=tl.load(b+xk,mask=valid&mk,other=0.).to(tl.float32)
        G=tl.load(g+xk,mask=valid&mk,other=0.).to(tl.float32)
        Val=tl.load(v+xv,mask=valid&mv,other=0.).to(tl.float32)
        R=tl.sum(A[:,None]*S,axis=0)
        S=S*tl.exp(G)[:,None]+Kt[:,None]*Val[None,:]+B[:,None]*R[None,:]

@triton.jit
def _bwd(q,k,v,a,b,g,do,ss,p,dv,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,NC:tl.constexpr,N:tl.constexpr,SCALE:tl.constexpr):
    iv=tl.program_id(0); z=tl.program_id(1).to(tl.int64); ih=z%H; ib=z//H
    ik=tl.arange(0,BK); ivs=iv*32+tl.arange(0,32); mk=ik<K; mv=ivs<V
    DS=tl.zeros((BK,32),tl.float32)
    for cr in range(NC):
        c=NC-1-cr
        for jr in tl.static_range(0,16):
            t=c*16+(16-1-jr); valid=t<T
            xk=((ib*T+t)*H+ih)*K+ik; xv=((ib*T+t)*H+ih)*V+ivs
            ps=(((z*T+t)*K+ik[:,None])*V+ivs[None,:])
            SP=tl.load(ss+ps,mask=valid&mk[:,None]&mv[None,:],other=0.)
            Q=tl.load(q+xk,mask=valid&mk,other=0.).to(tl.float32)*SCALE
            Kt=tl.load(k+xk,mask=valid&mk,other=0.).to(tl.float32)
            A=tl.load(a+xk,mask=valid&mk,other=0.).to(tl.float32)
            B=tl.load(b+xk,mask=valid&mk,other=0.).to(tl.float32)
            G=tl.load(g+xk,mask=valid&mk,other=0.).to(tl.float32); E=tl.exp(G)
            Val=tl.load(v+xv,mask=valid&mv,other=0.).to(tl.float32)
            DO=tl.load(do+xv,mask=valid&mv,other=0.).to(tl.float32)
            R=tl.sum(A[:,None]*SP,axis=0)
            ST=SP*E[:,None]+Kt[:,None]*Val[None,:]+B[:,None]*R[None,:]
            DQ=tl.sum(ST*DO[None,:],axis=1)*SCALE
            DS+=Q[:,None]*DO[None,:]
            DK=tl.sum(DS*Val[None,:],axis=1)
            tl.store(dv+xv,tl.sum(DS*Kt[:,None],axis=0),mask=valid&mv)
            DB=tl.sum(DS*R[None,:],axis=1)
            DR=tl.sum(DS*B[:,None],axis=0)
            DA=tl.sum(SP*DR[None,:],axis=1)
            DG=tl.sum(DS*SP*E[:,None],axis=1)
            pos=((ib*T+t)*H+ih)*K+ik; base=iv*5*N
            tl.store(p+base+0*N+pos,DQ,mask=valid&mk); tl.store(p+base+1*N+pos,DK,mask=valid&mk)
            tl.store(p+base+2*N+pos,DA,mask=valid&mk); tl.store(p+base+3*N+pos,DB,mask=valid&mk)
            tl.store(p+base+4*N+pos,DG,mask=valid&mk)
            DS=DS*E[:,None]+A[:,None]*DR[None,:]

@triton.jit
def _reduce(p,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
    x=tl.program_id(0)*BS+tl.arange(0,BS); m=x<N
    for z in tl.static_range(0,5):
        y=tl.zeros((BS,),tl.float32)
        for iv in tl.static_range(0,NV): y+=tl.load(p+iv*5*N+z*N+x,mask=m,other=0.)
        if z==0: tl.store(dq+x,y,mask=m)
        elif z==1: tl.store(dk+x,y,mask=m)
        elif z==2: tl.store(da+x,y,mask=m)
        elif z==3: tl.store(db+x,y,mask=m)
        else: tl.store(dg+x,y,mask=m)


class _DPLR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,a,b,g):
        q,k,v,a,b,g=(x.contiguous() for x in (q,k,v,a,b,g)); B,T,H,K=q.shape; V=v.shape[-1]
        bk=max(16,triton.next_power_of_2(K)); nc=triton.cdiv(T,BT); nv=triton.cdiv(V,BV)
        o=torch.empty_like(v); h=torch.empty((B*H,nc,K,V),device=q.device,dtype=torch.float32)
        _fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,BK=bk,NC=nc,SCALE=K**-.5,num_warps=8,num_stages=2)
        ctx.save_for_backward(q,k,v,a,b,g,h); ctx.dims=B,T,H,K,V,bk,nc,nv
        return o
    @staticmethod
    def backward(ctx,do):
        q,k,v,a,b,g,h=ctx.saved_tensors; B,T,H,K,V,bk,nc,nv=ctx.dims; do=do.contiguous()
        ss=torch.empty((B*H,T,K,V),device=q.device,dtype=torch.float32)
        _expand[(nv,nc,B*H)](k,v,a,b,g,h,ss,T=T,H=H,K=K,V=V,BK=bk,NC=nc,num_warps=4,num_stages=2)
        n=B*T*H*K; p=torch.zeros((nv,5,n),device=q.device,dtype=torch.float32); dv=torch.empty_like(v)
        _bwd[(nv,B*H)](q,k,v,a,b,g,do,ss,p,dv,T=T,H=H,K=K,V=V,BK=bk,NC=nc,N=n,SCALE=K**-.5,num_warps=8,num_stages=2)
        gs=[torch.empty_like(q) for _ in range(5)]
        _reduce[(triton.cdiv(n,256),)](p,*gs,N=n,NV=nv,BS=256,num_warps=4)
        return gs[0],gs[1],dv,gs[2],gs[3],gs[4]

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
    return _DPLR.apply(q,k,v,alpha,beta,gk)
