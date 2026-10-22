import torch
import triton
import triton.language as tl
BT, BV = 16, 16

@triton.jit
def inv_lower(x):
    r=tl.arange(0,16); m=tl.where(r[:,None]>r[None,:],x,0.0)
    for i in tl.static_range(1,16):
        z=tl.sum(tl.where((r==i)[:,None],m,0.0),0); z+=tl.sum(z[:,None]*m,0)*(r<i); m=tl.where((r==i)[:,None],z,m)
    return m+(r[:,None]==r[None,:])

@triton.jit
def fwd(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV0:tl.constexpr):
    iv=tl.program_id(0); bh=tl.program_id(1); ib=bh//H; ih=bh%H
    ik=tl.arange(0,BK); ov=iv*BV0+tl.arange(0,BV0); it=tl.arange(0,16); mk=ik<K; mv=ov<V
    S=tl.zeros((BK,BV0),tl.float32); sc=1.0/tl.sqrt(K+0.0)
    for c in range(NC):
        tt=c*4+it; mt=(tt<T)&(it<4); pk=((ib*T+tt)*H+ih)*K; pv=((ib*T+tt)*H+ih)*V; mtk=mt[:,None]&mk[None,:]; mtv=mt[:,None]&mv[None,:]
        Q=tl.load(q+pk[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); K0=tl.load(k+pk[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); A=tl.load(a+pk[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); B=tl.load(b+pk[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); G=tl.load(g+pk[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); X=tl.load(v+pv[:,None]+ov[None,:],mask=mtv,other=0.).to(tl.float32)
        C=tl.cumsum(G,0); E=tl.exp(tl.maximum(C,-80.)); EN=tl.exp(tl.minimum(-C,80.)); AP=A*tl.exp(tl.maximum(C-G,-80.)); KH=K0*EN; BH=B*EN; QG=Q*E*sc
        strict=it[:,None]>it[None,:]; lower=it[:,None]>=it[None,:]
        LAB=tl.where(strict,tl.dot(AP,tl.trans(BH),input_precision='ieee'),0.); LAK=tl.where(strict,tl.dot(AP,tl.trans(KH),input_precision='ieee'),0.); I=inv_lower(LAB)
        R=tl.dot(I,tl.dot(AP,S,input_precision='ieee')+tl.dot(LAK,X,input_precision='ieee'),input_precision='ieee')
        Y=tl.dot(QG,S,input_precision='ieee')+tl.dot(tl.where(lower,tl.dot(QG,tl.trans(KH),input_precision='ieee'),0.),X,input_precision='ieee')+tl.dot(tl.where(lower,tl.dot(QG,tl.trans(BH),input_precision='ieee'),0.),R,input_precision='ieee')
        tl.store(o+pv[:,None]+ov[None,:],Y,mask=mtv); hp=((((ib*NC+c)*H+ih)*K+ik[:,None])*V+ov[None,:]); tl.store(h+hp,S,mask=mk[:,None]&mv[None,:])
        last=tl.minimum(T-c*4,4)-1; D=tl.exp(tl.maximum(tl.sum(tl.where((it==last)[:,None],C,0.),0),-80.)); S=D[:,None]*(S+tl.dot(tl.trans(KH),X,input_precision='ieee')+tl.dot(tl.trans(BH),R,input_precision='ieee'))

@triton.jit
def bwd(q,k,v,a,b,g,do,h,dsend,pq,pk0,pa,pb,pg,dv,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV0:tl.constexpr):
    iv=tl.program_id(0); bh=tl.program_id(1); ib=bh//H; ih=bh%H; ik=tl.arange(0,BK); ov=iv*BV0+tl.arange(0,BV0); it=tl.arange(0,16); mk=ik<K; mv=ov<V; DS=tl.zeros((BK,BV0),tl.float32); sc=1./tl.sqrt(K+0.)
    for z in range(NC):
        c=NC-1-z; tt=c*4+it; mt=(tt<T)&(it<4); xp=((ib*T+tt)*H+ih)*K; vp=((ib*T+tt)*H+ih)*V; mtk=mt[:,None]&mk[None,:]; mtv=mt[:,None]&mv[None,:]
        Q=tl.load(q+xp[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); K0=tl.load(k+xp[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); A=tl.load(a+xp[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); B=tl.load(b+xp[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); G=tl.load(g+xp[:,None]+ik[None,:],mask=mtk,other=0.).to(tl.float32); X=tl.load(v+vp[:,None]+ov[None,:],mask=mtv,other=0.).to(tl.float32); DY=tl.load(do+vp[:,None]+ov[None,:],mask=mtv,other=0.).to(tl.float32); hp=((((ib*NC+c)*H+ih)*K+ik[:,None])*V+ov[None,:]); S=tl.load(h+hp,mask=mk[:,None]&mv[None,:],other=0.)
        tl.store(dsend+hp,DS,mask=mk[:,None]&mv[None,:])
        C=tl.cumsum(G,0); E=tl.exp(tl.maximum(C,-80.)); EN=tl.exp(tl.minimum(-C,80.)); EP=tl.exp(tl.maximum(C-G,-80.)); QG=Q*E*sc; KH=K0*EN; BH=B*EN; AP=A*EP; strict=it[:,None]>it[None,:]; lower=it[:,None]>=it[None,:]; LAB=tl.where(strict,tl.dot(AP,tl.trans(BH),input_precision='ieee'),0.); LAK=tl.where(strict,tl.dot(AP,tl.trans(KH),input_precision='ieee'),0.); I=inv_lower(LAB); R=tl.dot(I,tl.dot(AP,S,input_precision='ieee')+tl.dot(LAK,X,input_precision='ieee'),input_precision='ieee'); AQK=tl.where(lower,tl.dot(QG,tl.trans(KH),input_precision='ieee'),0.); AQB=tl.where(lower,tl.dot(QG,tl.trans(BH),input_precision='ieee'),0.)
        last=tl.minimum(T-c*4,4)-1; D=tl.exp(tl.maximum(tl.sum(tl.where((it==last)[:,None],C,0.),0),-80.)); SD=D[:,None]*DS; dQ=tl.dot(DY,tl.trans(S),input_precision='ieee'); dS=tl.dot(tl.trans(QG),DY,input_precision='ieee')+SD; d1=tl.where(lower,tl.dot(DY,tl.trans(X),input_precision='ieee'),0.); d2=tl.where(lower,tl.dot(DY,tl.trans(R),input_precision='ieee'),0.); dX=tl.dot(tl.trans(AQK),DY,input_precision='ieee'); dR=tl.dot(tl.trans(AQB),DY,input_precision='ieee'); dQ+=tl.dot(d1,KH,input_precision='ieee')+tl.dot(d2,BH,input_precision='ieee'); dK=tl.dot(tl.trans(d1),QG,input_precision='ieee')+tl.dot(X,tl.trans(SD),input_precision='ieee'); dB=tl.dot(tl.trans(d2),QG,input_precision='ieee')+tl.dot(R,tl.trans(SD),input_precision='ieee'); dX+=tl.dot(KH,SD,input_precision='ieee'); dR+=tl.dot(BH,SD,input_precision='ieee'); Z=tl.dot(tl.trans(KH),X,input_precision='ieee')+tl.dot(tl.trans(BH),R,input_precision='ieee'); dlast=D*tl.sum(DS*(S+Z),1)
        dR0=tl.dot(tl.trans(I),dR,input_precision='ieee'); dA=tl.dot(dR0,tl.trans(S),input_precision='ieee'); dS+=tl.dot(tl.trans(AP),dR0,input_precision='ieee'); dLk=tl.where(strict,tl.dot(dR0,tl.trans(X),input_precision='ieee'),0.); dX+=tl.dot(tl.trans(LAK),dR0,input_precision='ieee'); dA+=tl.dot(dLk,KH,input_precision='ieee'); dK+=tl.dot(tl.trans(dLk),AP,input_precision='ieee'); dLb=tl.where(strict,tl.dot(dR0,tl.trans(R),input_precision='ieee'),0.); dA+=tl.dot(dLb,BH,input_precision='ieee'); dB+=tl.dot(tl.trans(dLb),AP,input_precision='ieee')
        dg0=dQ*QG-dK*KH-dB*BH+dA*AP; dgg=tl.flip(tl.cumsum(tl.flip(dg0,0),0),0)-dA*AP+dlast[None,:]; dgg=tl.where(tt[:,None]==0,0.0,dgg); base=iv*T*H*K+(((ib*T+tt)*H+ih)*K)[:,None]+ik[None,:]
        tl.store(pq+base,dQ*E*sc,mask=mtk); tl.store(pk0+base,dK*EN,mask=mtk); tl.store(pa+base,dA*EP,mask=mtk); tl.store(pb+base,dB*EN,mask=mtk); tl.store(pg+base,dgg,mask=mtk); tl.store(dv+vp[:,None]+ov[None,:],dX,mask=mtv); DS=dS

@triton.jit
def gate_exact(q,k,v,a,b,g,do,h,dsend,pg,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV0:tl.constexpr):
    iv=tl.program_id(0); c=tl.program_id(1); bh=tl.program_id(2); ib=bh//H; ih=bh%H
    ik=tl.arange(0,BK); ov=iv*BV0+tl.arange(0,BV0); mk=ik<K; mv=ov<V; hp=((((ib*NC+c)*H+ih)*K+ik[:,None])*V+ov[None,:]); S0=tl.load(h+hp,mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32); D=tl.load(dsend+hp,mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32); sc=1./tl.sqrt(K+0.)
    t0=c*4; x0=((ib*T+t0)*H+ih)*K; y0=((ib*T+t0)*H+ih)*V; m0=t0<T; g0=tl.load(g+x0+ik,mask=m0&mk,other=0.).to(tl.float32); a0=tl.load(a+x0+ik,mask=m0&mk,other=0.).to(tl.float32); b0=tl.load(b+x0+ik,mask=m0&mk,other=0.).to(tl.float32); k0=tl.load(k+x0+ik,mask=m0&mk,other=0.).to(tl.float32); v0=tl.load(v+y0+ov,mask=m0&mv,other=0.).to(tl.float32); S1=tl.where(m0,S0*tl.exp(g0)[:,None]+k0[:,None]*v0[None,:]+b0[:,None]*tl.sum(a0[:,None]*S0,0)[None,:],S0)
    t1=t0+1; x1=x0+H*K; y1=y0+H*V; m1=t1<T; g1=tl.load(g+x1+ik,mask=m1&mk,other=0.).to(tl.float32); a1=tl.load(a+x1+ik,mask=m1&mk,other=0.).to(tl.float32); b1=tl.load(b+x1+ik,mask=m1&mk,other=0.).to(tl.float32); k1=tl.load(k+x1+ik,mask=m1&mk,other=0.).to(tl.float32); v1=tl.load(v+y1+ov,mask=m1&mv,other=0.).to(tl.float32); S2=tl.where(m1,S1*tl.exp(g1)[:,None]+k1[:,None]*v1[None,:]+b1[:,None]*tl.sum(a1[:,None]*S1,0)[None,:],S1)
    t2=t0+2; x2=x1+H*K; y2=y1+H*V; m2=t2<T; g2=tl.load(g+x2+ik,mask=m2&mk,other=0.).to(tl.float32); a2=tl.load(a+x2+ik,mask=m2&mk,other=0.).to(tl.float32); b2=tl.load(b+x2+ik,mask=m2&mk,other=0.).to(tl.float32); k2=tl.load(k+x2+ik,mask=m2&mk,other=0.).to(tl.float32); v2=tl.load(v+y2+ov,mask=m2&mv,other=0.).to(tl.float32); S3=tl.where(m2,S2*tl.exp(g2)[:,None]+k2[:,None]*v2[None,:]+b2[:,None]*tl.sum(a2[:,None]*S2,0)[None,:],S2)
    t3=t0+3; x3=x2+H*K; y3=y2+H*V; m3=t3<T; g3=tl.load(g+x3+ik,mask=m3&mk,other=0.).to(tl.float32); a3=tl.load(a+x3+ik,mask=m3&mk,other=0.).to(tl.float32); b3=tl.load(b+x3+ik,mask=m3&mk,other=0.).to(tl.float32)
    q3=tl.load(q+x3+ik,mask=m3&mk,other=0.).to(tl.float32); dy3=tl.load(do+y3+ov,mask=m3&mv,other=0.).to(tl.float32); D3=D+q3[:,None]*sc*dy3[None,:]; z3=tl.sum(D3*(tl.exp(g3)[:,None]*S3),1); tl.store(pg+iv*T*H*K+x3+ik,z3,mask=m3&mk); D=tl.where(m3,D3*tl.exp(g3)[:,None]+a3[:,None]*tl.sum(b3[:,None]*D3,0)[None,:],D)
    q2=tl.load(q+x2+ik,mask=m2&mk,other=0.).to(tl.float32); dy2=tl.load(do+y2+ov,mask=m2&mv,other=0.).to(tl.float32); D2=D+q2[:,None]*sc*dy2[None,:]; z2=tl.sum(D2*(tl.exp(g2)[:,None]*S2),1); tl.store(pg+iv*T*H*K+x2+ik,z2,mask=m2&mk); D=tl.where(m2,D2*tl.exp(g2)[:,None]+a2[:,None]*tl.sum(b2[:,None]*D2,0)[None,:],D)
    q1=tl.load(q+x1+ik,mask=m1&mk,other=0.).to(tl.float32); dy1=tl.load(do+y1+ov,mask=m1&mv,other=0.).to(tl.float32); D1=D+q1[:,None]*sc*dy1[None,:]; z1=tl.sum(D1*(tl.exp(g1)[:,None]*S1),1); tl.store(pg+iv*T*H*K+x1+ik,z1,mask=m1&mk); D=tl.where(m1,D1*tl.exp(g1)[:,None]+a1[:,None]*tl.sum(b1[:,None]*D1,0)[None,:],D)
    q0=tl.load(q+x0+ik,mask=m0&mk,other=0.).to(tl.float32); dy0=tl.load(do+y0+ov,mask=m0&mv,other=0.).to(tl.float32); D0=D+q0[:,None]*sc*dy0[None,:]; z0=tl.sum(D0*(tl.exp(g0)[:,None]*S0),1); tl.store(pg+iv*T*H*K+x0+ik,z0,mask=m0&mk)

@triton.jit
def reduce(x,y,N:tl.constexpr,NV:tl.constexpr):
    j=tl.program_id(0)*256+tl.arange(0,256); m=j<N; z=tl.zeros((256,),tl.float32)
    for i in range(NV): z+=tl.load(x+i*N+j,mask=m,other=0.)
    tl.store(y+j,z,mask=m)

class Op(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,a,b,g):
    B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,4); bv=32; nv=triton.cdiv(V,bv); bk=max(16,triton.next_power_of_2(K)); o=torch.empty_like(v); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32); fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV0=bv,num_warps=8,num_stages=2); ctx.save_for_backward(q,k,v,a,b,g,h); return o
 @staticmethod
 def backward(ctx,do):
    q,k,v,a,b,g,h=ctx.saved_tensors; B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,4); bv=32; nv=triton.cdiv(V,bv); bk=max(16,triton.next_power_of_2(K)); n=B*T*H*K; ps=[torch.zeros((nv,n),device=q.device,dtype=torch.float32) for _ in range(5)]; dv=torch.empty_like(v); ds=torch.empty_like(h); bwd[(nv,B*H)](q,k,v,a,b,g,do.contiguous(),h,ds,*ps,dv,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV0=bv,num_warps=8,num_stages=2); gate_exact[(nv,nc,B*H)](q,k,v,a,b,g,do.contiguous(),h,ds,ps[4],T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV0=bv,num_warps=4); gs=[]
    for p,x in zip(ps,(q,k,a,b,g)):
      y=torch.empty_like(x); reduce[(triton.cdiv(n,256),)](p,y,N=n,NV=nv,num_warps=4); gs.append(y)
    return gs[0],gs[1],dv,gs[2],gs[3],gs[4]
def chunked_dplr_delta(q,k,v,alpha,beta,gk): return Op.apply(q,k,v,alpha,beta,gk)
