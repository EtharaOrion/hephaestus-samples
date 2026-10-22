import torch
import triton
import triton.language as tl

BV = 32

@triton.jit
def _wy(k,g,b,gc,a,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,IP:tl.constexpr):
 c=tl.program_id(0).to(tl.int64); bh=tl.program_id(1).to(tl.int64); ib=bh//HV; h=bh%HV; hk=h//(HV//H)
 i=tl.arange(0,16); x=tl.arange(0,BK); t=c*16+i; m=t<T
 z=tl.load(k+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 gg=tl.load(g+(ib*T+t)*HV+h,mask=m,other=0.).to(tl.float32); bb=tl.load(b+(ib*T+t)*HV+h,mask=m,other=0.).to(tl.float32)
 cg=tl.cumsum(gg,0); valid=m[:,None]&m[None,:]; low=i[:,None]>i[None,:]
 r=tl.exp(tl.minimum(0.,tl.maximum(-80.,cg[:,None]-cg[None,:]))); L=tl.where(valid&low,tl.dot(z,tl.trans(z),input_precision=IP)*bb[:,None]*r,0.)
 A=-L
 for row in range(1,16):
  v=tl.sum(tl.where(i[:,None]==row,-L,0.),0); v=tl.where(i<row,v,0.); v+=tl.sum(v[:,None]*A,0); A=tl.where(i[:,None]==row,v[None,:],A)
 A+=tl.where((i[:,None]==i[None,:])&valid,1.,0.)
 tl.store(gc+(ib*T+t)*HV+h,cg,mask=m); tl.store(a+((ib*T+t[:,None])*HV+h)*16+i[None,:],A,mask=valid)

@triton.jit
def _fscan(k,v,b,gc,a,hs,u,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,VB:tl.constexpr,IP:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,VB); j=p%nv; bh=p//nv; ib=bh//HV; h=bh%HV; hk=h//(HV//H)
 x=tl.arange(0,BK); y=j*VB+tl.arange(0,VB); i=tl.arange(0,16); S=tl.zeros((BK,VB),tl.float32)
 for c in range(NC):
  t=c*16+i; m=t<T; tl.store(hs+((((ib*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),S,mask=(x[:,None]<K)&(y[None,:]<V))
  z=tl.load(k+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
  vv=tl.load(v+((ib*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
  bb=tl.load(b+(ib*T+t)*HV+h,mask=m,other=0.).to(tl.float32); cg=tl.load(gc+(ib*T+t)*HV+h,mask=m,other=0.)
  A=tl.load(a+((ib*T+t[:,None])*HV+h)*16+i[None,:],mask=m[:,None]&m[None,:],other=0.)
  R=bb[:,None]*vv-tl.dot((bb*tl.exp(tl.minimum(0.,tl.maximum(-80.,cg))))[:,None]*z,S,input_precision=IP); U=tl.dot(A,R,input_precision=IP)
  tl.store(u+((ib*T+t[:,None])*HV+h)*V+y[None,:],U,mask=m[:,None]&(y[None,:]<V))
  last=tl.minimum(16-1,T-c*16-1); ce=tl.sum(tl.where(i==last,cg,0.)); zd=z*tl.exp(tl.minimum(0.,tl.maximum(-80.,ce-cg)))[:,None]
  S=tl.exp(tl.minimum(0.,tl.maximum(-80.,ce)))*S+tl.dot(tl.trans(zd),U,input_precision=IP)

@triton.jit
def _out(q,k,gc,hs,u,o,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,VB:tl.constexpr,SCALE:tl.constexpr,IP:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,VB); j=p%nv; w=p//nv; c=w%NC; bh=w//NC; ib=bh//HV; h=bh%HV; hk=h//(HV//H)
 i=tl.arange(0,16); x=tl.arange(0,BK); y=j*VB+tl.arange(0,VB); t=c*16+i; m=t<T
 Q=tl.load(q+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32); Z=tl.load(k+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 cg=tl.load(gc+(ib*T+t)*HV+h,mask=m,other=0.); S=tl.load(hs+((((ib*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
 U=tl.load(u+((ib*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.)
 R=tl.dot(Q,tl.trans(Z),input_precision=IP)*SCALE; R=tl.where((i[:,None]>=i[None,:])&m[:,None]&m[None,:],R*tl.exp(tl.minimum(0.,tl.maximum(-80.,cg[:,None]-cg[None,:]))),0.)
 O=tl.dot(Q*(SCALE*tl.exp(tl.minimum(0.,tl.maximum(-80.,cg))))[:,None],S,input_precision=IP)+tl.dot(R,U,input_precision=IP)
 tl.store(o+((ib*T+t[:,None])*HV+h)*V+y[None,:],O,mask=m[:,None]&(y[None,:]<V))

@triton.jit
def _ascan(q,k,b,gc,a,do,dh,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,VB:tl.constexpr,SCALE:tl.constexpr,IP:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,VB); j=p%nv; bh=p//nv; ib=bh//HV; h=bh%HV; hk=h//(HV//H)
 i=tl.arange(0,16); x=tl.arange(0,BK); y=j*VB+tl.arange(0,VB); D=tl.zeros((BK,VB),tl.float32)
 for zc in range(NC):
  c=NC-1-zc; t=c*16+i; m=t<T
  tl.store(dh+((((ib*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),D,mask=(x[:,None]<K)&(y[None,:]<V))
  Q=tl.load(q+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32); Z=tl.load(k+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
  O=tl.load(do+((ib*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32); cg=tl.load(gc+(ib*T+t)*HV+h,mask=m,other=0.)
  bb=tl.load(b+(ib*T+t)*HV+h,mask=m,other=0.).to(tl.float32); A=tl.load(a+((ib*T+t[:,None])*HV+h)*16+i[None,:],mask=m[:,None]&m[None,:],other=0.)
  last=tl.minimum(16-1,T-c*16-1); ce=tl.sum(tl.where(i==last,cg,0.)); ee=tl.exp(tl.minimum(0.,tl.maximum(-80.,cg)))
  R=tl.dot(Q,tl.trans(Z),input_precision=IP)*SCALE; R=tl.where((i[:,None]>=i[None,:])&m[:,None]&m[None,:],R*tl.exp(tl.minimum(0.,tl.maximum(-80.,cg[:,None]-cg[None,:]))),0.)
  E=tl.dot(tl.trans(R),O,input_precision=IP)+tl.dot(Z*tl.exp(tl.minimum(0.,tl.maximum(-80.,ce-cg)))[:,None],D,input_precision=IP)
  dr=tl.dot(tl.trans(A),E,input_precision=IP)
  D=tl.exp(tl.minimum(0.,tl.maximum(-80.,ce)))*D+tl.dot(tl.trans(Q*(SCALE*ee)[:,None]),O,input_precision=IP)-tl.dot(tl.trans(Z*(bb*ee)[:,None]),dr,input_precision=IP)

@triton.jit
def _grad(q,k,v,b,gc,a,hs,u,do,dh,pq,pk,dv,pg,pb,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,VB:tl.constexpr,SCALE:tl.constexpr,IP:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,VB); j=p%nv; w=p//nv; c=w%NC; bh=w//NC; ib=bh//HV; h=bh%HV; hk=h//(HV//H)
 i=tl.arange(0,16); x=tl.arange(0,BK); y=j*VB+tl.arange(0,VB); t=c*16+i; m=t<T; mx=x<K; my=y<V
 Q=tl.load(q+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&mx[None,:],other=0.).to(tl.float32); Z=tl.load(k+((ib*T+t[:,None])*H+hk)*K+x[None,:],mask=m[:,None]&mx[None,:],other=0.).to(tl.float32)
 V0=tl.load(v+((ib*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.).to(tl.float32); O=tl.load(do+((ib*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.).to(tl.float32)
 U=tl.load(u+((ib*T+t[:,None])*HV+h)*V+y[None,:],mask=m[:,None]&my[None,:],other=0.); S=tl.load(hs+((((ib*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=mx[:,None]&my[None,:],other=0.)
 D=tl.load(dh+((((ib*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=mx[:,None]&my[None,:],other=0.).to(tl.float32); cg=tl.load(gc+(ib*T+t)*HV+h,mask=m,other=0.)
 bb=tl.load(b+(ib*T+t)*HV+h,mask=m,other=0.).to(tl.float32); A=tl.load(a+((ib*T+t[:,None])*HV+h)*16+i[None,:],mask=m[:,None]&m[None,:],other=0.)
 last=tl.minimum(16-1,T-c*16-1); ce=tl.sum(tl.where(i==last,cg,0.)); ee=tl.exp(tl.minimum(0.,tl.maximum(-80.,cg))); er=tl.exp(tl.minimum(0.,tl.maximum(-80.,cg[:,None]-cg[None,:]))); causal=(i[:,None]>=i[None,:])&m[:,None]&m[None,:]; low=(i[:,None]>i[None,:])&m[:,None]&m[None,:]
 G=tl.dot(Q,tl.trans(Z),input_precision=IP)*SCALE; R=tl.where(causal,G*er,0.); W=Z*(bb*ee)[:,None]; ratio=tl.exp(tl.minimum(0.,tl.maximum(-80.,ce-cg))); E=Z*ratio[:,None]
 rhs=bb[:,None]*V0-tl.dot(W,S,input_precision=IP); dR=tl.dot(O,tl.trans(U),input_precision=IP); dE=tl.dot(U,tl.trans(D),input_precision=IP)
 dU=tl.dot(tl.trans(R),O,input_precision=IP)+tl.dot(E,D,input_precision=IP); dA=tl.dot(dU,tl.trans(rhs),input_precision=IP); dX=tl.dot(tl.trans(A),dU,input_precision=IP)
 dL=-tl.dot(tl.trans(A),tl.dot(dA,tl.trans(A),input_precision=IP),input_precision=IP); dL=tl.where(low,dL,0.); dW=-tl.dot(dX,tl.trans(S),input_precision=IP)
 zR=dR*tl.where(causal,SCALE*er,0.); dQ=tl.dot(O,tl.trans(S),input_precision=IP)*(SCALE*ee)[:,None]+tl.dot(zR,Z,input_precision=IP); dZ=tl.dot(tl.trans(zR),Q,input_precision=IP)
 dgc=tl.sum(dR*R,1)-tl.sum(dR*R,0)+tl.sum(O*tl.dot(Q*(SCALE*ee)[:,None],S,input_precision=IP),1); dZ+=dW*(bb*ee)[:,None]; db=tl.sum(dW*(ee[:,None]*Z),1)+tl.sum(dX*V0,1); dgc+=tl.sum(dW*W,1)
 gram=tl.dot(Z,tl.trans(Z),input_precision=IP); KL=tl.where(low,gram*er,0.); zL=dL*bb[:,None]*er; dZ+=tl.dot(zL,Z,input_precision=IP)+tl.dot(tl.trans(zL),Z,input_precision=IP); db+=tl.sum(dL*KL,1); dgc+=tl.sum(dL*(bb[:,None]*KL),1)-tl.sum(dL*(bb[:,None]*KL),0)
 dZ+=dE*ratio[:,None]; ze=tl.sum(dE*E,1); dgc-=ze; dend=tl.sum(ze)+tl.sum(D*(tl.exp(tl.minimum(0.,tl.maximum(-80.,ce)))*S)); dgc+=tl.where(i==last,dend,0.); dg=tl.flip(tl.cumsum(tl.flip(dgc,0),0),0)
 tl.store(pq+((((ib*T+t[:,None])*HV+h)*nv+j)*K+x[None,:]),dQ,mask=m[:,None]&mx[None,:]); tl.store(pk+((((ib*T+t[:,None])*HV+h)*nv+j)*K+x[None,:]),dZ,mask=m[:,None]&mx[None,:])
 tl.store(dv+((ib*T+t[:,None])*HV+h)*V+y[None,:],dX*bb[:,None],mask=m[:,None]&my[None,:]); tl.store(pg+(((ib*T+t)*HV+h)*nv+j),dg,mask=m); tl.store(pb+(((ib*T+t)*HV+h)*nv+j),db,mask=m)

@triton.jit
def _rqk(pq,pk,dq,dk,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); x=tl.arange(0,BK); t=p%T; h=(p//T)%H; ib=p//(T*H); aq=tl.zeros((BK,),tl.float32); ak=tl.zeros((BK,),tl.float32)
 for r in range(HV//H):
  vh=h*(HV//H)+r
  for j in range(NV):
   off=((((ib*T+t)*HV+vh)*NV+j)*K+x); aq+=tl.load(pq+off,mask=x<K,other=0.); ak+=tl.load(pk+off,mask=x<K,other=0.)
 tl.store(dq+((ib*T+t)*H+h)*K+x,aq,mask=x<K); tl.store(dk+((ib*T+t)*H+h)*K+x,ak,mask=x<K)

@triton.jit
def _rs(pg,pb,dg,db,N:tl.constexpr,NV:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); ag=0.; ab=0.
 for j in range(NV): ag+=tl.load(pg+p*NV+j); ab+=tl.load(pb+p*NV+j)
 tl.store(dg+p,ag); tl.store(db+p,ab)

class _Rule(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,b):
  B,T,H,K=q.shape; HV,V=v.shape[2:]; ip='ieee' if q.dtype == torch.float32 else 'tf32'; nc=triton.cdiv(T,16); nv=triton.cdiv(V,BV); bk=triton.next_power_of_2(K)
  gc=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); a=torch.empty((B,T,HV,16),device=q.device,dtype=torch.float32); hs=torch.empty((B,nc,HV,K,V),device=q.device,dtype=torch.float32); u=torch.empty((B,T,HV,V),device=q.device,dtype=torch.float32); o=torch.empty_like(v)
  _wy[(nc,B*HV)](k,g,b,gc,a,T=T,H=H,HV=HV,K=K,BK=bk,IP=ip,num_warps=4); _fscan[(B*HV*nv,)](k,v,b,gc,a,hs,u,T=T,H=H,HV=HV,K=K,V=V,NC=nc,BK=bk,VB=BV,IP=ip,num_warps=4); _out[(B*HV*nc*nv,)](q,k,gc,hs,u,o,T=T,H=H,HV=HV,K=K,V=V,NC=nc,BK=bk,VB=BV,SCALE=K**-.5,IP=ip,num_warps=4)
  ctx.save_for_backward(q,k,v,b,gc,a,hs,u); ctx.d=(B,T,H,HV,K,V,nc,nv,bk); return o
 @staticmethod
 def backward(ctx,do):
  q,k,v,b,gc,a,hs,u=ctx.saved_tensors; B,T,H,HV,K,V,nc,nv,bk=ctx.d; ip='ieee' if q.dtype == torch.float32 else 'tf32'
  do=do.contiguous()
  dh=torch.empty_like(hs); pq=torch.empty((B,T,HV,nv,K),device=q.device,dtype=torch.float32); pk=torch.empty_like(pq); pg=torch.empty((B,T,HV,nv),device=q.device,dtype=torch.float32); pb=torch.empty_like(pg); dq=torch.empty_like(q); dk=torch.empty_like(k); dv=torch.empty_like(v); dg=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); db=torch.empty_like(b)
  _ascan[(B*HV*nv,)](q,k,b,gc,a,do,dh,T=T,H=H,HV=HV,K=K,V=V,NC=nc,BK=bk,VB=BV,SCALE=K**-.5,IP=ip,num_warps=4); _grad[(B*HV*nc*nv,)](q,k,v,b,gc,a,hs,u,do,dh,pq,pk,dv,pg,pb,T=T,H=H,HV=HV,K=K,V=V,NC=nc,BK=bk,VB=BV,SCALE=K**-.5,IP=ip,num_warps=4); _rqk[(B*T*H,)](pq,pk,dq,dk,T=T,H=H,HV=HV,K=K,NV=nv,BK=bk,num_warps=4); _rs[(B*T*HV,)](pg,pb,dg,db,N=B*T*HV,NV=nv)
  return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta): return _Rule.apply(q,k,v,g,beta)
