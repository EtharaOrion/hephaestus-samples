import torch
import triton
import triton.language as tl

BT=16; FBV=32; BBV=8

@triton.jit
def _inv(l,r):
 e=(r[:,None]==r[None,:]).to(tl.float32); z=l; x=e+z
 for _ in tl.static_range(0,4):
  z=tl.dot(z,z,input_precision='ieee'); x+=tl.dot(z,x,input_precision='ieee')
 return x

@triton.jit
def _fwd(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv=tl.program_id(0); nh=tl.program_id(1); ih=nh%H; ib=nh//H
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); r=tl.arange(0,16)
 mk=x<K; mv=y<V; bk=ib*T*H*K+ih*K; bv=ib*T*H*V+ih*V
 s=tl.zeros((BK,BV),tl.float32); sc=K ** -0.5
 for c in range(NC):
  t=c*16+r; mt=t<T
  tl.store(h+((ib*NC+c)*H+ih)*K*V+x[:,None]*V+y[None,:],s,mask=mk[:,None]&mv[None,:])
  m=mt[:,None]&mk[None,:]
  Q=tl.load(q+bk+t[:,None]*H*K+x[None,:],mask=m,other=0.).to(tl.float32)*sc
  A=tl.load(a+bk+t[:,None]*H*K+x[None,:],mask=m,other=0.).to(tl.float32)
  B=tl.load(b+bk+t[:,None]*H*K+x[None,:],mask=m,other=0.).to(tl.float32)
  G=tl.load(g+bk+t[:,None]*H*K+x[None,:],mask=m,other=0.).to(tl.float32)
  K0=tl.load(k+bk+t[:,None]*H*K+x[None,:],mask=m,other=0.).to(tl.float32)
  V0=tl.load(v+bv+t[:,None]*H*V+y[None,:],mask=mt[:,None]&mv[None,:],other=0.).to(tl.float32)
  C=tl.cumsum(G,axis=0); mm=mt[:,None]&mt[None,:]; lo=r[:,None]>r[None,:]; ca=r[:,None]>=r[None,:]
  dq=tl.exp(C[:,None,:]-C[None,:,:]); da=tl.exp(C[:,None,:]-G[:,None,:]-C[None,:,:])
  qk=tl.sum(Q[:,None,:]*K0[None,:,:]*dq,axis=2); qb=tl.sum(Q[:,None,:]*B[None,:,:]*dq,axis=2)
  ak=tl.sum(A[:,None,:]*K0[None,:,:]*da,axis=2); ab=tl.sum(A[:,None,:]*B[None,:,:]*da,axis=2)
  qk=tl.where(mm&ca,qk,0.); qb=tl.where(mm&ca,qb,0.); ak=tl.where(mm&lo,ak,0.); ab=tl.where(mm&lo,ab,0.)
  ii=_inv(ab,r); u=tl.dot(ii,tl.dot(ak,V0,input_precision='ieee'),input_precision='ieee')
  w=tl.dot(ii,A*tl.exp(C-G),input_precision='ieee'); vn=u+tl.dot(w,s,input_precision='ieee')
  z=tl.dot(Q*tl.exp(C),s,input_precision='ieee')+tl.dot(qk,V0,input_precision='ieee')+tl.dot(qb,vn,input_precision='ieee')
  tl.store(o+bv+t[:,None]*H*V+y[None,:],z,mask=mt[:,None]&mv[None,:])
  last=tl.minimum(15,T-1-c*16); ge=tl.sum(tl.where(r[:,None]==last,C,0.),axis=0); rem=tl.exp(ge[None,:]-C)
  s=s*tl.exp(ge)[:,None]+tl.dot(tl.trans(tl.where(mt[:,None],K0*rem,0.)),V0,input_precision='ieee')+tl.dot(tl.trans(tl.where(mt[:,None],B*rem,0.)),vn,input_precision='ieee')

@triton.jit
def _checkpoint(k,v,a,b,g,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv=tl.program_id(0); nh=tl.program_id(1); ih=nh%H; ib=nh//H
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mk=x<K; mv=y<V; bk=ib*T*H*K+ih*K; bv=ib*T*H*V+ih*V
 s=tl.zeros((BK,BV),tl.float32)
 for c in range(NC):
  tl.store(h+((ib*NC+c)*H+ih)*K*V+x[:,None]*V+y[None,:],s,mask=mk[:,None]&mv[None,:])
  for j in tl.static_range(0,16):
   t=c*16+j; mt=t<T
   A=tl.load(a+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); B=tl.load(b+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
   K0=tl.load(k+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); G=tl.load(g+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
   V0=tl.load(v+bv+t*H*V+y,mask=mt&mv,other=0.).to(tl.float32); rd=tl.sum(s*A[:,None],axis=0)
   s=s*tl.exp(G)[:,None]+K0[:,None]*V0[None,:]+B[:,None]*rd[None,:]

@triton.jit
def _expand(k,v,a,b,g,h,ss,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv=tl.program_id(0); c=tl.program_id(1); nh=tl.program_id(2); ih=nh%H; ib=nh//H
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mk=x<K; mv=y<V
 bk=ib*T*H*K+ih*K; bv=ib*T*H*V+ih*V
 s=tl.load(h+((ib*NC+c)*H+ih)*K*V+x[:,None]*V+y[None,:],mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32)
 for j in tl.static_range(0,16):
  t=c*16+j; mt=t<T
  tl.store(ss+((ib*T+t)*H+ih)*K*V+x[:,None]*V+y[None,:],s,mask=mt&mk[:,None]&mv[None,:])
  A=tl.load(a+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); B=tl.load(b+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
  K0=tl.load(k+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); G=tl.load(g+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
  V0=tl.load(v+bv+t*H*V+y,mask=mt&mv,other=0.).to(tl.float32); rd=tl.sum(s*A[:,None],axis=0)
  s=s*tl.exp(G)[:,None]+K0[:,None]*V0[None,:]+B[:,None]*rd[None,:]

@triton.jit
def _bounds(q,a,b,g,do,dh,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv=tl.program_id(0); nh=tl.program_id(1); ih=nh%H; ib=nh//H
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mk=x<K; mv=y<V; bk=ib*T*H*K+ih*K; bv=ib*T*H*V+ih*V
 s=tl.zeros((BK,BV),tl.float32); sc=K ** -0.5
 for cr in range(NC):
  c=NC-1-cr
  for jr in tl.static_range(0,16):
   t=c*16+15-jr; mt=t<T
   Q=tl.load(q+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)*sc; A=tl.load(a+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
   B=tl.load(b+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); G=tl.load(g+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
   D=tl.load(do+bv+t*H*V+y,mask=mt&mv,other=0.).to(tl.float32); s+=Q[:,None]*D[None,:]
   rd=tl.sum(s*B[:,None],axis=0); s=s*tl.exp(G)[:,None]+A[:,None]*rd[None,:]
  tl.store(dh+((ib*NC+c)*H+ih)*K*V+x[:,None]*V+y[None,:],s,mask=mk[:,None]&mv[None,:])

@triton.jit
def _bwd(q,k,v,a,b,g,do,ss,dh,pq,pk,pv,pa,pb,pg,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv=tl.program_id(0); c=tl.program_id(1); nh=tl.program_id(2); ih=nh%H; ib=nh//H
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); mk=x<K; mv=y<V; bk=ib*T*H*K+ih*K; bv=ib*T*H*V+ih*V
 ds=tl.zeros((BK,BV),tl.float32); sc=K ** -0.5
 if c+1<NC: ds=tl.load(dh+((ib*NC+c+1)*H+ih)*K*V+x[:,None]*V+y[None,:],mask=mk[:,None]&mv[None,:],other=0.).to(tl.float32)
 for jr in tl.static_range(0,16):
  t=c*16+15-jr; mt=t<T
  X=tl.load(ss+((ib*T+t)*H+ih)*K*V+x[:,None]*V+y[None,:],mask=mt&mk[:,None]&mv[None,:],other=0.).to(tl.float32)
  Q=tl.load(q+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); K0=tl.load(k+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
  A=tl.load(a+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); B=tl.load(b+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32)
  G=tl.load(g+bk+t*H*K+x,mask=mt&mk,other=0.).to(tl.float32); V0=tl.load(v+bv+t*H*V+y,mask=mt&mv,other=0.).to(tl.float32)
  D=tl.load(do+bv+t*H*V+y,mask=mt&mv,other=0.).to(tl.float32); E=tl.exp(G); rd0=tl.sum(X*A[:,None],axis=0)
  Y=X*E[:,None]+K0[:,None]*V0[None,:]+B[:,None]*rd0[None,:]; dq=tl.sum(Y*D[None,:],axis=1)*sc
  ds+=Q[:,None]*sc*D[None,:]; dv=tl.sum(ds*K0[:,None],axis=0); dk=tl.sum(ds*V0[None,:],axis=1); db=tl.sum(ds*rd0[None,:],axis=1)
  rd=tl.sum(ds*B[:,None],axis=0); da=tl.sum(X*rd[None,:],axis=1); dg=tl.sum(ds*X*E[:,None],axis=1)
  pi=((ib*T+t)*H+ih)*NV*BK+iv*BK+x
  tl.store(pq+pi,dq,mask=mt&mk); tl.store(pk+pi,dk,mask=mt&mk); tl.store(pa+pi,da,mask=mt&mk); tl.store(pb+pi,db,mask=mt&mk); tl.store(pg+pi,dg,mask=mt&mk)
  tl.store(pv+bv+t*H*V+y,dv,mask=mt&mv); ds=ds*E[:,None]+A[:,None]*rd[None,:]

@triton.jit
def _reduce(x,y,N:tl.constexpr,K:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,BS:tl.constexpr):
 z=tl.program_id(0)*BS+tl.arange(0,BS); m=z<N*K; n=z//K; k=z-n*K; a=tl.zeros((BS,),tl.float32)
 for i in range(NV): a+=tl.load(x+n*NV*BK+i*BK+k,mask=m,other=0.)
 tl.store(y+z,a,mask=m)

class D(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,a,b,g):
  B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,BT); bk=max(16,triton.next_power_of_2(K)); nv=triton.cdiv(V,FBV)
  o=torch.empty_like(v); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
  _fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=FBV,num_warps=8,num_stages=2)
  _checkpoint[(triton.cdiv(V,BBV),B*H)](k,v,a,b,g,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=BBV,num_warps=4,num_stages=2)
  ctx.save_for_backward(q,k,v,a,b,g,h); ctx.d=(B,T,H,K,V,nc,bk); return o
 @staticmethod
 def backward(ctx,do):
  do=do.contiguous()
  q,k,v,a,b,g,h=ctx.saved_tensors; B,T,H,K,V,nc,bk=ctx.d; bv=BBV; nv=triton.cdiv(V,bv)
  ss=torch.empty((B,T,H,K,V),device=q.device,dtype=torch.float32); dh=torch.empty_like(h)
  _expand[(nv,nc,B*H)](k,v,a,b,g,h,ss,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=4,num_stages=2)
  _bounds[(nv,B*H)](q,a,b,g,do,dh,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=4,num_stages=2)
  ps=[torch.zeros((B,T,H,nv,bk),device=q.device,dtype=torch.float32) for _ in range(5)]; dv=torch.empty_like(v)
  _bwd[(nv,nc,B*H)](q,k,v,a,b,g,do,ss,dh,ps[0],ps[1],dv,ps[2],ps[3],ps[4],T=T,H=H,K=K,V=V,NC=nc,NV=nv,BK=bk,BV=bv,num_warps=4,num_stages=2)
  out=[]; n=B*T*H
  for p,z0 in zip(ps,(q,k,a,b,g)):
   z=torch.zeros_like(z0); _reduce[(triton.cdiv(n*K,256),)](p,z,N=n,K=K,NV=nv,BK=bk,BS=256); out.append(z)
  return out[0],out[1],dv,out[2],out[3],out[4]

def chunked_dplr_delta(q,k,v,alpha,beta,gk): return D.apply(q,k,v,alpha,beta,gk)
