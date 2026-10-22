import torch
import triton
import triton.language as tl

BT, BV = 16, 32

@triton.jit
def _fwd(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV_:tl.constexpr):
 iv,bh=tl.program_id(0),tl.program_id(1); ib,ih=bh//H,bh%H
 xk=tl.arange(0,BK); xv=iv*BV_+tl.arange(0,BV_); mh=(xk[:,None]<K)&(xv[None,:]<V)
 s=tl.zeros((BK,BV_),tl.float32); nc:tl.constexpr=tl.cdiv(T,16); sc:tl.constexpr=1./tl.sqrt(K+0.)
 for c in range(nc):
  ph=h+(((ib*H+ih)*nc+c)*K+xk[:,None])*V+xv[None,:]; tl.store(ph,s,mask=mh)
  xt=c*16+tl.arange(0,16); mt=xt<T
  pk=((ib*T+xt[:,None])*H+ih)*K+xk[None,:]; pv=((ib*T+xt[:,None])*H+ih)*V+xv[None,:]
  mtk=mt[:,None]&(xk[None,:]<K); mtv=mt[:,None]&(xv[None,:]<V)
  Q=tl.load(q+pk,mask=mtk,other=0.).to(tl.float32)*sc; Kt=tl.load(k+pk,mask=mtk,other=0.).to(tl.float32)
  A=tl.load(a+pk,mask=mtk,other=0.).to(tl.float32); B=tl.load(b+pk,mask=mtk,other=0.).to(tl.float32)
  G0=tl.load(g+pk,mask=mtk,other=0.).to(tl.float32); X=tl.load(v+pv,mask=mtv,other=0.).to(tl.float32)
  G=tl.cumsum(G0,axis=0); Qg=Q*tl.exp(G); Ki=Kt*tl.exp(-G); Bi=B*tl.exp(-G); Ag=A*tl.exp(G-G0)
  qk=tl.dot(Qg,tl.trans(Ki),input_precision='ieee'); qb=tl.dot(Qg,tl.trans(Bi),input_precision='ieee')
  ak=tl.dot(Ag,tl.trans(Ki),input_precision='ieee'); ab=tl.dot(Ag,tl.trans(Bi),input_precision='ieee')
  ii=tl.arange(0,16); le=ii[:,None]>=ii[None,:]; lt=ii[:,None]>ii[None,:]
  qk=tl.where(le,qk,0.); qb=tl.where(le,qb,0.); ak=tl.where(lt,ak,0.); ab=tl.where(lt,ab,0.)
  u=tl.dot(ak,X,input_precision='ieee')+tl.dot(Ag,s,input_precision='ieee')
  for j in range(16):
   uj=tl.sum(tl.where((ii==j)[:,None],u,0.),axis=0); abj=tl.sum(tl.where((ii[None,:]==j),ab,0.),axis=1); u+=tl.where((ii>j)[:,None],abj[:,None]*uj[None,:],0.)
  z=tl.dot(Qg,s,input_precision='ieee')+tl.dot(qk,X,input_precision='ieee')+tl.dot(qb,u,input_precision='ieee')
  tl.store(o+pv,z,mask=mtv)
  nvalid=tl.minimum(T-c*16,16); last=tl.sum(tl.where((ii==nvalid-1)[:,None],G,0.),axis=0); d=tl.exp(last-G)
  s=s*tl.exp(last)[:,None]+tl.dot(tl.trans(Kt*d),X,input_precision='ieee')+tl.dot(tl.trans(B*d),u,input_precision='ieee')

@triton.jit
def _expand(k,v,a,b,g,h,ss,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV_:tl.constexpr):
 iv,c,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); ib,ih=bh//H,bh%H; nc:tl.constexpr=tl.cdiv(T,16)
 xk=tl.arange(0,BK); xv=iv*BV_+tl.arange(0,BV_); mh=(xk[:,None]<K)&(xv[None,:]<V)
 s=tl.load(h+(((ib*H+ih)*nc+c)*K+xk[:,None])*V+xv[None,:],mask=mh,other=0.).to(tl.float32)
 for j in range(16):
  t=c*16+j
  if t<T:
   ps=ss+(((ib*T+t)*H+ih)*K+xk[:,None])*V+xv[None,:]; tl.store(ps,s,mask=mh)
   pk=((ib*T+t)*H+ih)*K+xk; pv=((ib*T+t)*H+ih)*V+xv; mk=xk<K; mv=xv<V
   Kt=tl.load(k+pk,mask=mk,other=0.).to(tl.float32); A=tl.load(a+pk,mask=mk,other=0.).to(tl.float32)
   B=tl.load(b+pk,mask=mk,other=0.).to(tl.float32); G=tl.load(g+pk,mask=mk,other=0.).to(tl.float32); X=tl.load(v+pv,mask=mv,other=0.).to(tl.float32)
   r=tl.sum(s*A[:,None],axis=0); s=s*tl.exp(G)[:,None]+Kt[:,None]*X[None,:]+B[:,None]*r[None,:]

@triton.jit
def _bwd(q,k,v,a,b,g,do,ss,p,dv,T:tl.constexpr,B_:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV_:tl.constexpr):
 iv,bh=tl.program_id(0),tl.program_id(1); ib,ih=bh//H,bh%H
 xk=tl.arange(0,BK); xv=iv*BV_+tl.arange(0,BV_); mk=xk<K; mv=xv<V; mh=mk[:,None]&mv[None,:]
 ds=tl.zeros((BK,BV_),tl.float32); sc:tl.constexpr=1./tl.sqrt(K+0.); N:tl.constexpr=B_*T*H*K
 for z in range(T):
  t=T-1-z; pk=((ib*T+t)*H+ih)*K+xk; pv=((ib*T+t)*H+ih)*V+xv
  s0=tl.load(ss+(((ib*T+t)*H+ih)*K+xk[:,None])*V+xv[None,:],mask=mh,other=0.).to(tl.float32)
  Q=tl.load(q+pk,mask=mk,other=0.).to(tl.float32)*sc; Kt=tl.load(k+pk,mask=mk,other=0.).to(tl.float32)
  A=tl.load(a+pk,mask=mk,other=0.).to(tl.float32); B=tl.load(b+pk,mask=mk,other=0.).to(tl.float32)
  G=tl.load(g+pk,mask=mk,other=0.).to(tl.float32); X=tl.load(v+pv,mask=mv,other=0.).to(tl.float32); D=tl.load(do+pv,mask=mv,other=0.).to(tl.float32)
  r=tl.sum(s0*A[:,None],axis=0); s1=s0*tl.exp(G)[:,None]+Kt[:,None]*X[None,:]+B[:,None]*r[None,:]
  ds+=Q[:,None]*D[None,:]; dq=tl.sum(s1*D[None,:],axis=1)*sc; dk=tl.sum(ds*X[None,:],axis=1); dx=tl.sum(ds*Kt[:,None],axis=0)
  dr=tl.sum(ds*B[:,None],axis=0); da=tl.sum(s0*dr[None,:],axis=1); db=tl.sum(ds*r[None,:],axis=1); dg=tl.sum(ds*s0,axis=1)*tl.exp(G)
  tl.store(dv+pv,dx,mask=mv); base=iv*5*N+((ib*T+t)*H+ih)*K+xk
  tl.store(p+base,dq,mask=mk); tl.store(p+base+N,dk,mask=mk); tl.store(p+base+2*N,da,mask=mk); tl.store(p+base+3*N,db,mask=mk); tl.store(p+base+4*N,dg,mask=mk)
  ds=ds*tl.exp(G)[:,None]+A[:,None]*dr[None,:]

@triton.jit
def _reduce(p,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
 x=tl.program_id(0)*BS+tl.arange(0,BS); m=x<N; z0=tl.zeros((BS,),tl.float32);z1=tl.zeros((BS,),tl.float32);z2=tl.zeros((BS,),tl.float32);z3=tl.zeros((BS,),tl.float32);z4=tl.zeros((BS,),tl.float32)
 for i in range(NV):
  y=i*5*N+x; z0+=tl.load(p+y,mask=m,other=0.);z1+=tl.load(p+y+N,mask=m,other=0.);z2+=tl.load(p+y+2*N,mask=m,other=0.);z3+=tl.load(p+y+3*N,mask=m,other=0.);z4+=tl.load(p+y+4*N,mask=m,other=0.)
 tl.store(dq+x,z0,mask=m);tl.store(dk+x,z1,mask=m);tl.store(da+x,z2,mask=m);tl.store(db+x,z3,mask=m);tl.store(dg+x,z4,mask=m)

class _DPLR(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,a,b,g):
  B,T,H,K=q.shape;V=v.shape[-1];bk=max(16,triton.next_power_of_2(K));nv=triton.cdiv(V,BV);nc=triton.cdiv(T,BT)
  o=torch.empty_like(v);h=torch.empty((B,H,nc,K,V),device=q.device,dtype=torch.float32)
  _fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,BK=bk,BV_=BV,num_warps=8,num_stages=2);ctx.save_for_backward(q,k,v,a,b,g,h);return o
 @staticmethod
 def backward(ctx,do):
  do=do.contiguous()
  q,k,v,a,b,g,h=ctx.saved_tensors;B,T,H,K=q.shape;V=v.shape[-1];bk=max(16,triton.next_power_of_2(K));nv=triton.cdiv(V,BV);nc=triton.cdiv(T,BT)
  ss=torch.empty((B,T,H,K,V),device=q.device,dtype=torch.float32);_expand[(nv,nc,B*H)](k,v,a,b,g,h,ss,T=T,H=H,K=K,V=V,BK=bk,BV_=BV,num_warps=8,num_stages=2)
  p=torch.empty((nv,5,B,T,H,K),device=q.device,dtype=torch.float32);dv=torch.empty_like(v);_bwd[(nv,B*H)](q,k,v,a,b,g,do,ss,p,dv,T=T,B_=B,H=H,K=K,V=V,BK=bk,BV_=BV,num_warps=8,num_stages=2)
  dq,dk,da,db=(torch.empty_like(q) for _ in range(4));dg=torch.empty_like(g);n=B*T*H*K;_reduce[(triton.cdiv(n,256),)](p,dq,dk,da,db,dg,N=n,NV=nv,BS=256,num_warps=4)
  return dq,dk,dv,da,db,dg

def chunked_dplr_delta(q,k,v,alpha,beta,gk): return _DPLR.apply(q,k,v,alpha,beta,gk)
