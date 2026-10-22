import torch
import triton
import triton.language as tl

@triton.jit
def _fwd(q,k,v,a,b,g,o,h,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr):
 iv=tl.program_id(0); z=tl.program_id(1); ib=z//H; ih=z%H
 rk=tl.arange(0,BK); rv=iv*16+tl.arange(0,16); rt=tl.arange(0,16); mk=rk<K; mv=rv<V
 S=tl.zeros((BK,16),tl.float32); sc=K ** -0.5
 for ic in range(NC):
  ts=ic*16+rt; mt=ts<T; ix=((ib*T+ts[:,None])*H+ih)*K+rk[None,:]; iy=((ib*T+ts[:,None])*H+ih)*V+rv[None,:]
  mx=mt[:,None]&mk[None,:]; my=mt[:,None]&mv[None,:]
  Q=tl.load(q+ix,mask=mx,other=0.).to(tl.float32)*sc; X=tl.load(k+ix,mask=mx,other=0.).to(tl.float32)
  A=tl.load(a+ix,mask=mx,other=0.).to(tl.float32); B=tl.load(b+ix,mask=mx,other=0.).to(tl.float32)
  G=tl.load(g+ix,mask=mx,other=0.).to(tl.float32); W=tl.load(v+iy,mask=my,other=0.).to(tl.float32)
  C=tl.cumsum(G,axis=0); P=C-G; mi=tl.minimum(7,T-ic*16-1); M=tl.sum(tl.where(rt[:,None]==mi,C,0.),axis=0)
  ec=tl.exp(tl.maximum(tl.minimum(C-M[None,:],60.),-60.)); en=tl.exp(tl.maximum(tl.minimum(M[None,:]-C,60.),-60.)); ep=tl.exp(tl.maximum(tl.minimum(P-M[None,:],60.),-60.))
  Qc=Q*ec; Ac=A*ep; Xc=X*en; Bc=B*en; lo=rt[:,None]>=rt[None,:]; st=rt[:,None]>rt[None,:]
  QX=tl.where(lo,tl.dot(Qc,tl.trans(Xc),input_precision='ieee'),0.); QB=tl.where(lo,tl.dot(Qc,tl.trans(Bc),input_precision='ieee'),0.)
  AX=tl.where(st,tl.dot(Ac,tl.trans(Xc),input_precision='ieee'),0.); AB=tl.where(st,tl.dot(Ac,tl.trans(Bc),input_precision='ieee'),0.)
  ph=((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]); tl.store(h+ph,S,mask=mk[:,None]&mv[None,:])
  R=tl.dot(A*tl.exp(tl.maximum(P,-60.)),S,input_precision='ieee')+tl.dot(AX,W,input_precision='ieee'); U=tl.zeros((16,16),tl.float32)
  for j in tl.static_range(0,16):
   row=tl.sum(tl.where(rt[:,None]==j,AB,0.),axis=0); rj=tl.sum(tl.where(rt[:,None]==j,R,0.),axis=0); uj=rj+tl.sum(row[:,None]*U,axis=0); U+=tl.where(rt[:,None]==j,uj[None,:],0.)
  O=tl.dot(Q*tl.exp(tl.maximum(C,-60.)),S,input_precision='ieee')+tl.dot(QX,W,input_precision='ieee')+tl.dot(QB,U,input_precision='ieee'); tl.store(o+iy,O,mask=my)
  for j in tl.static_range(0,16):
   active=(ic*16+j)<T; xj=tl.sum(tl.where(rt[:,None]==j,X,0.),axis=0); aj=tl.sum(tl.where(rt[:,None]==j,A,0.),axis=0); bj=tl.sum(tl.where(rt[:,None]==j,B,0.),axis=0); gj=tl.sum(tl.where(rt[:,None]==j,G,0.),axis=0); wj=tl.sum(tl.where(rt[:,None]==j,W,0.),axis=0)
   rj=tl.sum(aj[:,None]*S,axis=0); S=tl.where(active,S*tl.exp(gj)[:,None]+xj[:,None]*wj[None,:]+bj[:,None]*rj[None,:],S)

@triton.jit
def _bwd(q,k,v,a,b,g,do,h,pq,pk,pv,pa,pb,pg,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr):
 iv=tl.program_id(0); z=tl.program_id(1); ib=z//H; ih=z%H
 rk=tl.arange(0,BK); rv=iv*16+tl.arange(0,16); rt=tl.arange(0,16); mk=rk<K; mv=rv<V; DS=tl.zeros((BK,16),tl.float32); sc=K ** -0.5
 for iz in range(NC):
  ic=NC-1-iz; ts=ic*16+rt; mt=ts<T; ix=((ib*T+ts[:,None])*H+ih)*K+rk[None,:]; iy=((ib*T+ts[:,None])*H+ih)*V+rv[None,:]
  mx=mt[:,None]&mk[None,:]; my=mt[:,None]&mv[None,:]
  Q=tl.load(q+ix,mask=mx,other=0.).to(tl.float32)*sc; X=tl.load(k+ix,mask=mx,other=0.).to(tl.float32); A=tl.load(a+ix,mask=mx,other=0.).to(tl.float32); B=tl.load(b+ix,mask=mx,other=0.).to(tl.float32)
  G=tl.load(g+ix,mask=mx,other=0.).to(tl.float32); W=tl.load(v+iy,mask=my,other=0.).to(tl.float32); D=tl.load(do+iy,mask=my,other=0.).to(tl.float32)
  ph=((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]); S=tl.load(h+ph,mask=mk[:,None]&mv[None,:],other=0.)
  C=tl.cumsum(G,axis=0); P=C-G; mi=tl.minimum(7,T-ic*16-1); M=tl.sum(tl.where(rt[:,None]==mi,C,0.),axis=0)
  ec=tl.exp(tl.maximum(tl.minimum(C-M[None,:],60.),-60.)); en=tl.exp(tl.maximum(tl.minimum(M[None,:]-C,60.),-60.)); ep=tl.exp(tl.maximum(tl.minimum(P-M[None,:],60.),-60.)); EQ=tl.exp(tl.maximum(C,-60.)); EA=tl.exp(tl.maximum(P,-60.))
  Qc=Q*ec; Ac=A*ep; Xc=X*en; Bc=B*en; Qs=Q*EQ; As=A*EA; lo=rt[:,None]>=rt[None,:]; st=rt[:,None]>rt[None,:]
  QX=tl.where(lo,tl.dot(Qc,tl.trans(Xc),input_precision='ieee'),0.); QB=tl.where(lo,tl.dot(Qc,tl.trans(Bc),input_precision='ieee'),0.)
  AX=tl.where(st,tl.dot(Ac,tl.trans(Xc),input_precision='ieee'),0.); AB=tl.where(st,tl.dot(Ac,tl.trans(Bc),input_precision='ieee'),0.)
  R=tl.dot(As,S,input_precision='ieee')+tl.dot(AX,W,input_precision='ieee'); U=tl.zeros((16,16),tl.float32)
  for j in tl.static_range(0,16):
   row=tl.sum(tl.where(rt[:,None]==j,AB,0.),axis=0); rj=tl.sum(tl.where(rt[:,None]==j,R,0.),axis=0); uj=rj+tl.sum(row[:,None]*U,axis=0); U+=tl.where(rt[:,None]==j,uj[None,:],0.)
  li=tl.minimum(15,T-ic*16-1); E=tl.sum(tl.where(rt[:,None]==li,C,0.),axis=0); ee=tl.exp(tl.maximum(E,-60.)); er=tl.exp(tl.maximum(E[None,:]-C,-60.)); XE=X*er; BE=B*er
  dE=tl.sum(DS*(S*ee[:,None]+tl.dot(tl.trans(XE),W,input_precision='ieee')+tl.dot(tl.trans(BE),U,input_precision='ieee')),axis=1)
  dQs=tl.dot(D,tl.trans(S),input_precision='ieee'); dS=DS*ee[:,None]+tl.dot(tl.trans(Qs),D,input_precision='ieee'); dQX=tl.where(lo,tl.dot(D,tl.trans(W),input_precision='ieee'),0.); dQB=tl.where(lo,tl.dot(D,tl.trans(U),input_precision='ieee'),0.)
  dW=tl.dot(tl.trans(QX),D,input_precision='ieee')+tl.dot(XE,DS,input_precision='ieee'); dU=tl.dot(tl.trans(QB),D,input_precision='ieee')+tl.dot(BE,DS,input_precision='ieee')
  dXE=tl.dot(W,tl.trans(DS),input_precision='ieee'); dBE=tl.dot(U,tl.trans(DS),input_precision='ieee'); Y=tl.zeros((16,16),tl.float32)
  for jj in tl.static_range(0,16):
   j=15-jj; col=tl.sum(tl.where(rt[None,:]==j,AB,0.),axis=1); dj=tl.sum(tl.where(rt[:,None]==j,dU,0.),axis=0); yj=dj+tl.sum(col[:,None]*Y,axis=0); Y+=tl.where(rt[:,None]==j,yj[None,:],0.)
  dAB=tl.where(st,tl.dot(Y,tl.trans(U),input_precision='ieee'),0.); dAX=tl.where(st,tl.dot(Y,tl.trans(W),input_precision='ieee'),0.); dAs=tl.dot(Y,tl.trans(S),input_precision='ieee')
  dS+=tl.dot(tl.trans(As),Y,input_precision='ieee'); dW+=tl.dot(tl.trans(AX),Y,input_precision='ieee')
  dQc=tl.dot(dQX,Xc,input_precision='ieee')+tl.dot(dQB,Bc,input_precision='ieee'); dXc=tl.dot(tl.trans(dQX),Qc,input_precision='ieee')+tl.dot(tl.trans(dAX),Ac,input_precision='ieee')
  dBc=tl.dot(tl.trans(dQB),Qc,input_precision='ieee')+tl.dot(tl.trans(dAB),Ac,input_precision='ieee'); dAc=tl.dot(dAX,Xc,input_precision='ieee')+tl.dot(dAB,Bc,input_precision='ieee')
  dQ=dQs*EQ+dQc*ec; dA=dAs*EA+dAc*ep; dX=dXE*er+dXc*en; dB=dBE*er+dBc*en
  zG=dQ*Q+dA*A-dX*X-dB*B; off=dA*A; dG=tl.zeros((16,BK),tl.float32); rz=tl.zeros((BK,),tl.float32)
  for jj in tl.static_range(0,16):
   j=15-jj; xz=tl.sum(tl.where(rt[:,None]==j,zG,0.),axis=0); xo=tl.sum(tl.where(rt[:,None]==j,off,0.),axis=0); rz+=xz; dG+=tl.where(rt[:,None]==j,(rz-xo+dE)[None,:],0.)
  DS=dS; ip=iv*(T*H*K)+ix
  tl.store(pq+ip,dQ*sc,mask=mx); tl.store(pk+ip,dX,mask=mx); tl.store(pa+ip,dA,mask=mx); tl.store(pb+ip,dB,mask=mx); tl.store(pg+ip,dG,mask=mx); tl.store(pv+iy,dW,mask=my)

@triton.jit
def _reduce(pq,pk,pa,pb,pg,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr,BS:tl.constexpr):
 x=tl.program_id(0)*BS+tl.arange(0,BS); m=x<N; q=tl.zeros((BS,),tl.float32); k=tl.zeros((BS,),tl.float32); a=tl.zeros((BS,),tl.float32); b=tl.zeros((BS,),tl.float32); g=tl.zeros((BS,),tl.float32)
 for i in range(NV):
  z=i*N+x; q+=tl.load(pq+z,mask=m,other=0.); k+=tl.load(pk+z,mask=m,other=0.); a+=tl.load(pa+z,mask=m,other=0.); b+=tl.load(pb+z,mask=m,other=0.); g+=tl.load(pg+z,mask=m,other=0.)
 tl.store(dq+x,q,mask=m); tl.store(dk+x,k,mask=m); tl.store(da+x,a,mask=m); tl.store(db+x,b,mask=m); tl.store(dg+x,g,mask=m)

class _Dplr(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,a,b,g):
  B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,16); bk=max(16,triton.next_power_of_2(K)); nv=triton.cdiv(V,16); o=torch.empty_like(v); h=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
  _fwd[(nv,B*H)](q,k,v,a,b,g,o,h,T=T,H=H,K=K,V=V,NC=nc,BK=bk,num_warps=8,num_stages=2); ctx.save_for_backward(q,k,v,a,b,g,h); return o
 @staticmethod
 def backward(ctx,do):
  q,k,v,a,b,g,h=ctx.saved_tensors; B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,16); bk=max(16,triton.next_power_of_2(K)); bv=16; nv=triton.cdiv(V,bv); n=B*T*H*K; shape=(nv,B,T,H,K)
  pq=torch.zeros(shape,device=q.device,dtype=torch.float32); pk=torch.zeros_like(pq); pa=torch.zeros_like(pq); pb=torch.zeros_like(pq); pg=torch.zeros_like(pq); dv=torch.empty_like(v)
  _bwd[(nv,B*H)](q,k,v,a,b,g,do.contiguous(),h,pq,pk,dv,pa,pb,pg,T=T,H=H,K=K,V=V,NC=nc,NV=nv,BK=bk,num_warps=8,num_stages=2)
  _exact_bwd[(nv,B*H)](q,k,v,a,b,g,do.contiguous(),h,pq,pk,dv,pa,pb,pg,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=8,num_stages=2)
  dq=torch.empty_like(q); dk=torch.empty_like(k); da=torch.empty_like(a); db=torch.empty_like(b); dg=torch.empty_like(g); _reduce[(triton.cdiv(n,256),)](pq,pk,pa,pb,pg,dq,dk,da,db,dg,N=n,NV=nv,BS=256,num_warps=8)
  return dq,dk,dv,da,db,dg

def chunked_dplr_delta(q,k,v,alpha,beta,gk): return _Dplr.apply(q,k,v,alpha,beta,gk)

@triton.jit
def _exact_bwd(q,k,v,a,b,g,do,h,pq,pk,pv,pa,pb,pg,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
 iv=tl.program_id(0); z=tl.program_id(1); ib=z//H; ih=z%H; rk=tl.arange(0,BK); rv=iv*BV+tl.arange(0,BV); mk=rk<K; mv=rv<V
 DS=tl.zeros((BK,BV),tl.float32); sc=K**-0.5
 for iz in range(NC):
  ic=NC-1-iz
  for jj in tl.static_range(0,16):
   j=15-jj; t=ic*16+j; valid=t<T; ix=((ib*T+t)*H+ih)*K+rk; iy=((ib*T+t)*H+ih)*V+rv
   SO=tl.zeros((BK,BV),tl.float32); ph=((((ib*NC+ic)*H+ih)*K+rk[:,None])*V+rv[None,:]); SO=tl.load(h+ph,mask=mk[:,None]&mv[None,:],other=0.)
   for u in tl.static_range(0,16):
    active=u<j; tu=ic*16+u; iux=((ib*T+tu)*H+ih)*K+rk; iuy=((ib*T+tu)*H+ih)*V+rv
    XU=tl.load(k+iux,mask=active&valid&mk,other=0.).to(tl.float32); AU=tl.load(a+iux,mask=active&valid&mk,other=0.).to(tl.float32); BU=tl.load(b+iux,mask=active&valid&mk,other=0.).to(tl.float32)
    GU=tl.load(g+iux,mask=active&valid&mk,other=0.).to(tl.float32); WU=tl.load(v+iuy,mask=active&valid&mv,other=0.).to(tl.float32); RU=tl.sum(AU[:,None]*SO,axis=0)
    SO=tl.where(active,SO*tl.exp(GU)[:,None]+XU[:,None]*WU[None,:]+BU[:,None]*RU[None,:],SO)
   Q=tl.load(q+ix,mask=valid&mk,other=0.).to(tl.float32)*sc; X=tl.load(k+ix,mask=valid&mk,other=0.).to(tl.float32); A=tl.load(a+ix,mask=valid&mk,other=0.).to(tl.float32); B=tl.load(b+ix,mask=valid&mk,other=0.).to(tl.float32); G=tl.load(g+ix,mask=valid&mk,other=0.).to(tl.float32)
   W=tl.load(v+iy,mask=valid&mv,other=0.).to(tl.float32); D=tl.load(do+iy,mask=valid&mv,other=0.).to(tl.float32); R=tl.sum(A[:,None]*SO,axis=0); SN=SO*tl.exp(G)[:,None]+X[:,None]*W[None,:]+B[:,None]*R[None,:]
   dQ=tl.sum(SN*D[None,:],axis=1)*sc; DS+=Q[:,None]*D[None,:]; dX=tl.sum(DS*W[None,:],axis=1); dW=tl.sum(DS*X[:,None],axis=0); dB=tl.sum(DS*R[None,:],axis=1); dR=tl.sum(DS*B[:,None],axis=0); dA=tl.sum(SO*dR[None,:],axis=1); EG=tl.exp(G); dG=tl.sum(DS*SO*EG[:,None],axis=1); DS=DS*EG[:,None]+A[:,None]*dR[None,:]
   ip=iv*(T*H*K)+ix; tl.store(pg+ip,dG,mask=valid&mk)
