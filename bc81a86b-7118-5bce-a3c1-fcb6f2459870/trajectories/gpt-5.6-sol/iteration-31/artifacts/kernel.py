import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(32)

@triton.jit
def _wy(k,g,beta,gc,A,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
 c=tl.program_id(0).to(tl.int64); bh=tl.program_id(1).to(tl.int64); b=bh//HV; h=bh%HV
 i=tl.arange(0,C); t=c*C+i; mt=t<T; x=tl.arange(0,BK); kh=h//(HV//H)
 gb=tl.load(g+b*T*HV+t*HV+h,mask=mt,other=0.).to(tl.float32); cs=tl.cumsum(gb,axis=0)
 tl.store(gc+b*T*HV+t*HV+h,cs,mask=mt)
 kb=tl.load(k+b*T*H*K+t[:,None]*H*K+kh*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)
 gram=tl.dot(kb,tl.trans(kb),input_precision='tf32')
 bb=tl.load(beta+b*T*HV+t*HV+h,mask=mt,other=0.).to(tl.float32)
 strict=(i[:,None]>i[None,:])&mt[:,None]&mt[None,:]
 L=tl.where(strict,gram*bb[:,None]*tl.exp(tl.maximum(-80.,tl.minimum(80.,cs[:,None]-cs[None,:]))),0.)
 inv=tl.where(i[:,None]==i[None,:],1.,0.)
 for r in range(1,C):
  lr=tl.sum(tl.where(i[:,None]==r,L,0.),axis=0)
  row=tl.where(i==r,1.,0.)-tl.sum(lr[:,None]*inv,axis=0)
  inv=tl.where(i[:,None]==r,row[None,:],inv)
 tl.store(A+((b*T+t[:,None])*HV+h)*C+i[None,:],inv,mask=mt[:,None])

@triton.jit
def _fscan(k,v,beta,gc,A,Sbuf,Dbuf,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,BV); iv=p%nv; bh=p//nv; b=bh//HV; h=bh%HV; kh=h//(HV//H)
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); i=tl.arange(0,C); S=tl.zeros((BK,BV),tl.float32)
 for c in range(NC):
  t=c*C+i; mt=t<T
  tl.store(Sbuf+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),S,mask=(x[:,None]<K)&(y[None,:]<V))
  aa=tl.load(A+((b*T+t[:,None])*HV+h)*C+i[None,:],mask=mt[:,None],other=0.)
  gg=tl.load(gc+b*T*HV+t*HV+h,mask=mt,other=0.); bb=tl.load(beta+b*T*HV+t*HV+h,mask=mt,other=0.)
  kb=tl.load(k+b*T*H*K+t[:,None]*H*K+kh*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)
  vb=tl.load(v+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
  at=aa.to(kb.dtype); U=tl.dot(at,(vb*bb[:,None]).to(vb.dtype),input_precision='tf32')
  Z=(kb*(bb*tl.exp(gg))[:,None]).to(kb.dtype); W=tl.dot(at,Z,input_precision='tf32')
  D=U-tl.dot(W,S.to(W.dtype),input_precision='tf32')
  tl.store(Dbuf+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],D,mask=mt[:,None]&(y[None,:]<V))
  li=tl.minimum(C-1,T-1-c*C); last=tl.sum(tl.where(i==li,gg,0.),axis=0)
  rk=kb*tl.exp(last-gg)[:,None]
  S=S*tl.exp(last)+tl.dot(tl.trans(rk),D.to(rk.dtype),input_precision='tf32')

@triton.jit
def _out(q,k,gc,Sbuf,Dbuf,o,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); nv=tl.cdiv(V,BV); iv=p%nv; c=(p//nv)%NC; bh=p//(nv*NC); b=bh//HV; h=bh%HV; kh=h//(HV//H)
 i=tl.arange(0,C); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); t=c*C+i; mt=t<T
 qb=tl.load(q+b*T*H*K+t[:,None]*H*K+kh*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)*SCALE
 kb=tl.load(k+b*T*H*K+t[:,None]*H*K+kh*K+x[None,:],mask=mt[:,None]&(x[None,:]<K),other=0.)
 gg=tl.load(gc+b*T*HV+t*HV+h,mask=mt,other=0.)
 D=tl.load(Dbuf+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],mask=mt[:,None]&(y[None,:]<V),other=0.)
 S=tl.load(Sbuf+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
 mask=(i[:,None]>=i[None,:])&mt[:,None]&mt[None,:]
 M=tl.where(mask,tl.dot(qb,tl.trans(kb),input_precision='tf32')*tl.exp(tl.maximum(-80.,tl.minimum(80.,gg[:,None]-gg[None,:]))),0.)
 eq=(qb*tl.exp(gg)[:,None]).to(qb.dtype)
 ob=tl.dot(eq,S.to(qb.dtype),input_precision='tf32')+tl.dot(M.to(D.dtype),D,input_precision='tf32')
 tl.store(o+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],ob,mask=mt[:,None]&(y[None,:]<V))

@triton.jit
def _bscan(q,k,v,beta,gc,A,Sbuf,Dbuf,do,pq,pk,dv,pg,pb,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); iv=p%NV; bh=p//NV; b=bh//HV; h=bh%HV; kh=h//(HV//H)
 i=tl.arange(0,C); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); dS=tl.zeros((BK,BV),tl.float32)
 for z in range(NC):
  c=NC-1-z; t=c*C+i; mt=t<T; mx=x<K; my=y<V
  qb=tl.load(q+b*T*H*K+t[:,None]*H*K+kh*K+x[None,:],mask=mt[:,None]&mx[None,:],other=0.)*SCALE
  kb=tl.load(k+b*T*H*K+t[:,None]*H*K+kh*K+x[None,:],mask=mt[:,None]&mx[None,:],other=0.)
  vb=tl.load(v+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],mask=mt[:,None]&my[None,:],other=0.)
  dob=tl.load(do+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],mask=mt[:,None]&my[None,:],other=0.)
  D=tl.load(Dbuf+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],mask=mt[:,None]&my[None,:],other=0.)
  S=tl.load(Sbuf+((((b*NC+c)*HV+h)*K+x[:,None])*V+y[None,:]),mask=mx[:,None]&my[None,:],other=0.)
  aa=tl.load(A+((b*T+t[:,None])*HV+h)*C+i[None,:],mask=mt[:,None],other=0.)
  gg=tl.load(gc+b*T*HV+t*HV+h,mask=mt,other=0.); bb=tl.load(beta+b*T*HV+t*HV+h,mask=mt,other=0.)
  li=tl.minimum(C-1,T-1-c*C); last=tl.sum(tl.where(i==li,gg,0.),axis=0); ee=tl.exp(gg); rr=tl.exp(last-gg); a=tl.exp(last)
  causal=(i[:,None]>=i[None,:])&mt[:,None]&mt[None,:]; strict=(i[:,None]>i[None,:])&mt[:,None]&mt[None,:]
  qk=tl.dot(qb,tl.trans(kb),input_precision='tf32'); CE=tl.where(causal,qk*tl.exp(tl.maximum(-80.,tl.minimum(80.,gg[:,None]-gg[None,:]))),0.)
  rk=kb*rr[:,None]
  dD=tl.dot(tl.trans(CE.to(dob.dtype)),dob,input_precision='tf32')+tl.dot(rk,dS.to(rk.dtype),input_precision='tf32')
  dCE=tl.dot(dob,tl.trans(D.to(dob.dtype)),input_precision='tf32'); dEQ=tl.dot(dob,tl.trans(S.to(dob.dtype)),input_precision='tf32'); drk=tl.dot(D,tl.trans(dS.to(D.dtype)),input_precision='tf32')
  R=(vb*bb[:,None]).to(vb.dtype); Z=(kb*(bb*ee)[:,None]).to(kb.dtype); at=aa.to(kb.dtype); W=tl.dot(at,Z,input_precision='tf32')
  dW=-tl.dot(dD,tl.trans(S.to(dD.dtype)),input_precision='tf32')
  dA=tl.dot(dD,tl.trans(R.to(dD.dtype)),input_precision='tf32')+tl.dot(dW,tl.trans(Z.to(dW.dtype)),input_precision='tf32')
  dR=tl.dot(tl.trans(at),dD.to(at.dtype),input_precision='tf32'); dZ=tl.dot(tl.trans(at),dW.to(at.dtype),input_precision='tf32')
  mid=tl.dot(dA.to(at.dtype),tl.trans(at),input_precision='tf32').to(at.dtype)
  dL=-tl.dot(tl.trans(at),mid,input_precision='tf32'); dL=tl.where(strict,dL,0.)
  gram=tl.dot(kb,tl.trans(kb),input_precision='tf32'); E=tl.exp(tl.maximum(-80.,tl.minimum(80.,gg[:,None]-gg[None,:]))); P=tl.where(strict,gram*E,0.); dP=dL*bb[:,None]; dgram=dP*E
  X=tl.where(causal,dCE*E,0.)
  dq=ee[:,None]*dEQ+tl.dot(X.to(kb.dtype),kb,input_precision='tf32')
  dk=tl.dot(tl.trans(X.to(qb.dtype)),qb,input_precision='tf32')+rr[:,None]*drk+dZ*(bb*ee)[:,None]
  dk+=tl.dot(dgram.to(kb.dtype),kb,input_precision='tf32')+tl.dot(tl.trans(dgram.to(kb.dtype)),kb,input_precision='tf32')
  dvb=dR*bb[:,None]
  db=tl.sum(dR*vb,axis=1)+tl.sum(dZ*(ee[:,None]*kb),axis=1)+tl.sum(dL*P,axis=1)
  dgc=tl.sum(dEQ*(ee[:,None]*qb),axis=1)+tl.sum(dZ*Z,axis=1)
  dgc+=tl.sum(dCE*CE,axis=1)-tl.sum(dCE*CE,axis=0)+tl.sum(dP*P,axis=1)-tl.sum(dP*P,axis=0)
  zr=tl.sum(drk*rk,axis=1); dgc-=zr
  dlast=tl.sum(zr,axis=0)+tl.sum(tl.sum(dS*(a*S),axis=1),axis=0); dgc+=tl.where(i==li,dlast,0.)
  dg=tl.sum(dgc,axis=0)-tl.cumsum(dgc,axis=0)+dgc
  off=(((iv*HV+h)*T+t[:,None])*K+x[None,:]); tl.store(pq+b*NV*HV*T*K+off,dq*SCALE,mask=mt[:,None]&mx[None,:]); tl.store(pk+b*NV*HV*T*K+off,dk,mask=mt[:,None]&mx[None,:])
  tl.store(dv+b*T*HV*V+t[:,None]*HV*V+h*V+y[None,:],dvb,mask=mt[:,None]&my[None,:])
  so=(iv*HV+h)*T+t; tl.store(pg+b*NV*HV*T+so,dg,mask=mt); tl.store(pb+b*NV*HV*T+so,db,mask=mt)
  dS=a*dS+tl.dot(tl.trans((qb*ee[:,None]).to(dob.dtype)),dob,input_precision='tf32')-tl.dot(tl.trans(W),dD.to(W.dtype),input_precision='tf32')

@triton.jit
def _rqk(pq,pk,dq,dk,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); t=p%T; z=p//T; h=z%H; b=z//H; x=tl.arange(0,BK); aq=tl.zeros((BK,),tl.float32); ak=tl.zeros((BK,),tl.float32)
 for r in range(HV//H):
  vh=h*(HV//H)+r
  for iv in range(NV):
   off=b*NV*HV*T*K+((iv*HV+vh)*T+t)*K+x; aq+=tl.load(pq+off,mask=x<K,other=0.); ak+=tl.load(pk+off,mask=x<K,other=0.)
 dst=((b*T+t)*H+h)*K+x; tl.store(dq+dst,aq,mask=x<K); tl.store(dk+dst,ak,mask=x<K)

@triton.jit
def _rs(pg,pb,dg,db,T:tl.constexpr,HV:tl.constexpr,NV:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); h=p%HV; z=p//HV; t=z%T; b=z//T; ag=0.; ab=0.
 for iv in range(NV):
  off=b*NV*HV*T+(iv*HV+h)*T+t; ag+=tl.load(pg+off); ab+=tl.load(pb+off)
 dst=(b*T+t)*HV+h; tl.store(dg+dst,ag); tl.store(db+dst,ab)

class _Op(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta):
  B,T,H,K=q.shape; HV,V=v.shape[2:]; NC=triton.cdiv(T,C); NV=triton.cdiv(V,BV); BK=max(16,triton.next_power_of_2(K))
  gc=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); A=torch.empty((B,T,HV,C),device=q.device,dtype=torch.float32)
  S=torch.empty((B,NC,HV,K,V),device=q.device,dtype=torch.float32); D=torch.empty((B,T,HV,V),device=q.device,dtype=torch.float32); o=torch.empty_like(v)
  _wy[(NC,B*HV)](k,g,beta,gc,A,T=T,H=H,HV=HV,K=K,BK=BK)
  _fscan[(B*HV*NV,)](k,v,beta,gc,A,S,D,T=T,H=H,HV=HV,K=K,V=V,NC=NC,BK=BK)
  _out[(B*HV*NC*NV,)](q,k,gc,S,D,o,T=T,H=H,HV=HV,K=K,V=V,NC=NC,BK=BK,SCALE=K**-.5)
  ctx.save_for_backward(q,k,v,beta,gc,A,S,D); ctx.m=(B,T,H,HV,K,V,NC,NV,BK); return o
 @staticmethod
 def backward(ctx,do):
  do=do.contiguous()
  q,k,v,beta,gc,A,S,D=ctx.saved_tensors; B,T,H,HV,K,V,NC,NV,BK=ctx.m
  pq=torch.empty((B,NV,HV,T,K),device=q.device,dtype=torch.float32); pk=torch.empty_like(pq); pg=torch.empty((B,NV,HV,T),device=q.device,dtype=torch.float32); pb=torch.empty_like(pg)
  dq=torch.empty_like(q); dk=torch.empty_like(k); dv=torch.empty_like(v); dg=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); db=torch.empty_like(beta)
  _bscan[(B*HV*NV,)](q,k,v,beta,gc,A,S,D,do,pq,pk,dv,pg,pb,T=T,H=H,HV=HV,K=K,V=V,NC=NC,NV=NV,BK=BK,SCALE=K**-.5)
  _rqk[(B*H*T,)](pq,pk,dq,dk,T=T,H=H,HV=HV,K=K,NV=NV,BK=BK); _rs[(B*T*HV,)](pg,pb,dg,db,T=T,HV=HV,NV=NV)
  return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta): return _Op.apply(q,k,v,g,beta)
