import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(32)

@triton.jit
def _mm(a,b): return tl.dot(a,b,input_precision='tf32')

@triton.jit
def _wy(k,g,beta,c,A,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr):
 i=tl.program_id(0); z=tl.program_id(1); b=z//HV; hv=z%HV; h=hv//(HV//H)
 t=i*C+tl.arange(0,C); x=tl.arange(0,BK); m=t<T
 kk=tl.load(k+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
 gg=tl.load(g+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32); be=tl.load(beta+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32)
 cc=tl.cumsum(gg,axis=0); tl.store(c+(b*T+t)*HV+hv,cc,mask=m)
 r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]; lo=(r>s)&m[:,None]&m[None,:]
 arg=tl.where(lo,cc[:,None]-cc[None,:],0.); L=tl.where(lo,be[:,None]*tl.exp(tl.maximum(-80.,tl.minimum(0.,arg)))*_mm(kk,tl.trans(kk)),0.)
 M=tl.zeros((C,C),tl.float32)
 for y in tl.static_range(0,C):
  row=tl.where(tl.arange(0,C)==y,1.,0.)
  for j in tl.static_range(0,y):
   lij=tl.sum(tl.where((r==y)&(s==j),L,0.)); mj=tl.sum(tl.where(r==j,M,0.),axis=0)
   row-=lij*mj
  M=tl.where(r==y,row[None,:],M)
 tl.store(A+((((b*NC+i)*HV+hv)*C+r)*C+s),M)

@triton.jit
def _fscan(k,v,c,beta,A,hs,u,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); iv=p%NV; z=p//NV; b=z//HV; hv=z%HV; h=hv//(HV//H)
 x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); S=tl.zeros((BK,BV),tl.float32); r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]
 for i in range(NC):
  so=((((b*NC+i)*HV+hv)*K+x[:,None])*V+y[None,:]); tl.store(hs+so,S,mask=(x[:,None]<K)&(y[None,:]<V))
  t=i*C+tl.arange(0,C); m=t<T
  kk=tl.load(k+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)
  vv=tl.load(v+((b*T+t[:,None])*HV+hv)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.).to(tl.float32)
  cc=tl.load(c+(b*T+t)*HV+hv,mask=m,other=0.); be=tl.load(beta+(b*T+t)*HV+hv,mask=m,other=0.).to(tl.float32)
  M=tl.load(A+((((b*NC+i)*HV+hv)*C+r)*C+s)); e=tl.exp(tl.maximum(-80.,tl.minimum(0.,cc)))
  uu=_mm(M,(vv-_mm(kk*e[:,None],S))*be[:,None])
  tl.store(u+((b*T+t[:,None])*HV+hv)*V+y[None,:],uu,mask=m[:,None]&(y[None,:]<V))
  n=tl.sum(m.to(tl.int32),axis=0); last=tl.sum(tl.where(tl.arange(0,C)==n-1,cc,0.),axis=0)
  ey=tl.exp(tl.maximum(-80.,tl.minimum(0.,tl.where(m,last-cc,0.))))
  S=S*tl.exp(tl.maximum(-80.,tl.minimum(0.,last)))+_mm(tl.trans(kk*ey[:,None]),uu)

@triton.jit
def _out(q,k,c,hs,u,o,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); iv=p%NV; z=p//NV; i=z%NC; z//=NC; b=z//HV; hv=z%HV; h=hv//(HV//H)
 t=i*C+tl.arange(0,C); x=tl.arange(0,BK); y=iv*BV+tl.arange(0,BV); m=t<T
 qq=tl.load(q+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32)*SCALE
 kk=tl.load(k+((b*T+t[:,None])*H+h)*K+x[None,:],mask=m[:,None]&(x[None,:]<K),other=0.).to(tl.float32); cc=tl.load(c+(b*T+t)*HV+hv,mask=m,other=0.)
 S=tl.load(hs+((((b*NC+i)*HV+hv)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
 uu=tl.load(u+((b*T+t[:,None])*HV+hv)*V+y[None,:],mask=m[:,None]&(y[None,:]<V),other=0.)
 r=tl.arange(0,C)[:,None]; s=tl.arange(0,C)[None,:]; lo=(r>=s)&m[:,None]&m[None,:]
 arg=tl.where(lo,cc[:,None]-cc[None,:],0.); R=tl.where(lo,_mm(qq,tl.trans(kk))*tl.exp(tl.maximum(-80.,tl.minimum(0.,arg))),0.)
 oo=_mm(qq*tl.exp(tl.maximum(-80.,tl.minimum(0.,cc)))[:,None],S)+_mm(R,uu)
 tl.store(o+((b*T+t[:,None])*HV+hv)*V+y[None,:],oo,mask=m[:,None]&(y[None,:]<V))

@triton.jit
def _bscan(q,k,v,g,beta,hs,u,do,dq,dk,dv,dg,db,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV2:tl.constexpr,SCALE:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); NV=tl.cdiv(V,BV2); iv=p%NV; z=p//NV; b=z//HV; hv=z%HV; h=hv//(HV//H)
 x=tl.arange(0,BK); y=iv*BV2+tl.arange(0,BV2); Z=tl.zeros((BK,BV2),tl.float32)
 for ri in range(NC):
  ic=NC-1-ri
  S=tl.load(hs+((((b*NC+ic)*HV+hv)*K+x[:,None])*V+y[None,:]),mask=(x[:,None]<K)&(y[None,:]<V),other=0.)
  for jj in tl.static_range(0,C):
   tt=ic*C+jj; valid=tt<T
   ka=tl.load(k+((b*T+tt)*H+h)*K+x,mask=valid&(x<K),other=0.).to(tl.float32)
   va=tl.load(v+((b*T+tt)*HV+hv)*V+y,mask=valid&(y<V),other=0.).to(tl.float32)
   ea=tl.exp(tl.maximum(-80.,tl.minimum(0.,tl.load(g+(b*T+tt)*HV+hv,mask=valid,other=0.).to(tl.float32))))
   ba=tl.load(beta+(b*T+tt)*HV+hv,mask=valid,other=0.).to(tl.float32)
   pred=tl.sum(ka[:,None]*S,axis=0)*ea; nxt=S*ea+ka[:,None]*((va-pred)*ba)[None,:]
   S=tl.where(valid,nxt,S)
  for jj in tl.static_range(0,C):
   t=ic*C+C-1-jj; valid=t<T
   kk=tl.load(k+((b*T+t)*H+h)*K+x,mask=valid&(x<K),other=0.).to(tl.float32)
   vv=tl.load(v+((b*T+t)*HV+hv)*V+y,mask=valid&(y<V),other=0.).to(tl.float32)
   qq=tl.load(q+((b*T+t)*H+h)*K+x,mask=valid&(x<K),other=0.).to(tl.float32)*SCALE
   oo=tl.load(do+((b*T+t)*HV+hv)*V+y,mask=valid&(y<V),other=0.).to(tl.float32)
   ee=tl.exp(tl.maximum(-80.,tl.minimum(0.,tl.load(g+(b*T+t)*HV+hv,mask=valid,other=0.).to(tl.float32))))
   be=tl.load(beta+(b*T+t)*HV+hv,mask=valid,other=0.).to(tl.float32)
   uu=tl.load(u+((b*T+t)*HV+hv)*V+y,mask=valid&(y<V),other=0.)
   Sold=(S-kk[:,None]*uu[None,:])/ee
   pred=ee*tl.sum(kk[:,None]*Sold,axis=0); delta=(vv-pred)*be
   Z1=Z+qq[:,None]*oo[None,:]
   ztk=tl.sum(kk[:,None]*Z1,axis=0); dvi=be*ztk; dpi=-be*ztk
   dki=tl.sum(Z1*delta[None,:],axis=1)+tl.sum(dpi[None,:]*ee*Sold,axis=1)
   dqi=tl.sum(S*oo[None,:],axis=1)*SCALE; dbi=tl.sum(ztk*(vv-pred))
   dgi=tl.sum(Z1*(ee*Sold))+tl.sum(dpi*pred); po=((((b*T+t)*HV+hv)*NV+iv)*K+x)
   tl.store(dq+po,dqi,mask=valid&(x<K)); tl.store(dk+po,dki,mask=valid&(x<K))
   tl.store(dv+((b*T+t)*HV+hv)*V+y,dvi,mask=valid&(y<V)); tl.store(dg+((b*T+t)*HV+hv)*NV+iv,dgi,mask=valid); tl.store(db+((b*T+t)*HV+hv)*NV+iv,dbi,mask=valid)
   Z=tl.where(valid,ee*(Z1+kk[:,None]*dpi[None,:]),Z); S=tl.where(valid,Sold,S)

@triton.jit
def _reduce(pq,pk,pg,pb,dq,dk,dg,db,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr):
 p=tl.program_id(0).to(tl.int64); h=p%H; z=p//H; t=z%T; b=z//T; x=tl.arange(0,BK); aq=tl.zeros((BK,),tl.float32); ak=tl.zeros((BK,),tl.float32)
 for r in range(HV//H):
  hv=h*(HV//H)+r
  for iv in range(NV):
   off=((((b*T+t)*HV+hv)*NV+iv)*K+x); aq+=tl.load(pq+off,mask=x<K,other=0.); ak+=tl.load(pk+off,mask=x<K,other=0.)
 tl.store(dq+((b*T+t)*H+h)*K+x,aq,mask=x<K); tl.store(dk+((b*T+t)*H+h)*K+x,ak,mask=x<K)
 for r in range(HV//H):
  hv=h*(HV//H)+r; ag=tl.zeros((),tl.float32); ab=tl.zeros((),tl.float32)
  for iv in range(NV): ag+=tl.load(pg+((b*T+t)*HV+hv)*NV+iv); ab+=tl.load(pb+((b*T+t)*HV+hv)*NV+iv)
  tl.store(dg+(b*T+t)*HV+hv,ag); tl.store(db+(b*T+t)*HV+hv,ab)

class _Op(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta):
  B,T,H,K=q.shape; HV=v.shape[2]; V=v.shape[3]; nc=triton.cdiv(T,C); nv=triton.cdiv(V,BV); bk=triton.next_power_of_2(K)
  c=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); A=torch.empty((B,nc,HV,C,C),device=q.device,dtype=torch.float32); hs=torch.empty((B,nc,HV,K,V),device=q.device,dtype=torch.float32); u=torch.empty((B,T,HV,V),device=q.device,dtype=torch.float32); o=torch.empty_like(v)
  _wy[(nc,B*HV)](k,g,beta,c,A,T,H,HV,K,nc,BK=bk); _fscan[(B*HV*nv,)](k,v,c,beta,A,hs,u,T,H,HV,K,V,nc,nv,BK=bk); _out[(B*HV*nc*nv,)](q,k,c,hs,u,o,T,H,HV,K,V,nc,nv,BK=bk,SCALE=K**-.5)
  ctx.save_for_backward(q,k,v,g,beta,hs,u); ctx.m=(B,T,H,HV,K,V,nc,bk); return o
 @staticmethod
 def backward(ctx,do):
  do=do.contiguous()
  q,k,v,g,beta,hs,u=ctx.saved_tensors; B,T,H,HV,K,V,nc,bk=ctx.m; nv=triton.cdiv(V,BV)
  pq=torch.empty((B,T,HV,nv,K),device=q.device,dtype=torch.float32); pk=torch.empty_like(pq); pg=torch.empty((B,T,HV,nv),device=q.device,dtype=torch.float32); pb=torch.empty_like(pg); dv=torch.empty_like(v); dq=torch.empty_like(q); dk=torch.empty_like(k); dg=torch.empty_like(g); db=torch.empty_like(beta)
  _bscan[(B*HV*nv,)](q,k,v,g,beta,hs,u,do,pq,pk,dv,pg,pb,T,H,HV,K,V,nc,BK=bk,BV2=BV,SCALE=K**-.5); _reduce[(B*T*H,)](pq,pk,pg,pb,dq,dk,dg,db,T,H,HV,K,nv,BK=bk)
  return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta): return _Op.apply(q,k,v,g,beta)
