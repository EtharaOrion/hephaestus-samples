import torch
import triton
import triton.language as tl
R, C, K = tl.constexpr(64), tl.constexpr(64), tl.constexpr(32)
@triton.jit
def router(X,W,S,T,D,E,B:tl.constexpr):
 m=tl.program_id(0)*B+tl.arange(0,B); n=tl.program_id(1)*B+tl.arange(0,B); z=tl.zeros((B,B),tl.float32)
 for k0 in range(0,D,K):
  k=k0+tl.arange(0,K); a=tl.load(X+m[:,None]*D+k[None,:],mask=(m[:,None]<T)&(k[None,:]<D),other=0.).to(tl.float32); b=tl.load(W+k[:,None]*E+n[None,:],mask=(k[:,None]<D)&(n[None,:]<E),other=0.).to(tl.float32); z+=tl.dot(a,b,input_precision='ieee')
 tl.store(S+m[:,None]*E+n[None,:],tl.sigmoid(z),mask=(m[:,None]<T)&(n[None,:]<E))
@triton.jit
def keys(S,B,O,N,E,BS:tl.constexpr):
 q=tl.program_id(0)*BS+tl.arange(0,BS); mm=q<N; e=q%E; v=tl.load(S+q,mask=mm,other=0.)+tl.load(B+e,mask=mm,other=0.); v=tl.where(v==0.,0.,v); i=v.to(tl.int32,bitcast=True); j=i.to(tl.int64); u=tl.where(i>=0,j^0x80000000,(~j)&0xffffffff); tl.store(O+q,(u<<10)|(E-1-e),mask=mm)
@triton.jit
def dkeys(SEL,O,N,BS:tl.constexpr):
 q=tl.program_id(0)*BS+tl.arange(0,BS); m=q<N; e=tl.load(SEL+q,mask=m,other=0).to(tl.int64); tl.store(O+q,e*N+q,mask=m)
@triton.jit
def weights(S,SEL,W,T,E,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); a=tl.arange(0,B); m=a<A; e=tl.load(SEL+t*A+a,mask=m,other=0); s=tl.load(S+t*E+e,mask=m,other=0.); tl.store(W+t*A+a,s/tl.sum(tl.where(m,s,0.),axis=0),mask=m)
@triton.jit
def offsets(SEL,ORD,OFF,N,E):
 x=tl.program_id(0); lo=0; hi=N
 while lo<hi:
  md=(lo+hi)//2; p=tl.load(ORD+md); e=tl.load(SEL+p); c=e<x; lo=tl.where(c,md+1,lo); hi=tl.where(c,hi,md)
 tl.store(OFF+x,lo)
@triton.jit
def tiles(OFF,TE,TS,E,CAP:tl.constexpr):
 e=0; o=0
 while e<E:
  p=tl.load(OFF+e); end=tl.load(OFF+e+1)
  while p<end: tl.store(TE+o,e); tl.store(TS+o,p); o+=1; p+=R
  e+=1
 while o<CAP: tl.store(TE+o,-1); tl.store(TS+o,0); o+=1
@triton.jit
def inverse(ORD,INV,N,B:tl.constexpr):
 r=tl.program_id(0)*B+tl.arange(0,B); m=r<N; p=tl.load(ORD+r,mask=m,other=0); tl.store(INV+p,r,mask=m)
@triton.jit
def ein(X,W,ORD,OFF,TE,TS,O,N,D,F,A:tl.constexpr,BF:tl.constexpr):
 ti=tl.program_id(0); e=tl.load(TE+ti); st=tl.load(TS+ti); r=st+tl.arange(0,R); j=tl.program_id(1)*C+tl.arange(0,C); end=tl.load(OFF+e+1,mask=e>=0,other=0); rm=(e>=0)&(r<end)&(r<N); p=tl.load(ORD+r,mask=rm,other=0); t=p//A; z=tl.zeros((R,C),tl.float32)
 for k0 in range(0,D,K):
  k=k0+tl.arange(0,K); a=tl.load(X+t[:,None]*D+k[None,:],mask=rm[:,None]&(k[None,:]<D),other=0.); b=tl.load(W+e*D*2*F+k[:,None]*2*F+j[None,:],mask=(e>=0)&(k[:,None]<D)&(j[None,:]<2*F),other=0.); z+=tl.dot(a,b) if BF else tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision='tf32x3')
 tl.store(O+r[:,None]*2*F+j[None,:],z,mask=rm[:,None]&(j[None,:]<2*F))
@triton.jit
def act(G,A,N,F,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B); m=q<N; r=q//F; f=q%F; g=tl.load(G+r*2*F+f,mask=m,other=0.); u=tl.load(G+r*2*F+F+f,mask=m,other=0.); tl.store(A+q,g*tl.sigmoid(g)*u,mask=m)
@triton.jit
def seg(A,W,OFF,TE,TS,O,N,M,KD,TRANS:tl.constexpr,BF:tl.constexpr,SPLIT:tl.constexpr):
 ti=tl.program_id(0); e=tl.load(TE+ti); st=tl.load(TS+ti); r=st+tl.arange(0,R); j=tl.program_id(1)*C+tl.arange(0,C); end=tl.load(OFF+e+1,mask=e>=0,other=0); rm=(e>=0)&(r<end)&(r<N); z=tl.zeros((R,C),tl.float32)
 for k0 in range(0,KD,K):
  k=k0+tl.arange(0,K); a=tl.load(A+r[:,None]*KD+k[None,:],mask=rm[:,None]&(k[None,:]<KD),other=0.)
  if TRANS: b=tl.load(W+e*M*KD+j[None,:]*KD+k[:,None],mask=(e>=0)&(j[None,:]<M)&(k[:,None]<KD),other=0.)
  else: b=tl.load(W+e*KD*M+k[:,None]*M+j[None,:],mask=(e>=0)&(k[:,None]<KD)&(j[None,:]<M),other=0.)
  if BF:
   if SPLIT:
    ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); z+=tl.dot(ah,b)+tl.dot(al,b)
   else: z+=tl.dot(a.to(tl.bfloat16),b)
  else: z+=tl.dot(a,b.to(tl.float32),input_precision='tf32x3')
 tl.store(O+r[:,None]*M+j[None,:],z,mask=rm[:,None]&(j[None,:]<M))
@triton.jit
def combine(YP,W,INV,Y,T,D,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); z=tl.zeros((B,),tl.float32)
 for a in range(A):
  p=t*A+a; r=tl.load(INV+p); z+=tl.load(W+p)*tl.load(YP+r*D+d,mask=d<D,other=0.)
 tl.store(Y+t*D+d,z,mask=d<D)
@triton.jit
def dz(DY,W,ORD,O,N,D,A:tl.constexpr,B:tl.constexpr):
 r=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); p=tl.load(ORD+r); t=p//A; tl.store(O+r*D+d,tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32)*tl.load(W+p),mask=d<D)
@triton.jit
def pdot(DY,Y,ORD,O,N,D,A:tl.constexpr,B:tl.constexpr):
 r=tl.program_id(0); d=tl.arange(0,B); p=tl.load(ORD+r); t=p//A; x=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32); y=tl.load(Y+r*D+d,mask=d<D,other=0.); tl.store(O+p,tl.sum(x*y,axis=0))
@triton.jit
def dactivation(G,DA,O,N,F,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B); m=q<N; r=q//F; f=q%F; g=tl.load(G+r*2*F+f,mask=m,other=0.); u=tl.load(G+r*2*F+F+f,mask=m,other=0.); da=tl.load(DA+q,mask=m,other=0.); s=tl.sigmoid(g); si=g*s; tl.store(O+r*2*F+f,da*u*(s+si*(1-s)),mask=m); tl.store(O+r*2*F+F+f,da*si,mask=m)
@triton.jit
def dw(A,B,ORD,OFF,O,E,M,N,KD,D,AUX:tl.constexpr,GATHER:tl.constexpr):
 e=tl.program_id(0); i=tl.program_id(1)*128+tl.arange(0,128); j=tl.program_id(2)*64+tl.arange(0,64); st=tl.load(OFF+e); end=tl.load(OFF+e+1); z=tl.zeros((128,64),tl.float32); q=st
 while q<end:
  k=q+tl.arange(0,K); p=tl.load(ORD+k,mask=k<end,other=0); row=p//AUX if GATHER else k
  a=tl.load(A+row[None,:]*M+i[:,None],mask=(k[None,:]<end)&(i[:,None]<M),other=0.).to(tl.float32)
  b=tl.load(B+k[:,None]*N+j[None,:],mask=(k[:,None]<end)&(j[None,:]<N),other=0.).to(tl.float32)
  if GATHER: z+=tl.dot(a,b,input_precision='tf32x3')
  else:
   ah=a.to(tl.bfloat16); bh=b.to(tl.bfloat16); bl=(b-bh).to(tl.bfloat16)
   z+=tl.dot(ah,bh)+tl.dot(ah,bl)+tl.dot((a-ah).to(tl.bfloat16),bh)
  q+=K
 tl.store(O+e*M*N+i[:,None]*N+j[None,:],z,mask=(i[:,None]<M)&(j[None,:]<N))
@triton.jit
def routeds(S,SEL,W,CV,O,T,E,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); a=tl.arange(0,B); m=a<A; p=t*A+a; w=tl.load(W+p,mask=m,other=0.); c=tl.load(CV+p,mask=m,other=0.); mean=tl.sum(w*c,axis=0); e=tl.load(SEL+p,mask=m,other=0); s=tl.load(S+t*E+e,mask=m,other=0.); den=tl.sum(tl.where(m,s,0.),axis=0); tl.store(O+p,(c-mean)/den*s*(1-s),mask=m)
@triton.jit
def dense(SEL,DS,O,T,E,A:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B); m=q<T*E; t=q//E; e=q%E; z=tl.zeros((B,),tl.float32)
 for a in range(A):
  se=tl.load(SEL+t*A+a,mask=m,other=-1); z+=tl.where(se==e,tl.load(DS+t*A+a,mask=m,other=0.),0.)
 tl.store(O+q,z,mask=m)
@triton.jit
def mm(A,B,O,M,N,KD,TA:tl.constexpr,TB:tl.constexpr,IEEE:tl.constexpr):
 i=tl.program_id(0)*32+tl.arange(0,32); j=tl.program_id(1)*32+tl.arange(0,32); z=tl.zeros((32,32),tl.float32)
 for k0 in range(0,KD,32):
  k=k0+tl.arange(0,32); a=tl.load(A+k[None,:]*M+i[:,None] if TA else A+i[:,None]*KD+k[None,:],mask=(i[:,None]<M)&(k[None,:]<KD),other=0.).to(tl.float32); b=tl.load(B+j[None,:]*KD+k[:,None] if TB else B+k[:,None]*N+j[None,:],mask=(k[:,None]<KD)&(j[None,:]<N),other=0.).to(tl.float32); z+=tl.dot(a,b,input_precision='ieee' if IEEE else 'tf32x3')
 tl.store(O+i[:,None]*N+j[None,:],z,mask=(i[:,None]<M)&(j[None,:]<N))
@triton.jit
def addrows(P,RW,INV,O,T,D,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); z=tl.load(RW+t*D+d,mask=d<D,other=0.)
 for a in range(A): z+=tl.load(P+tl.load(INV+t*A+a)*D+d,mask=d<D,other=0.)
 tl.store(O+t*D+d,z,mask=d<D)
class Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
  A=int(A); T,D=x.shape; E=rw.shape[1]; F=w1.shape[2]//2; N=T*A; bf=x.dtype==torch.bfloat16; S=torch.empty((T,E),device=x.device,dtype=torch.float32); router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,S,T,D,E,B=32); ky=torch.empty((T,E),device=x.device,dtype=torch.int64); keys[(triton.cdiv(T*E,256),)](S,rb,ky,T*E,E,BS=256); sel=torch.topk(ky,A,dim=1,sorted=True).indices; W=torch.empty((T,A),device=x.device,dtype=torch.float32); weights[(T,)](S,sel,W,T,E,A=A,B=triton.next_power_of_2(A)); dk=torch.empty(N,device=x.device,dtype=torch.int64); dkeys[(triton.cdiv(N,256),)](sel,dk,N,BS=256); order=torch.argsort(dk); off=torch.empty(E+1,device=x.device,dtype=torch.int32); offsets[(E+1,)](sel,order,off,N,E,num_warps=1); cap=triton.cdiv(N,R)+E; te=torch.empty(cap,device=x.device,dtype=torch.int32); ts=torch.empty_like(te); tiles[(1,)](off,te,ts,E,CAP=cap,num_warps=1); inv=torch.empty(N,device=x.device,dtype=torch.int64); inverse[(triton.cdiv(N,256),)](order,inv,N,B=256); gu=torch.empty((N,2*F),device=x.device,dtype=torch.float32); ein[(cap,triton.cdiv(2*F,C))](x,w1,order,off,te,ts,gu,N,D,F,A=A,BF=bf); ac=torch.empty((N,F),device=x.device,dtype=torch.float32); act[(triton.cdiv(N*F,256),)](gu,ac,N*F,F,B=256); yp=torch.empty((N,D),device=x.device,dtype=torch.float32); seg[(cap,triton.cdiv(D,C))](ac,w2,off,te,ts,yp,N,D,F,TRANS=False,BF=bf,SPLIT=True); y=torch.empty_like(x); combine[(T,triton.cdiv(D,256))](yp,W,inv,y,T,D,A=A,B=256); ctx.save_for_backward(x,rw,w1,w2,S,sel,W,order,off,te,ts,inv,gu,ac,yp); ctx.sh=(T,D,E,F,A,cap,bf); return y
 @staticmethod
 def backward(ctx,dy):
  x,rw,w1,w2,S,sel,W,order,off,te,ts,inv,gu,ac,yp=ctx.saved_tensors; T,D,E,F,A,cap,bf=ctx.sh; N=T*A; dy=dy.contiguous(); DZ=torch.empty((N,D),device=x.device,dtype=torch.float32); dz[(N,triton.cdiv(D,256))](dy,W,order,DZ,N,D,A=A,B=256); cv=torch.empty(N,device=x.device,dtype=torch.float32); pdot[(N,)](dy,yp,order,cv,N,D,A=A,B=triton.next_power_of_2(D)); DA=torch.empty((N,F),device=x.device,dtype=torch.float32); seg[(cap,triton.cdiv(F,C))](DZ,w2,off,te,ts,DA,N,F,D,TRANS=True,BF=bf,SPLIT=True); DG=torch.empty((N,2*F),device=x.device,dtype=torch.float32); dactivation[(triton.cdiv(N*F,256),)](gu,DA,DG,N*F,F,B=256); DXP=torch.empty((N,D),device=x.device,dtype=torch.float32); seg[(cap,triton.cdiv(D,C))](DG,w1,off,te,ts,DXP,N,D,2*F,TRANS=True,BF=bf,SPLIT=True); gw2=torch.empty_like(w2); dw[(E,triton.cdiv(F,128),triton.cdiv(D,64))](ac,DZ,order,off,gw2,E,F,D,N,D,AUX=A,GATHER=False); gw1=torch.empty_like(w1); dw[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,DG,order,off,gw1,E,D,2*F,N,D,AUX=A,GATHER=True); DS=torch.empty((T,A),device=x.device,dtype=torch.float32); routeds[(T,)](S,sel,W,cv,DS,T,E,A=A,B=triton.next_power_of_2(A)); DE=torch.empty((T,E),device=x.device,dtype=torch.float32); dense[(triton.cdiv(T*E,256),)](sel,DS,DE,T,E,A=A,B=256); DXR=torch.empty((T,D),device=x.device,dtype=torch.float32); mm[(triton.cdiv(T,32),triton.cdiv(D,32))](DE,rw,DXR,T,D,E,TA=False,TB=True,IEEE=True); grw=torch.empty_like(rw); mm[(triton.cdiv(D,32),triton.cdiv(E,32))](x,DE,grw,D,E,T,TA=True,TB=False,IEEE=True); gx=torch.empty_like(x); addrows[(T,triton.cdiv(D,256))](DXP,DXR,inv,gx,T,D,A=A,B=256); return gx,grw,None,gw1,gw2,None
def fused_moe(x,router_w,router_b,w1,w2,A): return Moe.apply(x,router_w,router_b,w1,w2,A)
