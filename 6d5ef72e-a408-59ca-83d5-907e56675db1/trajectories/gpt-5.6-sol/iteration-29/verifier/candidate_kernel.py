import torch
import triton
import triton.language as tl

R, C, K = 64, 64, 32

@triton.jit
def _router(X,W,S,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=32,BK:tl.constexpr=32):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); a=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); x=tl.load(X+m[:,None]*D+q[None,:],mask=(m[:,None]<T)&(q[None,:]<D),other=0.); w=tl.load(W+q[:,None]*E+n[None,:],mask=(q[:,None]<D)&(n[None,:]<E),other=0.)
  a+=tl.dot(x.to(tl.float32),w.to(tl.float32),input_precision="ieee")
 tl.store(S+m[:,None]*E+n[None,:],tl.sigmoid(a),mask=(m[:,None]<T)&(n[None,:]<E))

@triton.jit
def _keys(S,B,O,N:tl.constexpr,E:tl.constexpr,BS:tl.constexpr=256):
 p=tl.program_id(0)*BS+tl.arange(0,BS); z=p<N; e=p%E; v=tl.load(S+p,mask=z,other=0.)+tl.load(B+e,mask=z,other=0.); v=tl.where(v==0.,0.,v); i=v.to(tl.int32,bitcast=True).to(tl.int64); u=tl.where(i>=0,i^0x80000000,(~i)&0xFFFFFFFF); tl.store(O+p,(u<<10)|(E-1-e),mask=z)

@triton.jit
def _route(S,SEL,W,SK,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BS:tl.constexpr=256):
 t=tl.program_id(0)*BS+tl.arange(0,BS); z=t<T; den=tl.zeros((BS,),tl.float32)
 for a in range(A):
  e=tl.load(SEL+t*A+a,mask=z,other=0); den+=tl.load(S+t*E+e,mask=z,other=0.)
 for a in range(A):
  p=t*A+a; e=tl.load(SEL+p,mask=z,other=0); tl.store(W+p,tl.load(S+t*E+e,mask=z,other=0.)/den,mask=z); tl.store(SK+p,e.to(tl.int64)*(T*A)+p,mask=z)

@triton.jit
def _inv(O,I,N:tl.constexpr,BS:tl.constexpr=256):
 p=tl.program_id(0)*BS+tl.arange(0,BS); z=p<N; q=tl.load(O+p,mask=z,other=0); tl.store(I+q,p,mask=z)

@triton.jit
def _offs(O,S,F,N:tl.constexpr,E:tl.constexpr):
 e=tl.program_id(0); lo=0; hi=N
 while lo<hi:
  m=(lo+hi)//2; q=tl.load(O+m); v=tl.load(S+q); go=v<e; lo=tl.where(go,m+1,lo); hi=tl.where(go,hi,m)
 tl.store(F+e,lo)
 if e==E-1: tl.store(F+E,N)

@triton.jit
def _tiles(F,TE,TS,CAP:tl.constexpr,E:tl.constexpr,RB:tl.constexpr):
 q=tl.program_id(0); rem=q; e=0; found=False; st=0
 for j in range(E):
  lo=tl.load(F+j); hi=tl.load(F+j+1); n=tl.cdiv(hi-lo,RB); take=(~found)&(rem<n); e=tl.where(take,j,e); st=tl.where(take,lo+rem*RB,st); found|=take; rem=tl.where(found,rem,rem-n)
 tl.store(TE+q,tl.where(found,e,-1)); tl.store(TS+q,st)

@triton.jit
def _ein(X,W,O,TE,TS,OFF,Y,N:tl.constexpr,D:tl.constexpr,P:tl.constexpr,A:tl.constexpr,BF:tl.constexpr,RB:tl.constexpr,CB:tl.constexpr,KB:tl.constexpr):
 q,nc=tl.program_id(0),tl.program_id(1); e=tl.load(TE+q); st=tl.load(TS+q); r=tl.arange(0,RB); c=nc*CB+tl.arange(0,CB); pos=st+r; z=(e>=0)&(pos<N)&(pos<tl.load(OFF+tl.maximum(e,0)+1)); pair=tl.load(O+pos,mask=z,other=0); tok=pair//A; acc=tl.zeros((RB,CB),tl.float32)
 for k in range(0,D,KB):
  d=k+tl.arange(0,KB); x=tl.load(X+tok[:,None]*D+d[None,:],mask=z[:,None]&(d[None,:]<D),other=0.); w=tl.load(W+e*D*P+d[:,None]*P+c[None,:],mask=(e>=0)&(d[:,None]<D)&(c[None,:]<P),other=0.)
  if BF: acc+=tl.dot(x,w)
  else: acc+=tl.dot(x.to(tl.float32),w.to(tl.float32),input_precision="tf32x3")
 tl.store(Y+pos[:,None]*P+c[None,:],acc,mask=z[:,None]&(c[None,:]<P))

@triton.jit
def _eout(G,W,TE,TS,OFF,Y,N:tl.constexpr,F:tl.constexpr,D:tl.constexpr,BF:tl.constexpr,RB:tl.constexpr,CB:tl.constexpr,KB:tl.constexpr):
 q,nc=tl.program_id(0),tl.program_id(1); e=tl.load(TE+q); st=tl.load(TS+q); r=tl.arange(0,RB); d=nc*CB+tl.arange(0,CB); pos=st+r; z=(e>=0)&(pos<N)&(pos<tl.load(OFF+tl.maximum(e,0)+1)); acc=tl.zeros((RB,CB),tl.float32)
 for k in range(0,F,KB):
  f=k+tl.arange(0,KB); g=tl.load(G+pos[:,None]*(2*F)+f[None,:],mask=z[:,None]&(f[None,:]<F),other=0.); u=tl.load(G+pos[:,None]*(2*F)+F+f[None,:],mask=z[:,None]&(f[None,:]<F),other=0.); a=g*tl.sigmoid(g)*u; w=tl.load(W+e*F*D+f[:,None]*D+d[None,:],mask=(e>=0)&(f[:,None]<F)&(d[None,:]<D),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); acc+=tl.dot(ah,w)+tl.dot(al,w)
  else: acc+=tl.dot(a,w.to(tl.float32),input_precision="tf32x3")
 tl.store(Y+pos[:,None]*D+d[None,:],acc,mask=z[:,None]&(d[None,:]<D))

@triton.jit
def _combine(Z,W,I,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BS:tl.constexpr=256):
 t=tl.program_id(0); d=tl.program_id(1)*BS+tl.arange(0,BS); acc=tl.zeros((BS,),tl.float32)
 for a in range(A):
  p=t*A+a; q=tl.load(I+p); acc+=tl.load(W+p)*tl.load(Z+q*D+d,mask=d<D,other=0.)
 tl.store(Y+t*D+d,acc,mask=d<D)

@triton.jit
def _dz(DY,Z,W,O,DZ,Q,N:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BS:tl.constexpr=256):
 p=tl.program_id(0); pair=tl.load(O+p); t=pair//A; d=tl.arange(0,BS); q=0.
 for k in range(0,D,BS):
  x=k+d; dy=tl.load(DY+t*D+x,mask=x<D,other=0.); z=tl.load(Z+p*D+x,mask=x<D,other=0.); tl.store(DZ+p*D+x,dy*tl.load(W+pair),mask=x<D); q+=tl.sum(dy*z)
 tl.store(Q+pair,q)

@triton.jit
def _dgu(DZ,W,G,TE,TS,OFF,O,N:tl.constexpr,F:tl.constexpr,D:tl.constexpr,BF:tl.constexpr,RB:tl.constexpr,CB:tl.constexpr,KB:tl.constexpr):
 q,nc=tl.program_id(0),tl.program_id(1); e=tl.load(TE+q); st=tl.load(TS+q); r=tl.arange(0,RB); f=nc*CB+tl.arange(0,CB); p=st+r; z=(e>=0)&(p<N)&(p<tl.load(OFF+tl.maximum(e,0)+1)); acc=tl.zeros((RB,CB),tl.float32)
 for k in range(0,D,KB):
  d=k+tl.arange(0,KB); a=tl.load(DZ+p[:,None]*D+d[None,:],mask=z[:,None]&(d[None,:]<D),other=0.); w=tl.load(W+e*F*D+f[:,None]*D+d[None,:],mask=(e>=0)&(f[:,None]<F)&(d[None,:]<D),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); acc+=tl.dot(ah,tl.trans(w))+tl.dot(al,tl.trans(w))
  else: acc+=tl.dot(a,tl.trans(w.to(tl.float32)),input_precision="tf32x3")
 g=tl.load(G+p[:,None]*(2*F)+f[None,:],mask=z[:,None]&(f[None,:]<F),other=0.); u=tl.load(G+p[:,None]*(2*F)+F+f[None,:],mask=z[:,None]&(f[None,:]<F),other=0.); s=tl.sigmoid(g); si=g*s; m=z[:,None]&(f[None,:]<F); tl.store(O+p[:,None]*(2*F)+f[None,:],acc*u*(s+si*(1-s)),mask=m); tl.store(O+p[:,None]*(2*F)+F+f[None,:],acc*si,mask=m)

@triton.jit
def _dxr(G,W,TE,TS,OFF,O,N:tl.constexpr,D:tl.constexpr,P:tl.constexpr,BF:tl.constexpr,RB:tl.constexpr,CB:tl.constexpr,KB:tl.constexpr):
 q,nc=tl.program_id(0),tl.program_id(1); e=tl.load(TE+q); st=tl.load(TS+q); r=tl.arange(0,RB); d=nc*CB+tl.arange(0,CB); p=st+r; z=(e>=0)&(p<N)&(p<tl.load(OFF+tl.maximum(e,0)+1)); acc=tl.zeros((RB,CB),tl.float32)
 for k in range(0,P,KB):
  c=k+tl.arange(0,KB); a=tl.load(G+p[:,None]*P+c[None,:],mask=z[:,None]&(c[None,:]<P),other=0.); w=tl.load(W+e*D*P+d[:,None]*P+c[None,:],mask=(e>=0)&(d[:,None]<D)&(c[None,:]<P),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); acc+=tl.dot(ah,tl.trans(w))+tl.dot(al,tl.trans(w))
  else: acc+=tl.dot(a,tl.trans(w.to(tl.float32)),input_precision="tf32x3")
 tl.store(O+p[:,None]*D+d[None,:],acc,mask=z[:,None]&(d[None,:]<D))

@triton.jit
def _dw1(X,G,O,OFF,OUT,D:tl.constexpr,P:tl.constexpr,A:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=128,BN:tl.constexpr=64,BK:tl.constexpr=32):
 e,mi,ni=tl.program_id(0),tl.program_id(1),tl.program_id(2); d=mi*BM+tl.arange(0,BM); c=ni*BN+tl.arange(0,BN); lo=tl.load(OFF+e); hi=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(lo,hi,BK):
  p=k+tl.arange(0,BK); pair=tl.load(O+p,mask=p<hi,other=0); t=pair//A; x=tl.load(X+d[:,None]+t[None,:]*D,mask=(d[:,None]<D)&(p[None,:]<hi),other=0.); g=tl.load(G+p[:,None]*P+c[None,:],mask=(p[:,None]<hi)&(c[None,:]<P),other=0.)
  if BF:
   gh=g.to(tl.bfloat16); gl=(g-gh).to(tl.bfloat16); acc+=tl.dot(x,gh)+tl.dot(x,gl)
  else: acc+=tl.dot(x.to(tl.float32),g,input_precision="ieee")
 tl.store(OUT+(e*D*P+d[:,None]*P+c[None,:]),acc,mask=(d[:,None]<D)&(c[None,:]<P))

@triton.jit
def _dw2(G,Z,OFF,OUT,F:tl.constexpr,D:tl.constexpr,BF:tl.constexpr,BM:tl.constexpr=128,BN:tl.constexpr=64,BK:tl.constexpr=32):
 e,mi,ni=tl.program_id(0),tl.program_id(1),tl.program_id(2); f=mi*BM+tl.arange(0,BM); d=ni*BN+tl.arange(0,BN); lo=tl.load(OFF+e); hi=tl.load(OFF+e+1); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(lo,hi,BK):
  p=k+tl.arange(0,BK); g=tl.load(G+p[None,:]*(2*F)+f[:,None],mask=(p[None,:]<hi)&(f[:,None]<F),other=0.); u=tl.load(G+p[None,:]*(2*F)+F+f[:,None],mask=(p[None,:]<hi)&(f[:,None]<F),other=0.); a=g*tl.sigmoid(g)*u; z=tl.load(Z+p[:,None]*D+d[None,:],mask=(p[:,None]<hi)&(d[None,:]<D),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah).to(tl.bfloat16); zh=z.to(tl.bfloat16); zl=(z-zh).to(tl.bfloat16); acc+=tl.dot(ah,zh)+tl.dot(al,zh)+tl.dot(ah,zl)
  else: acc+=tl.dot(a,z,input_precision="ieee")
 tl.store(OUT+e*F*D+f[:,None]*D+d[None,:],acc,mask=(f[:,None]<F)&(d[None,:]<D))

@triton.jit
def _rgrad(S,SEL,W,Q,L,T:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BS:tl.constexpr=256):
 t=tl.program_id(0)*BS+tl.arange(0,BS); z=t<T; den=tl.zeros((BS,),tl.float32); av=tl.zeros((BS,),tl.float32)
 for e0 in range(E): tl.store(L+t*E+e0,0.,mask=z)
 for a in range(A):
  q=t*A+a; e=tl.load(SEL+q,mask=z,other=0); den+=tl.load(S+t*E+e,mask=z,other=0.); av+=tl.load(W+q,mask=z,other=0.)*tl.load(Q+q,mask=z,other=0.)
 for a in range(A):
  q=t*A+a; e=tl.load(SEL+q,mask=z,other=0); s=tl.load(S+t*E+e,mask=z,other=0.); tl.store(L+t*E+e,(tl.load(Q+q,mask=z,other=0.)-av)/den*s*(1-s),mask=z)

@triton.jit
def _mm(A,B,C,M:tl.constexpr,N:tl.constexpr,K0:tl.constexpr,TA:tl.constexpr,TB:tl.constexpr,BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); acc=tl.zeros((BM,BN),tl.float32)
 for k in range(0,K0,BK):
  q=k+tl.arange(0,BK)
  if TA: a=tl.trans(tl.load(A+q[:,None]*M+m[None,:],mask=(q[:,None]<K0)&(m[None,:]<M),other=0.))
  else: a=tl.load(A+m[:,None]*K0+q[None,:],mask=(m[:,None]<M)&(q[None,:]<K0),other=0.)
  if TB: b=tl.trans(tl.load(B+n[:,None]*K0+q[None,:],mask=(n[:,None]<N)&(q[None,:]<K0),other=0.))
  else: b=tl.load(B+q[:,None]*N+n[None,:],mask=(q[:,None]<K0)&(n[None,:]<N),other=0.)
  acc+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="ieee")
 tl.store(C+m[:,None]*N+n[None,:],acc,mask=(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _sumdx(RX,I,RT,X,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BS:tl.constexpr=256):
 t=tl.program_id(0); d=tl.program_id(1)*BS+tl.arange(0,BS); v=tl.load(RT+t*D+d,mask=d<D,other=0.)
 for a in range(A): v+=tl.load(RX+tl.load(I+t*A+a)*D+d,mask=d<D,other=0.)
 tl.store(X+t*D+d,v,mask=d<D)

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
  T,D=x.shape; E=rw.shape[1]; F=w2.shape[1]; A=int(A); N=T*A; bf=x.dtype==torch.bfloat16; s=torch.empty((T,E),device=x.device); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E); k=torch.empty((T,E),device=x.device,dtype=torch.int64); _keys[(triton.cdiv(T*E,256),)](s,rb,k,N=T*E,E=E); sel=torch.topk(k,A,dim=1).indices; w=torch.empty(N,device=x.device); sk=torch.empty(N,device=x.device,dtype=torch.int64); _route[(triton.cdiv(T,256),)](s,sel,w,sk,T=T,E=E,A=A); order=torch.argsort(sk); inv=torch.empty_like(order); _inv[(triton.cdiv(N,256),)](order,inv,N=N); off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offs[(E,)](order,sel,off,N=N,E=E,num_warps=1); cap=triton.cdiv(N,R)+E; te=torch.empty(cap,device=x.device,dtype=torch.int32); ts=torch.empty_like(te); _tiles[(cap,)](off,te,ts,CAP=cap,E=E,RB=R,num_warps=1); gu=torch.empty((N,2*F),device=x.device); _ein[(cap,triton.cdiv(2*F,C))](x,w1,order,te,ts,off,gu,N=N,D=D,P=2*F,A=A,BF=bf,RB=R,CB=C,KB=K,num_warps=4,num_stages=3); z=torch.empty((N,D),device=x.device); _eout[(cap,triton.cdiv(D,C))](gu,w2,te,ts,off,z,N=N,F=F,D=D,BF=bf,RB=R,CB=C,KB=K,num_warps=4,num_stages=3); y=torch.empty((T,D),device=x.device); _combine[(T,triton.cdiv(D,256))](z,w,inv,y,T=T,D=D,A=A); ctx.save_for_backward(x,rw,w1,w2,s,sel,w,order,inv,off,te,ts,gu,z); ctx.m=(T,D,E,F,A,N,cap,bf,x.dtype,rw.dtype,w1.dtype,w2.dtype); return y.to(x.dtype)
 @staticmethod
 def backward(ctx,dy):
  x,rw,w1,w2,s,sel,w,order,inv,off,te,ts,gu,z=ctx.saved_tensors; T,D,E,F,A,N,cap,bf,xd,rd,a1,a2=ctx.m; dy=dy.float().contiguous(); dz=torch.empty((N,D),device=x.device); q=torch.empty(N,device=x.device); _dz[(N,)](dy,z,w,order,dz,q,N=N,D=D,A=A); dg=torch.empty((N,2*F),device=x.device); _dgu[(cap,triton.cdiv(F,C))](dz,w2,gu,te,ts,off,dg,N=N,F=F,D=D,BF=bf,RB=R,CB=C,KB=K,num_warps=4,num_stages=3); rx=torch.empty((N,D),device=x.device); _dxr[(cap,triton.cdiv(D,C))](dg,w1,te,ts,off,rx,N=N,D=D,P=2*F,BF=bf,RB=R,CB=C,KB=K,num_warps=4,num_stages=3); g1=torch.empty_like(w1,dtype=torch.float32); _dw1[(E,triton.cdiv(D,128),triton.cdiv(2*F,64))](x,dg,order,off,g1,D=D,P=2*F,A=A,BF=bf,num_warps=4,num_stages=3); g2=torch.empty_like(w2,dtype=torch.float32); _dw2[(E,triton.cdiv(F,128),triton.cdiv(D,64))](gu,dz,off,g2,F=F,D=D,BF=bf,num_warps=4,num_stages=3); dl=torch.empty((T,E),device=x.device); _rgrad[(max(triton.cdiv(T*E,256),triton.cdiv(T,256)),)](s,sel,w,q,dl,T=T,E=E,A=A); gr=torch.empty((D,E),device=x.device); _mm[(triton.cdiv(D,64),triton.cdiv(E,64))](x,dl,gr,M=D,N=E,K0=T,TA=True,TB=False); rt=torch.empty((T,D),device=x.device); _mm[(triton.cdiv(T,64),triton.cdiv(D,64))](dl,rw,rt,M=T,N=D,K0=E,TA=False,TB=True); gx=torch.empty((T,D),device=x.device); _sumdx[(T,triton.cdiv(D,256))](rx,inv,rt,gx,T=T,D=D,A=A); return gx.to(xd),gr.to(rd),None,g1.to(a1),g2.to(a2),None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
