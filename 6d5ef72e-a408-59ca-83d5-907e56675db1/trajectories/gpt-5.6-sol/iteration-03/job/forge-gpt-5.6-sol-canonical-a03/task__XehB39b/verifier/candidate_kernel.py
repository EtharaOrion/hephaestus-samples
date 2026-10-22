import torch
import triton
import triton.language as tl

@triton.jit
def _router(X,R,S,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); a=tl.load(X+m[:,None]*D+q[None,:],mask=(m[:,None]<T)&(q[None,:]<D),other=0.).to(tl.float32); b=tl.load(R+q[:,None]*E+n[None,:],mask=(q[:,None]<D)&(n[None,:]<E),other=0.).to(tl.float32); z+=tl.dot(a,b,input_precision="ieee")
 tl.store(S+m[:,None]*E+n[None,:],tl.sigmoid(z),mask=(m[:,None]<T)&(n[None,:]<E))

@triton.jit
def _meta(SEL,ORD,S,TOK,EXP,WS,INV,N:tl.constexpr,A:tl.constexpr,E:tl.constexpr,B:tl.constexpr):
 p=tl.program_id(0)*B+tl.arange(0,B); ok=p<N; q=tl.load(ORD+p,mask=ok,other=0).to(tl.int64); t=q//A; e=tl.load(SEL+q,mask=ok,other=0).to(tl.int64); den=tl.zeros((B,),tl.float32)
 for a in range(A):
  ea=tl.load(SEL+t*A+a,mask=ok,other=0).to(tl.int64); den+=tl.load(S+t*E+ea,mask=ok,other=0.)
 w=tl.load(S+t*E+e,mask=ok,other=0.)/den; tl.store(TOK+p,t,mask=ok); tl.store(EXP+p,e,mask=ok); tl.store(WS+p,w,mask=ok); tl.store(INV+q,p,mask=ok)

@triton.jit
def _offsets(EXP,OFF,N:tl.constexpr,E:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
 e=tl.program_id(0)*B+tl.arange(0,B); lo=tl.zeros((B,),tl.int32); hi=tl.full((B,),N,tl.int32)
 for _ in range(L):
  md=(lo+hi)//2; v=tl.load(EXP+md,mask=(e<=E)&(md<N),other=E); left=v<e; lo=tl.where(left,md+1,lo); hi=tl.where(left,hi,md)
 tl.store(OFF+e,lo,mask=e<=E)

@triton.jit
def _tile_init(TE,TS,N:tl.constexpr,B:tl.constexpr):
 p=tl.program_id(0)*B+tl.arange(0,B); tl.store(TE+p,-1,mask=p<N); tl.store(TS+p,0,mask=p<N)

@triton.jit
def _tiles(OFF,TE,TS,E:tl.constexpr,BM:tl.constexpr):
 out=0
 for e in range(E):
  lo=tl.load(OFF+e); hi=tl.load(OFF+e+1); nt=(hi-lo+BM-1)//BM
  for j in range(0,nt):
   tl.store(TE+out,e); tl.store(TS+out,lo+j*BM); out+=1

@triton.jit
def _f1(X,W,TOK,OFF,TE,TS,GU,D:tl.constexpr,F2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 tile=tl.program_id(0); e=tl.load(TE+tile); st=tl.load(TS+tile); end=tl.load(OFF+e+1,mask=e>=0,other=0); r=st+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); ok=r<end; tok=tl.load(TOK+r,mask=ok,other=0).to(tl.int64); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  q=k+tl.arange(0,BK); a=tl.load(X+tok[:,None]*D+q[None,:],mask=ok[:,None]&(q[None,:]<D),other=0.); b=tl.load(W+e*D*F2+q[:,None]*F2+n[None,:],mask=(e>=0)&(q[:,None]<D)&(n[None,:]<F2),other=0.)
  if BF: z+=tl.dot(a,b)
  else: z+=tl.dot(a.to(tl.float32),b.to(tl.float32),input_precision="tf32x3")
 tl.store(GU+r[:,None]*F2+n[None,:],z,mask=ok[:,None]&(n[None,:]<F2))

@triton.jit
def _f2(GU,W,OFF,TE,TS,FE,F:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 tile=tl.program_id(0); e=tl.load(TE+tile); st=tl.load(TS+tile); end=tl.load(OFF+e+1,mask=e>=0,other=0); r=st+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); ok=r<end; z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,F,BK):
  q=k+tl.arange(0,BK); g=tl.load(GU+r[:,None]*(2*F)+q[None,:],mask=ok[:,None]&(q[None,:]<F),other=0.); u=tl.load(GU+r[:,None]*(2*F)+F+q[None,:],mask=ok[:,None]&(q[None,:]<F),other=0.); a=g*tl.sigmoid(g)*u; b=tl.load(W+e*F*D+q[:,None]*D+n[None,:],mask=(e>=0)&(q[:,None]<F)&(n[None,:]<D),other=0.)
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah.to(tl.float32)).to(tl.bfloat16); z+=tl.dot(ah,b)+tl.dot(al,b)
  else: z+=tl.dot(a,b.to(tl.float32),input_precision="tf32x3")
 tl.store(FE+r[:,None]*D+n[None,:],z,mask=ok[:,None]&(n[None,:]<D))

@triton.jit
def _combine(FE,INV,WS,Y,T:tl.constexpr,D:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); z=tl.zeros((B,),tl.float32)
 for a in range(A):
  p=tl.load(INV+t*A+a).to(tl.int64); z+=tl.load(WS+p)*tl.load(FE+p*D+d,mask=d<D,other=0.)
 tl.store(Y+t*D+d,z,mask=d<D)

@triton.jit
def _droute(DY,FE,Y,INV,S,SEL,DL,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); a=tl.program_id(1); p=tl.load(INV+t*A+a).to(tl.int64); e=tl.load(SEL+t*A+a).to(tl.int64); z=tl.zeros((B,),tl.float32)
 for k in range(0,D,B):
  d=k+tl.arange(0,B); z+=tl.load(DY+t*D+d,mask=d<D,other=0.).to(tl.float32)*(tl.load(FE+p*D+d,mask=d<D,other=0.)-tl.load(Y+t*D+d,mask=d<D,other=0.))
 dot=tl.sum(z,axis=0); den=tl.zeros((),tl.float32)
 for j in range(A):
  ej=tl.load(SEL+t*A+j).to(tl.int64); den+=tl.load(S+t*E+ej)
 s=tl.load(S+t*E+e); tl.store(DL+t*A+a,dot/den*s*(1-s))

@triton.jit
def _b2(DY,GU,W,TOK,WS,OFF,TE,TS,DGU,F:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 tile=tl.program_id(0); e=tl.load(TE+tile); st=tl.load(TS+tile); end=tl.load(OFF+e+1,mask=e>=0,other=0); r=st+tl.arange(0,BM); f=tl.program_id(1)*BN+tl.arange(0,BN); ok=r<end; tok=tl.load(TOK+r,mask=ok,other=0).to(tl.int64); scale=tl.load(WS+r,mask=ok,other=0.); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,D,BK):
  d=k+tl.arange(0,BK); a=tl.load(DY+tok[:,None]*D+d[None,:],mask=ok[:,None]&(d[None,:]<D),other=0.).to(tl.float32)*scale[:,None]; b=tl.load(W+e*F*D+f[:,None]*D+d[None,:],mask=(e>=0)&(f[:,None]<F)&(d[None,:]<D),other=0.)
  if BF: z+=tl.dot(a.to(tl.bfloat16),tl.trans(b))
  else: z+=tl.dot(a,tl.trans(b.to(tl.float32)),input_precision="tf32x3")
 g=tl.load(GU+r[:,None]*(2*F)+f[None,:],mask=ok[:,None]&(f[None,:]<F),other=0.); u=tl.load(GU+r[:,None]*(2*F)+F+f[None,:],mask=ok[:,None]&(f[None,:]<F),other=0.); sig=tl.sigmoid(g); mask=ok[:,None]&(f[None,:]<F); tl.store(DGU+r[:,None]*(2*F)+f[None,:],z*u*(sig+g*sig*(1-sig)),mask=mask); tl.store(DGU+r[:,None]*(2*F)+F+f[None,:],z*g*sig,mask=mask)

@triton.jit
def _bx(DGU,W,OFF,TE,TS,DXE,D:tl.constexpr,F2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 tile=tl.program_id(0); e=tl.load(TE+tile); st=tl.load(TS+tile); end=tl.load(OFF+e+1,mask=e>=0,other=0); r=st+tl.arange(0,BM); d=tl.program_id(1)*BN+tl.arange(0,BN); ok=r<end; z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,F2,BK):
  q=k+tl.arange(0,BK); a=tl.load(DGU+r[:,None]*F2+q[None,:],mask=ok[:,None]&(q[None,:]<F2),other=0.); b=tl.load(W+e*D*F2+d[:,None]*F2+q[None,:],mask=(e>=0)&(d[:,None]<D)&(q[None,:]<F2),other=0.)
  if BF: z+=tl.dot(a.to(tl.bfloat16),tl.trans(b))
  else: z+=tl.dot(a,tl.trans(b.to(tl.float32)),input_precision="tf32x3")
 tl.store(DXE+r[:,None]*D+d[None,:],z,mask=ok[:,None]&(d[None,:]<D))

@triton.jit
def _dw2(DY,GU,TOK,WS,OFF,DW,F:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 e=tl.program_id(0); f=tl.program_id(1)*BM+tl.arange(0,BM); d=tl.program_id(2)*BN+tl.arange(0,BN); lo=tl.load(OFF+e); hi=tl.load(OFF+e+1); z=tl.zeros((BM,BN),tl.float32)
 for k in range(lo,hi,BK):
  r=k+tl.arange(0,BK); tok=tl.load(TOK+r,mask=r<hi,other=0).to(tl.int64); g=tl.load(GU+r[None,:]*(2*F)+f[:,None],mask=(r[None,:]<hi)&(f[:,None]<F),other=0.); u=tl.load(GU+r[None,:]*(2*F)+F+f[:,None],mask=(r[None,:]<hi)&(f[:,None]<F),other=0.); a=g*tl.sigmoid(g)*u; b=tl.load(DY+tok[:,None]*D+d[None,:],mask=(r[:,None]<hi)&(d[None,:]<D),other=0.).to(tl.float32)*tl.load(WS+r,mask=r<hi,other=0.)[:,None]
  if BF:
   ah=a.to(tl.bfloat16); al=(a-ah.to(tl.float32)).to(tl.bfloat16); bh=b.to(tl.bfloat16); bl=(b-bh.to(tl.float32)).to(tl.bfloat16); z+=tl.dot(ah,bh)+tl.dot(al,bh)+tl.dot(ah,bl)
  else: z+=tl.dot(a,b,input_precision="tf32x3")
 tl.store(DW+e*F*D+f[:,None]*D+d[None,:],z,mask=(f[:,None]<F)&(d[None,:]<D))

@triton.jit
def _dw1(X,DGU,TOK,OFF,DW,D:tl.constexpr,F2:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 e=tl.program_id(0); d=tl.program_id(1)*BM+tl.arange(0,BM); q=tl.program_id(2)*BN+tl.arange(0,BN); lo=tl.load(OFF+e); hi=tl.load(OFF+e+1); z=tl.zeros((BM,BN),tl.float32)
 for k in range(lo,hi,BK):
  r=k+tl.arange(0,BK); tok=tl.load(TOK+r,mask=r<hi,other=0).to(tl.int64); a=tl.load(X+tok[None,:]*D+d[:,None],mask=(r[None,:]<hi)&(d[:,None]<D),other=0.); b=tl.load(DGU+r[:,None]*F2+q[None,:],mask=(r[:,None]<hi)&(q[None,:]<F2),other=0.)
  if BF:
   bh=b.to(tl.bfloat16); bl=(b-bh.to(tl.float32)).to(tl.bfloat16); z+=tl.dot(a,bh)+tl.dot(a,bl)
  else: z+=tl.dot(a.to(tl.float32),b,input_precision="tf32x3")
 tl.store(DW+e*D*F2+d[:,None]*F2+q[None,:],z,mask=(d[:,None]<D)&(q[None,:]<F2))

@triton.jit
def _dx(DXE,RW,DL,SEL,INV,DX,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B); z=tl.zeros((B,),tl.float32)
 for a in range(A):
  p=tl.load(INV+t*A+a).to(tl.int64); e=tl.load(SEL+t*A+a).to(tl.int64); z+=tl.load(DXE+p*D+d,mask=d<D,other=0.)+tl.load(DL+t*A+a)*tl.load(RW+d*E+e,mask=d<D,other=0.).to(tl.float32)
 tl.store(DX+t*D+d,z,mask=d<D)

@triton.jit
def _drw(X,DL,SEL,DRW,T:tl.constexpr,D:tl.constexpr,E:tl.constexpr,A:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BF:tl.constexpr):
 d=tl.program_id(0)*BM+tl.arange(0,BM); n=tl.program_id(1)*BN+tl.arange(0,BN); z=tl.zeros((BM,BN),tl.float32)
 for k in range(0,T,BK):
  t=k+tl.arange(0,BK); x=tl.load(X+t[None,:]*D+d[:,None],mask=(t[None,:]<T)&(d[:,None]<D),other=0.); v=tl.zeros((BK,BN),tl.float32)
  for a in range(A):
   e=tl.load(SEL+t*A+a,mask=t<T,other=-1); dl=tl.load(DL+t*A+a,mask=t<T,other=0.); v+=tl.where(e[:,None]==n[None,:],dl[:,None],0.)
  z+=tl.dot(x.to(tl.float32),v,input_precision="ieee")
 tl.store(DRW+d[:,None]*E+n[None,:],z,mask=(d[:,None]<D)&(n[None,:]<E))

def _select(s,b,A):
 E=s.shape[1]; v=s+b; v=torch.where(v==0,torch.zeros_like(v),v).contiguous(); ib=v.view(torch.int32).to(torch.int64); u=torch.where(ib>=0,ib^0x80000000,(~ib)&0xffffffff); e=torch.arange(E,device=s.device,dtype=torch.int64); return torch.topk((u<<10)|(E-1-e),A,dim=1,largest=True,sorted=True).indices

def _forward(x,rw,rb,w1,w2,A):
 T,D=x.shape; E,_,F2=w1.shape; F=F2//2; N=T*A; bf=x.dtype==torch.bfloat16; s=torch.empty((T,E),device=x.device,dtype=torch.float32); _router[(triton.cdiv(T,32),triton.cdiv(E,32))](x,rw,s,T=T,D=D,E=E,BM=32,BN=32,BK=32,num_warps=4)
 with torch.no_grad(): sel=_select(s,rb,A); order=torch.argsort(sel.reshape(-1),stable=True)
 tok=torch.empty(N,device=x.device,dtype=torch.int32); exp=torch.empty_like(tok); ws=torch.empty(N,device=x.device,dtype=torch.float32); inv=torch.empty_like(tok); _meta[(triton.cdiv(N,256),)](sel,order,s,tok,exp,ws,inv,N=N,A=A,E=E,B=256,num_warps=4); off=torch.empty(E+1,device=x.device,dtype=torch.int32); _offsets[(1,)](exp,off,N=N,E=E,L=max(1,(N+1).bit_length()),B=128,num_warps=4)
 bm=64 if N<128*E else 128; mt=triton.cdiv(N,bm)+E; te=torch.empty(mt,device=x.device,dtype=torch.int32); ts=torch.empty_like(te); _tile_init[(triton.cdiv(mt,256),)](te,ts,N=mt,B=256); _tiles[(1,)](off,te,ts,E=E,BM=bm,num_warps=1)
 gu=torch.empty((N,F2),device=x.device,dtype=torch.float32); _f1[(mt,triton.cdiv(F2,64))](x,w1,tok,off,te,ts,gu,D=D,F2=F2,BM=bm,BN=64,BK=32,BF=bf,num_warps=4,num_stages=3); fe=torch.empty((N,D),device=x.device,dtype=torch.float32); _f2[(mt,triton.cdiv(D,128))](gu,w2,off,te,ts,fe,F=F,D=D,BM=bm,BN=128,BK=32,BF=bf,num_warps=8,num_stages=3); y=torch.empty((T,D),device=x.device,dtype=torch.float32); _combine[(T,triton.cdiv(D,256))](fe,inv,ws,y,T=T,D=D,A=A,B=256,num_warps=8); return y,(s,sel,tok,ws,inv,off,te,ts,gu,fe)

class _Moe(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,rw,rb,w1,w2,A):
  A=int(A); y,a=_forward(x,rw,rb,w1,w2,A); ctx.save_for_backward(x,rw,w1,w2,y,*a); ctx.A=A; return y.to(x.dtype)
 @staticmethod
 def backward(ctx,dy):
  x,rw,w1,w2,y,s,sel,tok,ws,inv,off,te,ts,gu,fe=ctx.saved_tensors; A=ctx.A; T,D=x.shape; E,_,F2=w1.shape; F=F2//2; N=T*A; bf=x.dtype==torch.bfloat16; bm=64 if N<128*E else 128; dy=dy.contiguous(); dl=torch.empty((T,A),device=x.device,dtype=torch.float32); _droute[(T,A)](dy,fe,y,inv,s,sel,dl,D=D,E=E,A=A,B=256,num_warps=8); mt=te.numel(); dgu=torch.empty_like(gu); _b2[(mt,triton.cdiv(F,64))](dy,gu,w2,tok,ws,off,te,ts,dgu,F=F,D=D,BM=bm,BN=64,BK=32,BF=bf,num_warps=4,num_stages=3); dxe=torch.empty((N,D),device=x.device,dtype=torch.float32); _bx[(mt,triton.cdiv(D,128))](dgu,w1,off,te,ts,dxe,D=D,F2=F2,BM=bm,BN=128,BK=32,BF=bf,num_warps=8,num_stages=3); dw2=torch.empty_like(w2); _dw2[(E,triton.cdiv(F,128),triton.cdiv(D,128))](dy,gu,tok,ws,off,dw2,F=F,D=D,BM=128,BN=128,BK=64,BF=bf,num_warps=8,num_stages=3); dw1=torch.empty_like(w1); _dw1[(E,triton.cdiv(D,128),triton.cdiv(F2,128))](x,dgu,tok,off,dw1,D=D,F2=F2,BM=128,BN=128,BK=64,BF=bf,num_warps=8,num_stages=3); dx=torch.empty_like(x); _dx[(T,triton.cdiv(D,256))](dxe,rw,dl,sel,inv,dx,D=D,E=E,A=A,B=256,num_warps=8); drw=torch.empty_like(rw); _drw[(triton.cdiv(D,32),triton.cdiv(E,32))](x,dl,sel,drw,T=T,D=D,E=E,A=A,BM=32,BN=32,BK=32,BF=bf,num_warps=4,num_stages=3); return dx,drw,None,dw1,dw2,None

def fused_moe(x,router_w,router_b,w1,w2,A): return _Moe.apply(x,router_w,router_b,w1,w2,A)
