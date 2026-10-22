import torch
import triton
import triton.language as tl

_BT = 16
_BV = 16

@triton.jit
def _inv_lower(x):
    ii = tl.arange(0, 16)
    r = tl.where(ii[:, None] == ii[None, :], 1.0, 0.0)
    p = x
    for _ in tl.static_range(0, 4):
        r += tl.dot(p, r, input_precision="ieee")
        p = tl.dot(p, p, input_precision="ieee")
    return r

@triton.jit
def _data(qp, kp, ap, bp, gp, base, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, valid):
    it, ik = tl.arange(0, 16), tl.arange(0, BK)
    off = base + it[:, None] * H * K + ik[None, :]
    mask = valid[:, None] & (ik[None, :] < K)
    g = tl.load(gp + off, mask=mask, other=0.).to(tl.float32)
    c = tl.cumsum(g, axis=0)
    e = tl.sum(tl.where((it[:, None] == 15) & valid[:, None], c, 0.0), axis=0)
    r = e * 0.5
    q = tl.load(qp + off, mask=mask, other=0.).to(tl.float32) * tl.exp(c-r[None, :]) * (K ** -0.5)
    k = tl.load(kp + off, mask=mask, other=0.).to(tl.float32) * tl.exp(r[None, :]-c)
    a = tl.load(ap + off, mask=mask, other=0.).to(tl.float32) * tl.exp(c-g-r[None, :])
    b = tl.load(bp + off, mask=mask, other=0.).to(tl.float32) * tl.exp(r[None, :]-c)
    lower = it[:, None] > it[None, :]
    causal = it[:, None] >= it[None, :]
    lab = tl.where(lower, tl.dot(a, tl.trans(b), input_precision="ieee"), 0.)
    pak = tl.where(lower, tl.dot(a, tl.trans(k), input_precision="ieee"), 0.)
    nqk = tl.where(causal, tl.dot(q, tl.trans(k), input_precision="ieee"), 0.)
    nqb = tl.where(causal, tl.dot(q, tl.trans(b), input_precision="ieee"), 0.)
    return q, k, a, b, g, c, e, _inv_lower(lab), pak, nqk, nqb

@triton.jit
def _fwd(qp, kp, vp, ap, bp, gp, op, hp, T: tl.constexpr, H: tl.constexpr,
         K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    b0, h0 = bh // H, bh % H
    ik, ivv = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    s = tl.zeros((BK, BV), tl.float32)
    for ch in range(NC):
        it = tl.arange(0, 16); ts = ch*16 + it; valid = ts < T
        base = ((b0*T + ch*16)*H + h0)*K
        ho = (((b0*NC + ch)*H + h0)*K + ik[:, None])*V + ivv[None, :]
        tl.store(hp+ho, s, mask=(ik[:, None] < K) & (ivv[None, :] < V))
        q,k,a,bb,g,c,e,m,pak,nqk,nqb = _data(qp,kp,ap,bp,gp,base,H,K,BK,valid)
        vo = ((b0*T + ts[:, None])*H + h0)*V + ivv[None, :]
        vm = valid[:, None] & (ivv[None, :] < V)
        v = tl.load(vp+vo, mask=vm, other=0.).to(tl.float32)
        se = s * tl.exp(e*0.5)[:, None]
        r = tl.dot(m, tl.dot(pak,v,input_precision="ieee") + tl.dot(a,se,input_precision="ieee"), input_precision="ieee")
        o = tl.dot(q,se,input_precision="ieee") + tl.dot(nqk,v,input_precision="ieee") + tl.dot(nqb,r,input_precision="ieee")
        tl.store(op+vo, o, mask=vm)
        s = (se + tl.dot(tl.trans(k),v,input_precision="ieee") + tl.dot(tl.trans(bb),r,input_precision="ieee")) * tl.exp(e*0.5)[:,None]

@triton.jit
def _bounds(qp,kp,vp,ap,bp,gp,dop,hp,dsp,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,
            V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); b0,h0=bh//H,bh%H
    ik,ivv=tl.arange(0,BK),iv*BV+tl.arange(0,BV); ds=tl.zeros((BK,BV),tl.float32)
    for rr in range(NC):
        ch=NC-1-rr; it=tl.arange(0,16); ts=ch*16+it; valid=ts<T
        base=((b0*T+ch*16)*H+h0)*K
        so=(((b0*NC+ch)*H+h0)*K+ik[:,None])*V+ivv[None,:]
        sm=(ik[:,None]<K)&(ivv[None,:]<V); s=tl.load(hp+so,mask=sm,other=0.)
        tl.store(dsp+so,ds,mask=sm)
        q,k,a,bb,g,c,e,m,pak,nqk,nqb=_data(qp,kp,ap,bp,gp,base,H,K,BK,valid)
        vo=((b0*T+ts[:,None])*H+h0)*V+ivv[None,:]; vm=valid[:,None]&(ivv[None,:]<V)
        v=tl.load(vp+vo,mask=vm,other=0.).to(tl.float32); do=tl.load(dop+vo,mask=vm,other=0.).to(tl.float32)
        se=s*tl.exp(e*0.5)[:,None]
        r=tl.dot(m,tl.dot(pak,v,input_precision="ieee")+tl.dot(a,se,input_precision="ieee"),input_precision="ieee")
        dz=ds*tl.exp(e*0.5)[:,None]
        dr=tl.dot(tl.trans(nqb),do,input_precision="ieee")+tl.dot(bb,dz,input_precision="ieee")
        dc=tl.dot(tl.trans(m),dr,input_precision="ieee")
        dse=dz+tl.dot(tl.trans(q),do,input_precision="ieee")+tl.dot(tl.trans(a),dc,input_precision="ieee")
        ds=dse*tl.exp(e*0.5)[:,None]

@triton.jit
def _bwd(qp,kp,vp,ap,bp,gp,dop,hp,dsp,dvp,pqp,pkp,pap,pbp,pgp,T:tl.constexpr,H:tl.constexpr,
         K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,ch,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2); b0,h0=bh//H,bh%H
    it,ik,ivv=tl.arange(0,16),tl.arange(0,BK),iv*BV+tl.arange(0,BV)
    ts=ch*16+it; valid=ts<T; base=((b0*T+ch*16)*H+h0)*K
    so=(((b0*NC+ch)*H+h0)*K+ik[:,None])*V+ivv[None,:]; sm=(ik[:,None]<K)&(ivv[None,:]<V)
    s=tl.load(hp+so,mask=sm,other=0.); ds1=tl.load(dsp+so,mask=sm,other=0.)
    q,k,a,bb,g,c,e,m,pak,nqk,nqb=_data(qp,kp,ap,bp,gp,base,H,K,BK,valid)
    vo=((b0*T+ts[:,None])*H+h0)*V+ivv[None,:]; vm=valid[:,None]&(ivv[None,:]<V)
    v=tl.load(vp+vo,mask=vm,other=0.).to(tl.float32); do=tl.load(dop+vo,mask=vm,other=0.).to(tl.float32)
    se=s*tl.exp(e*0.5)[:,None]
    r=tl.dot(m,tl.dot(pak,v,input_precision="ieee")+tl.dot(a,se,input_precision="ieee"),input_precision="ieee")
    z=se+tl.dot(tl.trans(k),v,input_precision="ieee")+tl.dot(tl.trans(bb),r,input_precision="ieee")
    dz=ds1*tl.exp(e*0.5)[:,None]
    causal=it[:,None]>=it[None,:]; lower=it[:,None]>it[None,:]
    dnqk=tl.where(causal,tl.dot(do,tl.trans(v),input_precision="ieee"),0.)
    dnqb=tl.where(causal,tl.dot(do,tl.trans(r),input_precision="ieee"),0.)
    dq=tl.dot(do,tl.trans(se),input_precision="ieee")+tl.dot(dnqk,k,input_precision="ieee")+tl.dot(dnqb,bb,input_precision="ieee")
    dk=tl.dot(tl.trans(dnqk),q,input_precision="ieee")+tl.dot(v,tl.trans(dz),input_precision="ieee")
    db=tl.dot(tl.trans(dnqb),q,input_precision="ieee")+tl.dot(r,tl.trans(dz),input_precision="ieee")
    dv=tl.dot(tl.trans(nqk),do,input_precision="ieee")+tl.dot(k,dz,input_precision="ieee")
    dr=tl.dot(tl.trans(nqb),do,input_precision="ieee")+tl.dot(bb,dz,input_precision="ieee")
    dc=tl.dot(tl.trans(m),dr,input_precision="ieee")
    dl=tl.where(lower,tl.dot(dc,tl.trans(r),input_precision="ieee"),0.)
    dp=tl.where(lower,tl.dot(dc,tl.trans(v),input_precision="ieee"),0.)
    da=tl.dot(dc,tl.trans(se),input_precision="ieee")+tl.dot(dp,k,input_precision="ieee")+tl.dot(dl,bb,input_precision="ieee")
    dk+=tl.dot(tl.trans(dp),a,input_precision="ieee"); db+=tl.dot(tl.trans(dl),a,input_precision="ieee")
    dv+=tl.dot(tl.trans(pak),dc,input_precision="ieee")
    dse=dz+tl.dot(tl.trans(q),do,input_precision="ieee")+tl.dot(tl.trans(a),dc,input_precision="ieee")
    direct=dq*q-dk*k+da*a-db*bb
    prefix=tl.cumsum(direct,axis=0)-direct
    edge=0.5*(tl.sum(direct,axis=0)+tl.sum(dse*se,axis=1)+tl.sum(ds1*z,axis=1)*tl.exp(e*0.5))
    dg=0.5*tl.sum(direct,axis=0)[None,:]-prefix-da*a+0.5*(tl.sum(dse*se,axis=1)+tl.sum(ds1*z,axis=1)*tl.exp(e*0.5))[None,:]
    rr=e*0.5
    dq*=tl.exp(c-rr[None,:])*(K**-0.5); dk*=tl.exp(rr[None,:]-c)
    da*=tl.exp(c-g-rr[None,:]); db*=tl.exp(rr[None,:]-c)
    dg=tl.where(ts[:,None]==0,0.0,dg)
    go=iv*(T*tl.num_programs(2)*K)+((b0*T+ts[:,None])*H+h0)*K+ik[None,:]; gm=valid[:,None]&(ik[None,:]<K)
    tl.store(pqp+go,dq,mask=gm); tl.store(pkp+go,dk,mask=gm); tl.store(pap+go,da,mask=gm)
    tl.store(pbp+go,db,mask=gm); tl.store(pgp+go,dg,mask=gm); tl.store(dvp+vo,dv,mask=vm)

@triton.jit
def _reduce(pq,pk,pa,pb,pg,dq,dk,da,db,dg,N:tl.constexpr,NV:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256); mask=i<N
    xq=tl.zeros((256,),tl.float32); xk=tl.zeros((256,),tl.float32); xa=tl.zeros((256,),tl.float32)
    xb=tl.zeros((256,),tl.float32); xg=tl.zeros((256,),tl.float32)
    for j in range(NV):
        o=j*N+i; xq+=tl.load(pq+o,mask=mask,other=0.); xk+=tl.load(pk+o,mask=mask,other=0.)
        xa+=tl.load(pa+o,mask=mask,other=0.); xb+=tl.load(pb+o,mask=mask,other=0.); xg+=tl.load(pg+o,mask=mask,other=0.)
    tl.store(dq+i,xq,mask=mask); tl.store(dk+i,xk,mask=mask); tl.store(da+i,xa,mask=mask)
    tl.store(db+i,xb,mask=mask); tl.store(dg+i,xg,mask=mask)

class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,a,b,g):
        B,T,H,K=q.shape; V=v.shape[-1]; nc=triton.cdiv(T,16); bv=32; nv=triton.cdiv(V,bv); bk=max(16,triton.next_power_of_2(K))
        o=torch.empty_like(v); hs=torch.empty((B,nc,H,K,V),device=q.device,dtype=torch.float32)
        _fwd[(nv,B*H)](q,k,v,a,b,g,o,hs,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=4,num_stages=2)
        _exact_h[(nv,B*H)](k,v,a,b,g,hs,T=T,H=H,K=K,V=V,NC=nc,BK=bk,BV=bv,num_warps=4,num_stages=2)
        ctx.save_for_backward(q,k,v,a,b,g,hs); return o
    @staticmethod
    def backward(ctx,do):
        q,k,v,a,b,g,hs=ctx.saved_tensors; do=do.contiguous(); B,T,H,K=q.shape; V=v.shape[-1]
        nc=triton.cdiv(T,16); bv=8; nv=triton.cdiv(V,bv); bk=max(16,triton.next_power_of_2(K))
        n=B*T*H*K; ps=[torch.zeros((nv,n),device=q.device,dtype=torch.float32) for _ in range(5)]; dv=torch.empty_like(v)
        _exact_bwd[(nv,B*H)](q,k,v,a,b,g,do,hs,dv,*ps,T=T,H=H,K=K,V=V,NC=nc,NV=nv,BK=bk,BV=bv,num_warps=4,num_stages=2)
        gs=[torch.empty_like(q) for _ in range(5)]; _reduce[(triton.cdiv(n,256),)](*ps,*gs,N=n,NV=nv,num_warps=4)
        return gs[0],gs[1],dv,gs[2],gs[3],gs[4]

def chunked_dplr_delta(q,k,v,alpha,beta,gk):
    return _Dplr.apply(q,k,v,alpha,beta,gk)

@triton.jit
def _exact_bwd(qp,kp,vp,ap,bp,gp,dop,hp,dvp,pqp,pkp,pap,pbp,pgp,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,NV:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); b0,h0=bh//H,bh%H; ik=tl.arange(0,BK); ivv=iv*BV+tl.arange(0,BV); ds=tl.zeros((BK,BV),tl.float32)
    for rr in range(NC):
      ch=NC-1-rr
      for rj in tl.static_range(0,16):
        j=15-rj; t=ch*16+j; active=t<T; so=(((b0*NC+ch)*H+h0)*K+ik[:,None])*V+ivv[None,:]
        sold=tl.load(hp+so,mask=(ik[:,None]<K)&(ivv[None,:]<V),other=0.)
        for p in tl.static_range(0,16):
          use=(p<j)&(ch*16+p<T); xo=((b0*T+ch*16+p)*H+h0)*K+ik; vo=((b0*T+ch*16+p)*H+h0)*V+ivv
          gg=tl.load(gp+xo,mask=use&(ik<K),other=0.).to(tl.float32); aa=tl.load(ap+xo,mask=use&(ik<K),other=0.).to(tl.float32); bb=tl.load(bp+xo,mask=use&(ik<K),other=0.).to(tl.float32); kk=tl.load(kp+xo,mask=use&(ik<K),other=0.).to(tl.float32); vv=tl.load(vp+vo,mask=use&(ivv<V),other=0.).to(tl.float32)
          read=tl.sum(sold.to(tl.float64)*aa[:,None].to(tl.float64),axis=0).to(tl.float32); sn=sold*tl.exp(gg)[:,None]+kk[:,None]*vv[None,:]+bb[:,None]*read[None,:]; sold=tl.where(use,sn,sold)
        xo=((b0*T+t)*H+h0)*K+ik; vo=((b0*T+t)*H+h0)*V+ivv; gm=active&(ik<K); vm=active&(ivv<V)
        gg=tl.load(gp+xo,mask=gm,other=0.).to(tl.float32); dd=tl.exp(gg); aa=tl.load(ap+xo,mask=gm,other=0.).to(tl.float32); bb=tl.load(bp+xo,mask=gm,other=0.).to(tl.float32); kk=tl.load(kp+xo,mask=gm,other=0.).to(tl.float32); qq=tl.load(qp+xo,mask=gm,other=0.).to(tl.float32)*(K**-0.5); vv=tl.load(vp+vo,mask=vm,other=0.).to(tl.float32); do=tl.load(dop+vo,mask=vm,other=0.).to(tl.float32)
        read=tl.sum(sold.to(tl.float64)*aa[:,None].to(tl.float64),axis=0).to(tl.float32); snew=sold*dd[:,None]+kk[:,None]*vv[None,:]+bb[:,None]*read[None,:]; cur=ds+qq[:,None]*do[None,:]
        dq=tl.sum(snew*do[None,:],axis=1)*(K**-0.5); dk=tl.sum(cur*vv[None,:],axis=1); dv=tl.sum(cur*kk[:,None],axis=0); db=tl.sum(cur*read[None,:],axis=1); dread=tl.sum(cur.to(tl.float64)*bb[:,None].to(tl.float64),axis=0).to(tl.float32); da=tl.sum(sold*dread[None,:],axis=1); dg=tl.sum(cur*sold*dd[:,None],axis=1); ds=cur*dd[:,None]+aa[:,None]*dread[None,:]
        go=iv*(T*tl.num_programs(1)*K)+((b0*T+t)*H+h0)*K+ik; tl.store(pqp+go,dq,mask=gm);tl.store(pkp+go,dk,mask=gm);tl.store(pap+go,da,mask=gm);tl.store(pbp+go,db,mask=gm);tl.store(pgp+go,dg,mask=gm);tl.store(dvp+vo,dv,mask=vm)

@triton.jit
def _exact_h(kp,vp,ap,bp,gp,hp,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,NC:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    iv,bh=tl.program_id(0),tl.program_id(1); b0,h0=bh//H,bh%H; ik=tl.arange(0,BK); ivv=iv*BV+tl.arange(0,BV); s=tl.zeros((BK,BV),tl.float32)
    for t in range(T):
        if t%16==0:
            ch=t//16; ho=(((b0*NC+ch)*H+h0)*K+ik[:,None])*V+ivv[None,:]; tl.store(hp+ho,s,mask=(ik[:,None]<K)&(ivv[None,:]<V))
        xo=((b0*T+t)*H+h0)*K+ik; vo=((b0*T+t)*H+h0)*V+ivv
        a=tl.load(ap+xo,mask=ik<K,other=0.).to(tl.float32); bb=tl.load(bp+xo,mask=ik<K,other=0.).to(tl.float32); k=tl.load(kp+xo,mask=ik<K,other=0.).to(tl.float32); g=tl.load(gp+xo,mask=ik<K,other=0.).to(tl.float32); v=tl.load(vp+vo,mask=ivv<V,other=0.).to(tl.float32)
        read=tl.sum(s.to(tl.float64)*a[:,None].to(tl.float64),axis=0).to(tl.float32); s=s*tl.exp(g)[:,None]+k[:,None]*v[None,:]+bb[:,None]*read[None,:]
