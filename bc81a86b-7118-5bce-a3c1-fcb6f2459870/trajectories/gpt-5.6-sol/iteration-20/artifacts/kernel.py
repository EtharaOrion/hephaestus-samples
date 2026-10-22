import torch
import triton
import triton.language as tl

C = tl.constexpr(16)
BV = tl.constexpr(32)

@triton.jit
def _build_wy(k, g, beta, gc, A, W, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    ic, ibh = tl.program_id(0), tl.program_id(1)
    ib, ihv = ibh // HV, ibh % HV
    ih = ihv // (HV // H)
    r, x = tl.arange(0, C), tl.arange(0, BK)
    t = ic * C + r
    mt = t < T
    gv = tl.load(g + (ib*T+t)*HV+ihv, mask=mt, other=0.).to(tl.float32)
    gv = tl.cumsum(gv, axis=0)
    tl.store(gc + (ib*T+t)*HV+ihv, gv, mask=mt)
    b = tl.load(beta + (ib*T+t)*HV+ihv, mask=mt, other=0.).to(tl.float32)
    pk = k + ((ib*T+t[:, None])*H+ih)*K+x[None, :]
    kv = tl.load(pk, mask=mt[:, None] & (x[None, :] < K), other=0.)
    gram = tl.dot(kv, tl.trans(kv), input_precision="tf32")
    low = (r[:, None] > r[None, :]) & mt[:, None] & mt[None, :]
    L = tl.where(low, gram * tl.exp(gv[:, None]-gv[None, :]) * b[:, None], 0.)
    inv = -L
    for i in range(2, C):
        row = tl.sum(tl.where((r == i)[:, None], -L, 0.), axis=0)
        row = tl.where(r < i, row, 0.) + tl.sum(tl.where(r < i, row, 0.)[:, None] * inv, axis=0)
        inv = tl.where((r == i)[:, None], row, inv)
    inv += r[:, None] == r[None, :]
    inv = tl.where(mt[:, None] & mt[None, :], inv, 0.)
    tl.store(A + ((ib*T+t[:, None])*HV+ihv)*C+r[None, :], inv, mask=mt[:, None])
    X = kv * (b * tl.exp(gv))[:, None]
    wv = tl.dot(inv.to(kv.dtype), X.to(kv.dtype), input_precision="tf32")
    tl.store(W + ((ib*T+t[:, None])*HV+ihv)*K+x[None, :], wv, mask=mt[:, None] & (x[None, :] < K))

@triton.jit
def _build_u(v, beta, A, U, T: tl.constexpr, HV: tl.constexpr, V: tl.constexpr):
    ic, ibh, iv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // HV, ibh % HV
    r, z = tl.arange(0, C), tl.program_id(2)*BV + tl.arange(0, BV)
    t, mt, mz = ic*C+r, ic*C+r < T, z < V
    av = tl.load(A + ((ib*T+t[:, None])*HV+ih)*C+r[None, :], mask=mt[:, None], other=0.)
    b = tl.load(beta+(ib*T+t)*HV+ih, mask=mt, other=0.)
    pv = v + ((ib*T+t[:, None])*HV+ih)*V+z[None, :]
    vv = tl.load(pv, mask=mt[:, None] & mz[None, :], other=0.)
    uv = tl.dot(av.to(vv.dtype), (vv*b[:, None]).to(vv.dtype), input_precision="tf32")
    tl.store(U + ((ib*T+t[:, None])*HV+ih)*V+z[None, :], uv, mask=mt[:, None] & mz[None, :])

@triton.jit
def _scan_state(k, gc, W, U, states, D, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv: tl.constexpr = tl.cdiv(V, BV)
    iv, ibh = pid % nv, pid // nv
    ib, ihv = ibh // HV, ibh % HV
    ih = ihv // (HV // H)
    x, z, r = tl.arange(0, BK), iv*BV+tl.arange(0, BV), tl.arange(0, C)
    mx, mz = x < K, z < V
    S = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        t, mt = ic*C+r, ic*C+r < T
        ps = states + ((((ib*NC+ic)*HV+ihv)*K+x[:, None])*V+z[None, :])
        tl.store(ps, S, mask=mx[:, None] & mz[None, :])
        G = tl.load(gc+(ib*T+t)*HV+ihv, mask=mt, other=0.)
        pk = k+((ib*T+t[:, None])*H+ih)*K+x[None, :]
        Kb = tl.load(pk, mask=mt[:, None]&mx[None, :], other=0.)
        Wb = tl.load(W+((ib*T+t[:, None])*HV+ihv)*K+x[None, :], mask=mt[:, None]&mx[None, :], other=0.)
        Ub = tl.load(U+((ib*T+t[:, None])*HV+ihv)*V+z[None, :], mask=mt[:, None]&mz[None, :], other=0.)
        Db = Ub.to(tl.float32)-tl.dot(Wb, S.to(Wb.dtype), input_precision="tf32")
        tl.store(D+((ib*T+t[:, None])*HV+ihv)*V+z[None, :], Db, mask=mt[:, None]&mz[None, :])
        n = tl.minimum(C, T-ic*C)
        last = tl.sum(tl.where(r == n-1, G, 0.), axis=0)
        N = Kb * tl.exp(last-G)[:, None]
        S = S*tl.exp(last) + tl.dot(tl.trans(N), Db.to(N.dtype), input_precision="tf32")

@triton.jit
def _emit(q, k, gc, D, states, out, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv: tl.constexpr = tl.cdiv(V, BV)
    iv, p = pid % nv, pid // nv
    ic, ibh = p % NC, p // NC
    ib, ihv = ibh // HV, ibh % HV
    ih = ihv // (HV // H)
    r, x, z = tl.arange(0, C), tl.arange(0, BK), iv*BV+tl.arange(0, BV)
    t, mt, mx, mz = ic*C+r, ic*C+r<T, x<K, z<V
    G = tl.load(gc+(ib*T+t)*HV+ihv, mask=mt, other=0.)
    Q = tl.load(q+((ib*T+t[:, None])*H+ih)*K+x[None, :], mask=mt[:, None]&mx[None, :], other=0.)*SCALE
    Kb = tl.load(k+((ib*T+t[:, None])*H+ih)*K+x[None, :], mask=mt[:, None]&mx[None, :], other=0.)
    S = tl.load(states+((((ib*NC+ic)*HV+ihv)*K+x[:, None])*V+z[None, :]), mask=mx[:, None]&mz[None, :], other=0.)
    Db = tl.load(D+((ib*T+t[:, None])*HV+ihv)*V+z[None, :], mask=mt[:, None]&mz[None, :], other=0.)
    M = tl.dot(Q, tl.trans(Kb), input_precision="tf32") * tl.exp(G[:, None]-G[None, :])
    M = tl.where((r[:, None]>=r[None, :])&mt[:, None]&mt[None, :], M, 0.)
    O = tl.dot(Q, S.to(Q.dtype), input_precision="tf32")*tl.exp(G)[:, None] + tl.dot(M.to(Db.dtype), Db, input_precision="tf32")
    tl.store(out+((ib*T+t[:, None])*HV+ihv)*V+z[None, :], O, mask=mt[:, None]&mz[None, :])

@triton.jit
def _adjoint_scan(q, k, gc, W, dout, dstates, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv: tl.constexpr = tl.cdiv(V, BV)
    iv, ibh = pid % nv, pid // nv
    ib, ihv = ibh // HV, ibh % HV
    ih = ihv // (HV // H)
    r, x, z = tl.arange(0, C), tl.arange(0, BK), iv*BV+tl.arange(0, BV)
    mx, mz = x<K, z<V
    dS = tl.zeros((BK, BV), tl.float32)
    for jj in range(NC):
        ic = NC-1-jj
        t, mt = ic*C+r, ic*C+r<T
        pds = dstates+((((ib*NC+ic)*HV+ihv)*K+x[:, None])*V+z[None, :])
        tl.store(pds, dS, mask=mx[:, None]&mz[None, :])
        G = tl.load(gc+(ib*T+t)*HV+ihv, mask=mt, other=0.)
        Q = tl.load(q+((ib*T+t[:, None])*H+ih)*K+x[None, :], mask=mt[:, None]&mx[None, :], other=0.)*SCALE
        Kb = tl.load(k+((ib*T+t[:, None])*H+ih)*K+x[None, :], mask=mt[:, None]&mx[None, :], other=0.)
        do = tl.load(dout+((ib*T+t[:, None])*HV+ihv)*V+z[None, :], mask=mt[:, None]&mz[None, :], other=0.)
        M = tl.dot(Q, tl.trans(Kb), input_precision="tf32")*tl.exp(G[:, None]-G[None, :])
        M = tl.where((r[:, None]>=r[None, :])&mt[:, None]&mt[None, :], M, 0.)
        n = tl.minimum(C, T-ic*C)
        last = tl.sum(tl.where(r==n-1, G, 0.), axis=0)
        N = Kb*tl.exp(last-G)[:, None]
        dD = tl.dot(tl.trans(M.to(do.dtype)), do, input_precision="tf32") + tl.dot(N, dS.to(N.dtype), input_precision="tf32")
        Wb = tl.load(W+((ib*T+t[:, None])*HV+ihv)*K+x[None, :], mask=mt[:, None]&mx[None, :], other=0.)
        dS = tl.dot(tl.trans((Q*tl.exp(G)[:, None]).to(do.dtype)), do, input_precision="tf32") + tl.exp(last)*dS - tl.dot(tl.trans(Wb), dD.to(Wb.dtype), input_precision="tf32")

@triton.jit
def _chunk_grad(q, k, v, beta, gc, A, W, D, states, dstates, dout, pdq, pdk, dv_out, pdg, pdb, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv: tl.constexpr = tl.cdiv(V, BV)
    iv, p = pid%nv, pid//nv
    ic, ibh = p%NC, p//NC
    ib, ihv = ibh//HV, ibh%HV
    ih = ihv//(HV//H)
    r, x, z = tl.arange(0,C), tl.arange(0,BK), iv*BV+tl.arange(0,BV)
    t, mt, mx, mz = ic*C+r, ic*C+r<T, x<K, z<V
    G = tl.load(gc+(ib*T+t)*HV+ihv, mask=mt, other=0.)
    b = tl.load(beta+(ib*T+t)*HV+ihv, mask=mt, other=0.).to(tl.float32)
    Q = tl.load(q+((ib*T+t[:,None])*H+ih)*K+x[None,:], mask=mt[:,None]&mx[None,:], other=0.)*SCALE
    Kb = tl.load(k+((ib*T+t[:,None])*H+ih)*K+x[None,:], mask=mt[:,None]&mx[None,:], other=0.)
    Vb = tl.load(v+((ib*T+t[:,None])*HV+ihv)*V+z[None,:], mask=mt[:,None]&mz[None,:], other=0.)
    do = tl.load(dout+((ib*T+t[:,None])*HV+ihv)*V+z[None,:], mask=mt[:,None]&mz[None,:], other=0.)
    Db = tl.load(D+((ib*T+t[:,None])*HV+ihv)*V+z[None,:], mask=mt[:,None]&mz[None,:], other=0.)
    Wb = tl.load(W+((ib*T+t[:,None])*HV+ihv)*K+x[None,:], mask=mt[:,None]&mx[None,:], other=0.)
    S = tl.load(states+((((ib*NC+ic)*HV+ihv)*K+x[:,None])*V+z[None,:]), mask=mx[:,None]&mz[None,:], other=0.)
    dSend = tl.load(dstates+((((ib*NC+ic)*HV+ihv)*K+x[:,None])*V+z[None,:]), mask=mx[:,None]&mz[None,:], other=0.)
    Ab = tl.load(A+((ib*T+t[:,None])*HV+ihv)*C+r[None,:], mask=mt[:,None], other=0.)
    tri = (r[:,None]>=r[None,:])&mt[:,None]&mt[None,:]
    R = tl.where(tri, tl.exp(G[:,None]-G[None,:]), 0.)
    M = tl.dot(Q,tl.trans(Kb),input_precision="tf32")*R
    n = tl.minimum(C,T-ic*C)
    last = tl.sum(tl.where(r==n-1,G,0.),axis=0)
    Rend = tl.exp(last-G)
    N = Kb*Rend[:,None]
    dD = tl.dot(tl.trans(M.to(do.dtype)),do,input_precision="tf32") + tl.dot(N,dSend.to(N.dtype),input_precision="tf32")
    dM = tl.dot(do,tl.trans(Db),input_precision="tf32")
    dN = tl.dot(Db.to(tl.float32),tl.trans(dSend),input_precision="tf32")
    dW = -tl.dot(dD,tl.trans(S),input_precision="tf32")
    X = Kb*(b*tl.exp(G))[:,None]
    Y = Vb*b[:,None]
    dA = tl.dot(dW,X.to(tl.float32).trans(),input_precision="tf32") + tl.dot(dD,Y.to(tl.float32).trans(),input_precision="tf32")
    dX = tl.dot(Ab.trans(),dW,input_precision="tf32")
    dY = tl.dot(Ab.trans(),dD,input_precision="tf32")
    dL = -tl.dot(tl.dot(Ab.trans(),dA,input_precision="tf32"),Ab.trans(),input_precision="tf32")
    strict = (r[:,None]>r[None,:])&mt[:,None]&mt[None,:]
    dL = tl.where(strict,dL,0.)
    gram = tl.dot(Kb.to(tl.float32),Kb.to(tl.float32).trans(),input_precision="tf32")
    RL = tl.where(strict,tl.exp(G[:,None]-G[None,:]),0.)
    Cg = dL*b[:,None]*RL
    Z = dM*R
    dQ = tl.dot(Z,Kb.to(tl.float32),input_precision="tf32") + tl.dot((do*tl.exp(G)[:,None]).to(tl.float32),S.trans(),input_precision="tf32")
    dK = tl.dot(Z.trans(),Q.to(tl.float32),input_precision="tf32") + dN*Rend[:,None] + dX*(b*tl.exp(G))[:,None] + tl.dot(Cg+Cg.trans(),Kb.to(tl.float32),input_precision="tf32")
    dbv = tl.sum(dX*(Kb*tl.exp(G)[:,None]),axis=1)+tl.sum(dY*Vb,axis=1)+tl.sum(dL*RL*gram,axis=1)
    dV = dY*b[:,None]
    dG = tl.sum(do*(tl.dot(Q,S.to(Q.dtype),input_precision="tf32")*tl.exp(G)[:,None]),axis=1)+tl.sum(dX*X,axis=1)
    lM, lL = dM*M, dL*(b[:,None]*RL*gram)
    dG += tl.sum(lM,axis=1)-tl.sum(lM,axis=0)+tl.sum(lL,axis=1)-tl.sum(lL,axis=0)
    lN = tl.sum(dN*N,axis=1)
    dG -= lN
    end = tl.sum(lN,axis=0)+tl.sum(tl.sum(dSend*(S*tl.exp(last)),axis=1),axis=0)
    dG += tl.where(r==n-1,end,0.)
    dg = tl.sum(tl.where(r[None,:]>=r[:,None],dG[None,:],0.),axis=1)
    base = (((ib*T+t[:,None])*HV+ihv)*nv+iv)*K+x[None,:]
    tl.store(pdq+base,dQ*SCALE,mask=mt[:,None]&mx[None,:])
    tl.store(pdk+base,dK,mask=mt[:,None]&mx[None,:])
    tl.store(dv_out+((ib*T+t[:,None])*HV+ihv)*V+z[None,:],dV,mask=mt[:,None]&mz[None,:])
    sb = ((ib*T+t)*HV+ihv)*nv+iv
    tl.store(pdg+sb,dg,mask=mt)
    tl.store(pdb+sb,dbv,mask=mt)

@triton.jit
def _sum_qk(pdq,pdk,dq,dk,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,NV:tl.constexpr):
    pid=tl.program_id(0).to(tl.int64); x=tl.arange(0,BK); ih=pid%H; bt=pid//H
    aq=tl.zeros((BK,),tl.float32); ak=tl.zeros((BK,),tl.float32)
    REP:tl.constexpr=HV//H
    for j in range(REP):
        for iv in range(NV):
            off=(((bt*HV+ih*REP+j)*NV+iv)*K+x)
            aq+=tl.load(pdq+off,mask=x<K,other=0.); ak+=tl.load(pdk+off,mask=x<K,other=0.)
    tl.store(dq+(bt*H+ih)*K+x,aq,mask=x<K); tl.store(dk+(bt*H+ih)*K+x,ak,mask=x<K)

@triton.jit
def _sum_scalar(pg,pb,dg,db,NV:tl.constexpr):
    pid=tl.program_id(0).to(tl.int64); ag=0.; ab=0.
    for iv in range(NV):
        ag+=tl.load(pg+pid*NV+iv); ab+=tl.load(pb+pid*NV+iv)
    tl.store(dg+pid,ag); tl.store(db+pid,ab)

class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,g,beta):
        B,T,H,K=q.shape; HV,V=v.shape[2:]; nc=triton.cdiv(T,C); nv=triton.cdiv(V,BV); bk=triton.next_power_of_2(K)
        gc=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); A=torch.empty((B,T,HV,C),device=q.device,dtype=torch.float32)
        W=torch.empty((B,T,HV,K),device=q.device,dtype=q.dtype); U=torch.empty_like(v); D=torch.empty_like(v)
        states=torch.empty((B,nc,HV,K,V),device=q.device,dtype=torch.float32); out=torch.empty_like(v)
        _build_wy[(nc,B*HV)](k,g,beta,gc,A,W,T=T,H=H,HV=HV,K=K,BK=bk,num_warps=4)
        _build_u[(nc,B*HV,nv)](v,beta,A,U,T=T,HV=HV,V=V,num_warps=4)
        _scan_state[(B*HV*nv,)](k,gc,W,U,states,D,T=T,H=H,HV=HV,K=K,V=V,BK=bk,NC=nc,num_warps=4)
        _emit[(B*HV*nc*nv,)](q,k,gc,D,states,out,T=T,H=H,HV=HV,K=K,V=V,BK=bk,NC=nc,SCALE=K**-.5,num_warps=4)
        ctx.save_for_backward(q,k,v,beta,gc,A,W,D,states); ctx.meta=(B,T,H,HV,K,V,nc,nv,bk)
        return out
    @staticmethod
    def backward(ctx,dout):
        q,k,v,beta,gc,A,W,D,states=ctx.saved_tensors; B,T,H,HV,K,V,nc,nv,bk=ctx.meta; dout=dout.contiguous()
        dstates=torch.empty_like(states)
        _adjoint_scan[(B*HV*nv,)](q,k,gc,W,dout,dstates,T=T,H=H,HV=HV,K=K,V=V,BK=bk,NC=nc,SCALE=K**-.5,num_warps=4)
        shape=(B,T,HV,nv,K); pq=torch.empty(shape,device=q.device,dtype=torch.float32); pk=torch.empty_like(pq)
        pg=torch.empty((B,T,HV,nv),device=q.device,dtype=torch.float32); pb=torch.empty_like(pg); dv=torch.empty_like(v)
        _chunk_grad[(B*HV*nc*nv,)](q,k,v,beta,gc,A,W,D,states,dstates,dout,pq,pk,dv,pg,pb,T=T,H=H,HV=HV,K=K,V=V,BK=bk,NC=nc,SCALE=K**-.5,num_warps=4)
        dq=torch.empty_like(q); dk=torch.empty_like(k); dg=torch.empty((B,T,HV),device=q.device,dtype=torch.float32); db=torch.empty_like(beta)
        _sum_qk[(B*T*H,)](pq,pk,dq,dk,H=H,HV=HV,K=K,BK=bk,NV=nv,num_warps=4)
        _sum_scalar[(B*T*HV,)](pg,pb,dg,db,NV=nv,num_warps=1)
        return dq,dk,dv,dg,db

def gated_delta_rule(q,k,v,g,beta):
    return _GatedDeltaRule.apply(q,k,v,g,beta)
