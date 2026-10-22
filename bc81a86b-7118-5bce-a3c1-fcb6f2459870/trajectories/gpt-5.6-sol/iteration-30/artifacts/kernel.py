import torch
import triton
import triton.language as tl



@triton.jit
def _wy(k, g, beta, gc, a, w, T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr,
        K: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IP: tl.constexpr):
    c, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // HV, bh % HV
    hq = h // (HV // HQ)
    i, j, x = tl.arange(0, 16), tl.arange(0, 16), tl.arange(0, BK)
    t = c * 16 + i
    m = t < T
    G = tl.cumsum(tl.load(g + (b*T+t)*HV+h, mask=m, other=0.).to(tl.float32), axis=0)
    be = tl.load(beta + (b*T+t)*HV+h, mask=m, other=0.).to(tl.float32)
    kk = tl.load(k + ((b*T+t[:, None])*HQ+hq)*K+x[None, :],
                 mask=m[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
    gram = tl.dot(kk, tl.trans(kk), input_precision=IP)
    E = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.))
    L = tl.where((i[:, None] > j[None, :]) & m[:, None] & m[None, :],
                 be[:, None] * E * gram, 0.)
    A = tl.where(i[:, None] == j[None, :], 1., 0.)
    for r in range(1, 16):
        lr = tl.sum(tl.where(i[:, None] == r, L, 0.), axis=0)
        row = tl.where(j == r, 1., 0.) - tl.sum(tl.where(j < r, lr, 0.)[:, None] * A, axis=0)
        A = tl.where(i[:, None] == r, row[None, :], A)
    W = tl.dot(A, be[:, None] * tl.exp(tl.minimum(G, 0.))[:, None] * kk, input_precision=IP)
    ba = ((b*HV+h)*NC+c)*16*16
    tl.store(a + ba + i[:, None]*16+j[None, :], A, mask=m[:, None] & m[None, :])
    tl.store(gc + (b*T+t)*HV+h, G, mask=m)
    tl.store(w + ((b*T+t[:, None])*HV+h)*K+x[None, :], W,
             mask=m[:, None] & (x[None, :] < K))


@triton.jit
def _fscan(k, v, gc, beta, a, w, hs, z, T: tl.constexpr, HQ: tl.constexpr,
           HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
           NC: tl.constexpr, IP: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, 32)
    vb, bh = p % nv, p // nv
    b, h = bh // HV, bh % HV
    hq = h // (HV // HQ)
    x, y = tl.arange(0, BK), vb*32+tl.arange(0, 32)
    i, j = tl.arange(0, 16), tl.arange(0, 16)
    S = tl.zeros((BK, 32), tl.float32)
    for c in range(NC):
        t, m = c*16+i, c*16+i < T
        ph = (((b*HV+h)*NC+c)*K+x[:, None])*V+y[None, :]
        tl.store(hs+ph, S, mask=(x[:, None] < K) & (y[None, :] < V))
        ba = ((b*HV+h)*NC+c)*16*16
        A = tl.load(a+ba+i[:, None]*16+j[None, :], mask=m[:, None] & m[None, :], other=0.)
        be = tl.load(beta+(b*T+t)*HV+h, mask=m, other=0.).to(tl.float32)
        vv = tl.load(v+((b*T+t[:, None])*HV+h)*V+y[None, :],
                     mask=m[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
        W = tl.load(w+((b*T+t[:, None])*HV+h)*K+x[None, :],
                    mask=m[:, None] & (x[None, :] < K), other=0.)
        Z = tl.dot(A, be[:, None]*vv, input_precision=IP) - tl.dot(W, S, input_precision=IP)
        tl.store(z+((b*T+t[:, None])*HV+h)*V+y[None, :], Z,
                 mask=m[:, None] & (y[None, :] < V))
        kk = tl.load(k+((b*T+t[:, None])*HQ+hq)*K+x[None, :],
                     mask=m[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
        G = tl.load(gc+(b*T+t)*HV+h, mask=m, other=0.)
        li = tl.max(tl.where(m, i, -1), axis=0)
        ge = tl.sum(tl.where(i == li, G, 0.), axis=0)
        S = tl.exp(tl.minimum(ge, 0.))*S + tl.dot(tl.trans(kk*tl.exp(tl.minimum(ge-G, 0.))[:, None]), Z, input_precision=IP)


@triton.jit
def _out(q, k, gc, hs, z, o, T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr,
         K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
         IP: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, 32)
    vb, p = p % nv, p // nv
    c, bh = p % NC, p // NC
    b, h = bh // HV, bh % HV
    hq = h // (HV // HQ)
    i, j, x = tl.arange(0, 16), tl.arange(0, 16), tl.arange(0, BK)
    y, t = vb*32+tl.arange(0, 32), c*16+i
    m = t < T
    Q = tl.load(q+((b*T+t[:, None])*HQ+hq)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.).to(tl.float32)*SCALE
    K0 = tl.load(k+((b*T+t[:, None])*HQ+hq)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.).to(tl.float32)
    G = tl.load(gc+(b*T+t)*HV+h, mask=m, other=0.)
    ph = (((b*HV+h)*NC+c)*K+x[:, None])*V+y[None, :]
    H0 = tl.load(hs+ph, mask=(x[:, None]<K)&(y[None, :]<V), other=0.)
    Z = tl.load(z+((b*T+t[:, None])*HV+h)*V+y[None, :], mask=m[:, None]&(y[None, :]<V), other=0.).to(tl.float32)
    R = tl.dot(Q, tl.trans(K0), input_precision=IP)
    R *= tl.where((i[:, None]>=j[None, :])&m[None, :], tl.exp(tl.minimum(G[:, None]-G[None, :], 0.)), 0.)
    O = tl.exp(tl.minimum(G, 0.))[:, None]*tl.dot(Q, H0, input_precision=IP) + tl.dot(R, Z, input_precision=IP)
    tl.store(o+((b*T+t[:, None])*HV+h)*V+y[None, :], O, mask=m[:, None]&(y[None, :]<V))


@triton.jit
def _ascan(q, k, gc, w, z, do, dhs, T: tl.constexpr, HQ: tl.constexpr,
           HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
           NC: tl.constexpr, IP: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, 32)
    vb, bh = p % nv, p // nv
    b, h = bh // HV, bh % HV
    hq = h // (HV // HQ)
    x, y = tl.arange(0, BK), vb*32+tl.arange(0, 32)
    i, j = tl.arange(0, 16), tl.arange(0, 16)
    dS = tl.zeros((BK, 32), tl.float32)
    for rc in range(NC):
        c = NC-1-rc
        t, m = c*16+i, c*16+i < T
        ph = (((b*HV+h)*NC+c)*K+x[:, None])*V+y[None, :]
        tl.store(dhs+ph, dS, mask=(x[:, None]<K)&(y[None, :]<V))
        Q = tl.load(q+((b*T+t[:, None])*HQ+hq)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.).to(tl.float32)*SCALE
        K0 = tl.load(k+((b*T+t[:, None])*HQ+hq)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.).to(tl.float32)
        G = tl.load(gc+(b*T+t)*HV+h, mask=m, other=0.)
        li = tl.max(tl.where(m, i, -1), axis=0)
        ge = tl.sum(tl.where(i == li, G, 0.), axis=0)
        D = tl.load(do+((b*T+t[:, None])*HV+h)*V+y[None, :], mask=m[:, None]&(y[None, :]<V), other=0.).to(tl.float32)
        W = tl.load(w+((b*T+t[:, None])*HV+h)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.)
        E = tl.where((i[:, None]>=j[None, :])&m[None, :], tl.exp(tl.minimum(G[:, None]-G[None, :], 0.)), 0.)
        R = tl.dot(Q, tl.trans(K0), input_precision=IP)*E
        dZ = tl.dot(tl.trans(R), D, input_precision=IP) + tl.dot(K0*tl.exp(tl.minimum(ge-G, 0.))[:, None], dS, input_precision=IP)
        dS = tl.exp(tl.minimum(ge, 0.))*dS + tl.dot(tl.trans(Q), tl.exp(tl.minimum(G, 0.))[:, None]*D, input_precision=IP) - tl.dot(tl.trans(W), dZ, input_precision=IP)


@triton.jit
def _grad(q, k, v, gc, beta, a, w, hs, z, do, dhs, dqp, dkp, dv, dgp, dbp,
          T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
          V: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IP: tl.constexpr,
          SCALE: tl.constexpr):
    p0 = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, 32)
    vb, p = p0 % nv, p0 // nv
    c, bh = p % NC, p // NC
    b, h = bh // HV, bh % HV
    hq = h // (HV // HQ)
    i, j, x = tl.arange(0, 16), tl.arange(0, 16), tl.arange(0, BK)
    y, t = vb*32+tl.arange(0, 32), c*16+i
    m = t < T
    Q = tl.load(q+((b*T+t[:, None])*HQ+hq)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.).to(tl.float32)*SCALE
    K0 = tl.load(k+((b*T+t[:, None])*HQ+hq)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.).to(tl.float32)
    V0 = tl.load(v+((b*T+t[:, None])*HV+h)*V+y[None, :], mask=m[:, None]&(y[None, :]<V), other=0.).to(tl.float32)
    D = tl.load(do+((b*T+t[:, None])*HV+h)*V+y[None, :], mask=m[:, None]&(y[None, :]<V), other=0.).to(tl.float32)
    Z = tl.load(z+((b*T+t[:, None])*HV+h)*V+y[None, :], mask=m[:, None]&(y[None, :]<V), other=0.).to(tl.float32)
    G = tl.load(gc+(b*T+t)*HV+h, mask=m, other=0.)
    be = tl.load(beta+(b*T+t)*HV+h, mask=m, other=0.).to(tl.float32)
    ph = (((b*HV+h)*NC+c)*K+x[:, None])*V+y[None, :]
    H0 = tl.load(hs+ph, mask=(x[:, None]<K)&(y[None, :]<V), other=0.)
    dSn = tl.load(dhs+ph, mask=(x[:, None]<K)&(y[None, :]<V), other=0.)
    W = tl.load(w+((b*T+t[:, None])*HV+h)*K+x[None, :], mask=m[:, None]&(x[None, :]<K), other=0.)
    ba = ((b*HV+h)*NC+c)*16*16
    A = tl.load(a+ba+i[:, None]*16+j[None, :], mask=m[:, None]&m[None, :], other=0.)
    li = tl.max(tl.where(m, i, -1), axis=0)
    ge = tl.sum(tl.where(i == li, G, 0.), axis=0)
    e, ek = tl.exp(tl.minimum(G, 0.)), tl.exp(tl.minimum(ge-G, 0.))
    E = tl.where((i[:, None]>=j[None, :])&m[None, :], tl.exp(tl.minimum(G[:, None]-G[None, :], 0.)), 0.)
    R = tl.dot(Q, tl.trans(K0), input_precision=IP)*E
    dR = tl.dot(D, tl.trans(Z), input_precision=IP)
    dRE = tl.where(i[:, None]>=j[None, :], dR*E, 0.)
    dZ = tl.dot(tl.trans(R), D, input_precision=IP) + tl.dot(K0*ek[:, None], dSn, input_precision=IP)
    dQ = e[:, None]*tl.dot(D, tl.trans(H0), input_precision=IP) + tl.dot(dRE, K0, input_precision=IP)
    dK = tl.dot(tl.trans(dRE), Q, input_precision=IP) + ek[:, None]*tl.dot(Z, tl.trans(dSn), input_precision=IP)
    dW = -tl.dot(dZ, tl.trans(H0), input_precision=IP)
    X, P = be[:, None]*V0, be[:, None]*e[:, None]*K0
    dA = tl.dot(dZ, tl.trans(X), input_precision=IP) + tl.dot(dW, tl.trans(P), input_precision=IP)
    dX, dP = tl.dot(tl.trans(A), dZ, input_precision=IP), tl.dot(tl.trans(A), dW, input_precision=IP)
    dL = -tl.dot(tl.dot(tl.trans(A), dA, input_precision=IP), tl.trans(A), input_precision=IP)
    Es = tl.where((i[:, None]>j[None, :])&m[:, None]&m[None, :], tl.exp(tl.minimum(G[:, None]-G[None, :], 0.)), 0.)
    gram = tl.dot(K0, tl.trans(K0), input_precision=IP)
    dLE = dL*Es
    dK += be[:, None]*tl.dot(dLE, K0, input_precision=IP) + tl.dot(tl.trans(dLE), be[:, None]*K0, input_precision=IP) + be[:, None]*e[:, None]*dP
    dV = be[:, None]*dX
    db = tl.sum(dX*V0, axis=1) + tl.sum(e[:, None]*dP*K0, axis=1)
    db += tl.sum(dL*Es*gram, axis=1)
    dG = tl.sum(D*(e[:, None]*tl.dot(Q, H0, input_precision=IP)), axis=1)
    gr = dR*R
    dG += tl.sum(gr, axis=1)-tl.sum(gr, axis=0)
    gl = dL*(be[:, None]*Es*gram)
    dG += tl.sum(gl, axis=1)-tl.sum(gl, axis=0) + tl.sum(dP*P, axis=1)
    kg = tl.sum(ek[:, None]*tl.dot(Z, tl.trans(dSn), input_precision=IP)*K0, axis=1)
    dG -= kg
    end = tl.sum(kg, axis=0) + tl.sum(dSn*(tl.exp(tl.minimum(ge, 0.))*H0))
    dG += tl.where(i == li, end, 0.)
    dG = tl.sum(dG, axis=0) - tl.cumsum(dG, axis=0) + dG
    part = ((b*T+t)*HV+h)*nv+vb
    tl.store(dqp+part[:, None]*K+x[None, :], dQ*SCALE, mask=m[:, None]&(x[None, :]<K))
    tl.store(dkp+part[:, None]*K+x[None, :], dK, mask=m[:, None]&(x[None, :]<K))
    tl.store(dv+((b*T+t[:, None])*HV+h)*V+y[None, :], dV, mask=m[:, None]&(y[None, :]<V))
    tl.store(dgp+part, dG, mask=m)
    tl.store(dbp+part, db, mask=m)


@triton.jit
def _rqk(dqp, dkp, dq, dk, HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
         BK: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    x, hq, bt = tl.arange(0, BK), p % HQ, p // HQ
    aq, ak = tl.zeros((BK,), tl.float32), tl.zeros((BK,), tl.float32)
    for r in range(HV // HQ):
        for vb in range(NV):
            z = (bt*HV+hq*(HV//HQ)+r)*NV+vb
            aq += tl.load(dqp+z*K+x, mask=x<K, other=0.)
            ak += tl.load(dkp+z*K+x, mask=x<K, other=0.)
    tl.store(dq+p*K+x, aq, mask=x<K)
    tl.store(dk+p*K+x, ak, mask=x<K)


@triton.jit
def _rs(dgp, dbp, dg, db, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    x, y = 0., 0.
    for vb in range(NV):
        x += tl.load(dgp+p*NV+vb)
        y += tl.load(dbp+p*NV+vb)
    tl.store(dg+p, x)
    tl.store(db+p, y)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, HQ, K = q.shape
        HV, V = v.shape[2:]
        nc = triton.cdiv(T, 16)
        nv = triton.cdiv(V, 32)
        bk = max(16, triton.next_power_of_2(K))
        ip = 'ieee' if q.dtype == torch.float32 else 'tf32'
        gc = torch.empty((B,T,HV), device=q.device, dtype=torch.float32)
        a = torch.empty((B,HV,nc,16,16), device=q.device, dtype=torch.float32)
        w = torch.empty((B,T,HV,K), device=q.device, dtype=torch.float32)
        hs = torch.empty((B,HV,nc,K,V), device=q.device, dtype=torch.float32)
        z, o = torch.empty_like(v), torch.empty_like(v)
        _wy[(nc,B*HV)](k,g,beta,gc,a,w,T=T,HQ=HQ,HV=HV,K=K,BK=bk,NC=nc,IP=ip,num_warps=4)
        _fscan[(B*HV*nv,)](k,v,gc,beta,a,w,hs,z,T=T,HQ=HQ,HV=HV,K=K,V=V,BK=bk,NC=nc,IP=ip,num_warps=4)
        _out[(B*HV*nc*nv,)](q,k,gc,hs,z,o,T=T,HQ=HQ,HV=HV,K=K,V=V,BK=bk,NC=nc,IP=ip,SCALE=K**-0.5,num_warps=4)
        ctx.save_for_backward(q,k,v,g,beta,gc,a,w,hs,z)
        ctx.meta = B,T,HQ,HV,K,V,nc,nv,bk,ip
        return o

    @staticmethod
    def backward(ctx, do):
        q,k,v,g,beta,gc,a,w,hs,z = ctx.saved_tensors
        B,T,HQ,HV,K,V,nc,nv,bk,ip = ctx.meta
        do = do.contiguous()
        dhs = torch.empty_like(hs)
        dqp = torch.empty((B,T,HV,nv,K),device=q.device,dtype=torch.float32)
        dkp = torch.empty_like(dqp)
        dgp = torch.empty((B,T,HV,nv),device=q.device,dtype=torch.float32)
        dbp = torch.empty_like(dgp)
        dq,dk,dv,dg,db = torch.empty_like(q),torch.empty_like(k),torch.empty_like(v),torch.empty_like(g),torch.empty_like(beta)
        _ascan[(B*HV*nv,)](q,k,gc,w,z,do,dhs,T=T,HQ=HQ,HV=HV,K=K,V=V,BK=bk,NC=nc,IP=ip,SCALE=K**-0.5,num_warps=4)
        _grad[(B*HV*nc*nv,)](q,k,v,gc,beta,a,w,hs,z,do,dhs,dqp,dkp,dv,dgp,dbp,T=T,HQ=HQ,HV=HV,K=K,V=V,BK=bk,NC=nc,IP=ip,SCALE=K**-0.5,num_warps=4)
        _rqk[(B*T*HQ,)](dqp,dkp,dq,dk,HQ=HQ,HV=HV,K=K,BK=bk,NV=nv,num_warps=4)
        _rs[(B*T*HV,)](dgp,dbp,dg,db,NV=nv,num_warps=1)
        return dq,dk,dv,dg,db


def gated_delta_rule(q,k,v,g,beta):
    return _GatedDelta.apply(q,k,v,g,beta)
