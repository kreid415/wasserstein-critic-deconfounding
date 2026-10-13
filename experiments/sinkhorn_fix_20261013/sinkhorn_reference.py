"""Audit-only converged entropy-regularized OT; no repository changes.
Cost/epsilon/masks are exactly those of alignment.py. Float64 positive-kernel
scaling gives an optimal plan, whose envelope cost gradient is used once.
"""
import math
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.optimize import minimize


def solve_ot(C, a, b, eps, tol=1e-10, max_iter=200):
    C=np.asarray(C,dtype=np.float64); a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    # Constant shift protects representability; it does not change the transport plan.
    shift=float(C.min()); K=np.exp(-(C-shift)/eps)
    u=np.ones(len(a)); v=np.ones(len(b)); residual=np.inf
    for it in range(max_iter):
        u=a/(K@v)
        v=b/(K.T@u)
        if it%20==19:
            residual=max(np.max(np.abs(u*(K@v)-a)),np.max(np.abs(v*(K.T@u)-b)))
            if residual<tol: break
            # Gauge normalization avoids overflow without changing the plan.
            gauge=np.exp(np.mean(np.log(u))); u/=gauge; v*=gauge
    if residual>=tol:
        # Newton solve the reduced smooth dual. This also handles near-disconnected
        # transport where plain balancing converges slowly.
        g=eps*(np.log(v)-np.log(b)); g-=g[-1]
        def fun(x):
            gg=np.r_[x,0.]
            logits=np.log(b)[None,:]+(gg[None,:]-C)/eps
            lse=logsumexp(logits,axis=1)
            P=a[:,None]*np.exp(logits-lse[:,None])
            val=eps*np.dot(a,lse)-np.dot(b,gg)
            grad=P.sum(0)-b
            H=(np.diag(P.sum(0))-(P.T/a)@P)/eps
            return val,grad[:-1],H[:-1,:-1]
        sol=minimize(lambda x:fun(x)[0],g[:-1],jac=lambda x:fun(x)[1],hess=lambda x:fun(x)[2],method='trust-exact',options={'gtol':tol/10,'maxiter':200})
        gg=np.r_[sol.x,0.]
        logits=np.log(b)[None,:]+(gg[None,:]-C)/eps
        f=-eps*logsumexp(logits,axis=1)
        P=a[:,None]*np.exp(logits+f[:,None]/eps)
        residual=max(np.max(np.abs(P.sum(1)-a)),np.max(np.abs(P.sum(0)-b)))
        value=np.dot(a,f)+np.dot(b,gg)-eps*(P.sum()-1)
        it+=sol.nit
    else:
        P=u[:,None]*K*v[None,:]
        f=eps*(np.log(u)-np.log(a))+shift
        g=eps*(np.log(v)-np.log(b))
        value=np.dot(a,f)+np.dot(b,g)-eps*(P.sum()-1)
    if not np.isfinite(value) or residual>max(tol*20,2e-8):
        raise RuntimeError(f'OT failed to converge: residual={residual}, iterations={it+1}')
    return float(value),P,dict(residual=float(residual),iterations=it+1)


def cost_matrix(z):
    # Direct differences avoid diagonal cancellation in the repository FP32
    # expansion. Double precision also handles very small collapsed distances.
    zd=z.double()
    d2=(zd[:,None,:]-zd[None,:,:]).square().sum(-1)
    return torch.sqrt(d2+1e-12)


def reference_cost(C,b,eps,tol=1e-10,max_iter=200):
    cd=C.detach().cpu().double().numpy(); bn=b.detach().cpu().numpy().reshape(-1)
    groups=np.unique(bn); n=len(bn); value=0.; coeff=np.zeros((n,n),dtype=np.float64); stats=[]
    if len(groups)<2: return C.sum()*0,stats
    cache={}
    def ot(ii,jj):
        key=(tuple(ii),tuple(jj))
        if key in cache: return cache[key]
        rev=(key[1],key[0])
        if rev in cache:
            vv,pp,ss=cache[rev]; return vv,pp.T,ss
        a=np.full(len(ii),1/len(ii)); bb=np.full(len(jj),1/len(jj))
        ans=solve_ot(cd[np.ix_(ii,jj)],a,bb,eps,tol,max_iter)
        cache[key]=ans
        return ans
    for k in groups:
        ii=np.flatnonzero(bn==k); jj=np.flatnonzero(bn!=k)
        for x,y,w in ((ii,jj,1.),(ii,ii,-.5),(jj,jj,-.5)):
            vv,pp,ss=ot(x,y); value+=w*vv/len(groups)
            coeff[np.ix_(x,y)]+=w*pp/len(groups); stats.append(ss)
    surrogate=(C*torch.as_tensor(coeff,dtype=C.dtype,device=C.device)).sum()
    # Exact value, optimal-plan derivative, no doubled extrapolation gradient.
    out=surrogate+(C.new_tensor(value)-surrogate.detach())
    return out,stats


def corrected_sinkhorn(z,b,tol=1e-9):
    C=cost_matrix(z); eps=.05*math.sqrt(2*z.shape[1])
    return reference_cost(C,b,eps,tol=tol)[0]
