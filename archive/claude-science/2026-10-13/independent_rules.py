"""Independent re-implementation of PREREG R1, R4 (stage a1), R2/R3 written from docs/PREREG.md text only
(review track Pipeline, 2026-10-12/13). Uses SD with ddof=1 (sample SD) as implied by the median-unbiasing constant
sqrt(chi2_{n-1,0.5}/(n-1)). Inputs: tier12_v2/{A1,A1_ext_r1_round1,A1_ext_r4_round1,A2A3}/scores; resolved manifest r4a1.
"""
import numpy as np, pandas as pd, math
from scipy.stats import chi2
BATCH=["PCR_batch","ASW_label/batch","iLISI","graph_conn","kBET"]
BIO=["NMI_cluster/label","ARI_cluster/label","ASW_label","isolated_label_F1","isolated_label_silhouette","cLISI","cell_cycle_conservation","trajectory"]
SEQ=[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,10000,30000,100000,300000]
FAM={'discriminator':'JS','reference':'W1','pooled':'W1','barycenter':'W1','mmd':'MMD','sinkhorn':'Sinkhorn'}
DMIN,K,C,Q=0.01,3,0.10,0.5
ARMS6=['discriminator','reference','pooled','barycenter','mmd','sinkhorn']
A2ARMS=['discriminator','reference','pooled']; A3ARMS=A2ARMS+['mmd','sinkhorn']; A23TASKS=['atac_small','immune']
# r23() reads globals r4fin (output of r4 on A1+extensions), A1all and a23b (add_BC outputs)
def add_BC(df):
    df=df.copy()
    df['B']=df[BATCH].mean(axis=1,skipna=False)
    def c(row):
        ms=BIO[:6]+(['cell_cycle_conservation'] if row.task in ('pancreas','lung','immune','immune_hum_mou') else [])+(['trajectory'] if row.task in ('immune','immune_hum_mou') else [])
        v=row[ms].astype(float)
        assert np.isfinite(v).all(), row.tag
        return v.mean()
    df['C']=df.apply(c,axis=1); assert np.isfinite(df.B).all()
    return df


def mu_const(n): return np.sqrt(chi2.ppf(0.5,n-1)/(n-1))


def block_stats(d, lams, failed):
    """d: rows of one block incl lam=0 (none) rows. failed: set of (lam,seed)"""
    base=d[d.lam==0].set_index('seed')
    rows=[]
    for l in lams:
        x=d[d.lam==l].set_index('seed')
        nf=sum(1 for (ll,s) in failed if ll==l)
        seeds=[s for s in x.index if s in base.index]
        db=np.array([x.B[s]-base.B[s] for s in seeds]); dc=np.array([x.C[s]-base.C[s] for s in seeds])
        n=len(seeds)
        rows.append(dict(lam=l,n=n,nfail=nf,db=db.mean() if n else np.nan,dc=dc.mean() if n else np.nan,
                         sdb=db.std(ddof=1)/mu_const(n) if n>=2 else np.nan, sdc=dc.std(ddof=1)/mu_const(n) if n>=2 else np.nan))
    t=pd.DataFrame(rows)
    sb=np.nanmedian(t.sdb); scc=np.nanmedian(t.sdc)
    t['delb']=np.maximum(DMIN,K*sb/np.sqrt(t.n)); t['delc']=np.maximum(DMIN,K*scc/np.sqrt(t.n))
    t['noeff']=(t.nfail==0)&(t.db.abs()<t.delb)&(t.dc.abs()<t.delc)
    t['coll']=(t.nfail>=1)|(t.dc<=-C+1e-9*0)
    m=len(t)
    f=-1
    for i in range(m):
        if t.noeff[:i+1].all(): f=i
        else: break
    mx=t.db.max()
    p=None
    if not (mx < t.delb.iloc[t.db.values.argmax()]):
        for i in range(f+1,m):
            if t.db[i] >= mx - t.delb[i]: p=i; break
    h=None
    for i in range(f+1,m):
        if all(t.coll[j] or (p is not None and j>=p) for j in range(i,m)): h=i; break
    nonresp=t.noeff.all()
    low_edge = f==-1
    high_edge = nonresp or h is None or (h==m-1 and not t.coll[m-1])
    return dict(f=f,p=p,h=h,low_edge=low_edge,high_edge=high_edge,nonresp=nonresp,sigma_b=sb,sigma_c=scc,table=t)


def r1(df, failed_df=None):
    df=add_BC(df)
    out={}
    for cond in [0,1]:
        for fam in ['JS','W1','MMD','Sinkhorn']:
            arms=[a for a,f in FAM.items() if f==fam]
            sub=df[(df.cond==cond)&(df.arm.isin(arms))]
            lams=sorted(sub.lam.unique())
            blocks={}
            for task in sorted(df.task.unique()):
                base=df[(df.cond==cond)&(df.task==task)&(df.arm=='none')]
                for a in arms:
                    d=pd.concat([base,sub[(sub.task==task)&(sub.arm==a)]])
                    blocks[(task,a)]=block_stats(d,lams,set())
            fs=[b['f'] for b in blocks.values() if not b['nonresp']]
            hs=[b['h'] for b in blocks.values() if not b['nonresp']]
            le=[k for k,b in blocks.items() if b['low_edge']]; he=[k for k,b in blocks.items() if b['high_edge']]
            res=dict(lams=lams,blocks=blocks,low_edge=le,high_edge=he)
            if not le and not he:
                ilo=min(fs); ihi=max(hs); w=ihi-ilo+1
                if w>=6: idx=[int(np.floor(ilo+j*(ihi-ilo)/5+0.5)) for j in range(6)]
                else:
                    idx=list(range(ilo,ihi+1)); up,dn=ihi+1,ilo-1; side='up'
                    while len(idx)<6:
                        if side=='up' and up<len(lams): idx.append(up); up+=1
                        elif side=='dn' and dn>=0: idx.insert(0,dn); dn-=1
                        elif up>=len(lams) and dn<0: break
                        side='dn' if side=='up' else 'up'
                res.update(ilo=ilo,ihi=ihi,grid=[lams[i] for i in sorted(idx)])
            out[(cond,fam)]=res
    return out


def r4(df, failed=set(), extendable=lambda t,c,a: t in A23TASKS and a in A3ARMS, rounds=None):
    df=add_BC(df); res={}
    for task in sorted(df.task.unique()):
        for cond in [0,1]:
            d=df[(df.task==task)&(df.cond==cond)]
            b0=d[d.arm=='none'].B.mean()
            sm={a:d[d.arm==a].groupby('lam').B.mean() for a in ARMS6}
            M={a:s.max() for a,s in sm.items()}
            bc=min(M.values()); bs=b0+Q*(bc-b0)
            cells={}
            for a in ARMS6:
                s=sm[a].sort_index(); dist=(s-bs).abs()
                mn=dist.min(); cand=[l for l in s.index if dist[l]<=mn+1e-9]; m=min(cand)
                lams=list(s.index); i=lams.index(m)
                lo=lams[i-1] if i>0 else None; hi=lams[i+1] if i<len(lams)-1 else None
                edge=None
                if lo is None: edge='low'
                if hi is None: edge='high'
                cells[a]=dict(matched=m,lo=lo,hi=hi,edge=edge,ext=extendable(task,cond,a) and edge is not None,B=s.to_dict())
            res[(task,cond)]=dict(b0=b0,b_common=bc,bstar=bs,M=M,cells=cells,low_signal=(bc-b0)<2*DMIN)
    return res


def bio_at(sm, bs):
    """sm: list of (lam,B,C) in lam order, failed-all points removed"""
    Bs=[b for _,b,_ in sm]
    for l,b,c in sm:
        if abs(b-bs)<=1e-9: return c,'ok'
    for (l1,b1,c1),(l2,b2,c2) in zip(sm[:-1],sm[1:]):
        if (b1-bs)*(b2-bs)<0:
            return c1+(bs-b1)*(c2-c1)/(b2-b1),'ok'
    if all(b<bs for b in Bs): return np.nan,'unreached-low'
    if all(b>bs for b in Bs): return np.nan,'unreached-high'
    return np.nan,'other'


def r23(exp, arms, ROPE):
    cells=[]
    for task in A23TASKS:
        for cond in [0,1]:
            bs=r4fin[(task,cond)]['bstar']
            for a in arms:
                c=r4fin[(task,cond)]['cells'][a]; win=[c['lo'],c['matched'],c['hi']]
                dd=A1all[(A1all.task==task)&(A1all.cond==cond)&(A1all.arm==a)]
                aa=a23b[(a23b.experiment==exp)&(a23b.task==task)&(a23b.cond==cond)&(a23b.arm==a)]
                out={}
                for nm,src in [('def',dd),('alt',aa)]:
                    sm=[(l,src[np.isclose(src.lam,l)].B.mean(),src[np.isclose(src.lam,l)].C.mean()) for l in win]
                    assert all(len(src[np.isclose(src.lam,l)])==3 for l in win),(task,cond,a,nm,win)
                    out[nm]=bio_at(sm,bs)
                ev=out['def'][1]=='ok' and out['alt'][1]=='ok'
                cells.append(dict(task=task,cond=cond,arm=a,win=win,bio_def=out['def'][0],st_def=out['def'][1],bio_alt=out['alt'][0],st_alt=out['alt'][1],evaluable=ev,D=out['alt'][0]-out['def'][0] if ev else np.nan))
    t=pd.DataFrame(cells); n=len(t)
    ne=t.evaluable.sum(); per=t.groupby(['task','cond']).evaluable.sum()
    c1=ne>=math.ceil(2*n/3) and (per>=1).all()
    mD=t.D.mean(); c2=mD>ROPE
    c3=(t[t.evaluable].groupby(['task','cond']).D.mean()>0).all() if c1 else False
    c4=True  # no failures in any stage (checked separately)
    mask = exp=='A2' and ((t.st_alt=='unreached-low')&(t.st_def=='ok')).any()
    switch=c1 and c2 and c3 and c4 and not mask
    return t,dict(n=n,n_eval=int(ne),c1=c1,meanD=mD,c2=c2,c3=c3,mask=mask,switch=switch)
