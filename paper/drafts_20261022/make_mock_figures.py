# make_mock_figures.py -- regenerates figures/mock/fig1-6.png (synthetic data, fixed seed).
# Run after loading the figure-style skill: apply_figure_style(sizes=(8,7,6)); panel_letter must be in scope.
import re

import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
CM = 1/2.54
W2 = 17.4*CM; W1 = 8.4*CM
SEED = 20261022
TASKS8 = ['pancreas','lung','immune','ATAC small','ATAC large','human-mouse immune','simulation 1','simulation 2']
REAL6 = TASKS8[:6]
METHODS = ['plain scVI','scvi-tools adversary','JS discriminator','reference critic','pooled critic','barycenter critic','MMD','Sinkhorn']
# Okabe-Ito based; focal pair = discriminator (vermilion) and reference critic (blue)
COL = {'plain scVI':'#000000','scvi-tools adversary':'#999999','JS discriminator':'#D55E00',
       'reference critic':'#0072B2','pooled critic':'#56B4E9','barycenter critic':'#009E73',
       'MMD':'#CC79A7','Sinkhorn':'#E69F00'}
BASELINES = ['Harmony','Scanorama','sysVI','scANVI','PCA']
BMARK = {'Harmony':'s','Scanorama':'D','sysVI':'^','scANVI':'v','PCA':'X'}
DECODERS = ['conditioned decoder','unconditioned decoder']

def watermark(fig):
    fig.text(0.5,0.5,'MOCK',rotation=30,fontsize=min(110, 0.55*fig.get_size_inches()[1]*72),color='#C00000',alpha=0.16,
             ha='center',va='center',zorder=1000,fontweight='bold',transform=fig.transFigure)

def bbox_check(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t.get_text()!='MOCK']
    ticklabels = {ax: set(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both')) for ax in fig.axes}
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    ov = [(a.get_text(), b.get_text()) for i,(a,ba) in enumerate(texts) for b,bb in texts[i+1:] if ba.overlaps(bb)]
    ov += [(t.get_text(),'spine') for t,bt in texts for s,bs in spines if bt.overlaps(bs) and t not in ticklabels[s.axes]]
    out = [t.get_text() for t,bt in texts if not fig.bbox.contains(bt.x0,bt.y0) or not fig.bbox.contains(bt.x1,bt.y1)]
    return ov, out

def finish(fig, name):
    watermark(fig); mpl.rcParams['savefig.bbox']=None
    fig.savefig(f'figures/mock/{name}.png', dpi=300, bbox_inches=None)
    ov, out = bbox_check(fig)
    print(name, 'overlaps:', ov[:8], 'outside:', out[:5])
    plt.close(fig)


from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
rng = np.random.default_rng(SEED)
fig = plt.figure(figsize=(W2, 6.4*CM))
gs = fig.add_gridspec(1,2,width_ratios=[1.35,1],wspace=0.32,left=0.03,right=0.985,bottom=0.17,top=0.9)
ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(0,10); ax.set_ylim(0,5.8)
def box(x,y,w,h,text,fc='#F2F2F2',ec='#333333',ls='-',fs=7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',fc=fc,ec=ec,ls=ls,lw=0.9))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arrow(x0,y0,x1,y1,ls='-',color='#333333'):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=8,lw=0.9,color=color,ls=ls))
box(0.2,2.3,1.6,1.3,'counts\n$x$')
arrow(1.95,2.95,2.55,2.95)
box(2.6,2.3,1.9,1.3,'encoder')
arrow(4.6,2.95,5.2,2.95)
box(5.25,2.3,1.5,1.3,'latent\n$z$ (10-d)',fc='#DCE9F5',ec='#0072B2')
arrow(6.85,2.95,7.45,2.95)
box(7.5,2.3,1.9,1.3,'decoder')
arrow(9.5,2.95,9.95,2.95)
ax.text(9.97,2.95,'$\\hat{x}$',ha='left',va='center',fontsize=7)
# batch label into decoder (optional)
box(7.5,0.3,1.9,1.1,'batch label $s$\n(conditioned only)',fc='white',ls='--')
arrow(8.45,1.5,8.45,2.2,ls='--')
# alignment term
box(4.4,4.5,3.2,1.2,'alignment term (swappable)\nJS discriminator | W1 critic\nMMD | Sinkhorn',fc='#FBE5D6',ec='#D55E00',fs=6.5)
arrow(6.0,3.7,6.0,4.4,color='#D55E00')
ax.text(6.1,4.05,'penalty $\\times$ strength',fontsize=6,ha='left',va='center',color='#D55E00')
ax.text(0.2,0.25,'loss = scVI loss + strength $\\times$ penalty',fontsize=7,va='center')
ax.text(0.2,1.15,'Batch label to decoder: with (conditioned) or without\n(unconditioned, adversary is the only correction)',fontsize=6,va='center',color='#444444')
panel_letter(ax,'a',dx=0.0,dy=1.0)

# (b) reading biology at b*
ax2 = fig.add_subplot(gs[1])
b0, bio0 = 0.62, 0.70
bstar = 0.72
strengths = np.logspace(-1,2,6)
def curve(bmax, drop, bend):
    t = np.linspace(0,1,6)
    b = b0 + (bmax-b0)*(1-np.exp(-3*t))/(1-np.exp(-3))
    bio = bio0 - drop*t**bend
    return b, bio
for m,(bmax,drop,bend) in {'JS discriminator':(0.82,0.11,1.6),'reference critic':(0.80,0.08,2.2)}.items():
    b,bio = curve(bmax,drop,bend)
    b = np.r_[b0,b]; bio = np.r_[bio0,bio]
    ax2.plot(b,bio,'-o',color=COL[m],ms=3,lw=1.2,label=m)
    # interpolation at b*
    i = np.searchsorted(b,bstar)
    y = bio[i-1]+(bio[i]-bio[i-1])*(bstar-b[i-1])/(b[i]-b[i-1])
    ax2.plot([bstar],[y],marker='o',mfc='white',mec=COL[m],ms=6,mew=1.3,zorder=5)
    ax2.annotate('',xy=(b0-0.005,y),xytext=(bstar,y),arrowprops=dict(arrowstyle='-|>',color=COL[m],lw=0.8,ls=':'))
ax2.plot([b0],[bio0],marker='*',color='k',ms=9,ls='none',label='plain scVI',zorder=6)
ax2.axvline(bstar,color='#555555',lw=0.9,ls='--')
ax2.text(bstar+0.004,0.715,'$b^*$ matched\nbatch removal',fontsize=6,ha='left',va='top')
ax2.text(0.74,0.602,'strength increases\nalong each curve',fontsize=6,ha='left',color='#555555')
ax2.set_xlabel('batch score (higher = more batch removal)'); ax2.set_ylabel('biology score (higher = better)')
ax2.set_xlim(0.60,0.84); ax2.set_ylim(0.585,0.72); ax2.set_yticks([0.60,0.64,0.68,0.72]); ax2.set_xticks([0.60,0.65,0.70,0.75,0.80])
ax2.legend(loc='lower left',fontsize=6,handlelength=1.5)
ax2.set_title('biology is read where each curve reaches $b^*$',loc='left',fontsize=7)
panel_letter(ax2,'b',dx=-0.2)
finish(fig,'fig1')


rng = np.random.default_rng(SEED+2)
# Figure 2: biology at matched batch removal b* (colour) with within-task rank (number); mean rank per penalty
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
PEN = ['JS discriminator','reference critic','pooled critic','barycenter critic','MMD','Sinkhorn']
# (decoder, penalty, task) cells that are not ranked (low signal or not evaluable); the first two are also not evaluable in Figure 4
NOT_RANKED = {(0,'pooled critic','ATAC small'),(1,'Sinkhorn','human-mouse immune'),(1,'MMD','ATAC large'),(0,'MMD','ATAC large')}
NOT_EVAL = {(0,'pooled critic','ATAC small'),(1,'Sinkhorn','human-mouse immune')}
task_base = np.array([0.68,0.72,0.62,0.58,0.64,0.70])
# ranks: independent random permutation of the ranked penalties in each task column and decoder (no method is favoured);
# biology scores are drawn independently and assigned in rank order so the colour agrees with the printed rank
bio = np.zeros((2,6,6)); rank = np.full((2,6,6),np.nan)
for d in range(2):
    for t in range(6):
        ok = [p for p in range(6) if (d,PEN[p],REAL6[t]) not in NOT_RANKED]
        perm = rng.permutation(len(ok))            # perm[i] = rank-1 of the i-th ranked penalty
        vals = np.sort(task_base[t]-0.02*d+rng.normal(0,0.02,len(ok)))[::-1]
        for i,p in enumerate(ok): rank[d,p,t]=perm[i]+1; bio[d,p,t]=vals[perm[i]]
        for p in range(6):
            if p not in ok: bio[d,p,t]=task_base[t]
fig = plt.figure(figsize=(W2,9.8*CM))
gs = fig.add_gridspec(1,5,width_ratios=[1,1,0.04,0.85,1.1],wspace=0.09,left=0.125,right=0.975,bottom=0.37,top=0.86)
axh = [fig.add_subplot(gs[0]),fig.add_subplot(gs[1])]; axc = fig.add_subplot(gs[2]); axb = fig.add_subplot(gs[4])
cmap = plt.cm.viridis; vmin,vmax = 0.50,0.78; norm = mpl.colors.Normalize(vmin,vmax)
def lum(c):
    c = [x/12.92 if x<=0.03928 else ((x+0.055)/1.055)**2.4 for x in c[:3]]; return 0.2126*c[0]+0.7152*c[1]+0.0722*c[2]
for d,ax in enumerate(axh):
    im = ax.imshow(bio[d],cmap=cmap,norm=norm,aspect='auto')
    for p in range(6):
        for t in range(6):
            if (d,PEN[p],REAL6[t]) in NOT_RANKED:
                ax.add_patch(Rectangle((t-0.5,p-0.5),1,1,fc='#F2F2F2',ec='#7F7F7F',hatch='////',lw=0,zorder=3))
                continue
            L = lum(cmap(norm(bio[d,p,t]))); col = 'white' if 1.05/(L+0.05) > (L+0.05)/0.05 else 'black'
            ax.text(t,p,f'{int(rank[d,p,t])}',ha='center',va='center',fontsize=6,color=col)
    ax.set_xticks(range(6)); ax.set_xticklabels(REAL6,rotation=55,ha='right',rotation_mode='anchor')
    ax.set_yticks(range(6)); ax.set_yticklabels(PEN if d==0 else [])
    ax.tick_params(length=2,left=(d==0))
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-0.5,6,1),minor=True); ax.set_yticks(np.arange(-0.5,6,1),minor=True)
    ax.grid(which='minor',color='white',lw=0.8); ax.tick_params(which='minor',length=0)
    ax.set_title(DECODERS[d],loc='left',fontsize=7)
cb = fig.colorbar(im,cax=axc); cb.set_label('biology score at $b^*$',fontsize=7); cb.ax.tick_params(labelsize=6,length=2); cb.outline.set_visible(False)
cb.set_ticks([0.5,0.6,0.7])
fig.text(0.125,0.955,'biology score at matched batch removal (colour); rank within task column (number, 1 = best)',fontsize=7,ha='left',va='center')
fig.text(0.012,0.955,'a',fontsize=9,fontweight='bold',ha='left',va='center')
# (b) mean rank over ranked real tasks, range over tasks
ax = axb; DM = {0:'o',1:'s'}
for p,m in enumerate(PEN):
    for d in range(2):
        v = rank[d,p][~np.isnan(rank[d,p])]; y = p+(-0.17 if d==0 else 0.17)
        ax.plot([v.min(),v.max()],[y,y],color=COL[m],lw=1.2,solid_capstyle='butt',zorder=2)
        ax.plot([v.mean()],[y],DM[d],color=COL[m],ms=4.2,mec='white',mew=0.4,zorder=3)
ax.set_ylim(5.5,-0.5); ax.set_yticks(range(6)); ax.set_yticklabels(PEN); ax.tick_params(axis='y',length=2); ax.set_xlim(0.6,6.4); ax.set_xticks(range(1,7))
ax.set_xlabel('mean rank over tasks\n$\\leftarrow$ better (1 = best)',fontsize=7)
for y in np.arange(0.5,5.5,1): ax.axhline(y,color='#E5E5E5',lw=0.5,zorder=0)
ax.set_title('mean rank; bars = range',loc='left',fontsize=7)
panel_letter(ax,'b',dx=-0.12,dy=1.04)
h=[Patch(fc='#F2F2F2',ec='#7F7F7F',hatch='////',lw=0,label='not ranked (low signal or not evaluable)')]
fig.legend(handles=h,loc='lower left',bbox_to_anchor=(0.115,0.0),fontsize=6,frameon=False,handlelength=1.6)
h2=[Line2D([],[],marker='o',ls='none',color='#777777',ms=4,label='conditioned decoder (upper of each pair)'),
    Line2D([],[],marker='s',ls='none',color='#777777',ms=4,label='unconditioned decoder (lower)')]
fig.legend(handles=h2,loc='lower left',bbox_to_anchor=(0.62,0.0),fontsize=6,frameon=False,ncol=1,handletextpad=0.3)
finish(fig,'fig2')


rng = np.random.default_rng(SEED+3)
PEN = ['JS discriminator','reference critic','pooled critic','barycenter critic','MMD','Sinkhorn']
fig, axs = plt.subplots(2,6,figsize=(W2,9.0*CM),sharex=False,sharey=False,
                        gridspec_kw=dict(left=0.085,right=0.955,bottom=0.2,top=0.9,wspace=0.42,hspace=0.35))
def curve(b0,bio0,bmax,drop,bend,n=6):
    t=np.linspace(0,1,n); b=b0+(bmax-b0)*(1-np.exp(-3*t))/(1-np.exp(-3)); bio=bio0-drop*t**bend
    return np.r_[b0,b],np.r_[bio0,bio]
for r,dec in enumerate(DECODERS):
    for c,task in enumerate(REAL6):
        ax=axs[r,c]
        b0=0.55+0.15*rng.random()-0.05*r; bio0=0.62+0.12*rng.random()
        bmaxs=[]
        for m in PEN:
            bmax=b0+0.12+0.1*rng.random(); drop=0.04+0.12*rng.random(); bend=1.2+2*rng.random()
            b,bio=curve(b0,bio0,bmax,drop,bend)
            if rng.random()<0.25: b,bio=b[:5],bio[:5]  # collapse: curve ends early
            focal = m in ('JS discriminator','reference critic')
            ax.plot(b,bio,'-o',color=COL[m],ms=1.8 if not focal else 2.4,lw=1.1 if focal else 0.7,alpha=1 if focal else 0.8,label=m if (r,c)==(0,0) else None)
            bmaxs.append(b[-1])
        if r==0:
            ax.plot([b0+0.02],[bio0-0.01],'o',color=COL['scvi-tools adversary'],ms=3,label='scvi-tools adversary' if c==0 else None)
        ax.plot([b0],[bio0],'*',color='k',ms=7,zorder=6,label='plain scVI' if (r,c)==(0,0) else None)
        bstar=(b0+min(bmaxs))/2; ax.axvline(bstar,color='#555555',ls='--',lw=0.7)
        for k,bl in enumerate(BASELINES):
            ax.plot([b0-0.02+0.25*rng.random()],[bio0-0.12+0.14*rng.random()],BMARK[bl],color='#7F7F7F',ms=3,mfc='none',mew=0.8,label=bl if (r,c)==(0,0) else None)
        ax.set_xlim(b0-0.06,b0+0.3); ax.set_ylim(bio0-0.2,bio0+0.05)
        ax.set_xticks([round(b0,1),round(b0+0.2,1)]); ax.set_yticks([round(bio0-0.15,2),round(bio0-0.05,2)])
        if r==0: ax.set_title(task.replace('human-mouse immune','human-mouse imm.'),loc='left',fontsize=7)
        if c==0: ax.set_ylabel('biology score')
        if r==1: ax.set_xlabel('batch score')
axs[0,0].text(axs[0,0].get_xlim()[1]-0.005,axs[0,0].get_ylim()[0]+0.005,'$b^*$',fontsize=6,ha='right',va='bottom',color='#555555')
fig.text(0.975,0.55,'higher = better (both axes)',rotation=90,fontsize=6,va='center',ha='center',color='#555555')
for r,dec in enumerate(DECODERS): fig.text(0.012,[0.72,0.33][r],dec,rotation=90,fontsize=7,va='center',ha='center')
h,l=axs[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',bbox_to_anchor=(0.52,0.0),ncol=7,fontsize=6,handlelength=1.6,columnspacing=1.0)
for ax,let in zip([axs[0,0],axs[1,0]],'ab'): panel_letter(ax,let,dx=-0.6,dy=1.08)
finish(fig,'fig3')


rng = np.random.default_rng(SEED+4)
# Figure 4: per-penalty dot plots, both decoders, one point per real task (vertical slot = task order) plus mean tick
from scipy.stats import beta
from matplotlib.lines import Line2D
PEN = ['JS discriminator','reference critic','pooled critic','barycenter critic','MMD','Sinkhorn']
fig, axs = plt.subplots(1,3,figsize=(W2,10.5*CM),gridspec_kw=dict(left=0.145,right=0.985,bottom=0.2,top=0.88,wspace=0.15))
titles = ['trade-off area','collapse rate, exact 95% interval','local minus global removal at $b^*$']
xlabels = ['trade-off area\n(higher = better)','collapse rate (percentage points)\n(lower = better)','local minus global batch removal\n(positive = stronger locally)']
step = 0.085
ty = (np.arange(6)-2.5)*step   # task slot offsets within a decoder block
dblock = {0:-0.2,1:0.2}         # decoder block centre (conditioned above)
pen_area = np.array([0.52,0.58,0.56,0.55,0.50,0.51]); pen_col = np.array([14,4,8,6,12,10]); pen_loc=np.array([0.01,0.04,0.02,0.03,-0.01,0.00])
NREP = 24
MK = {0:'o',1:'s'}
for pi,ax in enumerate(axs):
    for p,m in enumerate(PEN):
        for d in range(2):
            vals=[]; 
            for t in range(6):
                y = p+dblock[d]+ty[t]
                ne = (d,m,REAL6[t]) in NOT_EVAL and pi!=1
                if pi==0: v = pen_area[p]-0.02*d+rng.normal(0,0.045)
                elif pi==1:
                    k = int(rng.binomial(NREP,min(0.95,max(0.0,(pen_col[p]+(3 if d else 0))/100*1.0+rng.normal(0,0.03)))*1.0)); v=100*k/NREP
                    lo = 0 if k==0 else beta.ppf(0.025,k,NREP-k+1)*100; hi = 100 if k==NREP else beta.ppf(0.975,k+1,NREP-k)*100
                else: v = pen_loc[p]-0.005*d+rng.normal(0,0.03)
                if ne:
                    ax.text(0.02,y,'n.e.',transform=ax.get_yaxis_transform(),fontsize=6,ha='left',va='center',color='#555555'); continue
                vals.append(v)
                if pi==1: ax.plot([lo,hi],[y,y],color=COL[m],lw=0.6,alpha=0.7,solid_capstyle='butt',zorder=2)
                ax.plot([v],[y],MK[d],color=COL[m],ms=2.6,mec='none',zorder=3)
            ax.plot([np.mean(vals)]*2,[p+dblock[d]-3*step,p+dblock[d]+3*step],color='k',lw=1.1,zorder=4)
    for y in np.arange(0.5,5.5,1): ax.axhline(y,color='#E5E5E5',lw=0.5,zorder=0)
    if pi==2: ax.axvline(0,color='#555555',lw=0.7,ls='--',zorder=1)
    ax.set_ylim(5.55,-0.55); ax.set_yticks(range(6)); ax.set_yticklabels(PEN if pi==0 else [])
    ax.set_title(titles[pi],loc='left',fontsize=7); ax.set_xlabel(xlabels[pi],fontsize=7)
    panel_letter(ax,'abc'[pi],dx=-0.52 if pi==0 else -0.1,dy=1.04)
axs[0].set_xlim(0.28,0.82); axs[0].set_xticks([0.3,0.5,0.7]); axs[1].set_xlim(-4,104); axs[1].set_xticks([0,25,50,75,100]); axs[2].set_xlim(-0.15,0.2); axs[2].set_xticks([-0.1,0,0.1])
# 'n.e.' text sits at the left edge of the panel; leave room
h=[Line2D([],[],marker='o',ls='none',color='#777777',ms=3.5,label='conditioned decoder (upper block)'),
   Line2D([],[],marker='s',ls='none',color='#777777',ms=3.5,label='unconditioned decoder (lower block)'),
   Line2D([],[],color='k',lw=1.1,label='mean over tasks'),
   Line2D([],[],ls='none',label='n.e. = not evaluable')]
fig.legend(handles=h,loc='lower center',bbox_to_anchor=(0.56,0.0),ncol=4,fontsize=6,frameon=False,handlelength=1.3,columnspacing=1.2,handletextpad=0.4)
finish(fig,'fig4')


rng = np.random.default_rng(SEED+5)
# Figure 5: width of each penalty's range of good strengths (decades), one point per real task and decoder, median per penalty
from matplotlib.lines import Line2D
PEN=['JS discriminator','reference critic','pooled critic','barycenter critic','MMD','Sinkhorn']
fig, ax = plt.subplots(figsize=(W2,7.4*CM),gridspec_kw=dict(left=0.075,right=0.985,bottom=0.22,top=0.9))
centre={'JS discriminator':1.0,'reference critic':1.4,'pooled critic':1.3,'barycenter critic':1.2,'MMD':1.8,'Sinkhorn':1.6}
for i,m in enumerate(PEN):
    allw=[]
    for j,mk in enumerate(['o','s']):
        w=np.clip(rng.normal(centre[m]-0.2*j,0.4,6),0,2.5); allw+=list(w)
        ax.plot(i+(-0.17 if j==0 else 0.17)+rng.normal(0,0.04,6),w,mk,ms=3.4,color=COL[m],alpha=0.8,mec='none')
    ax.plot([i-0.32,i+0.32],[np.median(allw)]*2,color='k',lw=1.3,zorder=4)
ax.set_xticks(range(6)); ax.set_xticklabels(PEN)
ax.set_ylabel('width of good-strength range\n(decades; wider = broader)'); ax.set_ylim(-0.3,3.0); ax.set_yticks([0,1,2,3]); ax.set_xlim(-0.6,5.6)
ax.set_title('width of the range of good strengths, one point per real task and decoder',loc='left',fontsize=7)
h=[Line2D([],[],marker='o',ls='none',color='#777777',ms=3.8,label='conditioned decoder (left of each pair)'),
   Line2D([],[],marker='s',ls='none',color='#777777',ms=3.8,label='unconditioned decoder (right)'),
   Line2D([],[],color='k',lw=1.3,label='median over 12 points')]
fig.legend(handles=h,loc='lower center',bbox_to_anchor=(0.54,0.0),ncol=3,fontsize=6,frameon=False,handlelength=1.5,columnspacing=1.5)
finish(fig,'fig5')


rng = np.random.default_rng(SEED+6)
fig, axs = plt.subplots(1,2,figsize=(W2,7.8*CM),gridspec_kw=dict(width_ratios=[1,1.1],left=0.07,right=0.985,bottom=0.31,top=0.9,wspace=0.35))
# (a) expected best selection score vs number of tries k=1..6, pooled over real tasks, conditioned decoder
ax=axs[0]; k=np.arange(1,7)
fam = {'JS discriminator':0.655,'reference critic':0.66,'pooled critic':0.658,'barycenter critic':0.652,'MMD':0.65,'Sinkhorn':0.648}
for m,top in fam.items():
    y = top-0.03*np.exp(-(k-1)/1.6)+rng.normal(0,0.001,6)
    focal = m in ('JS discriminator','reference critic')
    ax.plot(k,y,'-o',color=COL[m],ms=2.5,lw=1.1 if focal else 0.7,label=m)
ax.axhline(0.651,color='k',lw=1.0,ls='-'); ax.text(6.05,0.651,'plain scVI',fontsize=6,va='center',color='k')
for bl,lv,ls in [('Harmony',0.645,'--'),('Scanorama',0.63,':'),('sysVI',0.64,'-.')]:
    y=lv-0.02*np.exp(-(k-1)/1.6); ax.plot(k,y,ls=ls,color='#7F7F7F',lw=0.8,label=bl)
ax.plot([1],[0.668],BMARK['scANVI'],color='#7F7F7F',ms=4,mfc='none',label='scANVI (uses labels)'); ax.text(1.12,0.668,'scANVI (uses labels)',fontsize=6,va='center',color='#555555')
ax.set_xlabel('number of settings tried, $k$'); ax.set_ylabel('expected best selection score'); ax.set_xticks(k); ax.set_xlim(0.7,7.3); ax.set_ylim(0.578,0.675)
ax.set_title('expected best score against number of tries',loc='left',fontsize=7)
h,l=ax.get_legend_handles_labels(); keep=[i for i,lab in enumerate(l) if lab!='scANVI (uses labels)']
ax.legend([h[i] for i in keep],[l[i] for i in keep],fontsize=5.5,loc='lower right',ncol=2,handlelength=1.6,columnspacing=0.8)
fig.text(0.07,0.0,'(a) higher = better; optimistic: best setting chosen and scored on the same data.  (b) higher = better on both axes; one star per task.',fontsize=6,color='#555555',va='bottom')
panel_letter(ax,'a',dx=-0.17,dy=1.03)
# (b) batch vs biology score at the single transferred strength, per real task, conditioned decoder
ax=axs[1]; PEN=list(fam)
for ti,t in enumerate(REAL6):
    b0=[0.50,0.56,0.62,0.68,0.74,0.60][ti]; bio0=[0.62,0.70,0.58,0.66,0.60,0.72][ti]
    ax.plot(b0,bio0,'*',color='k',ms=6,zorder=5,label='plain scVI' if ti==0 else None)
    for m in PEN:
        ax.plot(b0+0.03+0.1*rng.random(),bio0-0.05+0.06*rng.random(),'o',color=COL[m],ms=3.2,mec='white',mew=0.3,label=m if ti==0 else None)
    for bl in BASELINES:
        ax.plot(b0-0.03+0.2*rng.random(),bio0-0.08+0.1*rng.random(),BMARK[bl],color='#7F7F7F',ms=3,mfc='none',mew=0.8,label=bl if ti==0 else None)
    ax.annotate(t,xy=(b0,bio0),xytext=(b0+(0.012 if ti==5 else -0.012),bio0+0.012),fontsize=6,ha='left' if ti==5 else 'right',color='#333333',arrowprops=dict(arrowstyle='-',lw=0.4,color='#999999'))
ax.set_xlabel('batch score at single transferred strength'); ax.set_ylabel('biology score')
ax.set_title('single strength chosen on other task families',loc='left',fontsize=7)
ax.set_xlim(0.4,0.95); ax.set_ylim(0.48,0.76)
h,l=ax.get_legend_handles_labels(); fig.legend(h,l,fontsize=6,loc='lower center',bbox_to_anchor=(0.56,0.045),ncol=6,handlelength=1.4,columnspacing=0.9)
panel_letter(ax,'b',dx=-0.17,dy=1.03)
finish(fig,'fig6')
