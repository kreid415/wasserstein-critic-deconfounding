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
fig, axs = plt.subplots(1,3,figsize=(W2,8.0*CM),gridspec_kw=dict(width_ratios=[1.25,1,1],wspace=0.42,left=0.08,right=0.985,bottom=0.33,top=0.92))
# (a) separability AUC per task, two decoders, 5 seeds
ax = axs[0]
base = np.array([0.97,0.93,0.99,0.96,0.98,0.995,0.90,0.88])
for j,(dec,mk,off) in enumerate(zip(DECODERS,['o','s'],[-0.17,0.17])):
    for i,t in enumerate(TASKS8):
        v = np.clip(base[i]-0.01*j + rng.normal(0,0.008,5),0.5,1)
        ax.plot(np.full(5,i+off)+rng.normal(0,0.03,5),v,mk,ms=2.6,color=['#0072B2','#E69F00'][j],alpha=0.75,mec='none',label=dec if i==0 else None)
        ax.plot([i+off-0.12,i+off+0.12],[v.mean()]*2,color='k',lw=1.0)
ax.axhline(0.95,color='#555555',ls='--',lw=0.8); ax.text(7.45,0.951,'0.95 nearly\nseparable',fontsize=6,ha='right',va='bottom',color='#555555')
ax.axhline(0.5,color='#AAAAAA',ls=':',lw=0.8); ax.text(0.0,0.515,'chance',fontsize=6,color='#777777')
ax.set_xticks(range(8)); ax.set_xticklabels(TASKS8,rotation=55,ha='right',rotation_mode='anchor'); ax.set_ylim(0.45,1.02)
ax.set_ylabel('batch separability (AUC, plain scVI)'); ax.set_title('batches nearly separable',loc='left',fontsize=7)
ax.legend(loc='center left',bbox_to_anchor=(0.0,0.35),fontsize=6,markerscale=1.5)
panel_letter(ax,'a',dx=-0.22)
# (b) gradient norm per epoch, normalised to its peak
ax = axs[1]; ep = np.arange(1,401)
shapes = {'JS discriminator':np.exp(-ep/25)*(1+0.1*rng.normal(size=400)*0.3)+0.02,
          'reference critic':0.6+0.4*np.exp(-ep/150)+0.05*np.sin(ep/15),
          'pooled critic':0.55+0.45*np.exp(-ep/120)+0.04*np.sin(ep/20+1),
          'barycenter critic':0.5+0.5*np.exp(-ep/200)+0.04*np.sin(ep/12+2),
          'MMD':0.4+0.6*np.exp(-ep/80),'Sinkhorn':0.45+0.55*np.exp(-ep/100)}
for m,y in shapes.items():
    y = np.clip(y,0.005,None); y = y/y.max(); ax.plot(ep,y,color=COL[m],lw=1.0 if m in('JS discriminator','reference critic') else 0.7,label=m)
ax.axhline(0.1,color='#555555',ls='--',lw=0.8); ax.text(150,0.108,'10% of own peak',fontsize=6,va='bottom',color='#555555')
ax.axvline(40,color='#AAAAAA',ls=':',lw=0.8); ax.text(44,0.98,'first 10%\nof epochs',fontsize=6,va='top',color='#777777')
ax.set_yscale('log'); ax.set_ylim(0.008,1.3); ax.set_yticks([0.01,0.1,1]); ax.set_yticklabels(['0.01','0.1','1'])
ax.set_xlabel('epoch'); ax.set_ylabel('adversary gradient norm / peak'); ax.set_xlim(0,400)
ax.set_title('discriminator gradient fades early',loc='left',fontsize=7); h,l=ax.get_legend_handles_labels(); fig.legend(h,l,fontsize=6,loc='lower center',bbox_to_anchor=(0.68,0.0),ncol=3,handlelength=1.4,columnspacing=1.2); panel_letter(ax,'b',dx=-0.3)
# (c) discriminator accuracy per epoch, per task
ax = axs[2]; accs=[]
for i,t in enumerate(TASKS8):
    acc = 1/(1+np.exp(-(ep-20)/8))*(base[i]-1/8)+1/8; acc = acc + rng.normal(0,0.01,400)
    ax.plot(ep,acc,lw=0.7,color=plt.cm.Greys(0.35+0.6*i/7)); accs.append(acc)
ax.axhline(1/8,color='#AAAAAA',ls=':',lw=0.8); ax.text(390,0.14,'chance (8 batches, example)',fontsize=6,ha='right',color='#777777')
ax.set_ylim(0,1.05); ax.set_xlim(0,400); ax.set_xlabel('epoch'); ax.set_ylabel('discriminator batch accuracy')
ax.set_title('accuracy saturates quickly',loc='left',fontsize=7)
ax.text(200,0.55,'one line per task (8 tasks),\nconditioned decoder, seed mean',fontsize=6,ha='center',color='#555555')
ax.annotate('human-mouse immune',xy=(300,accs[5][299]),xytext=(250,0.78),fontsize=6,arrowprops=dict(arrowstyle='-',lw=0.5),ha='center')
ax.annotate('simulation 2',xy=(330,accs[7][329]),xytext=(330,0.70),fontsize=6,arrowprops=dict(arrowstyle='-',lw=0.5),ha='center')
panel_letter(ax,'c',dx=-0.3)
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
fig, axs = plt.subplots(1,4,figsize=(W2,7.6*CM),gridspec_kw=dict(left=0.16,right=0.985,bottom=0.3,top=0.84,wspace=0.14))
rows = REAL6+['pooled (6 real tasks)']
titles = ['biology kept at $b^*$','trade-off area','collapse rate','local vs global removal']
xlabels = ['biology-score difference','area difference','percentage-point difference','score difference']
margins = [0.01,0.01,5,None]; scales=[0.03,0.04,8,0.03]
DCOL = {'conditioned decoder':'#0072B2','unconditioned decoder':'#E69F00'}
for p,ax in enumerate(axs):
    sc=scales[p]
    for j,dec in enumerate(DECODERS):
        off = 0.18 if j==0 else -0.18
        est = rng.normal(0.2*sc,0.6*sc,7); est[6]=est[:6].mean()
        ci = np.abs(rng.normal(1.0*sc,0.3*sc,7)); ci[6]=ci[:6].mean()/2.2
        for i in range(7):
            if p<2 and i==2 and j==0: # low-signal pair example: not evaluable
                ax.text(0,i+off,'n.e.',fontsize=6,ha='center',va='center',color=DCOL[dec]); continue
            ax.errorbar(est[i],i+off,xerr=ci[i],fmt='o' if i<6 else 'D',color=DCOL[dec],ms=3 if i<6 else 4.5,capsize=1.5,lw=0.8,label=dec if (p==0 and i==0) else None)
    ax.axvline(0,color='k',lw=0.7)
    if margins[p] is not None:
        ax.axvspan(-margins[p],margins[p],color='#DDDDDD',alpha=0.6,lw=0,zorder=0)
    ax.axhline(5.5,color='#999999',lw=0.6,ls=':')
    ax.set_ylim(-0.6,6.6); ax.set_yticks(range(7))
    ax.set_yticklabels(rows if p==0 else []); ax.invert_yaxis()
    ax.set_title(titles[p],loc='left',fontsize=7); ax.set_xlabel(xlabels[p])
    ax.set_xlim(-3.2*sc,3.2*sc)
    panel_letter(ax,'abcd'[p],dx=-0.8 if p==0 else -0.14,dy=1.02)
axs[0].text(0.011,-0.45,'$\\pm$0.01 margin',fontsize=6,ha='left',va='center',color='#555555')
axs[2].text(5.5,-0.45,'$\\pm$5 pp margin',fontsize=6,ha='left',va='center',color='#555555')
fig.suptitle('reference critic minus discriminator, per task and pooled',x=0.57,y=0.98,fontsize=8,ha='center'); fig.text(0.57,0.0,'favours discriminator  $\\leftarrow$   $\\rightarrow$  favours reference critic (a, b);   lower collapse with critic $\\leftarrow$ (c);   $\\rightarrow$ critic removes batch more locally (d)',fontsize=6,ha='center',va='bottom',color='#555555')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',bbox_to_anchor=(0.57,0.07),ncol=2,fontsize=6)
finish(fig,'fig4')


rng = np.random.default_rng(SEED+5)
fig, axs = plt.subplots(1,2,figsize=(W2,7.4*CM),gridspec_kw=dict(width_ratios=[1,1.0],left=0.07,right=0.985,bottom=0.3,top=0.9,wspace=0.7))
# (a) good-strength width in decades per method; points = task x decoder (8 tasks x 2), median tick
ax=axs[0]; PEN=['JS discriminator','reference critic','pooled critic','barycenter critic','MMD','Sinkhorn']
centre={'JS discriminator':1.0,'reference critic':1.4,'pooled critic':1.3,'barycenter critic':1.2,'MMD':1.8,'Sinkhorn':1.6}
for i,m in enumerate(PEN):
    for j,(dec,mk) in enumerate(zip(DECODERS,['o','s'])):
        w=np.clip(rng.normal(centre[m]-0.2*j,0.4,8),0,2.5)
        ax.plot(i+(-0.17 if j==0 else 0.17)+rng.normal(0,0.04,8),w,mk,ms=2.8,color=COL[m],alpha=0.7,mec='none')
    allw=np.clip(rng.normal(centre[m]-0.1,0.4,16),0,2.5); ax.plot([i-0.3,i+0.3],[np.median(allw)]*2,color='k',lw=1.2)
ax.plot([],[],'o',color='#777777',label='conditioned decoder (left)'); ax.plot([],[],'s',color='#777777',label='unconditioned decoder (right)')
ax.plot([],[],'-',color='k',lw=1.2,label='median'); ax.legend(fontsize=6,loc='upper left',bbox_to_anchor=(0.0,1.0),handlelength=1.3)
ax.set_xticks(range(6)); ax.set_xticklabels(PEN,rotation=40,ha='right',rotation_mode='anchor')
ax.set_ylabel('good-strength width (decades)'); ax.set_ylim(-0.3,3.0); ax.set_yticks([0,1,2,3])
ax.set_title('good-strength width, one point per task',loc='left',fontsize=7)
ax.text(5.45,0.0,'wider = less tuning needed',fontsize=6,ha='right',va='bottom',color='#555555')
panel_letter(ax,'a',dx=-0.2,dy=1.04)
# (b) follow-up variants minus plain discriminator, biology at matched strength
ax=axs[1]
variants=['discriminator + SN','discriminator + R1','discriminator + SN\n+ critic optimiser *','pooled critic, GP, 1 step','pooled critic, GP, 5 steps','pooled critic, GP, 10 steps','pooled critic, SN, 1 step *','pooled critic, SN, 5 steps']
DCOL={'conditioned decoder':'#0072B2','unconditioned decoder':'#E69F00'}
for j,dec in enumerate(DECODERS):
    off=0.18 if j==0 else -0.18
    est=rng.normal(0.005,0.015,8); ci=np.abs(rng.normal(0.02,0.006,8))
    ax.errorbar(est,np.arange(8)+off,xerr=ci,fmt='o',color=DCOL[dec],ms=3,capsize=1.5,lw=0.8,label=dec)
ax.axvline(0,color='k',lw=0.7); ax.axvspan(-0.01,0.01,color='#DDDDDD',alpha=0.6,lw=0,zorder=0)
ax.set_yticks(range(8)); ax.set_yticklabels(variants); ax.invert_yaxis(); ax.set_ylim(7.75,-0.75)
ax.set_xlim(-0.065,0.065); ax.set_xlabel('biology difference vs plain discriminator')
ax.set_title('follow-up: which ingredient moves biology?',loc='left',fontsize=7)
ax.text(0.011,7.45,'$\\pm$0.01 margin',fontsize=6,ha='left',va='center',color='#555555')
h,l=ax.get_legend_handles_labels(); fig.legend(h,l,fontsize=6,loc='lower center',bbox_to_anchor=(0.72,0.09),ncol=2)
ax.text(0.064,-0.45,'* arms of the confirmatory contrast',fontsize=6,ha='right',va='center',color='#555555')
fig.text(0.72,0.0,'$\\leftarrow$ favours plain discriminator      favours variant $\\rightarrow$',fontsize=6,ha='center',va='bottom',color='#555555')
panel_letter(ax,'b',dx=-0.62,dy=1.04)
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
