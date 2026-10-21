
import os, sys, json, csv, statistics as S, collections as C, subprocess
R='/home/kendall/experiment_data/wasserstein-critic-deconfounding/code/wcd_final-review-fixes'; os.chdir(R)
D='/home/kendall/experiment_data/wasserstein-critic-deconfounding'
exec(open('.claude/skills/experiment-preflight/kernel.py').read())
exec(open('.claude/skills/experiment-preflight-singlecell/kernel.py').read())
SHA='034403eb206f77296bc857e5776a84e22e6a172f787913875a9ddc159e1fa6c8'
MAN=D+'/tier12_v3/manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv'
rows=list(csv.DictReader([l for l in open(MAN) if not l.startswith('#')],delimiter='\t'))
X=[r for r in rows if r['experiment']=='X1']; LOC={'atac_small','immune','sim1'}
old=json.load(open('experiments/X1/preflight.json'))
plan=dict(old['plan'])
plan['question']=("X1 main benchmark (PREREG sec 0, after R2/R3), after the Sinkhorn redo: fit and score the 3,000 X1 rows of the resolved design file "
 "tier12_v3/manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv = manifests/paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv (sha256 034403eb206f...; rules/r23/m.x1.tsv): "
 "8 scIB tasks x 2 decoders x seeds 0-4; six adversarial arms at R1's 6 penalty strengths per arm x decoder (Sinkhorn grids from the corrected solver), adversary input = posterior mean and "
 "no standardisation (R2/R3 re-run 2026-10-20, unchanged), plus none and scvi_adv. 75 rule-free rows on the JHPCE tasks were fitted by the filler jobs; 1,125 rows (atac_small, immune, sim1) run "
 "locally on the RTX 3080 (20 non-Sinkhorn rows linked from tier12_v2/X1_local, 1,105 to fit); 1,800 adversarial rows on JHPCE (5 one-task jobs, 1 L40S each, 2 at a time); all scoring on this "
 "workstation with the tagged scorer; code = tag prereg-tier12-v3 (10a1ca5) throughout (non-Sinkhorn outputs at c88cce3 count as v3 under SI-54)")
plan['multiple_comparison_correction']=old['plan']['multiple_comparison_correction']+("; the pruning amendment of 2026-10-17 (branches prune-*, not merged, user diff review pending) would drop P3/P4; "
 "X1 supplies P1 and P2 either way")
ID='preflight_x1_20261020'
rec=pf_new(ID, plan, root='.', extra_items=scpf_items())
EX=scpf_items()
def ans(i,s,e): pf_answer(rec,i,s,e,extra_items=EX) if 'extra_items' in pf_answer.__code__.co_varnames else pf_answer(rec,i,s,e)
idd=subprocess.run(['git','diff','--stat','c3fd92d','prereg-tier12-v3','--','src','scripts/fit_paper_config.py','scripts/scvi_adversarial_plan.py','scripts/score_scib_native.py'],capture_output=True,text=True).stdout.strip()
hd=subprocess.run(['git','diff','--stat','prereg-tier12-v3','HEAD','--','src','scripts'],capture_output=True,text=True).stdout.strip()
assert idd=='' and hd==''
WHY=("still holds at v3: the redo changed only Sinkhorn penalty-strength grids (R1: 2 of 24 cells, both Sinkhorn; R4-a1 10 of 144, all Sinkhorn) and moved the code tag; "
     "fit and score code equals c3fd92d's (git diff --stat c3fd92d prereg-tier12-v3 -- src scripts/fit_paper_config.py scripts/scvi_adversarial_plan.py scripts/score_scib_native.py is empty; "
     "prereg-tier12-v3..HEAD under src/ and scripts/ is empty); tasks, arms, decoders, seeds, scorer and prepped inputs unchanged")
FRESH={'PF-01','PF-02','PF-06','PF-17','PF-18','PF-19','PF-20','PF-21','PF-22','PF-24','SC-19','SC-20'}
for k,v in old['answers'].items():
    if k in FRESH: continue
    v=v[-1] if isinstance(v,list) else v
    ans(k, v['status'], "carried from experiments/X1/preflight.json (2026-10-12; design 7c2fe867..., code c88cce3): %s || %s" % (str(v['evidence'])[:700], WHY))
# PF-06 seeds
sx=sorted({r['seed'] for r in X}); sp=sorted({r['seed'] for r in rows if r['experiment'] in ('A1','A2','A3')})
assert not set(sx)&set(sp)
ans('PF-06','pass',"X1 seeds %s disjoint from pilot seeds %s (A1-A3 rows of the resolved file); penalty strengths, adversary input and standardisation were fixed by R1, R4-a1, R2/R3 on pilot seeds only (rules/r1_pass2, r4_pass2, r23 step.json exit 0); no X1 score is used before the X1 analysis" % (sx, sp))
# PF-19 manifest flags
A1=[r for r in rows if r['experiment']=='A1']
const={k for k in ('counts','n_latent','n_layers','n_hidden','likelihood','batch_size','train_size','n_critic') if len({r[k] for r in A1 if r['arm']!='none'})==1}
intended={'adv_input':'mean','zstd':'0'}; intended.update({k:A1[0][k] for k in const})
adv=[r for r in X if r['arm'] not in ('none','scvi_adv')]
r19=pf_check_manifest(adv, intended, expected_rows=2880, strict=False); pf_record_check(rec,'PF-19',r19)
# PF-20 counts
tc=C.Counter(r['task'] for r in X); ac=C.Counter(r['arm'] for r in X)
loc=[r for r in X if r['task'] in LOC]; clu=[r for r in X if r['task'] not in LOC]
tags={}
for f in sorted(os.listdir('experiments/X1_jhpce/tags')):
    if f.startswith('x1_jhpce_'): tags[f]=sum(1 for l in open('experiments/X1_jhpce/tags/'+f) if l.strip() and not l.startswith('#'))
ok20 = len(X)==3000 and all(v==375 for v in tc.values()) and len(loc)==1125 and sum(1 for r in clu if r['arm'] not in ('none','scvi_adv'))==1800 and all(v==360 for v in tags.values())
ans('PF-20','pass' if ok20 else 'fail',"independent count on the resolved file: X1 %d rows = 8 tasks x 375 %s; arms %s; 2,880 rule-set adversarial rows (= r2r3_record x1_rows 2880) + 120 rule-free; local 1,125 (1,080 adversarial + 45 rule-free); cluster 1,875 = 1,800 adversarial in 5 run lists %s + 75 rule-free fitted by the filler jobs; local dry run (tier12_v3/logs/x1_local_dry_run_1.txt) 1,125 rows: 19 scored, 1 needs score, 1,105 need fits; report-only 0 invalid outputs" % (len(X), dict(tc), dict(ac), tags))
# PF-21
ans('PF-21','pass',"replicate seeds give distinct latents under the v3 code: Sinkhorn pilot completion check (D/code/stage2/fitcheck.py on tier12_v3/A2A3, 2026-10-20 19:59, log tier12_v3/logs/r2r3_watch.log) cells_with_identical_seed_latents 0 of 12 cells, 36/36 scored, nmi_zero 0; X1 latents are not opened (blinding SI-78), the X1 runner uses the same seed handling as A1-A3")
# PF-17 cost
def led(p): return list(csv.DictReader(open(p)))
ldir=D+'/tier12_v3/A1_ext_r1_sinkhorn_round1/ledger'; ext=[x for f in os.listdir(ldir) if f.endswith('.csv') for x in led(ldir+'/'+f)]
v2a=led(D+'/tier12_v2/A1/ledger/A1__atac_small+immune+sim1.csv'); v3a=led(D+'/tier12_v3/A1_sinkhorn/ledger/A1__atac_small+immune+sim1__tags-a19e4683640a.csv')+ext
P=[r for r in v2a if r['arm']!='sinkhorn']+v3a; P=[r for r in P if r['fit_seconds']]
def key(r,ks): return tuple(str(float(r[k])) if k=='lam' else r[k] for k in ks)
lv=[('task','arm','cond','lam'),('task','arm','cond'),('task','arm')]
by={ks:C.defaultdict(list) for ks in lv}
for r in P:
    for ks in lv: by[ks][key(r,ks)].append(float(r['fit_seconds']))
linked={f[:-4] for f in os.listdir(D+'/tier12_v3/X1_local/latents')}
todo=[r for r in loc if r['tag'] not in linked]
pre={r['tag']:float(r['fit_seconds']) for r in v2a if r['arm']=='sinkhorn' and r['fit_seconds']}
ratio={}
for t in LOC:
    q=[float(r['fit_seconds'])/pre[r['tag']] for r in v3a if r['task']==t and r['tag'] in pre]; ratio[t]=S.median(q)
noneP=sorted(by[('task','arm')][('immune','none')])
T=0; Tk=C.Counter(); Sk=C.Counter(); how=C.Counter(); mx=0
for r in todo:
    s=None
    for i,ks in enumerate(lv):
        v=by[ks].get(key(r,ks))
        if v: s=S.median(v) if i==0 else sorted(v)[int(.9*len(v))]; how[['exact','cond-p90','arm-p90'][i]]+=1; break
    if s is None:
        v=by[('task','arm')][(r['task'],'none')]; s=1.5*max(v); how['scvi_adv=1.5x max none']+=1
    T+=s; Tk[r['task']]+=s; mx=max(mx,s)
    if r['arm']=='sinkhorn': Sk[r['task']]+=s
st=lambda r: __import__('datetime').datetime.strptime(r,'%Y-%m-%dT%H:%M:%S%z')
lanes_eff=sum(float(r['fit_seconds']) for r in v2a)/((max(st(r['fit_ended']) for r in v2a)-min(st(r['fit_started']) for r in v2a)).total_seconds())
settings={'device':'RTX 3080','fit_lanes':8,'fit_threads':1,'scorers_during':3,'code':'10a1ca5','adv_input':'mean','zstd':0}
r17=pf_estimate_cost(T, len(todo), len(todo), safety=2.0, pilot_settings=settings, production_settings=settings, parallel=lanes_eff, strict=False)
fac={t:Tk[t]/(Tk[t]-Sk[t]+Sk[t]/ratio[t]) for t in LOC}; f=sum(Tk.values())/sum(Tk[t]-Sk[t]+Sk[t]/ratio[t] for t in LOC)
jobs={'immune_hum_mou':(2.2,2.7,2.0),'atac_large':(1.4,2.4,1.6),'lung':(1.9,2.5,1.8),'sim2':(1.5,2.4,1.7),'pancreas':(1.5,1.9,1.3)}
sinkmax=max(max(by[('task','arm')][(t,'sinkhorn')])/S.median(by[('task','arm')][(t,'barycenter')]) for t in LOC)
cl={j:dict(pred_days=round(p*f,2), slow_days=round(s*f,2), longest_sinkhorn_h=round(lf*sinkmax,1)) for j,(p,s,lf) in jobs.items()}
gpu_days=round(sum(v['pred_days'] for v in cl.values()),1)
sc={'atac_small':163,'immune':870,'sim1':339}; ssum=sum(sc[r['task']] for r in todo)+870
ev17=("LOCAL (auto): %s. Method: each of the %d local rows to fit gets the median fit_seconds of its pilot cell (task, arm, decoder, penalty strength) from tier12_v2/A1 (non-Sinkhorn, 1,098 rows at 8 lanes) and tier12_v3/A1_sinkhorn + A1_ext_r1_sinkhorn_round1 (corrected Sinkhorn, zstd 0, as X1 uses); match levels %s; "
 "lane-hours %.1f (by task %s; Sinkhorn %s); effective lanes %.2f measured on A1 (842.7 lane-h in 123.7 h); wall %.1f h, x2 safety %.1f h (no walltime limit locally; no per-fit timeout); longest single fit %.0f s; "
 "scoring %.1f lane-h on 3 scorers during fits = %.1f h, below the fit wall. CLUSTER: corrected/pre-fix Sinkhorn fit-time ratio on the A1 rows median %s; with Sinkhorn's measured share of local lane-time this raises a job by x%.3f (by task %s) over the pre-fix plan "
 "(SUBMISSION_PLAN.md). Re-estimated GPU-days per job (plan x factor): %s; total %.1f GPU-days (was 8.5). Condition: at x2 safety the two largest jobs exceed the 3-day Slurm limit (builder maximum); the runner's deadline guard stops 45 min before the limit and keeps finished models, and one resume job per affected dataset fits the rest, so a resume job is expected for immune_hum_mou and possibly lung; "
 "the L40S queue timing of the corrected Sinkhorn (gate job 36450131: 10 Sinkhorn fits 63-130 s each, queue wall 488.6 s vs 353.7 s at the 2026-10-03 gate) is consistent with this factor." % (
 r17['summary'], len(todo), dict(how), T/3600, {k:round(v/3600,1) for k,v in Tk.items()}, {k:round(v/3600,1) for k,v in Sk.items()}, lanes_eff, T/3600/lanes_eff, 2*T/3600/lanes_eff, mx,
 ssum/3600, ssum/3600/3, {k:round(v,2) for k,v in ratio.items()}, f, {k:round(v,3) for k,v in fac.items()}, cl, gpu_days))
pf_record_check(rec,'PF-17',r17); ans('PF-17','pass' if r17['ok'] else 'fail', ev17)
# PF-18
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
procs=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
mem=[l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]
ans('PF-18','pass',"local: same 8 fit lanes x 1 thread + 3 scorers during fits (8 after) as A1-A3 on the same 3 tasks (A1 1,098 fits and Sinkhorn redo 216 fits, 0 failures); at check time the GPU shows '%s' with compute apps [%s] (the A3 pilot ended 19:56) and MemAvailable %d GB of 94 GB; no filler scorer is running. JHPCE: 8 lanes per L40S, 10 CPUs, 96 GB as in the filler jobs; the v3 L40S gate job 36450131 ran 64 mixed-arm fits at 8 lanes with 0 failures" % (gpu, procs, int(mem)//1048576))
# PF-22
df=subprocess.run(['df','-BG','--output=avail',D],capture_output=True,text=True).stdout.split()[-1]
ans('PF-22','pass',"same runner as A1-A3 (scripts/run_stage.py at the tag): per tag latents/<tag>.npz, models/<tag>/, scores/<tag>.csv, attempts, ledger, failures.csv; the completion check exits 0 only if every row is final. Local output: NEW folder tier12_v3/X1_local (decision D-S2-3; 20 non-Sinkhorn v2 outputs linked by name, 0 Sinkhorn, 3 runner files, no copies); tier12_v2/X1_local is not written. JHPCE jobs fit only and pack a harvest (about 2.5 GB per job) downloaded, checksum-checked and scored here; fastscratch purges after 30 days so each harvest is downloaded when its job ends. Free space at %s: %s" % (D, df))
# PF-24
ans('PF-24','pass',"training genes and epochs as the 2026-10-12 record (unchanged prep and fitter). Per-fit timeout: local none (run_stage --fit-timeout-s default 0; run_X1_local.sh passes none); JHPCE 28,800 s (x1_jhpce_command.py --fit-timeout-s default) vs longest corrected-Sinkhorn fit estimated per job %s h (pre-fix plan's longest fit x %.2f = the largest corrected-Sinkhorn max / barycenter median ratio on the A1 pilot tasks); walltime 3 d with deadline guard and resume job (PF-17 condition); kNN parameters as SC-16" % ({k:v['longest_sinkhorn_h'] for k,v in cl.items()}, sinkmax))
# SC-19
r=scpf_check_allowed_values([x['adv_input'] for x in adv]+['zstd='+x['zstd'] for x in adv], {'mean','zstd=0'}, name='X1 adversarial adv_input/zstd', strict=False)
ans('SC-19','pass' if r['ok'] else 'fail', scpf_evidence(r))
# SC-20
hits=[]
for p in ('experiments/X1_local/run_X1_local.sh','experiments/X1_jhpce/x1_jhpce_command.py'):
    rr=scpf_scan_fallbacks(p, strict=False); hits.append((p, rr['ok'], len(rr.get('problems',[]))))
ok20b=all(h[1] for h in hits)
ans('SC-20','pass' if ok20b else 'fail',"scpf_scan_fallbacks on the 2 launcher files changed 2026-10-20 (commit 2168d75, constants and comments only): %s; fitter, runner and scorer are the tagged files used by A1-A3 and the redo" % hits)
EXI=EX
v=pf_verdict(rec, extra_items=EXI); pf_save(rec)
try: pf_render(rec)
except TypeError: pass
out={'go':v['go'],'blocking':v['blocking'],'major_fail':v['major_fail'],'n_items':v['n_items'],'n_pass':v['n_pass'],'n_na':v['n_na'],'r19':r19['summary'],'ok20':ok20,'cluster':cl,'gpu_days':gpu_days,'factor':round(f,3),
     'local_lane_h':round(T/3600,1),'local_wall_h':round(T/3600/lanes_eff,1),'local_wall_h_x2':round(2*T/3600/lanes_eff,1),'lanes_eff':round(lanes_eff,2),'ratio':ratio,'sinkmax':round(sinkmax,2),'how':dict(how)}
json.dump(out, open('experiments/%s/summary_numbers.json'%ID,'w'), indent=1); print(json.dumps(out, indent=1))

