
import os, json
R='/home/kendall/experiment_data/wasserstein-critic-deconfounding/code/wcd_final-review-fixes'; os.chdir(R)
exec(open('.claude/skills/experiment-preflight/kernel.py').read()); exec(open('.claude/skills/experiment-preflight-singlecell/kernel.py').read())
EX=scpf_items(); rec=pf_load('preflight_x1_20261020', root='.')
tags={f:sum(1 for l in open('experiments/X1_jhpce/tags/'+f) if l.strip() and not l.startswith('#')) for f in sorted(os.listdir('experiments/X1_jhpce/tags')) if f.startswith('x1_jhpce_') and f.endswith('.tags')}
assert len(tags)==5 and all(v==360 for v in tags.values())
e=rec['answers']['PF-20']['evidence']
e=e.split(' in 5 run lists ')[0]+' in 5 run lists '+str(tags)+' + 75 rule-free'+e.split(' + 75 rule-free',1)[1]
e+=' (re-recorded: the first answer was a fail caused by the checking script counting experiments/X1_jhpce/tags/x1_jhpce_counts.csv as a run list; only *.tags files are run lists)'
pf_answer(rec,'PF-20','pass',e,extra_items=EX)
v=pf_verdict(rec, extra_items=EX); pf_save(rec); pf_render(rec)
print(json.dumps({k:v[k] for k in ('go','blocking','major_fail','minor_fail','unanswered','n_items','n_pass','n_na','plan_changed')}))

