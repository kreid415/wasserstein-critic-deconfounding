#!/usr/bin/env python3
"""Exit 0 iff every tag of the shard's tags file has a latent or a recorded failure row in the pilot5_jhpce output folder."""
import csv, os, sys
R = "/fastscratch/myscratch/kreid/wcd/atac_resume_20261028"
OUT = "/fastscratch/myscratch/kreid/wcd/tier12/pilot5_jhpce"
n = sys.argv[1]
tags = [l.strip() for l in open(f"{R}/pilot5_jhpce_atac_large_resume_s{n}.tags") if l.strip() and not l.startswith("#")]
failed = {r["tag"] for r in csv.DictReader(open(f"{OUT}/failures.csv"))} & set(tags)
missing = [t for t in tags if not os.path.isfile(f"{OUT}/latents/{t}.npz") and t not in failed]
print(f"shard {n}: {len(tags)} tags, {len(tags)-len(missing)-len(failed)} with a latent, {len(failed)} recorded failures, {len(missing)} unresolved")
sys.exit(0 if not missing else 1)
