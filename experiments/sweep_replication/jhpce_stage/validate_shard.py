#!/usr/bin/env python3
"""Exit 0 iff every tag of the shard's tags file has a latent or a recorded failure row in the sweep_replication output folder."""
import csv, os, sys
R = "/fastscratch/myscratch/kreid/wcd/replication_20261102"
OUT = "/fastscratch/myscratch/kreid/wcd/tier12/sweep_replication"
n = sys.argv[1]
tags = [l.strip() for l in open(f"{R}/replication_s{n}.tags") if l.strip() and not l.startswith("#")]
fp = f"{OUT}/failures.csv"
failed = ({r["tag"] for r in csv.DictReader(open(fp))} if os.path.isfile(fp) else set()) & set(tags)
missing = [t for t in tags if not os.path.isfile(f"{OUT}/latents/{t}.npz") and t not in failed]
print(f"shard {n}: {len(tags)} tags, {len(tags)-len(missing)-len(failed)} with a latent, {len(failed)} recorded failures, {len(missing)} unresolved")
sys.exit(0 if not missing else 1)
