#!/usr/bin/env python
"""Delete embeddings that have no corresponding result row.

# WHY THIS IS A SCRIPT WITH A SAFETY GATE, NOT AN AD-HOC GLOB:
#   `save_embedding` runs BEFORE `full_metric_suite` in evaluate_config, so a config
#   currently inside its metric window (35-150 s) legitimately has an .npz and NO row
#   yet. Deleting "orphans" while the wave runs therefore destroys live embeddings. That
#   happened once: an ad-hoc cleanup removed the lam=0.01 latent of a shard that was
#   mid-flight, and the loss only surfaced 20 minutes later when the shard finished and
#   its row appeared with no file. Repairing it cost a re-run of that config.
#
#   So this refuses to run while any shard is in flight, and requires --force to override.
#   The recovery path if it ever does happen: drop the affected row from its shard CSV,
#   delete the shard's .done marker, and re-run -- the resume key includes d_coef, so only
#   that one config recomputes.
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wave_status import tag  # noqa: E402


def wave_is_running():
    """A shard is in flight if its log exists but its .done marker does not."""
    live = []
    for log in glob.glob("logs/wave/*.log"):
        t = os.path.basename(log)[:-4]
        if t == "_driver":
            continue
        if not os.path.exists(f"results/wave/{t}.done"):
            live.append(t)
    return live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="prune even while shards are in flight (WILL destroy live latents)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    live = wave_is_running()
    if live and not args.force:
        print(f"REFUSING: {len(live)} shard(s) in flight -- their in-progress embeddings have "
              f"no row yet and would be deleted as orphans:")
        for t in live[:8]:
            print(f"  {t}")
        print("Stop the wave first, or pass --force if you accept losing live latents.")
        return 1

    emb_root = os.environ.get("WCD_EMBED_OUT", os.path.join(os.path.abspath(".."), "embeddings"))
    want = set()
    for f in glob.glob("results/wave/*.csv"):
        for _, r in pd.read_csv(f).iterrows():
            want.add(f"{r['dataset']}/{tag(r)}")
    orphans = [p for p in glob.glob(os.path.join(emb_root, "*", "*.npz"))
               if f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)[:-4]}" not in want]

    print(f"rows->tags {len(want)}   orphan embeddings {len(orphans)}")
    for p in orphans[:10]:
        print(f"  {'would remove' if args.dry_run else 'removing'}: {os.path.basename(p)}")
    if not args.dry_run:
        for p in orphans:
            os.remove(p)
        print(f"removed {len(orphans)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
