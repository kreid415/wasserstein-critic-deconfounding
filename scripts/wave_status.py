#!/usr/bin/env python
"""One-shot health report for a running wave.

Checks the invariant that actually catches damage: every result ROW must have its
embedding on disk. Embeddings may EXCEED rows -- save_embedding runs before
full_metric_suite, so a config inside its metric window has an .npz but no row yet,
bounded by the lane count. Equality is not the invariant; zero missing is.
"""
import glob
import json
import os
import sys

import pandas as pd

WS = os.path.abspath("..")
EMB = os.environ.get("WCD_EMBED_OUT", os.path.join(WS, "embeddings"))
with open("configs/dataset_registry.json") as _fh:
    _REG = json.load(_fh)


def tag(r):
    """Rebuild the embed tag exactly as evaluate_config does."""
    opt = ""
    bs = r.get("batch_size", 1024)
    if pd.notna(bs) and int(bs) != 1024:
        opt += f"_bs{int(bs)}"
    lr = r.get("lr_g", 1e-3)
    if pd.notna(lr) and abs(float(lr) - 1e-3) > 1e-12:
        opt += f"_lr{str(float(lr)).replace('.', 'p').replace('-', 'm')}"
    # WHY THE REGISTRY, NOT THE ROW: result rows carry `batch_count` (the level actually
    # trained) but NOT `n_batches` (the dataset's full count), so comparing the two row
    # fields silently yields no suffix -- and the check then looks for the PLAIN filename,
    # which exists because E1/E2 wrote it, so a genuinely missing E8 latent passes
    # unnoticed. The full count has to come from the registry, exactly as
    # run_experiment.py passes it to evaluate_config as full_n_batches.
    bc = r.get("batch_count")
    full = _REG.get(r.get("dataset"), {}).get("n_batches")
    if pd.notna(bc) and full is not None and int(bc) != int(full):
        opt += f"_bc{int(bc)}"
    # E4's fixed_refN designs: must mirror evaluate_config exactly, or every refN>0 latent
    # is reported as an orphan and its row as missing. Omitted for reference_batch=0 (the
    # default) and for the discriminator, which has no reference batch.
    rb, rm = r.get("reference_batch"), r.get("reference_mode", "fixed")
    if rm == "fixed" and pd.notna(rb) and int(rb) != 0:
        opt += f"_ref{int(rb)}"
    return (f"{r['method']}_{r['backbone']}_lam{str(r['d_coef']).replace('.', 'p')}"
            f"_s{int(r['seed'])}_{r.get('reference_mode', 'fixed')}"
            f"_{r.get('formulation', 'reference')}{opt}")


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "scripts/wave_manifest.tsv"
    M = pd.read_csv(manifest, sep="\t")
    done = {os.path.basename(p)[:-5] for p in glob.glob("results/wave/*.done")}
    fs = [f for f in glob.glob("results/wave/*.csv")]
    frames, want = [], set()
    for f in fs:
        d = pd.read_csv(f)
        if not len(d):
            continue
        frames.append(d)
        ds = os.path.basename(f).split("_")[0]
        for _, r in d.iterrows():
            want.add(f"{ds}/{tag(r)}")
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    have = {f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)[:-4]}"
            for p in glob.glob(os.path.join(EMB, "*", "*.npz"))}

    budget = M.est_hours.sum()
    done_budget = M[M.tag.isin(done)].est_hours.sum()
    print(f"shards      {len(done)}/{len(M)}   ({done_budget:.1f}/{budget:.1f} budgeted worker-h = "
          f"{done_budget / budget * 100:.1f}%)")
    print(f"rows        {len(d)}   embeddings {len(have)}")
    print(f"MISSING emb {len(want - have)}   (must be 0)   in-flight {len(have - want)}")
    if len(d):
        failed = d["failed"].notna().sum() if "failed" in d.columns else 0
        print(f"kbet        {d.kbet.notna().sum()}/{len(d)}   failed rows {failed}")
        print(f"epochs      {d.epochs_run.min()}-{d.epochs_run.max()}   hit 500 ceiling: {(d.epochs_run >= 500).sum()}")
        print(f"experiments {d.groupby('experiment').size().to_dict()}")
        print(f"datasets    {sorted(d.dataset.unique()) if 'dataset' in d.columns else 'n/a'}")
    for m in sorted(want - have)[:5]:
        print(f"  MISSING: {m}")


if __name__ == "__main__":
    main()
