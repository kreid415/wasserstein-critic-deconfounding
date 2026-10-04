#!/usr/bin/env python
"""Fit ONE manifest row and write its latent + provenance (replaces scvi_adv_fit.py; code review N9).

Every design setting of a row comes from its manifest columns; a row that is missing a column, or still carries a
lambda grid INDEX instead of a frozen value, is refused. Settings that are code defaults rather than columns
(adversary width 128, critic Adam lr 1e-4 betas (0, 0.9), discriminator Adam lr 1e-3, lambda_GP 10, 10 barycenter
iterations, scvi-tools' generator optimiser and KL warm-up; code check CR-09) are recorded as RESOLVED values in each
latent's config under 'plan', read from the trained training plan(s) and their optimizers (resolved_plan).
scANVI and sysVI fits (X13) run under the same non-finite-loss guard as the adversarial plan (guarded_plan_class;
code check CR-06), so their divergences are recorded as 'diverged' too.

Env:  MANIFEST (tsv), TAG (row tag), PREPPED_DIR (prep_scib_task.py outputs), OUT_DIR, WCD_SRC
Out:  OUT_DIR/latents/<tag>.npz   z (posterior mean), batch, celltype, config json, history json
      OUT_DIR/models/<tag>/       trained scvi-tools model (decoder needed for marker/DE analyses)
      OUT_DIR/status/<tag>.json   only if the fit diverged (a training loss term became non-finite; NonFiniteLossError
                                  of the training plan): status 'diverged', epoch, step, detail, row, provenance; the
                                  process then exits with fit_outcome.EXIT_DIVERGED and writes no latent. A row with a
                                  status file is refused (delete it to refit). Any other error exits non-zero (1) and
                                  writes nothing: an infrastructure error, rerun by the caller (docs/PREREG.md sec. 1).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ["tag", "experiment", "task", "counts", "arm", "lam", "n_critic", "adv_input", "zstd", "cond",
            "decoder", "n_latent", "n_layers", "n_hidden", "likelihood", "batch_size", "max_epochs",
            "train_size", "seed", "reference", "extra"]
ADVERSARIAL = {"discriminator", "discriminator_sn", "reference", "reference_fixed", "pooled", "pooled_sn",
               "barycenter", "barycenter_sn", "reference_sn", "mmd", "mmd_ref", "sinkhorn",
               "discriminator_r1", "discriminator_ref"}
CPU_BASELINES = ("harmony", "scanorama", "pca")   # X13 CPU rows: scripts/run_cpu_baselines.py (SI-34..SI-36)
# every key of a row's `extra` must be consumed here; anything else is refused (fail-loud R4)
EXTRA_KEYS = {"subsample", "adv_width", "adv_lr", "bary_iter", "bary_warm_iter",
              "factorial_run",            # X12 run label: provenance only (the factors it encodes are row columns)
              "r1_gamma",                 # X6 discriminator_r1 (docs/SPECS_missing_arms.md section 1, SI-22)
              "sampler",                  # X8 'stratified' (section 3, SI-24)
              "iw"}                       # X3 'depletion_oracle' (section 4, SI-25)
# X3 (SI-41): every composition spec carries draw=X3_DRAW (x3_draw); X3_PERMUTATION_SEED seeds the ONE dose-independent
# permutation of a task's cells that every dose of the task draws from. Both are written into each X3 latent's config
# (subsample_info); a spec with another or no draw is refused.
X3_DRAW = "nested_v1"
X3_PERMUTATION_SEED = 40_000
X3_SPEC_KEYS = {"kind", "batch", "types", "deplete_pct", "n_cells", "draw"}


def resolve_reference(tag, value, batches, subsample_spec=None, select=None):
    """Reference batch of a row. 'auto' = the repo rule on the cells actually used (select()); a non-negative
    integer = index into the sorted batch names (X7 sweeps every reference); any other value = a batch name that
    must be among the fitted cells. An X3 composition row must name its reference (the dose-0 automatic choice,
    fixed for every dose) and the reference must not be the depleted batch (SI-31)."""
    names = sorted(map(str, batches))
    comp = bool(subsample_spec) and subsample_spec.get("kind") == "composition"
    if comp and (value == "auto" or value.isdigit()):
        raise ValueError(f"{tag}: an X3 composition row must name its reference batch (SI-31), got {value!r}")
    if value == "auto":
        return str(select())
    if value.isdigit():
        if int(value) >= len(names):
            raise IndexError(f"{tag}: reference index {value} but only {len(names)} batches")
        return names[int(value)]
    if value not in names:
        raise ValueError(f"{tag}: reference {value!r} is not a batch of the fitted cells ({names})")
    if comp and value == str(subsample_spec["batch"]):
        raise ValueError(f"{tag}: reference {value!r} is the depleted batch; X3 depletes a non-reference batch (SI-31)")
    return value


def check_extra(tag, arm, extra):
    """Refuse unknown or inconsistent `extra` keys before any data is read."""
    unknown = sorted(set(extra) - EXTRA_KEYS)
    if unknown:
        raise KeyError(f"{tag}: extra has keys the runner does not consume: {unknown}")
    if arm.endswith("_r1") != ("r1_gamma" in extra):
        raise ValueError(f"{tag}: r1_gamma must be given for *_r1 arms and only for them (arm {arm!r})")
    if "sampler" in extra and extra["sampler"] != "stratified":
        raise ValueError(f"{tag}: sampler must be 'stratified', got {extra['sampler']!r}")
    if "iw" in extra:
        if extra["iw"] != "depletion_oracle":
            raise ValueError(f"{tag}: iw must be 'depletion_oracle', got {extra['iw']!r}")
        if extra.get("subsample", {}).get("kind") != "composition":
            raise ValueError(f"{tag}: iw='depletion_oracle' needs an X3 composition subsample")


def group_counts(obs):
    """{batch: {cell type: number of cells}} of an obs table (str labels, pairs with at least one cell)."""
    out = {}
    for (b, c), n in obs[["batch", "celltype"]].astype(str).value_counts().items():
        out.setdefault(b, {})[c] = int(n)
    return out


def depletion_oracle_weights(a, spec, info):
    """X3 oracle importance weights (SI-25 target, exact under SI-41; lead decision 2026-10-03): the (batch, cell type)
    table w[b, y] = n_K0(b, y) / n_Kd(b, y), with K0 the dose-0 cells and K_d the cells of this dose (`a`), both drawn
    by subsample() (info['k0_counts'] is its record of K0). The losses use the weights self-normalised, so the
    weighted (batch, type) counts of K_d equal those of K0 exactly: every batch's dose-0 composition and the dose-0
    batch sizes are restored (Tachet des Combes et al. 2020, Eq. 4 on the joint label, target = dose 0). Pairs drawn
    by the refill but absent from K0 get weight 0. Without refill this is SI-25's 1 / keep fraction for the depleted
    pairs and 1 elsewhere. Defined for 0 < deplete_pct < 100 (doses 50/80/95): at dose 0 every weight is 1 and at
    dose 100 the depleted pairs are empty, so K0 cannot be restored.
    Returns ({batch: {cell type: w}} over the pairs present in a, {declared type: kept / K0 count})."""
    pct = spec["deplete_pct"]
    if not 0 < pct < 100:
        raise ValueError(f"importance weights are defined for 0 < deplete_pct < 100 (doses 50/80/95), got {pct}: "
                         f"at dose 0 every weight is 1 and at dose 100 the depleted pairs are empty")
    k0, kd = info["k0_counts"], group_counts(a.obs)
    lost = [(b, c) for b, row in k0.items() for c in row if kd.get(b, {}).get(c, 0) == 0]
    if lost:
        raise ValueError(f"(batch, cell type) pairs of K0 absent at deplete_pct={pct}: {lost[:5]}; their dose-0 counts "
                         f"cannot be restored")
    w = {b: {c: k0.get(b, {}).get(c, 0) / n for c, n in row.items()} for b, row in kd.items()}
    for b, row in k0.items():            # the defining property, checked on the realised counts
        for c, n0 in row.items():
            if abs(w[b][c] * kd[b][c] - n0) > 1e-9 * n0:
                raise AssertionError(f"weighted count of ({b}, {c}) is {w[b][c] * kd[b][c]}, K0 has {n0}")
    keep = {t: v["kept"] / v["h0"] for t, v in info["per_type"].items()}
    return w, keep


def load_row(manifest, tag):
    m = pd.read_csv(manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        raise KeyError(f"manifest lacks columns {missing}")
    r = m[m.tag == tag]
    if len(r) != 1:
        raise KeyError(f"tag {tag!r} matches {len(r)} rows")
    r = r.iloc[0].to_dict()
    empty = [c for c in REQUIRED if r[c] == ""]
    if empty:
        raise ValueError(f"{tag}: empty fields {empty}")
    return r


def subsample(adata, spec, seed, return_info=False):
    """Deterministic subsamples for X3 (composition shift; x3_draw) and X8 (number of batches).
    return_info=True also returns the realised X3 selection (x3_draw's info: per declared type the dose-0, dropped
    and kept counts, and the dose-0 (batch, cell type) counts that the importance weights restore), computed by the
    same code that selects the cells."""
    import anndata as ad  # noqa: F401
    # The subsample depends on the design cell (task, V, subset, dose) only, never on the training
    # seed, so every arm and seed of one design cell sees the same cells.
    obs = adata.obs
    if spec["kind"] == "batches":
        rng = np.random.default_rng(30_000 + 100 * spec["subset"] + spec["n_batches"])
        sizes = obs["batch"].value_counts()
        ok = sizes[sizes >= spec["n_cells"] // spec["n_batches"]].index.sort_values()
        if len(ok) < spec["n_batches"]:
            raise ValueError(f"only {len(ok)} batches have >= {spec['n_cells'] // spec['n_batches']} cells")
        pick = np.random.default_rng(20_000 + 100 * spec["subset"] + spec["n_batches"]).choice(
            np.asarray(ok), spec["n_batches"], replace=False)
        per = spec["n_cells"] // spec["n_batches"]
        idx = np.concatenate([rng.choice(np.where(obs["batch"].values == b)[0], per, replace=False) for b in sorted(pick)])
        if return_info:
            raise ValueError("return_info is defined for composition subsamples only")
        return adata[np.sort(idx)].copy()
    if spec["kind"] == "composition":
        idx, info = x3_draw(obs, spec)
        if return_info:
            return adata[idx].copy(), info
        return adata[idx].copy()
    raise ValueError(spec)


def x3_draw(obs, spec):
    """X3 cells of one dose, draw 'nested_v1' (SI-41; code check CR-04; lead decision 2026-10-03). Every dose of a
    task keeps the same number of cells N = spec['n_cells'] (the task's dose-100 size), drawn from ONE
    dose-independent permutation of the task's cells (default_rng(X3_PERMUTATION_SEED), common random numbers):
      K0   = the first N cells of the permutation (dose 0);
      dose d removes, for EACH declared type t, the first round(d/100 * h0_t) cells of type t in the declared batch
           among K0 (permutation order; h0_t = their number in K0), and refills the freed slots with the next
           non-declared cells of the permutation (cells that are not of a declared type in the declared batch).
    Every declared type keeps exactly h0_t - round(d/100 * h0_t) cells (none empties before dose 100), no K0 cell of
    another kind is removed, the kept declared cells and the refill are nested across doses, and two doses share
    the largest possible number of cells (N minus the difference of their declared-cell counts).
    Returns (sorted row indices into obs, info) with info = {draw, permutation_seed, n_cells, n_hit (declared cells
    of the whole task), per_type {t: {h0, dropped, kept}}, k0_counts {batch: {cell type: n}}}."""
    unknown = sorted(set(spec) - X3_SPEC_KEYS)
    if unknown:
        raise KeyError(f"composition subsample has keys x3_draw does not consume: {unknown}")
    if spec.get("draw") != X3_DRAW:
        raise ValueError(f"composition subsample needs draw={X3_DRAW!r} (SI-41), got {spec.get('draw')!r}: the "
                         f"pre-SI-41 X3 rows (one RNG draw per dose, unequal cell counts) are retired")
    b, pct, n = str(spec["batch"]), spec["deplete_pct"], int(spec["n_cells"])
    types = [str(t) for t in spec["types"]]
    if not 0 <= pct <= 100:
        raise ValueError(f"deplete_pct must be in [0, 100], got {pct}")
    if len(set(types)) != len(types) or not types:
        raise ValueError(f"declared types must be distinct and non-empty, got {types}")
    bat, ct = obs["batch"].astype(str).values, obs["celltype"].astype(str).values
    is_hit = (bat == b) & np.isin(ct, types)
    n_nonhit = int((~is_hit).sum())
    if not 0 < n <= n_nonhit:
        raise ValueError(f"n_cells={n} must be in (0, {n_nonhit}], the cells left at dose 100 (SI-41)")
    perm = np.random.default_rng(X3_PERMUTATION_SEED).permutation(len(obs))
    k0 = perm[:n]
    kept_decl, per_type = [], {}
    for t in types:
        h0 = k0[is_hit[k0] & (ct[k0] == t)]                 # declared cells of type t in K0, permutation order
        if len(h0) == 0:
            raise ValueError(f"declared type {t!r} of batch {b!r} has no cell among the dose-0 cells: nothing to deplete")
        nd = int(round(len(h0) * pct / 100))
        kept_decl.append(h0[nd:])
        per_type[t] = dict(h0=int(len(h0)), dropped=nd, kept=int(len(h0) - nd))
    kept_decl = np.concatenate(kept_decl)
    fill = perm[~is_hit[perm]][:n - len(kept_decl)]        # K0's non-declared cells, then the next ones
    idx = np.sort(np.concatenate([kept_decl, fill]))
    if len(idx) != n or len(np.unique(idx)) != n:
        raise AssertionError(f"selected {len(idx)} cells ({len(np.unique(idx))} distinct), expected {n}")
    info = dict(draw=X3_DRAW, permutation_seed=X3_PERMUTATION_SEED, n_cells=n, n_hit=int(is_hit.sum()),
                per_type=per_type, k0_counts=group_counts(obs.iloc[np.sort(k0)]))
    return idx, info


def provenance():
    import scvi
    import torch
    sha = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip() != ""
    return dict(git_sha=sha, git_dirty=dirty, scvi=scvi.__version__, torch=torch.__version__,
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))


def main():
    r = load_row(os.environ["MANIFEST"], os.environ["TAG"])
    out_dir = os.environ["OUT_DIR"]
    npz = os.path.join(out_dir, "latents", f"{r['tag']}.npz")
    status = os.path.join(out_dir, "status", f"{r['tag']}.json")
    if os.path.exists(status):
        raise FileExistsError(f"{status} exists: {r['tag']} already has a recorded outcome (delete it to refit)")
    if os.path.exists(npz):
        print(f"[skip] {r['tag']} exists", flush=True)
        return
    import anndata as ad
    import scvi
    import torch
    import importlib.util
    spec = importlib.util.spec_from_file_location("wcd_data", os.path.join(os.environ["WCD_SRC"], "wcd_vae", "wcd", "data.py"))
    wcd_data = importlib.util.module_from_spec(spec); spec.loader.exec_module(wcd_data)
    select_reference_batch = wcd_data.select_reference_batch   # the repo's rule, loaded without the package __init__

    seed = int(r["seed"])
    arm = r["arm"]
    if arm in CPU_BASELINES:
        raise ValueError(f"{r['tag']}: {arm} is an X13 CPU baseline; run it with scripts/run_cpu_baselines.py "
                         f"--manifest <manifest> --tag {r['tag']}")
    extra = json.loads(r["extra"])
    check_extra(r["tag"], arm, extra)
    if arm in ("scanvi", "sysvi") and {"r1_gamma", "sampler", "iw"} & set(extra):
        raise ValueError(f"{r['tag']}: r1_gamma / sampler / iw are adversary options, not defined for {arm}")
    try:
        lam = float(r["lam"])
    except ValueError:
        raise ValueError(f"{r['tag']}: lam={r['lam']!r} is a grid index; freeze the lambda grid first")
    if arm in ADVERSARIAL and lam <= 0:
        raise ValueError(f"{r['tag']}: adversarial arm with lam={lam}")

    a = ad.read_h5ad(os.path.join(os.environ["PREPPED_DIR"], f"{r['task']}__{r['counts']}.h5ad"))
    iw_weights, iw_keep, sub_info = None, None, None
    if "subsample" in extra and extra["subsample"].get("kind") == "composition":
        a, sub_info = subsample(a, extra["subsample"], seed, return_info=True)
        if "iw" in extra:
            iw_weights, iw_keep = depletion_oracle_weights(a, extra["subsample"], sub_info)
    elif "subsample" in extra:
        a = subsample(a, extra["subsample"], seed)
    a = a[:, a.var["highly_variable"].values].copy()
    a.obs["batch"] = a.obs["batch"].astype(str).astype("category")

    # reference batch: 'auto' = repo rule (max cell-type entropy, ties by size) on the cells actually used;
    # an index (X7) or a batch name (X3, fixed at the dose-0 choice, SI-31)
    ref_name = resolve_reference(r["tag"], r["reference"], a.obs["batch"].cat.categories, extra.get("subsample"),
                                 select=lambda: select_reference_batch(a, "batch", "celltype"))

    common = dict(n_latent=int(r["n_latent"]), max_epochs=int(r["max_epochs"]), batch_size=int(r["batch_size"]),
                  seed=seed, conditioned=bool(int(r["cond"])))
    backbone = dict(n_layers=int(r["n_layers"]), n_hidden=int(r["n_hidden"]), gene_likelihood=r["likelihood"],
                    train_size=float(r["train_size"]))
    t0 = time.time()
    if arm in ("scanvi", "sysvi"):
        z, model, plan_rec = fit_baseline(a, arm, lam, r, common, backbone)
    else:
        from scvi_adversarial_plan import fit_adversarial_scvi
        n_critic = int(r["n_critic"])
        z, model = fit_adversarial_scvi(
            a, "batch", adversary=arm, d_coef=lam, n_critic=(n_critic if n_critic > 0 else None),
            reference_batch=ref_name, adv_input=r["adv_input"], zstd=bool(int(r["zstd"])),
            model_name=r["decoder"], adv_hidden=int(extra.get("adv_width", 128)),
            critic_lr=float(extra.get("adv_lr", 1e-4)), disc_lr=float(extra.get("adv_lr", 1e-3)),
            bary_iter=int(extra.get("bary_iter", 10)), bary_warm_iter=extra.get("bary_warm_iter"),
            r1_gamma=(None if "r1_gamma" not in extra else float(extra["r1_gamma"])),
            sampler=extra.get("sampler"), iw_weights=iw_weights,
            **common, **backbone)
        plan_rec = {"fit": resolved_plan(model)}
    secs = time.time() - t0
    hist = {k: v.iloc[:, 0].astype(float).tolist() for k, v in getattr(model, "history_", {}).items()
            if hasattr(v, "iloc")}
    cfg = dict(row=r, reference_name=ref_name, n_cells=int(a.n_obs), n_batches=int(a.obs["batch"].nunique()),
               fit_seconds=round(secs, 1), iw_keep_fraction=iw_keep, iw_weights=iw_weights, subsample_info=sub_info,
               plan=plan_rec, **provenance())
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, z=np.asarray(z, dtype=np.float32), obs_names=a.obs_names.to_numpy(dtype="U128"),
                        batch=a.obs["batch"].astype(str).to_numpy(dtype="U64"),
                        celltype=a.obs["celltype"].astype(str).to_numpy(dtype="U64"),
                        config=json.dumps(cfg), history=json.dumps(hist))
    model.save(os.path.join(out_dir, "models", r["tag"]), overwrite=True, save_anndata=False)
    os.replace(tmp, npz)
    print(f"[fit] {r['tag']} {secs:.0f}s z{np.asarray(z).shape} finite={bool(np.isfinite(z).all())}", flush=True)


def fit_baseline(a, arm, lam, r, common, backbone):
    """X13 neural baselines. Returns (posterior-mean latent, model, {phase: resolved_plan}). Every training phase runs
    under guarded_plan_class (one step counter over the whole fit); the latent is read with torch.distributions
    validation off (scvi_adversarial_plan._posterior_mean), so a NaN latent is saved and recorded as nonfinite_latent."""
    import scvi
    from scvi.train import SemiSupervisedTrainingPlan, TrainingPlan
    from scvi_adversarial_plan import _posterior_mean
    scvi.settings.seed = common["seed"]
    counter = {"step": -1}
    if arm == "scanvi":
        # scIB's scANVI protocol (scib 1.1.7 integration.scanvi): scVI with the same backbone, then
        # SCANVI.from_scvi_model trained min(10, max(2, round(epochs / 3))) epochs, at the backbone's train_size.
        s = a.copy(); s.X = s.layers["counts"].copy()
        scvi.model.SCVI.setup_anndata(s, batch_key="batch", labels_key="celltype")
        vae = scvi.model.SCVI(s, n_latent=common["n_latent"], n_layers=backbone["n_layers"],
                              n_hidden=backbone["n_hidden"], gene_likelihood=backbone["gene_likelihood"])
        vae._training_plan_cls = guarded_plan_class(vae._training_plan_cls, TrainingPlan, counter, "scvi pretraining")
        vae.train(max_epochs=common["max_epochs"], batch_size=common["batch_size"], train_size=backbone["train_size"],
                  early_stopping=False, enable_progress_bar=False)
        pre_plan = resolved_plan(vae)
        m = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="UnknownUnknown")
        m._training_plan_cls = guarded_plan_class(m._training_plan_cls, SemiSupervisedTrainingPlan, counter, "scanvi")
        m.train(max_epochs=int(min(10, max(2, round(common["max_epochs"] / 3.0)))), batch_size=common["batch_size"],
                train_size=backbone["train_size"], early_stopping=False, enable_progress_bar=False)
        return _posterior_mean(m), m, {"scvi_pretraining": pre_plan, "scanvi": resolved_plan(m)}
    # sysVI (scvi-tools 1.4.2 scvi.external.SysVI): Gaussian likelihood on scIB's normalised X,
    # VampPrior (default), strength knob = z_distance_cycle_weight (= lam; scvi-tools default 2.0).
    from scvi.external import SysVI
    s = a.copy()
    SysVI.setup_anndata(s, batch_key="batch")
    m = SysVI(s, n_latent=common["n_latent"], n_hidden=backbone["n_hidden"], n_layers=backbone["n_layers"])
    m._training_plan_cls = guarded_plan_class(m._training_plan_cls, TrainingPlan, counter, "sysvi")
    m.train(max_epochs=common["max_epochs"], batch_size=common["batch_size"], train_size=backbone["train_size"],
            early_stopping=False, enable_progress_bar=False,
            plan_kwargs=dict(z_distance_cycle_weight=lam))   # TrainingPlan forwards extra kwargs to SysVAE.loss
    return _posterior_mean(m), m, {"fit": resolved_plan(m)}


def guarded_plan_class(base, expected, counter, phase):
    """scvi-tools training plan `base` (must be `expected`) under the non-finite-loss guard of
    WassersteinAdversarialTrainingPlan (docs/PREREG.md section 1; code check CR-06): every training step checks that
    its loss is finite, a forward pass that torch.distributions refuses for a non-finite parameter counts as a
    non-finite loss, and every parameter must be finite after the last optimizer step; each raises
    NonFiniteLossError (epoch of this phase, step = 0-based minibatch index over the whole fit, counted in
    counter['step'] across phases). The guard only reads values: losses, gradients and latents are those of the
    stock plan (tests/scvi/test_baseline_guard.py checks bit identity against the tagged fitter)."""
    import torch
    from fit_outcome import NonFiniteLossError
    from scvi_adversarial_plan import _is_distribution_validation_error
    if base is not expected:
        raise TypeError(f"{phase}: training plan {base.__name__}, expected {expected.__name__} (scvi-tools changed?)")

    class Guarded(base):
        def _nonfinite_parameters(self):
            return [n for n, p in self.named_parameters() if not bool(torch.isfinite(p).all())]

        def forward(self, *args, **kwargs):
            try:
                return super().forward(*args, **kwargs)
            except ValueError as e:
                if not _is_distribution_validation_error(e):
                    raise
                bad = self._nonfinite_parameters()
                raise NonFiniteLossError(
                    self.current_epoch, counter["step"], {},
                    detail=f"{phase}: forward pass refused a non-finite distribution parameter: "
                           f"{str(e).splitlines()[0]} non-finite model parameters: {bad[:5] if bad else 'none'}") from e

        def training_step(self, batch, batch_idx):
            counter["step"] += 1
            loss = super().training_step(batch, batch_idx)
            value = loss.detach().float().reshape(())
            if not bool(torch.isfinite(value)):
                raise NonFiniteLossError(self.current_epoch, counter["step"], {"train_loss": float(value)},
                                         detail=f"{phase}: minibatch {batch_idx} of epoch {self.current_epoch}")
            return loss

        def on_train_end(self):
            super().on_train_end()
            bad = self._nonfinite_parameters()
            if bad:
                raise NonFiniteLossError(self.current_epoch, counter["step"], {},
                                         detail=f"{phase}: non-finite parameters after the last optimizer step: {bad[:5]}")

    Guarded.__name__ = Guarded.__qualname__ = f"Guarded{base.__name__}"
    return Guarded


PLAN_FIELDS = ("optimizer_name", "lr", "eps", "weight_decay", "n_epochs_kl_warmup", "n_steps_kl_warmup",
               "max_kl_weight", "min_kl_weight")
ADVERSARY_FIELDS = ("adversary", "d_coef", "adv_steps", "adv_hidden", "critic_lr", "critic_betas", "disc_lr",
                    "bary_iter", "bary_warm_iter", "bary_weights", "adv_input", "zstd", "spectral_norm", "r1_gamma")


def _plain(v):
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def resolved_plan(model):
    """The settings the trained plan actually used (code check CR-09): scvi-tools' generator optimiser and KL warm-up,
    the adversary settings of WassersteinAdversarialTrainingPlan, lambda_GP where the gradient penalty applies (the
    default of wcd critic.multi_class_gradient_penalty, which the head calls without lambda_gp), and every optimizer's
    param groups (lr, betas, eps, weight_decay, amsgrad), read from model.trainer after training."""
    import inspect
    plan = model.trainer.lightning_module
    rec = {"plan_class": type(plan).__name__}
    rec.update({k: _plain(getattr(plan, k)) for k in PLAN_FIELDS})
    if hasattr(plan, "adversary_base"):
        rec.update({k: _plain(getattr(plan, k)) for k in ADVERSARY_FIELDS})
        gp = bool(plan.is_critic and not plan.spectral_norm)
        rec["gradient_penalty"] = gp
        rec["lambda_gp"] = None
        if gp:
            fn = sys.modules[type(plan._wcd_head).__module__].multi_class_gradient_penalty
            rec["lambda_gp"] = float(inspect.signature(fn).parameters["lambda_gp"].default)
    rec["optimizers"] = [dict(cls=type(o).__name__,
                              param_groups=[{k: _plain(g[k]) for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")
                                             if k in g} for g in o.param_groups])
                         for o in model.trainer.optimizers]
    return rec


def record_divergence(manifest, out_dir, tag, exc):
    """Write OUT_DIR/status/<tag>.json for a NonFiniteLossError (status 'diverged', docs/PREREG.md section 1) with
    the epoch, step, loss terms, manifest row and provenance; returns fit_outcome.EXIT_DIVERGED."""
    import fit_outcome
    r = load_row(manifest, tag)
    path = fit_outcome.write_status(out_dir, tag, "diverged", detail=str(exc), row=r, epoch=exc.epoch,
                                    step=exc.step, terms=exc.terms, error=repr(exc.__cause__ or exc), **provenance())
    print(f"[diverged] {tag}: {exc} -> {path}", flush=True)
    return fit_outcome.EXIT_DIVERGED


if __name__ == "__main__":
    from fit_outcome import NonFiniteLossError
    try:
        main()
    except NonFiniteLossError as e:     # the fit diverged: an outcome, recorded; every other error propagates
        sys.exit(record_divergence(os.environ["MANIFEST"], os.environ["OUT_DIR"], os.environ["TAG"], e))
