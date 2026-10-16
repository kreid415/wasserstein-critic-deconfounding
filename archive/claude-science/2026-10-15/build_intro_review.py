# build_intro_review.py
# Builds the Introduction review report, the flat ledger entries and the claims table.
# Run inside the python kernel that has the claim-gate helpers (cg_readback, cg_render, cg_receipt, cg_reconcile):
#   exec(open('intro_review/build_intro_review.py').read())
# Inputs (all under intro_review/): findings_data.py, tables_data.py, intro_checks_results.json (from intro_checks.py),
# gap_search_screen.csv, gap_forward_citations.csv, gap_search_numbers.json; work/rv = checkout of add2f73.
import json, re, os, sys
import pandas as pd
sys.path.insert(0, 'intro_review')
import importlib, findings_data as fd, tables_data as td
importlib.reload(fd); importlib.reload(td)

WS = 'intro_review'
RES = f'{WS}/intro_checks_results.json'
res = json.load(open(RES))
gs = pd.read_csv(f'{WS}/gap_search_screen.csv')
gf = pd.read_csv(f'{WS}/gap_forward_citations.csv')
gn = json.load(open(f'{WS}/gap_search_numbers.json'))
paper = open('work/rv/paper/paper.tex', encoding='utf-8').read().split('\n')
FIND = fd.FINDINGS
sev = {s: [f['id'] for f in FIND if f['severity'] == s] for s in ('blocker', 'fix', 'note')}

# ---- derived scalar values, written to a file so that every number in the report is read back from it ----
blocks = res['r4_a1_blocks']
setters = [b['setter'] for b in blocks.values()]
queries = sorted({k.split('|', 1)[1] for k in gn['coverage']})
rated = gs['overall'].isin(['none', 'partial', 'substantial'])
dl_ids = sorted(set(re.findall(r'\bC(\d{2})\b', '\n'.join(paper[184:216]))))
all_ids = [f'{i:02d}' for i in range(1, 26)]
x6 = res['x6_arm_steps_rows']; assert len(set(x6.values())) == 1
x13 = res['x13_matches_x1_epochs_batch_latent']
intro_txt = '\n'.join(paper[113:182])
cite_cmds = re.findall(r'\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]+)\}', intro_txt)
cite_keys = [k.strip() for c in cite_cmds for k in c.split(',')]
conf_class = lambda v: 'incomplete' if v.startswith('Resolution incomplete') else ('outside' if v.startswith('Outside') else 'ok')
cc = [conf_class(c[4]) for c in td.CONFLICTS]
ca = pd.read_csv(f'{WS}/original_manuscript_claims_audit.csv')
_claim_status = dict(zip(ca['id'], ca['status_now']))
_smap = {'partially supported': 'partially_supported', 'untested': 'untested', 'invalidated': 'invalidated_by_bugs', 'contradicted': 'contradicted'}
_dl = ' '.join(l.lstrip('% ').strip() for l in paper[184:216])
_pairs = []
for _m in re.finditer(r'\(?(C\d{2})((?:,\s*C\d{2})*)\s+(partially supported|untested|invalidated|contradicted)(?:;\s*(C\d{2})\s+(untested|invalidated|contradicted|partially supported))?', _dl):
    _pairs.append((_m.group(1), _m.group(3)))
    if _m.group(4):
        _pairs.append((_m.group(4), _m.group(5)))
n_status_checked = len(_pairs)
n_status_mismatch = sum(1 for c, s_ in _pairs if _claim_status[c] != _smap[s_])
vals = dict(
    manifest_sha=res['manifest_sha256_prefix'], manifest_rows=res['manifest_rows'], x1_rows=res['x1_rows'],
    scvi_adv_rows=res['scvi_adv_rows'],
    scvi_adv_adv_input_values=', '.join(f"'{v}'" for v in res['scvi_adv_adv_input_values']),
    immune1_diff=blocks['immune|1']['diff'], sim1_1_diff=blocks['sim1|1']['diff'],
    r4_git=res['r4_a1_git_sha'][:7], r4_extensions=res['r4_a1_extensions'],
    r4_bary_blocks=setters.count('barycenter'), r4_mmd_blocks=setters.count('mmd'),
    r4_sink_blocks=setters.count('sinkhorn'), r4_blocks_total=len(setters),
    x1_batch=', '.join(map(str, res['x1_batch_sizes'])), x1_latent=', '.join(map(str, res['x1_n_latent'])),
    x13_match=('all %d tasks, for sysVI and for scANVI' % len(next(iter(x13.values())))
               if all(all(d.values()) for d in x13.values()) else 'NOT ALL MATCH'),
    x6_rows_each=next(iter(x6.values())),
    gap_queries=len(queries), gap_raw=gn['raw_total'], gap_unique=len(gs), gap_forward=len(gf),
    gap_rated=int(rated.sum()), gap_rated_none=int((gs['overall'] == 'none').sum()),
    gap_rated_partial=int((gs['overall'] == 'partial').sum()),
    gap_rated_substantial=int((gs['overall'] == 'substantial').sum()), gap_not_rated=int((~rated).sum()),
    fwd_scibe=int((gf['cites'] == 'scIB-E').sum()), fwd_sysvi=int((gf['cites'] == 'sysVI').sum()),
    fwd_sccraft=int((gf['cites'] == 'scCRAFT').sum()), fwd_respan=int((gf['cites'] == 'ResPAN').sum()),
    droplist_listed=len(dl_ids), droplist_missing=', '.join('C' + i for i in all_ids if i not in dl_ids),
    droplist_ids=', '.join('C' + i for i in dl_ids),
    n_blocker=len(sev['blocker']), n_fix=len(sev['fix']), n_note=len(sev['note']), n_total=len(FIND),
    n_claims=len(td.CLAIMS), n_cite_cmds=len(cite_cmds), n_cite_inst=len(cite_keys), n_cite_keys=len(set(cite_keys)),
    n_src=len(re.findall(r'\\src\{', intro_txt)), n_pending=len(re.findall(r'\\pending\{', intro_txt)),
    n_tag_lines=len(td.STATUS), n_conflicts=len(td.CONFLICTS), n_conf_ok=cc.count('ok'),
    n_conf_incomplete=cc.count('incomplete'), n_conf_outside=cc.count('outside'),
    n_droplist_rows=len(td.DROPLIST), n_edits=len(td.EDITS),
    n_status_checked=n_status_checked, n_status_mismatch=n_status_mismatch,
)
def _count_lines(fp, pat):
    return sum(1 for l in open(fp, encoding='utf-8').read().split('\n') if re.search(pat, l, re.I))
vals['ledger_mmd_lines'] = _count_lines('work/rv/CONSTRAINTS.md', r'\bmmd\b|sinkhorn')
vals['spec_mmd_lines'] = _count_lines('work/rv/docs/ANALYSIS_SPEC.md', r'\bmmd\b|sinkhorn')
vals['prereg_mmd_lines'] = _count_lines('work/rv/docs/PREREG.md', r'\bmmd\b|sinkhorn')
cck = json.load(open(f'{WS}/compile_check.json'))
for k in ('lines_replaced', 'lines_inserted', 'tex_errors', 'undefined_citations_or_references', 'underfull_original', 'underfull_patched'):
    vals[k] = cck[k]
json.dump(vals, open(f'{WS}/report_values.json', 'w'), indent=1)
V = {k: cg_readback(f'{WS}/report_values.json', {'key': k}) for k in vals}

def render(text):
    phs = set(re.findall(r'\{\{([^}:]+)(?::[^}]*)?\}\}', text))
    r = cg_render(text, {k: V[k] for k in phs}, allow_unused=True)
    assert str(r['ok']) == 'True' and not r['problems'], (r['problems'], text[:120])
    out = r['text']
    assert '{{' not in out, out[:200]
    return out

def cell(s):
    return render(str(s)).replace('|', '\\|').replace('\n', ' ')

# ---- findings ----
def finding_md(f):
    L = [f"### {f['id']} ({f['severity']}) - {render(f['location'])}", '',
         f"**Problem.** {render(f['problem'])}", '',
         f"**Evidence.** {render(f['evidence'])}", '',
         f"**Recommended fix.** {render(f['fix'])}"]
    for r in f['rep']:
        L += ['', f"Replacement text for paper.tex, {r['target']}:", '', '```latex', render(r['text']), '```']
    return '\n'.join(L)

def suggested_fix(f):
    s = render(f['fix'])
    for r in f['rep']:
        s += f"\n\n[paper.tex, {r['target']}]\n{render(r['text'])}"
    return s

ledger = [dict(id=f['id'], severity=f['severity'], status='open', track='intro_review', reviewed_commit='add2f73',
               first_seen='2026-10-15', location=render(f['location']), problem=render(f['problem']),
               evidence=render(f['evidence']), suggested_fix=suggested_fix(f)) for f in FIND]
out_findings = [dict(id=e['id'], severity=e['severity'], location=e['location'], problem=e['problem'],
                     evidence=e['evidence'], suggested_fix=e['suggested_fix']) for e in ledger]

# ---- tables ----
def table(header, rows):
    L = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        L.append('| ' + ' | '.join(cell(c) for c in r) + ' |')
    return '\n'.join(L)

claims_md = table(['id', 'paper.tex line', 'claim', 'source cited', 'what was read', 'verdict'], td.CLAIMS)
conf_md = table(['#', 'conflict (PAPER_PLAN, July text or earlier review)', 'resolution in the outline', 'what I checked', 'verdict'], td.CONFLICTS)
drop_md = table(['lines', 'item dropped or softened', 'audit status (CSV)', 'what I checked', 'verdict'], td.DROPLIST)
edit_md = table(['#', 'where', 'edit by the lead', 'what I checked', 'verdict'], td.EDITS)
stat_md = table(['paper.tex line', 'tags on the line', 'classification check', 'finding'],
                [(f':{k}',) + v for k, v in sorted(td.STATUS.items())])
claims_csv = pd.DataFrame([[render(str(c)) for c in row] for row in td.CLAIMS],
                          columns=['id', 'paper_tex_line', 'claim', 'source_cited', 'what_was_read', 'verdict'])
claims_csv.to_csv(f'{WS}/intro_claims_checked.csv', index=False)

sev_rows = [(s, str(len(sev[s])), ', '.join(sev[s]) or 'none') for s in ('blocker', 'fix', 'note')]
sev_md = table(['severity', 'count', 'ids'], sev_rows)

TERMS = [
 ('paper.tex:N', 'line N of paper/paper.tex at commit add2f73 (the outline the paper is written from)'),
 ('source tag, pending marker', r'\src{...} = locked statement with its source; \pending{what}{when it locks} = not locked; author-year \citep = literature claim checked against the paper'),
 ('SI-nn', 'entry nn of the decisions ledger CONSTRAINTS.md (a decision by the user)'),
 ('X1, X3, X6, X7, X8, X12, X13, X15', 'experiments: X1 = core benchmark (3,000 model fits); X3 = cell types removed from one batch at rising doses; X6 = regularisation and critic-step follow-up; X7 = choice of reference batch; X8 = number of batches; X12, X15 = robustness checks; X13 = standard integration tools'),
 ('A1, A2, A3', 'pilot experiments: A1 = penalty strengths; A2 = adversary input (posterior sample instead of mean); A3 = per-dimension standardisation of that input'),
 ('rules R1-R4; results sections R0-R5', 'rules R1-R4 = pre-registered decision rules (R1 strengths, R2 adversary input, R3 standardisation, R4 matched strengths); R0-R5 = results sections of the paper (R0 = premise check)'),
 ('P1-P4', 'primary endpoints; P1 = biology score at matched batch removal, reference critic minus discriminator'),
 ('b0, b_common, low-signal block', 'b0 = batch score of plain scVI; b_common = batch score that every adversarial arm reaches (the lowest of the arms\' highest); low-signal block = one task with one decoder where b_common - b0 < 0.02'),
 ('JS, W1, KL, MMD, OT', 'Jensen-Shannon divergence; Wasserstein-1 distance; Kullback-Leibler divergence; maximum mean discrepancy; optimal transport'),
 ('GAN, WGAN-GP, HSIC, GLUE', 'generative adversarial network; Wasserstein GAN with gradient penalty; Hilbert-Schmidt independence criterion; GLUE (graph-linked embedding, as named in the sysVI reference list)'),
 ('scVI, scIB, sysVI, scIB-E', 'single-cell variational inference; single-cell integration benchmarking; a published scVI-based integration method; a published benchmark of batch losses in scVI'),
 ('D-06, D-07, I089, I015, I077, I078', 'labels of earlier design-review findings: D-06 = X7 variant arms not at matched batch removal; D-07 = X8 per-batch cell numbers and composition change with batch count; I089 = X7 reference coverage confounded; I015, I077, I078 = X8 design issues'),
 ('C01-C25', 'claim numbers of the claims audit of the original submission'),
 ('D1-D9', 'pipeline defects of the original submission listed in PAPER_PLAN section 1 (D5 = 10 critic updates versus 1 discriminator update)'),
 ('R-11, G-07', 'R-11 = finding of the earlier Methods-track review: the outcome of rules R2 and R3 for X1 is fixed by SI-66 and the frozen rule-R4 record (defaults kept), only the formal re-run is pending; G-07 = guard-review note that sysVI rows carry placeholder columns the fit does not read'),
 ('PREREG, PAPER_PLAN, ANALYSIS_SPEC, plan.py', 'docs/PREREG.md (pre-registration), docs/PAPER_PLAN.md (paper plan), docs/ANALYSIS_SPEC.md (analysis specification), scripts/scvi_adversarial_plan.py (training code of the arms)'),
 ('INT-nn', 'findings of this review'),
]
terms_md = table(['term', 'plain meaning'], TERMS)

# closest works found by the gap search
CLOSE = [
 ('scIB-E (Yi et al., Genome Biol 26:398, 2025)', 'benchmark', 'six level-1 batch losses (GAN, HSIC, orthogonal projection, mutual information, reverse backpropagation, reverse cross-entropy) in one scVI framework', 'no W1, MMD, OT or Sinkhorn arm (0 hits in the full text)', 'full text'),
 ("Authors' preprint (Topology Matters, bioRxiv 10.1101/2025.11.12.688061)", 'comparison', 'discriminator versus a multi-headed reference-based W1 critic with the backbone held fixed (scCRAFT)', 'two objectives only; no MMD or OT; not scVI', 'July submission text; web-search entries for v1 and v3; bioRxiv full text unreadable (HTTP 429)'),
 ('sysVI (Hrovatin et al., BMC Genomics 26:974, 2025)', 'method with ablations', 'KL weight, adversarial module, VampPrior and cycle-consistency inside one conditional VAE', 'no W1, MMD or OT arm', 'full text'),
 ('iDLC (bioRxiv 10.64898/2026.02.17.706490, 2026)', 'new method', 'explicit disentanglement plus optimal transport; evidence.md:66 records a WGAN-GP discriminator with a Sinkhorn regulariser', 'a method, not a controlled comparison', 'abstract'),
 ('scBIOT (Research Square 10.21203/rs.3.rs-8310482/v1, 2026)', 'new method (by title)', 'atlas integration method named scBIOT', 'method; no abstract in the OpenAlex record', 'title only'),
 ('ResPAN (Wang et al., Bioinformatics 2022)', 'new method', 'WGAN-GP with a residual generator and mutual-nearest-neighbour pairing', 'one model, no comparison of objectives', 'bioRxiv text and published abstract'),
 ('iMAP (Wang et al., Genome Biol 22:63, 2021)', 'new method', 'autoencoder plus WGAN-GP with paired transfer; one anchor dataset', 'one model', 'full text'),
 ('IMAAE (Int J Mol Sci 24:5502, 2023)', 'new method', 'adversarial autoencoder with WGAN-GP-form losses; its design exploration reports that a plain GAN under-corrects with three or more batches', 'design motivation, no JS-versus-W1 comparison', 'full text (PMC)'),
 ('uniPort, IMGG, transmorph, scPoli, Iterative Alignment Flows', 'new methods or frameworks', 'optimal-transport or GAN-based integration frameworks', 'each proposes one method', 'title and abstract in the screening table'),
 ('Maan et al., Nat Biotechnol 42:1899, 2024 (and its 2022 preprint)', 'benchmark of imbalance', 'five integration methods (BBKNN, Harmony, Scanorama, scVI, Seurat in the preprint)', 'none adversarial; no loss comparison', 'published abstract; preprint text'),
 ('Con-AAE (Bioinformatics 39:btad162, 2023)', 'new method (multi-omics)', 'contrastive cycle adversarial autoencoder for multi-omics alignment', 'multi-omics alignment, not scVI batch-loss comparison', 'seen as a web-search result only; not opened'),
]
close_md = table(['work', 'type', 'what it does', 'why it is not the comparison', 'what I read'], CLOSE)
cov_rows = []
for q in queries:
    cov_rows.append((q, str(gn['coverage'][f'openalex|{q}']), str(gn['coverage'][f'arxiv|{q}']), str(gn['coverage'][f'europepmc|{q}'])))
cov_md = table(['query', 'OpenAlex hits', 'arXiv hits', 'Europe PMC hits'], cov_rows)

DEVIATIONS = [
 "Goh 2022 and Maan 2024: publisher full texts are closed; the claims were checked against the abstracts (Maan also against its 2022 preprint, a different version), not against full text.",
 "Arjovsky and Bottou 2017: read as arXiv v1 only; the ICLR camera-ready on OpenReview was not opened, so theorem numbers are those of v1.",
 "Gap search: of the {{gap_unique}} unique hits, {{gap_rated}} (the top 60 minus one without a rating) were rated by a language model from title and abstract; the other {{gap_not_rated}} were screened by title by me, with abstracts read for all partial matches and all 2025-2026 candidates; the {{gap_forward}} forward-citation works were screened by title; Con-AAE was seen only as a web-search result and scBIOT by title only.",
 "SAUCIE and AD-AE were read as bioRxiv versions and MMD-ResNet as arXiv, not the published versions; Stegle 2015 was not retrieved; the HSIC-equals-MMD relation was seen only as a search-result entry.",
 "The shared project rigour ledger was not modified (the ledger loader found no project ledger file); findings are in a separate flat file, ids INT-01 to INT-{{n_total}}.",
 "Related work and Methods citation sites were not reviewed (parallel tracks); only cross-section inconsistencies were noted (INT-24, section 6).",
 "The ponytail skill (smallest-change rule) is not in this profile's catalogue, so it was not loaded; the check script was kept to read-only readbacks.",
]
DEVIATIONS = [render(d) for d in DEVIATIONS]
SOURCES_NOT_RECHECKED = [
 "maan2024characterizing: published full text closed; abstract (Europe PMC) and the 2022 preprint only",
 "goh2022batch: published full text closed; abstract only",
 "arjovsky2017towards: ICLR camera-ready (OpenReview) not opened; arXiv v1 read",
 "saucie (Amodio 2019), dincer2020adversarial, mmd-resnet (Shaham 2017): bioRxiv or arXiv versions read, not published versions",
 "Stegle 2015 (dropped by the literature track): not retrieved",
 "HSIC equals MMD between joint and product of marginals (arXiv 2503.04820): seen as a search-result entry only",
 "Con-AAE (Bioinformatics 39:btad162) and scBIOT (Research Square 10.21203/rs.3.rs-8310482/v1): not opened / title only",
 "bioRxiv full text of the authors' own preprint 10.1101/2025.11.12.688061 (HTTP 429); July submission text read instead",
 "Related work and Methods citation sites (parallel tracks)",
]

W = []
A = W.append
A("# Independent rigour check: Introduction of paper/paper.tex (commit add2f73, lines 114-217)")
A("")
A("Reviewer: Rigor Reviewer (review only; paper.tex was not edited) | Date: 2026-10-15 | Track: intro_review | Repository: kreid415/wasserstein-critic-deconfounding, branch paper")
A("")
A("## 1. Verdict")
A("")
A("**PASS_WITH_FIXES.** {{n_total}} findings: {{n_blocker}} blockers, {{n_fix}} fixes, {{n_note}} notes.")
A("")
A(sev_md)
A("")
A("A blocker would be a false locked statement, a citation whose source does not say what is attributed (including a wrong theorem, section or page pointer), or bibliographic data that misidentifies a work. None was found. All {{n_src}} source tags in the Introduction (lines 114-182) point at facts that are true at commit 08f1e3c; three of them do not support the whole statement they follow: the two Gap lines (:166-167 carry design-document tags on a search outcome and a writing choice, INT-02) and the SI-18 tag at :177 (it fixes the arms and strength grids, not a comparison, INT-10). All {{n_cite_cmds}} citation commands ({{n_cite_inst}} key uses, {{n_cite_keys}} distinct works) support their claims at the level read, with the wording fixes below. The fixes are what a methods reviewer or a first-time reader would object to first:")
A("")
A("1. Gap paragraph (INT-01, INT-02, INT-03): 'budget-matched' is undefined (PAPER_PLAN uses 'budget' both for the number of configurations tried and for critic steps) and the primary contrast P1 compares 5 critic steps with 1; the search scope is not stated and the recorded search covers 2024 onward only; the authors' own earlier preprint (discriminator versus W1 critic on a fixed backbone) is not mentioned.")
A("2. Theory paragraphs (INT-04 to INT-08): the over-alignment sentence (:150) claims more than the two cited theorems; the precondition of Zhao's theorem, the noise model of Arjovsky and Bottou's Theorem 2.6, the fixed-distribution caveat and one theorem pointer need correcting.")
A("3. Locked versus pending (INT-09 to INT-12, INT-14, INT-21): the finding labels of X7, the W1-versus-MMD/Sinkhorn comparison tagged with a ledger entry that fixes only the arms and grids, a guidance claim without a tag, four untagged inferences, and three scCRAFT mentions under one pending marker.")
A("4. Scope and readability (INT-13, INT-15): the decoder choice (batch label given to the decoder or withheld) is never named in the Introduction; the paragraph at :139-140 uses terms that no sentence defines.")
A("5. Bibliography (INT-16): one author name in refs.bib.")
A("")
A("### Terms and labels used in this report")
A("")
A(terms_md)
A("")
A("## 2. Scope, inputs and method")
A("")
A("- Reviewed: paper/paper.tex at commit add2f73, lines 114-217: the Introduction (114-182) and the commented list 'CLAIMS TO DROP OR SOFTEN' (184-216). File:line tags inside paper.tex (for example scvi_adversarial_plan.py:359) refer to commit 08f1e3c; add2f73 changed only files under paper/ relative to 08f1e3c, so every source file I read is identical at both commits.")
A("- Sources were read as text (PDF to text, no page images). Literature: full texts or preprint versions of the works in section 5.1 except Goh 2022 and Maan 2024 (abstract only; section 9). Project facts: the decisions ledger CONSTRAINTS.md (SI-01 to SI-77), PAPER_PLAN, PREREG sections 0-9, ANALYSIS_SPEC, the scripts, the manifest of record, the frozen record docs/prereg/r4_a1_record.json, the design recheck, the decision letter and the claims audit. Code: scvi-tools 1.4.2 (wheel downloaded; the cited lines read) and the repository scripts at add2f73.")
A("- Task items map to sections: (1) literature truth, 5.1 and INT-04 to INT-08, INT-18 to INT-20; (2) source statements against source, 5.1 rows CL-17 to CL-23 and 5.5; (3) locked versus pending, 5.5; (4) the 14 framing conflicts and the drop list, 5.2 and 5.3; (5) the lead's integration edits, 5.4; (6) the gap search repeated, section 7; (7) coverage and contradictions, section 6; (8) readability, section 8.")
A("- Severity as defined in the delegation: blocker = false or wrongly sourced locked statement, citation whose source does not say what is attributed, or bibliographic metadata that misidentifies a work; fix = lock/pending misclassification, claim beyond what was read, missing locked item, inconsistency with Methods or the glossary; note = wording, readability, style.")
A("- Kept: paper.tex not edited (replacement text is given); nothing pushed; no write to the /home/kendall folders; the shared project rigour ledger was not modified (entries are in the separate flat file rigor_ledger_entries_intro_review.json, ids INT-01 to INT-{{n_total}}).")
A("")
A("## 3. Findings (severity order)")
A("")
A("No blockers. Fixes first, then notes. 'Replacement text' is exact LaTeX for the outline (ASCII only, same macros as the outline).")
A("")
for f in FIND:
    A(finding_md(f)); A("")
A("## 4. What already holds (no change needed)")
A("")
A("- Code and data facts behind the source tags: scvi-tools 1.4.2 draws the encoder sample with rsample and passes it to the stock adversary (_trainingplans.py:689, _base_components.py:304-305, _vae.py:390 and :413); scripts/scvi_adversarial_plan.py:359 uses the posterior mean by default for the authored arms; its fool loss (:292-298) is the cross-entropy to the uniform distribution over the other batches, as in scvi-tools (_trainingplans.py:651-667); the manifest of record ({{manifest_rows}} rows, SHA-256 prefix {{manifest_sha}}) has {{x1_rows}} X1 rows, equal epochs, minibatch and latent size across arms per task, and the X6 combinations described at :178.")
A("- R-11 wording (the rule-R4 record on A1) in the two Introduction places is correct. Monotonic argument: b_common is the lowest, over the six arms, of each arm's highest batch score. A refit of Sinkhorn changes only Sinkhorn's term, so it cannot raise b_common above the arm that caps it (barycenter in {{r4_bary_blocks}} of {{r4_blocks_total}} blocks, MMD in {{r4_mmd_blocks}}, Sinkhorn in {{r4_sink_blocks}}); in the flagged blocks b_common - b0 is {{immune1_diff:.4f}} (immune, conditioned) and {{sim1_1_diff:.4f}} (sim1, conditioned), both below 0.02, with {{r4_extensions}} extension fits.")
A("- Framing conflicts: {{n_conf_ok}} of {{n_conflicts}} resolutions are right as written (their Introduction-side residuals are separate findings), {{n_conf_incomplete}} is incomplete (conflict 12, INT-14), {{n_conf_outside}} lies outside the Introduction (conflict 14).")
A("- Drop list: the {{n_status_checked}} audit statuses written out in the list (C01, C02, C06, C08, C09, C12, C14, C18, C20) were matched to the claims audit CSV by script ({{n_status_mismatch}} mismatches); the July background lines (Tung, taxonomy, Stegle), the 'real counts' wording, the premise wording, the pilot hint (0.726 versus 0.646, PAPER_PLAN:46) and the August barycenter sentence check out against their sources.")
A("- The pending marker on the decomposition (:173), the X3, X6 analyses and the formal re-run of rules R2 and R3 are correctly classified.")
A("")
A("## 5. Verification tables")
A("")
A("### 5.1 Claims checked ({{n_claims}} rows; each against the source text named)")
A("")
A(claims_md)
A("")
A("### 5.2 The 14 framing conflicts")
A("")
A(conf_md)
A("")
A("### 5.3 Drop-list items")
A("")
A(drop_md)
A("")
A("### 5.4 Integration edits 1 to 5 (edits 6 to 8 concern Methods, the reference list and the legend, outside this section)")
A("")
A(edit_md)
A("")
A("### 5.5 Locked versus pending: every line of the Introduction that carries a tag ({{n_src}} source tags and {{n_pending}} pending markers on {{n_tag_lines}} lines)")
A("")
A(stat_md)
A("")
A("### 5.6 Replacement text compiled")
A("")
A("The replacement text of the findings that change lines of the outline (INT-01 to INT-22; INT-02 supplies :166-167) was applied to a copy of paper.tex at add2f73 ({{lines_replaced}} lines replaced, {{lines_inserted}} inserted, and the refs.bib author corrected) and the copy was compiled with tectonic: {{tex_errors}} TeX errors, {{undefined_citations_or_references}} undefined citations or references, {{underfull_patched}} underfull-line warnings against {{underfull_original}} in the unchanged file. The change set is intro_replacements.diff. Not applied in the test: the replacements of commented lines (INT-04 :191-192, INT-23, INT-25), because comments do not compile. paper.tex itself was not edited.")
A("")
A("## 6. Coverage and contradictions against Research questions, Methods and the glossary")
A("")
A(table(['topic', 'Introduction', 'other section', 'result'], [
 ('decoder (batch label given or withheld)', 'not named', 'abstract :106, glossary :82, Methods :344, results per decoder (SI-70)', 'missing in the Introduction (INT-13)'),
 ('rule-R4 record, low-signal flag', ':140, :177', 'Methods :369, glossary :77', 'consistent in substance; both Introduction and Methods :369 mention immune only while the record flags immune and sim1 (INT-15; Methods track)'),
 ('X6 design', ':178', 'glossary :62, Methods :448', 'consistent; citations missing in the Introduction (INT-21)'),
 ('X7 finding labels', ':179 (D-06, D-07)', 'glossary :63, Research questions :255, Methods :452 (D-06, I089)', 'contradiction (INT-09)'),
 ('X8, batch-count scaling, local mixing, X12, X15', 'absent from Contribution', 'Research questions :232, :239-240, :246-248', 'coverage gap (INT-22)'),
 ('P1 critic steps', ':175 (no mention)', 'Research questions :231 (one step), Methods :353 (5 steps)', 'unstated limit of P1 (INT-01)'),
 ('backbone scope', ':172 (narrowed)', 'Methods :334 (every arm and baseline, ZINB)', 'Methods line needs the same narrowing (INT-24)'),
 ('earlier preprint', 'not mentioned', 'abstract :111 (replaced by the next version), Related work :306', 'INT-03'),
 ('scCRAFT', ':152', 'Related work :305-306 (pending only on :306)', 'INT-14'),
 ('guidance for scVI users', ':181 (untagged first clause)', 'abstract :110 (pending)', 'classified differently (INT-11)'),
]))
A("")
A("## 7. Gap search repeated (task item 6)")
A("")
A("Scope: OpenAlex (with the project's API key), arXiv and Europe PMC (PubMed, bioRxiv and medRxiv records); publication year 2017 to 2026-10-15; run on 2026-10-15 without the 2024 cut used in evidence.md. {{gap_queries}} queries, up to 20 results per source and query:")
A("")
A(cov_md)
A("")
A("{{gap_raw}} raw hits gave {{gap_unique}} unique works. A language model rated the top 60 on title and abstract against the question 'compares JS, W1 and MMD/OT objectives on one scVI backbone': {{gap_rated}} were rated ({{gap_rated_none}} none, {{gap_rated_partial}} partial, {{gap_rated_substantial}} substantial); {{gap_not_rated}} (one inside the top 60 that had no rating, the rest below the cut) were screened by title by me, and I read the abstracts of all partial matches and of the 2025-2026 candidates. Forward citations (OpenAlex 'cites' filter) of scIB-E ({{fwd_scibe}}), sysVI ({{fwd_sysvi}}), scCRAFT ({{fwd_sccraft}}) and ResPAN ({{fwd_respan}}) gave {{gap_forward}} works, screened by title (the file records titles, not abstracts). Two further web searches (one for the comparison, one for adversarial versus MMD or optimal-transport losses on one autoencoder) returned the authors' own preprint pages and Con-AAE; iDLC came from the OpenAlex hits and is also recorded in evidence.md:66. Files: gap_search_screen.csv, gap_forward_citations.csv, gap_search_numbers.json.")
A("")
A("**Verdict (scope above): no controlled comparison of JS, W1 and MMD/OT objectives on one scVI backbone was found.** Partial overlap only with the authors' own preprint (JS versus reference-based W1 on the scCRAFT backbone; no MMD or OT). This supports a search-based absence, not an exhaustive one, and it is not a claim of novelty; the record in evidence.md covered 2024 onward only (INT-02).")
A("")
A(close_md)
A("")
A("## 8. Readability (task item 8)")
A("")
A("Test applied: could a genetics PhD student who has not read earlier documents follow each sentence (SI-67)? Passes: :118-121 (background), :126-128 (acronyms are spelled out at first use except JSD), :158-160, :165. Fails or needs a definition in the same sentence: the encoder-as-generator mapping, posterior mean and sample, 'authored arms', the rule-R4 record on A1, b_common, low-signal, refitting Sinkhorn, GLUE, the unexpanded scVI before :165, and a reviewer remark in plain text at :143. All are in INT-15 with replacement text; INT-08 spells out JS divergence at :127.")
A("")
A("## 9. Sources not re-checked, and limits")
A("")
for s in [
 "Maan et al. 2024 (maan2024characterizing) and Goh et al. 2022 (goh2022batch): publisher full texts closed (Unpaywall, Semantic Scholar, PMC, Crossref text-mining links negative); claims checked against the abstracts. For Maan the 2022 preprint (10.1101/2022.10.06.511156) was read as corroboration; it is a different version.",
 "Arjovsky and Bottou 2017 (arjovsky2017towards): read as arXiv v1 (1701.04862v1); the ICLR camera-ready on OpenReview was not opened (bot challenge), so theorem numbers are those of arXiv v1.",
 "SAUCIE, AD-AE and the weighted-distance preprint sources: read as bioRxiv or arXiv versions, not the published versions; Stegle 2015 not retrieved (not cited in the Introduction).",
 "The HSIC-equals-MMD relation (INT-20): seen as a search-result entry for arXiv 2503.04820; the paper itself was not opened.",
 "Con-AAE, scBIOT: not opened or title only (section 7).",
 "Related work and Methods citation sites belong to the parallel tracks; only cross-section inconsistencies are noted (INT-24, section 6).",
]: A("- " + s)
A("")
A("## 10. Deviations from the delegated task")
A("")
for d in DEVIATIONS: A("- " + d)
A("")
A("## 11. Reproducibility")
A("")
A("- Environment: conda env `python` (pandas, requests), plus the scvi-tools 1.4.2 wheel read without installation; repository checkout at add2f73 (origin removed, detached).")
A("- intro_checks.py reads the manifest, the frozen record and the cited scvi-tools and repository lines: `python intro_checks.py <repo_dir> <scvi_wheel_dir> intro_checks_results.json`. findings_data.py, tables_data.py and build_intro_review.py rebuild this report, the claims table and the ledger file from those JSON and CSV files; every number in the report is read back from report_values.json, which is derived from intro_checks_results.json, gap_search_screen.csv, gap_forward_citations.csv and gap_search_numbers.json.")
A("- Inputs copied from project artifacts: the claims audit of the original submission (version 6c946526-684e-4e4b-9306-95c648230aae), the outline review record (dd13e9fa-068d-4e83-9c4d-15b6fa0faf71) and the review input bundle (13f3c1f2-fa87-4438-bfd8-f2fa2ddc72b2); the repository at add2f73 (paper/paper.tex, paper/refs.bib, paper/evidence.md, docs/, scripts/, manifests/) was read from a local clone with its remote removed.")
A("- apply_replacements_test.py applies the replacement text to a copy of paper.tex and compiles it with tectonic (compile_check.json, intro_replacements.diff).")
A("- Search record: gap_search_screen.csv (175 rows, with year, DOI, rating, note, URL), gap_forward_citations.csv, gap_search_numbers.json.")
report_template = '\n'.join(W)
report = render(report_template)

# ---- claim-gate receipt: counts reconcile, numbers read back, then appended to the report ----
_rr = json.load(open(f'{WS}/paper_outline_review_record.json', encoding='utf-8'))
_fc = _rr['lit_review']['framing_conflicts']
rec_a = cg_reconcile(total=len(ledger), findings=len(out_findings), parts=[len(sev[s_]) for s_ in ('blocker', 'fix', 'note')],
                     partitions={s_: set(sev[s_]) for s_ in sev}, items=[e['id'] for e in ledger])
rec_b = cg_reconcile(claims_table_rows=len(claims_csv), claims_in_data=len(td.CLAIMS))
rec_c = cg_reconcile(conflicts_rows=len(td.CONFLICTS), conflict_sources=len(_fc))
_ev = [rec_a, rec_b, rec_c] + [V[k] for k in ('immune1_diff', 'sim1_1_diff', 'manifest_rows', 'x1_rows', 'manifest_sha', 'gap_unique',
                                              'gap_forward', 'gap_raw', 'n_status_checked', 'n_status_mismatch', 'tex_errors',
                                              'undefined_citations_or_references')]
_claim = (f"Introduction of paper/paper.tex at add2f73 reviewed: {len(FIND)} findings ({len(sev['blocker'])} blocker, {len(sev['fix'])} fix, "
          f"{len(sev['note'])} note); report, flat ledger entries and claims table written; numbers in the report read back from the result files; "
          f"replacement text compiled with {cck['tex_errors']} TeX errors.")
receipt = cg_receipt(_claim, _ev, SOURCES_NOT_RECHECKED)
_rtext = receipt['text'].replace(os.getcwd() + '/', './').replace(os.getcwd(), '.')
open(f'{WS}/claim_receipt.txt', 'w', encoding='utf-8').write(_rtext + '\n')
report += '\n\n## 12. Claim-gate receipt\n\n```\n' + _rtext + '\n```\n'
open(f'{WS}/rigor_review_intro_20261015.md', 'w', encoding='utf-8').write(report)
json.dump(ledger, open(f'{WS}/rigor_ledger_entries_intro_review.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump(out_findings, open(f'{WS}/intro_findings_for_submit.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(len(report), len(ledger), len(out_findings), len(claims_csv))
