"""Pass 11 edits to paper/outline.tex (tracked id P11): real Figure 1 (path, caption, header note).
Every old text must occur exactly once or the script stops. Numbers in the caption are read from
paper/figure_data/paper_numbers.json and checked against the values fig1.py asserts; written to paper/pass11_numbers.csv.
Run from the repository root after paper/figure_scripts/fig1.py:  python paper/tools/pass11_edits.py"""
import csv, json, os
from pathlib import Path

ROOT = Path(os.environ.get("PASS11_ROOT") or Path(__file__).resolve().parents[2])
TASK, COND, ARMS = "atac_small", 0, ["discriminator", "reference", "mmd"]
pn = json.load(open(ROOT / "paper/figure_data/paper_numbers.json"))
v = lambda k: pn[k]["value"]
ledger = []
def num(item, key, fmt):
    ledger.append(dict(item=item, value=format(v(key), fmt), source=f"paper/figure_data/paper_numbers.json:{key}"))
    return format(v(key), fmt)

bstar = num("b* (ATAC small, unconditioned decoder)", f"pair.{TASK}.cond{COND}.bstar", ".3f")
read = {a: num(f"biology at b*, {a}", f"bio_at_bstar.{a}.{TASK}.cond{COND}", ".3f") for a in ARMS}
nstr = {a: v(f"n_strengths.{a}.{TASK}.cond{COND}") for a in ARMS}
cross = {a: v(f"crossings.{a}.{TASK}.cond{COND}") for a in ARMS}
assert nstr == {"discriminator": 12, "reference": 10, "mmd": 10} and cross == {"discriminator": 2, "reference": 2, "mmd": 1}
for a in ARMS:
    ledger.append(dict(item=f"strengths fitted, {a}", value=nstr[a], source=f"paper_numbers.json:n_strengths.{a}.{TASK}.cond{COND}"))
    ledger.append(dict(item=f"times the curve crosses b*, {a}", value=cross[a], source=f"paper_numbers.json:crossings.{a}.{TASK}.cond{COND}"))

new_caption = (
    "Model and read-out. (a) One scVI encoder--decoder is shared by all methods; the decoder receives the batch label "
    "(conditioned) or not (unconditioned), and only the alignment penalty on the 10-dimensional latent embedding is swapped. "
    "Four adversarial penalties (a Jensen--Shannon discriminator, one update step; Wasserstein critics, reference, pooled and "
    "barycenter, five update steps with a gradient penalty) and two non-adversarial penalties (MMD with a fixed multi-scale "
    "Gaussian kernel; the Sinkhorn divergence) are compared. The loss is the scVI loss plus the strength times the penalty; "
    "strength 0 is plain scVI. (b) One real task and decoder from the sweep, ATAC small with the unconditioned decoder, for three of the six "
    "penalties (colours as in Figures 2--6). Each curve joins plain scVI (star) and the sweep strengths in increasing order "
    f"({nstr['reference']} strengths for the reference critic and MMD, {nstr['discriminator']} for the discriminator, which has two further low strengths); "
    "each point is a mean over the three seeds. Biology is read where the curve first reaches the matched batch removal "
    f"$b^*$ (dashed line, {bstar}), interpolating between neighbouring strengths (open circles; dotted arrows read the biology score): "
    f"JS discriminator {read['discriminator']}, reference critic {read['reference']}, MMD {read['mmd']}. "
    f"The discriminator and reference-critic curves cross $b^*$ a second time at higher strengths; the reading uses the first crossing.")

s = (ROOT / "paper/outline.tex").read_text()
def rep(old, new, why):
    global s
    assert s.count(old) == 1, (s.count(old), why, old[:90])
    s = s.replace(old, new)

rep(r"\includegraphics[width=\textwidth]{figures/mock/fig1.png}", r"\includegraphics[width=\textwidth]{figures/fig1.png}", "fig1 path")
i = s.index(r"\caption{MOCK -- synthetic data. Model and read-out.")
j = s.index("\n\\label{fig:1}", i)
old_caption_body = s[i + len(r"\caption{"): j - 1]          # up to the closing brace before the newline
assert s[j - 1] == "}" and old_caption_body.count("{") == old_caption_body.count("}")
s = s[:i] + r"\caption{\replaced[id=P11]{" + new_caption + "}{" + old_caption_body + "}}" + s[j:]
rep(r"Figure 1 is a placeholder schematic and Figures 2--6 are drawn from the sweep analysis (Pass 10, \texttt{paper/figure\_scripts}, change log \texttt{paper/pass10\_changelog.md})",
    r"\replaced[id=P11]{Figures 1--6 are drawn from the sweep analysis (Figures 2--6 in Pass 10, Figure 1 in Pass 11, \texttt{paper/figure\_scripts}, change logs \texttt{paper/pass10\_changelog.md} and \texttt{paper/pass11\_changelog.md})}{Figure 1 is a placeholder schematic and Figures 2--6 are drawn from the sweep analysis (Pass 10, \texttt{paper/figure\_scripts}, change log \texttt{paper/pass10\_changelog.md})}",
    "header note")
rep(r"\definechangesauthor[name={Pass 10}, color=teal]{P10}",
    r"\definechangesauthor[name={Pass 11}, color=blue]{P11}% real Figure 1 (schematic and one real sweep example); change log paper/pass11_changelog.md, numbers in paper/pass11_numbers.csv" + "\n" +
    r"\definechangesauthor[name={Pass 10}, color=teal]{P10}", "author def")
rep("% PNGs: figures/mock/fig1.png ... fig6.png (MOCK watermark, synthetic data, 300 dpi, 17.4 cm wide).",
    "% PNGs: figures/mock/fig1.png ... fig6.png (MOCK watermark, synthetic data, 300 dpi, 17.4 cm wide). Pass 11: Figure 1 is now figures/fig1.png (paper/figure_scripts/fig1.py).", "comment")
(ROOT / "paper/outline.tex").write_text(s)
with open(ROOT / "paper/pass11_numbers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["item", "value", "source"]); w.writeheader(); w.writerows(ledger)
print(new_caption[-420:]); print(len(ledger), "ledger rows")
