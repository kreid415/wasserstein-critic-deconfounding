import json

from replication_mean_ranks import OUT, ROOT, summary

ARMS = ["JS discriminator", "Reference critic", "Pooled critic", "Barycenter critic", "MMD", "Sinkhorn"]


def block(name, dec, reps):
    head = rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{name}, {dec} decoder; $b^*=0.5$}}}} \\"
    return [head] + [f"{a} & 1--3 & 0.1/0.2 & 0.3/0.4 & 0.5 & 0.5 & {r} ({6 - i}) &  \\\\" for i, (a, r) in enumerate(zip(ARMS, reps))]


def test_summary_skips_simulations_and_averages_ranks():
    tex = "\n".join(block("Lung", "unconditioned", [1, 2, 3, 4, 5, 6]) + block("Pancreas", "unconditioned", [5, 3, 1, 2, 4, 6])
                    + block("Simulation 1", "unconditioned", [6, 5, 4, 3, 2, 1]))
    got = summary(tex)["unconditioned"]
    assert got["n_tasks"] == 2 and got["discriminator"] == {"replication": 3.0, "sweep": 6.0}
    assert (got["replication_first"], got["replication_second"]) == ("pooled", "reference")


def test_committed_json_matches_table_s9():
    assert summary((ROOT / "paper/supp/tableS9.tex").read_text()) == json.loads(OUT.read_text())
