"""Word count of the active (final-mode) abstract of paper/outline.tex, as printed with \\usepackage[final]{changes}.

Usage, from the repository root:  python paper/tools/abstract_words.py [--show] [--limit 350] [--tex paper/outline.tex]
Method: take the enumerate block after \\hd{Abstract}; keep the new text of \\replaced{new}{old}, the text of \\added{...}, drop \\deleted{...}
and TeX comments; \\res{...} and \\tbd{...} placeholders are not words of the abstract and are dropped (they are listed);
\\textbf/\\emph/\\citep keep their argument (\\citep is dropped); \\item and the keywords line are removed; `--` and `---` become
a space-free dash; then split on white space (the count of `wc -w`). A formula such as $b^*$ counts as one word.
Exit 1 if the count exceeds --limit."""
import argparse, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def group(s, i):
    """Return (content, index after the closing brace) of the brace group that starts at s[i]."""
    assert s[i] == "{", s[i:i + 30]
    depth, j = 0, i
    while True:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        depth += c == "{"
        depth -= c == "}"
        j += 1
        if depth == 0:
            return s[i + 1:j - 1], j


def active_text(s, dropped):
    out, i = [], 0
    while i < len(s):
        m = re.compile(r"\\(replaced|added|deleted|textbf|emph|textit|res|tbd|citep|texttt)(\[[^\]]*\])?(?=\{)").match(s, i)
        if not m:
            out.append(s[i])
            i += 1
            continue
        name, j = m.group(1), m.end()
        a, j = group(s, j)
        if name == "replaced":
            _, j = group(s, j)  # the old text is dropped
        if name in ("replaced", "added", "textbf", "emph", "textit"):
            out.append(active_text(a, dropped))
        elif name in ("res", "tbd"):
            dropped.append(a)
        i = j
    return "".join(out)


def abstract(tex):
    """Source of the abstract: after the heading's budget argument, up to the keywords line. Until Pass 23 the abstract was a
    numbered list; since Pass 24 it is three paragraphs after the struck list, so list environments and \\vspace are dropped."""
    block = tex.split(r"\hd{Abstract}", 1)[1]
    block = re.split(r"\n[^\n]*Keywords:", block[group(block, 0)[1]:], maxsplit=1)[0]
    block = re.sub(r"(?m)(?<!\\)%.*$", "", block)
    return re.sub(r"\\(begin|end)\{enumerate\}(\[[^\]]*\])?|\\vspace\{[^}]*\}", " ", block)


def count(tex):
    dropped = []
    t = active_text(abstract(tex), dropped)
    t = t.replace(r"\item", " ").replace(r"\%", "%").replace("---", "-").replace("--", "-")
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t).replace("~", " ")
    t = re.sub(r"[{}]", "", t)
    return t.split(), dropped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(ROOT / "paper/outline.tex"))
    ap.add_argument("--limit", type=int, default=350)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args(argv)
    words, dropped = count(Path(a.tex).read_text())
    if a.show:
        print(" ".join(words))
    split = len(" ".join(words).replace("-", " ").split())  # stricter counter that also splits at hyphens and dashes
    print(f"abstract words (final text, markup and placeholders removed): {len(words)}; hyphen-split count {split}; "
          f"placeholders dropped: {len(dropped)} ({sum(len(x.split()) for x in dropped)} words of placeholder text); limit {a.limit}")
    return int(len(words) > a.limit)


if __name__ == "__main__":  # pragma: no cover  # command-line entry; main() is tested
    sys.exit(main())
