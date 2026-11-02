"""Read-only checks for paper/outline.tex after Pass 4 (no TeX engine available on the writing machine).
Usage: python3 -I paper/tools/check_outline_pass4.py paper/outline.tex paper/refs.bib

Checks: brace balance (escaped braces ignored, comments stripped); \\begin/\\end pairing; every \\cite key
exists in refs.bib; every \\ref has a \\label; `changes`-package rules (no \\todo inside deleted text, every
citation inside deleted text wrapped in \\mbox, no changes command nested inside another); counts of \\tbd and
\\res marks and of Pass 4 commands; the protected "registered" wording occurrences; leftover main-benchmark
wording in live text; a deslop scan of the Pass 4 text when the deslop skill is installed.
"""
import re, sys

tex_path, bib_path = sys.argv[1], sys.argv[2]
raw = open(tex_path, encoding="utf-8").read()
lines = raw.split("\n")

def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2]); i += 2; continue
        if c == "%":
            break
        out.append(c); i += 1
    return "".join(out)

code_lines = [strip_comment(l) for l in lines]
code = "\n".join(code_lines)

# 1. brace balance (escaped braces ignored), with the first line where depth goes negative
depth, neg_line = 0, None
for ln, l in enumerate(code_lines, 1):
    j = 0
    while j < len(l):
        if l[j] == "\\" and j + 1 < len(l):
            j += 2; continue
        if l[j] == "{": depth += 1
        elif l[j] == "}":
            depth -= 1
            if depth < 0 and neg_line is None: neg_line = ln
        j += 1
print("braces: final depth", depth, "first negative at line", neg_line)

# 2. environments
stack, env_err = [], []
for ln, l in enumerate(code_lines, 1):
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", l):
        if m.group(1) == "begin": stack.append((m.group(2), ln))
        else:
            if not stack or stack[-1][0] != m.group(2): env_err.append((ln, m.group(2), stack[-1] if stack else None))
            else: stack.pop()
print("environments: unmatched", stack, "errors", env_err)

# 3. citations vs bib
bib = open(bib_path, encoding="utf-8").read()
bibkeys = set(re.findall(r"^@\w+\{([^,\s]+),", bib, re.M))
cites = set()
for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", code):
    for k in m.group(1).split(","): cites.add(k.strip())
missing = sorted(k for k in cites if k not in bibkeys)
print("cite keys:", len(cites), "missing from bib:", missing)

# 4. refs vs labels
labels = set(re.findall(r"\\label\{([^}]*)\}", code))
refs = set(re.findall(r"\\(?:eq|page|auto|c)?ref\{([^}]*)\}", code))
print("labels:", sorted(labels), "refs without label:", sorted(refs - labels))

# 5. changes-package rules, by walking the brace tree
def arg_spans(cmd_pattern):
    """yield (start, end, text, line) for the first brace argument of each command match."""
    for m in re.finditer(cmd_pattern, code):
        i = m.end()
        if i >= len(code) or code[i] != "{": continue
        d, j = 0, i
        while j < len(code):
            if code[j] == "\\": j += 2; continue
            if code[j] == "{": d += 1
            elif code[j] == "}":
                d -= 1
                if d == 0: break
            j += 1
        yield i, j + 1, code[i + 1:j], code.count("\n", 0, m.start()) + 1

def second_arg(start_end):
    """given the end of the first argument, return the second argument span text."""
    i = start_end
    if i >= len(code) or code[i] != "{": return None
    d, j = 0, i
    while j < len(code):
        if code[j] == "\\": j += 2; continue
        if code[j] == "{": d += 1
        elif code[j] == "}":
            d -= 1
            if d == 0: break
        j += 1
    return code[i + 1:j], i, j + 1

opt = r"(?:\[[^\]]*\])?"
deleted_texts = []  # (line, text, start, end)
for s, e, t, ln in arg_spans(r"\\deleted" + opt):
    deleted_texts.append((ln, t, s, e))
replaced_new, replaced_old = [], []
for s, e, t, ln in arg_spans(r"\\replaced" + opt):
    replaced_new.append((ln, t, s, e))
    sec = second_arg(e)
    if sec: replaced_old.append((ln, sec[0], sec[1], sec[2]))
added_texts = [(ln, t, s, e) for s, e, t, ln in arg_spans(r"\\added" + opt)]

problems = []
for ln, t, s, e in deleted_texts + replaced_old:
    if "\\todo" in t: problems.append((ln, "\\todo inside deleted text"))
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{", t):
        pre = t[max(0, m.start() - 6):m.start()]
        if not pre.endswith("\\mbox{"): problems.append((ln, "citation in deleted text not wrapped in \\mbox: " + t[m.start():m.start() + 40]))
# nesting: any changes command inside the argument of another
all_spans = [(s, e, ln, "deleted") for ln, t, s, e in deleted_texts] + [(s, e, ln, "replaced-new") for ln, t, s, e in replaced_new] + [(s, e, ln, "replaced-old") for ln, t, s, e in replaced_old] + [(s, e, ln, "added") for ln, t, s, e in added_texts]
for s, e, ln, kind in all_spans:
    inner = code[s + 1:e - 1]
    if re.search(r"\\(added|deleted|replaced)\b", inner): problems.append((ln, "nested changes command inside " + kind))
print("changes-rule problems:", problems)

# 6. counts and protected wording
tbd = len(re.findall(r"\\tbd\{", code)); res = len(re.findall(r"\\res\{", code))
print("\\tbd marks:", tbd, " \\res marks:", res)
regs = [(ln, m.group(0)) for ln, l in enumerate(lines, 1) for m in re.finditer(r"[Pp]re-?regist\w*|\bregist\w*", l)]
print("regist* occurrences (all lines incl. comments):", len(regs), regs)
p4_lines = [ln for ln, l in enumerate(lines, 1) if "id=P4" in l]
print("lines with id=P4:", len(p4_lines), p4_lines)
print("P4 commands:", len(re.findall(r"\\(?:added|deleted|replaced)\[id=P4\]", code)))

# 6b. leftover wording in LIVE text (outside \deleted{} and outside the old argument of \replaced{}{})
dead = sorted([(s, e) for ln, t, s, e in deleted_texts] + [(s, e) for ln, t, s, e in replaced_old])
live_chars = list(code)
for s, e in dead:
    for k in range(s, e): live_chars[k] = " " if live_chars[k] != "\n" else "\n"
live = "".join(live_chars)
for phrase in ["main benchmark", "main-benchmark", "six strengths", "five seeds", "all-task pilot", "strength pilot", "pilot", "k=1$--6", "408", "3,000 fits", "follow-up"]:
    hits = [(live.count("\n", 0, m.start()) + 1, live[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")) for m in re.finditer(re.escape(phrase), live)]
    if hits:
        print(f"LIVE '{phrase}':", len(hits))
        for h in hits: print("   ", h)

# 7. deslop scan of the new Pass 4 text
try:
    sys.path.insert(0, "/home/kendall/.claude/skills/deslop")
    import kernel as ds
    new_text = "\n".join(t for ln, t, s, e in added_texts + replaced_new if "id=P4" in code[max(0, s - 12):s])
    new_text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", new_text).replace("{", " ").replace("}", " ")
    scan = ds.deslop_scan(new_text)
    print("deslop words:", scan["n_words"], "score:", ds.deslop_score(scan))
    for f in scan["findings"]:
        if f["severity"] == "hard": print("  HARD", f["category"], f["label"], "|", f["excerpt"][:80])
    soft = [f for f in scan["findings"] if f["severity"] != "hard"]
    print("  soft findings:", len(soft), sorted(set(f["label"] for f in soft)))
except Exception as ex:
    print("deslop scan not run:", repr(ex))
