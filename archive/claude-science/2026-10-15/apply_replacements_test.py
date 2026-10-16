# apply_replacements_test.py
# Applies the replacement texts of findings_data.py to a COPY of paper/paper.tex (add2f73) and compiles the copy with
# tectonic, to check that the proposed LaTeX is well formed and uses only macros the outline defines.
# Run in the python kernel after build_intro_review.py (needs `render` and `fd`). Nothing in the repository is edited.
import re, os, shutil, subprocess

SRC = 'work/rv/paper/paper.tex'
TD = 'intro_review/tex_test'
os.makedirs(TD, exist_ok=True)
L = open(SRC, encoding='utf-8').read().split('\n')      # L[n-1] is line n
orig = lambda n: L[n - 1]
edits = {}      # line number -> new text (replace)
after = {}      # line number -> list of new lines inserted after it

def rep_text(fid, i):
    return render([f for f in fd.FINDINGS if f['id'] == fid][0]['rep'][i]['text'])

# INT-02's version of :166 and :167 supersede INT-01's :166
edits[166] = rep_text('INT-02', 0)
edits[167] = rep_text('INT-02', 1)
after[166] = [rep_text('INT-03', 0)]
after[172] = [rep_text('INT-01', 1)]
edits[150] = rep_text('INT-04', 0)
edits[148] = rep_text('INT-05', 0)
edits[142] = rep_text('INT-06', 0)
edits[137] = rep_text('INT-07', 0)
edits[127] = rep_text('INT-08', 0)
edits[179] = rep_text('INT-09', 0)
r10 = rep_text('INT-10', 0)
edits[177] = '\\item ' + r10[:r10.index('per-dimension standardisation')] + orig(177)[orig(177).index('per-dimension standardisation'):]
edits[181] = rep_text('INT-11', 0)
r12 = rep_text('INT-12', 0)
edits[141] = orig(141)[:orig(141).index('at one adversary step')] + r12[3:].lstrip()
edits[160] = rep_text('INT-12', 1)
r13 = rep_text('INT-13', 0)
edits[172] = orig(172).replace('at scvi-tools defaults;', 'at scvi-tools defaults; ' + r13)
for n, i in ((152, 0), (305, 1)):
    r14 = rep_text('INT-14', i)
    pend = r14[r14.index('\\pending'):]
    edits[n] = orig(n).rstrip().rstrip('.') + ' ' + pend
after[132] = [rep_text('INT-15', 0)]       # before :133 = after :132 (:132 is \begin{itemize})
edits[139] = rep_text('INT-15', 1)
edits[140] = rep_text('INT-15', 2)
edits[143] = rep_text('INT-15', 3)
r151 = rep_text('INT-15', 4)
edits[151] = '\\item ' + r151[:r151.index(' ...')] + ' ' + orig(151)[orig(151).index('mixed unrelated'):]
edits[120] = rep_text('INT-18', 0)
r133 = rep_text('INT-18', 1)
edits[133] = orig(133)[:orig(133).index('differentiable almost everywhere')] + r133[r133.index('differentiable almost everywhere'):]
edits[135] = rep_text('INT-18', 2)
edits[138] = rep_text('INT-18', 3)
edits[119] = rep_text('INT-19', 0)
r20 = rep_text('INT-20', 0)
edits[165] = orig(165)[:orig(165).index('in one scVI')] + r20[r20.index('in one scVI'):]
edits[176] = rep_text('INT-21', 0)
r21 = rep_text('INT-21', 1)
edits[178] = '\\item ' + r21[:r21.index(' ... (rest unchanged)')] + orig(178)[orig(178).index(' \\src{SI-22; manifest}'):]
after[180] = [rep_text('INT-22', 0)]

out = []
for n, line in enumerate(L, start=1):
    out.append(edits.get(n, line))
    for extra in after.get(n, []):
        out.append(extra)
open(f'{TD}/paper_patched.tex', 'w', encoding='utf-8').write('\n'.join(out))
shutil.copy('work/rv/paper/refs.bib', f'{TD}/refs.bib')
# the bib fix of INT-16 (author name) is applied to the copy as well
b = open(f'{TD}/refs.bib', encoding='utf-8').read()
assert 'and Mario March and' in b
open(f'{TD}/refs.bib', 'w', encoding='utf-8').write(b.replace('and Mario March and', 'and Mario Marchand and'))
n_changed = len(edits) + sum(len(v) for v in after.values())
print('lines replaced:', len(edits), '| lines inserted:', sum(len(v) for v in after.values()))
