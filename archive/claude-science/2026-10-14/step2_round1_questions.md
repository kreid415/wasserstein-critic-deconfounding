# Step 2, round 1: five questions for the user (drafted 2026-10-13, not yet asked)

Ask with ask_user, one tab per question, recommended option first. Plain words (SI-67). Spec sections refer to
docs/ANALYSIS_SPEC.md (draft, commit 6d0ca1f). After the answers: record each as an SI entry (si_add with the option
text) plus a notebook DECISION; then ask round 2 (section 4 dataset structure, section 5 margin r, section 6 P2 rule,
section 7 local-global, section 8 selection score S).

## Q1 · header "Scope" (spec section 0)
Question: Which analyses should be written as code and locked before X1, the main experiment, starts? Locking means
the exact calculation is fixed and tested before anyone sees results, so results cannot shape how they are analysed.
Two later experiments must be redesigned anyway, because the final review found their current designs cannot answer
their questions: X7 (does the choice of reference batch matter? = comparison P4) and X8 (does the critic-discriminator
gap change with the number of batches? = comparison P3). Neither can start before X1 is finished and scored.
1. Lock all X1 analyses now; the rest during X1 (Recommended). Now: P1 (biology kept at equal batch removal) with its
   trade-off area, P2's collapse rate, the local-versus-global test, the single-setting summary tables, the comparison
   with standard tools (best-of-k) and the penalty-window width. Later, while X1 runs and before anyone looks at X1
   results: P3, P4 and the other follow-up analyses, together with the X7/X8 redesign. The premise check (do
   discriminator gradients vanish?) and the downstream-biology panel become exploratory, so neither can be claimed as a
   confirmed result. Pros: every analysis fixed before its own data exist; X1 not delayed by the redesign. Cons: those
   two lose confirmatory status; a second lock-in step during X1.
2. Lock only P1 now. Only P1 with its trade-off area; everything else exploratory unless locked before its data are
   looked at. Pros: fastest to X1. Cons: most X1 results become exploratory.
3. Lock everything now, including P3/P4. P3 and P4 locked as currently designed, amended with dates after X7/X8 are
   redesigned. Pros: all locked before X1. Cons: P3/P4 code written for designs that will change; delays X1.

## Q2 · header "Reading P1" (spec section 2.4)
Question: P1 compares how much biology the reference critic and the discriminator keep when both remove the same
amount of batch effect. Each method has 5 runs (seeds) at each of 6 penalty strengths, plus plain scVI. We read off the
biology score at the target batch removal (half-way between plain scVI and the batch removal every method reaches;
rule R4). Example: batch score 0.60 at strength 1 and 0.70 at strength 3, target 0.65: the biology score is read
half-way between the biology scores at strengths 1 and 3. Should the 5 runs be averaged before reading, or read one by
one? Either way, a curve that crosses the target twice uses the first crossing (weakest strength), as rules R2/R3 do,
and the option not chosen is reported as a sensitivity analysis.
1. Average the 5 runs first (Recommended). Average batch and biology over the 5 runs at each strength, read the target
   once; uncertainty from repeating the reading 5 times, each time leaving one run out. The reading rules R2/R3 already
   use. Pros: one reading in the whole project; one noisy run moves the result less. Cons: the leave-one-out
   uncertainty is approximate.
2. Read each run separately. Read the target on each run's own curve, then average the 5 readings. Pros: uncertainty
   straight from the spread of the 5 readings. Cons: single runs are noisier; more double crossings; runs that never
   reach the target drop out.

## Q3 · header "Decoders" (spec section 2.5)
Question: Every X1 method runs with two decoders: conditioned (the decoder is told each cell's batch; how people use
scVI) and unconditioned (the adversary is the only batch correction; the original paper's setting). How should P1
treat them? In the pilot, 2 of 3 datasets (immune, sim1) had too little batch effect left with the conditioned decoder
to compare methods (plain scVI's batch score within 0.02 of the best every method reached), and such datasets are left
out of P1 by your rule SI-56.
1. Two results, one per decoder (Recommended). Both main results; never averaged together. Pros: each answers a clear
   question; the unconditioned result likely has more usable datasets. Cons: two main results instead of one.
2. Conditioned decoder is the main result (scVI as people use it, SI-09); unconditioned secondary. Pros: one main
   result. Cons: may rest on few datasets.
3. One result averaged over both decoders, adjusted for decoder. Pros: most data per estimate. Cons: averages two
   different situations; harder to interpret.

## Q4 · header "Failed runs" (spec section 2.4)
Question: Some runs fail (training diverges, or the result holds invalid numbers). How should failed runs enter P1 and
the trade-off area? Either way every failure is counted in P2, the collapse rate.
1. Leave failed runs out of P1 (Recommended). Each strength's score is the average of its successful runs, the rule
   already in the pre-registration (PREREG section 1); a strength where all 5 runs failed is skipped. Sensitivity: drop
   every strength with any failure. Pros: registered rule; failures still reported in P2. Cons: if a method fails
   often, its surviving runs may look better than typical.
2. Count a failed run as plain scVI (the paper plan's "impute the unintegrated score"). Pros: failures stay inside the
   P1 numbers. Cons: a failed run at a strong penalty would look as if it removed no batch effect, which can create a
   false crossing of the target.

## Q5 · header "Trade-off area" (spec section 3)
Question: P1's second main measure is the area under each method's trade-off curve: how much biology it keeps across
all 6 penalty strengths, not only at the target. Example: a point that removes 0.10 more batch than plain scVI and keeps
biology 0.05 above the floor adds a rectangle of 0.10 x 0.05. The area needs a fixed floor corner. In the pilot, plain
scVI's biology score was 0.67-0.90 and the methods moved it by -0.59 to +0.23.
1. Floor at the collapse line (Recommended). Corner at plain scVI's batch score and 0.10 below plain scVI's biology
   score (the collapse threshold of rule R1). Only settings that remove more batch than scVI and keep biology above the
   collapse line add area. Pros: biology losses change the area substantially. Cons: settings past collapse add
   nothing, however much batch they remove.
2. Floor at biology 0 (the review's example). Pros: simplest. Cons: every rectangle includes the 0.67-0.90 baseline,
   so the area mostly measures how far batch removal goes.
3. Make the area secondary (descriptive); the reading at the target is P1's only main result. Pros: fewer main
   results. Cons: changes the registered plan, which named the area co-primary.
