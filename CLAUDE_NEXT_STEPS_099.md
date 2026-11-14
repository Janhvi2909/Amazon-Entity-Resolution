Execution is paused. I won’t run commands, modify files, or interfere with Claude’s jobs.

**Your confirmed score is 0.982564. Reaching 0.99 requires another +0.007436.** That means eliminating about **43% of the remaining leaderboard loss**. No measured experiment currently supports promising that gain.

Below is the sequence I would give Claude. The existing written handoff is [CURRENT_PROGRESS_AND_NEXT_STEPS.md](CURRENT_PROGRESS_AND_NEXT_STEPS.md).

**1. Finish Claude’s current experiments, but don’t spend an upload automatically.**

At my last verified check:

| Variant | Change | Status/result |
|---|---|---|
| V3d | Average five stackers, 63 leaves | Finished: validation **0.989650**, essentially tied with V3 |
| V3e | Five stackers, 127 leaves, learning rate 0.02 | Running |
| V3f | Five stackers, 31 leaves | Queued after V3e |

Ask Claude to report for each:

- ES score used for selection, held-out validation score, and US/India scores.
- Precision, recall, singleton performance.
- Number of validation entities improved and worsened versus V3.
- Number of test businesses changed versus V3, separately for France, India and US.
- Whether the predictions come from all five models or only one.

**Do not assume bagging adds a gain.** V3d has not shown one. V3b changes only 3,894 test businesses; it cannot close the full-test-equivalent gap by itself.

**2. Separate the remaining loss before choosing another expensive experiment.**

Have Claude produce an error table for the **current M2 fusion**, not the older M-v5 model:

| Error group | Definition | Potential intervention |
|---|---|---|
| Missing candidate | True target absent from M2 candidates | Additional retrieval |
| Below neural band | Candidate exists but never receives CE review | Selective band expansion |
| Scored but rejected | CE/model scored it, final probability too low | Better features or calibration |
| Lost to another S1 | Rival business wins the target | Better competitor evidence |
| False match | Predicted target belongs elsewhere or is unmatched | Better precision controls |
| Singleton mistake | Empty/nonempty decision is wrong | Entity-level decision improvement |

For every group, request:

- Country and source breakdown.
- Address present versus absent.
- Number of affected businesses—not just links.
- **Oracle macro-F improvement if that group were corrected perfectly.**

This identifies whether a proposed method has enough room to matter. Recovering a business’s only match can be much more valuable than recovering its fifth.

**3. Audit the French review before adding another model.**

France remains an important hypothesis because previous CE additions helped the leaderboard more than US/India validation. **Its actual score is unknown.**

Ask Claude to verify:

1. Did the intended French confident pairs all receive scores?
2. Were failed or restarted inference chunks omitted?
3. Do repeated pair scores come from the same checkpoint, tokenizer, maximum length and input text?
4. Are large duplicate-score disagreements concentrated in particular files?
5. Is the stacker seeing comparable confident-pair examples during training and inference?
6. Were confident training rows sampled differently from uncertain rows, and could that distort calibration?

The CE audit found **76,077 duplicated pair keys**, including **23,185 with score differences above 0.001**. That does not prove corruption, but “all differences are padding noise” should not be assumed.

If coverage or provenance is wrong, fix that first. A valid correction is more defensible than another arbitrary ensemble.

**4. Highest-priority new experiment: selective review below the existing CE band.**

The current neural models mostly review pairs selected by tree-model probability. A confidently wrong tree can prevent a useful candidate from receiving neural review.

Ask Claude to:

1. Find validation positives that exist in M2 but fall outside the reviewed set.
2. Examine which **label-free signals** identify them:
   - Strong address agreement despite a different business name.
   - Strong name agreement with missing address.
   - Disagreement between old M-v5 stage-2 and M2 stage-1 probabilities.
   - Similarity to another confidently matched record of the business.
3. Define a compact selection rule using those signals.
4. Apply that rule to **all eligible candidates**, including negatives—not only known missed positives.
5. Score a manageable ES pilot using the existing trained CEs.
6. Train/calibrate the expanded stack on corresponding REST examples.
7. Choose its decision settings on ES, then evaluate on EVAL.
8. Run test inference only if the measured benefit and remaining runtime justify it.

**Critical:** scoring only known validation misses produces an optimistic diagnostic, not a deployable improvement.

This branch can address errors untouched by seed averaging. Its benefit is still unmeasured.

**5. Add retrieval only if the missing-candidate oracle shows enough gain.**

M2 already deepened sparse retrieval. The next channel should recover genuinely different candidates.

The earlier multilingual-e5-small result—recovering 38% of known residual misses—was a **diagnostic**, not an end-to-end gain.

For a real experiment:

1. Define which businesses receive extra retrieval using inference-time information.
2. Retrieve a small number of candidates per target source.
3. Merge with existing candidates and deduplicate.
4. Compute the same matcher features.
5. Score both true and false new candidates.
6. Recompute target competition and final set decisions.
7. Measure the actual macro-F change and inference cost.

Potential query groups include businesses with no confident matches, strong model disagreement, or ambiguous name-only evidence. Validate the selection rule rather than assuming these groups are beneficial.

**Do not start a fresh all-record dense run if it cannot finish scoring, validation and export before the cutoff.**

**6. For contested targets, improve evidence—not arbitrary assignment rules.**

If the error table shows meaningful loss from wrong target ownership, test:

- Best and second-best candidate-business scores.
- Margins from each CE, not just the tree model.
- Whether the target agrees with other confident records of each competing business.
- Name frequency and missing-address indicators.

Use out-of-fold or held-out predictions for training those features. Otherwise, the apparent improvement can come from training leakage.

Do not arbitrarily force assignment to an underfilled business. The observed source maxima are not guaranteed test constraints.

**7. Do not repeat these techniques tonight.**

These have already produced little gain or caused harm:

- Blanket nearby-house-number filters.
- Name-only propagation through matched targets.
- Generic gate searches as the main improvement strategy.
- Exact expected-F calculation: only about +0.000009 validation.
- Entity-size weighting: below the unweighted refined stack.
- All-country MuRIL.
- More stacker seeds without evidence that prediction quality improves.

Target-to-target support can still be useful as a **model feature or retrieval proposal**, but unconditional name-only linking was harmful.

**8. Use the remaining two uploads deliberately.**

For the next upload:

- Prefer a completed candidate with a defensible mechanism, validation evidence, and enough changed businesses to matter.
- Keep the scored V3 file untouched.
- Record its exact hash and architecture before uploading.
- Do not label an unscored variant “better” based only on validation.

For the final upload:

- Let the next portal result guide the choice.
- If a combined candidate regresses, consider a country-spliced variant to isolate the French change.
- If it improves, add only another independently supported change.
- Confirm whether the portal uses the best, last, or explicitly selected submission.

**A country splice is an experiment, not a guaranteed improvement.** Two uploads cannot identify every component’s effect.

**9. Preserve enough time to submit correctly.**

Using the earlier deadline plan:

- **22:45:** stop launching substantial new training or retrieval.
- **23:10:** choose and freeze the final model and matching file.
- **23:35 onward:** reserve time for package upload and corrections.

The final package must contain:

- The exact selected matching file.
- The complete candidate set actually scored.
- Reproduction commands and all ensemble members.
- Correct dependency versions and model licenses.
- Validation reports and hashes.
- Documentation distinguishing validation results from portal results.

V3’s current candidate file contains a post-scoring top-15 subset. Fix its candidate provenance for the package without silently replacing its scored predictions with a different variant.

**What to tell Claude now:**

> Finish V3e/V3f and report paired validation plus changed test entities versus scored V3. Do not upload merely for a small validation increase. Next, decompose current M2-fusion misses into absent candidates, unreviewed low-score candidates, rejected reviewed candidates and wrong target ownership, with oracle macro-F gains. Audit French review coverage and duplicate CE provenance. Prioritize a small, label-free below-band CE expansion pilot if it has sufficient recoverable loss and can finish before the cutoff. Preserve V3 at 0.982564 and both remaining uploads until a stronger candidate is justified. Keep current progress, failures, commands and artifact hashes updated in the handoff.
