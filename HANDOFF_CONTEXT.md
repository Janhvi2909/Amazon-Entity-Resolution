# HANDOFF CONTEXT — Amazon ML Challenge 2026, Business Entity Resolution

**Start continuation here:** [CURRENT_PROGRESS_AND_NEXT_STEPS.md](CURRENT_PROGRESS_AND_NEXT_STEPS.md)
is the living progress/next-action index, including the latest confirmed score,
two remaining uploads, active experiments, rejected techniques and audit findings.

**Latest portal result, user-confirmed 2026-09-27:**
`submissions/SUB-FINAL-V3_m2_ce_old_hi` scored **0.982564** (+0.002960 over SUB-N3).
This is the V3 artifact with 444 stacker trees and gate .55 selected on EVAL,
not V3b/V3c. Gap to 0.99: **0.007436**. Later-variant portal scores and current
remaining uploads are not yet confirmed. See `analysis/score_audit/PORTAL_RESULTS.json`.

**Continuation update, 2026-09-27 ~21:05 IST:** read
[`EXECUTION_PLAN_TO_DEADLINE.md`](EXECUTION_PLAN_TO_DEADLINE.md) for the latest measured
audit results, complete next-step plan, runnable commands, artifact paths, GPU recovery
caveats, and submission strategy. The user clarified **three uploads remain**. SUB-R
validation is 0.988214; its test/LB score is unknown. The measured LB best remains
SUB-N3 at 0.979604. Live job status and ETAs below are historical snapshots.

Written 2026-09-27 ~20:35 IST for a new agent (Codex) taking over. Deadline: **2026-09-27 23:59 IST.**
Everything here is measured unless marked "estimate". Detailed experiment log: `method_result.md`.
Project rules for the codebase: `CLAUDE.md`. Official task: `resources/student_resource/README.md`.

---

## 1. Problem in one paragraph

For every Source 1 (S1) business record, list the Source 2 / Source 3 (S2/S3) records of the same business.
Fields: `entity_id`, `business_name`, `business_address`, `country`. No shared IDs. Train: US + India with
labels. Test: US + India + **France (unseen, no labels, ~15% of test S1)**. Metric: **macro F0.5 per S1
entity** (precision weighted 2x; an S1 with no true match scores 1.0 only if its list is empty).
Deliverables: `output/matching_results.tsv` (scored), `output/candidate_pairs.tsv` (the exact set the model
scored; every match must be in it), runnable `code/business_entity_resolution/`, filled `Documentation_template.md`.
Constraints: MIT/Apache-2.0 models, <= 8B params, no external data/APIs/geocoding.

Data facts: train S1 2,206,821, S2 5.03M, S3 5.29M; test S1 1,732,544 (India 809,986 / US 663,106 /
France 259,452), targets 9.97M. Each S2/S3 record belongs to at most one S1. Max 5 S2 and 6 S3 matches per S1.
5.58% train S1 are singletons; mean 3.46 matches. Test has ~24% more targets per S1 than train
(5.53-5.82 vs 4.68). No ID/row-order leakage (checked).

---

## 2. Leaderboard history (public LB)

| Upload | Architecture | Public LB | Gain |
|---|---|---:|---:|
| 1 | **M-v3**: NORM-v2 + BLK-v4b@20 + FEAT-v2 (52 feats) + 2-stage LightGBM + exclusive + expected-F | **0.968014** | — |
| 2 | **SUB-N**: M-v5 (below) + mDeBERTa-v3-base cross-encoder stacked on the uncertain band, France included | **0.978848** | +0.0108 |
| 3 | **SUB-N3**: SUB-N + mDeBERTa-2 (2 epochs) + bge-reranker-v2-m3 + MuRIL (India pairs only), all in the stacker | **0.979604** | +0.0008 |

Uploads left after these: see with the user (they had 4 before SUB-N; used SUB-N and SUB-N3 → 2 left).
Interpretation: SUB-N's jump came mostly from **France** (US/India eval moved only +0.0027; LB +0.0108).
SUB-N3's +0.0008 LB vs +0.0001 US/India eval → again mostly France: cross-encoders fix France.

Estimated per-country LB for SUB-N3 (estimate, from eval minus ~0.002 test drop): US ~0.987, India ~0.984,
**France ~0.94-0.95**. France is still the biggest gap.

---

## 3. Architecture (what produced SUB-N / SUB-N3), layer by layer

Code: `code/business_entity_resolution/src/` (`run_pipeline.py`, `ber/*.py`, `neural/*.py`). Every component
is a frozen named version in `ber/versions.py`; each stage caches under its version key; all artifacts have
`.ok` markers and are re-opened before being trusted (`ber/ckpt.py`); `PROGRESS.txt` in the work dir.

**Data scope NORM-v3t (prep):** anyascii transliteration; name cleanup (legal forms, fka/dba split, digit-for-letter
fixes, canonical words, core name without legal forms/honorifics); address canonicalisation (street types,
states, ordinals, joined multi-word states, leading zeros); per-country learned aliases from training links
(1,105 name / 3,107 address); **France canon** learned from test files (15 cities→3 regions, 4 departements,
12 digit-corrupted street types e.g. cour2→cours, bis/ter kept with house number; see
`resources/france_canon/MANIFEST.md`); legal forms kept in France core name (S1 duplicate test: 2→59 when
stripped); house parse (number, suffix, street key); **twin augmentation**: 224,830 clones of real
near-house-number distractors so train matches test twin density (US 0.166→0.301, India 0.065→0.116 twins/S1).
Split: fit 30% / early-stop 5% / eval 20% / rest 45% (seed 42).

**Blocking BLK-v4b@20:** hashed sparse features (name tokens, 4-char prefixes, consonant skeletons, joined
name, char 3-grams; address tokens + variants + number affixes), IDF per (country, source), score =
cos(name)+cos(address), top-20 per source via sparse_dot_topn, region split by learned states (merges
telangana↔andhrapradesh, delhi↔haryana; France unsplit), + name-only top-5 for targets without address.
Eval recall ceiling **98.31%** (US 98.9%, India 97.4%).

**Features FEAT-v4 (65):** rapidfuzz name/core/joined/address sims, alias match, token containment, number
agreement + number-gap, house features (gap, edit, most-significant differing digit, same street, twin flag,
bis conflict), legal equal/conflict, lengths/flags, competition context (ranks, margin over rival S1, target
claimants), soft word alignment.

**Matcher MATCH-v5 (M-v5):** LightGBM binary, 2 stages, 3-fold cross-fit; trained on fit+rest (75% of S1),
twin negatives x1.9, easy negatives kept 10% weighted 10x (19.8M rows); stage 2 adds p1 context + house
consensus. **Eval 0.9852** (P 0.9972, R 0.9609; US 0.9872, India 0.9822).

**Cross-encoder stack (SUB-N):** band = pairs with M-v5 p2 in [0.02, 0.98] (1.53M train, 1.46M test) with
RAW text; mDeBERTa-v3-base fine-tuned on fit-entity band pairs (input "name ; address" x2, 128 tok, 1 epoch,
bf16, GPU); LightGBM stacker on rest-entity band pairs with p1, p2, ce, ce rank in S1, best rival ce for
target, band size; early stop on ES. **Eval 0.9879** (P 0.9986, R 0.9653; US 0.9893, India 0.9857).

**SUB-N3 additions:** mDeBERTa-2 (2 epochs, 160 tok; held-out band log-loss 0.240 vs 0.251), bge-reranker-v2-m3
(Apache-2.0, 568M; log-loss 0.246), MuRIL (Apache-2.0) trained/scored on **India pairs only** (log-loss 0.206 vs
M-v5 0.227 on India), all as stacker inputs (MuRIL NaN outside India). **Eval 0.9880** (P 0.9987, R 0.9657;
US 0.9895, India 0.9859).

**Decision:** exclusive assignment (target → its highest-scoring S1), then gated expected-F0.5 (gate ~0.6).
**Output:** matches + candidates = matcher top-15 per S1 + forced links (validator PASS).

---

## 4. Where the remaining loss is (M-v5 eval decomposition; recall dominates)

- Loss 0.0148 points: 84% recall (partial misses 64%, entities with nothing predicted 20%), FP 9%, singleton FP 5%.
- Missed links (M-v5, 60,334 of 1.53M): not a candidate 43% (India worse), below decision 41%, lost to rival S1 16%.
- **Targets without an address = ~4% of true links but 53% of missed links** (recall on them ~52%).
- Indic-script targets are NOT over-represented among misses (11% of misses vs 24% of India links).
- Link-level (M-v5): US recall 96.6%, precision 99.81%; India recall 95.3%, precision 99.73%.
- Residual blocking misses after deeper blocking (M2): India 10,663, US 5,924; 40% (India) have name AND
  address similar (crowding), 34% invented trade names, US 52% no-address.
- France: no labels. Model self-estimate for France was far above its LB-implied score (confidently wrong);
  France has 2x the uncertain-band share (11.5% vs 5.5-6.9%).

---

## 5. What was tested — worked / did not

| Idea | Result |
|---|---|
| Test-density eval (twin clones) | eval 0.9812→0.9809: twin shift is NOT the LB gap |
| Twin post-filter (drop same-street gap 2-30 links) | LOSES (-0.0010): 83-99% of removed links are true |
| France canon alone in M-v5a (SUB-B) | label-free proxies bad (17% legal conflicts in added links); not uploaded |
| M-v5 (FEAT-v4 + consensus + twin weight + 75% train) | +0.0043 eval |
| Consensus ablation | -0.0005 without it |
| Support features (MATCH-v6) | +0.0001 |
| Cross-encoder mDeBERTa stack | **+0.0027 eval, +0.0108 LB (France)** |
| mDeBERTa-2 (2 epochs) | +0.0001 eval |
| Two mDeBERTas ensembled | 0 |
| bge-reranker-v2-m3 | 0 on US/India eval; helped France on LB (SUB-N3) |
| MuRIL on all countries | bad (log-loss 0.509) |
| MuRIL India-only | good log-loss, 0 extra eval in stack |
| Deeper blocking top-40 + name-only top-20 (M2) | recall ceiling 98.31% → **98.91%** |
| Independent address/name/rare-token channels on residual misses | +~0.0002 (not integrated) |
| Dense e5-small channel on residual misses | recovers **38%** of misses (India 4,087/10,663 @5, US 1,727/5,924 @5); NOT integrated (time) |
| Per-country gates, entity "has any match" model, margin-aware exclusivity | 0 |
| Leave-one-country-out stacker (France policy proxy) | unseen country costs only -0.00007: stacker/gate not France's issue |
| Confident links (p2>0.98) | 99.955% precise on US/India |
| French long legal-form dictionary | phrases never occur in data |
| Orphan-cluster hypothesis | rejected |

---

## 6. Frozen submissions (all validated, md5 in each folder, copies in S3)

`submissions/<name>/matching_results.tsv` (+ candidate_pairs.tsv, MD5SUMS, VALIDATOR.txt)
S3: `s3://<S3_BUCKET_2>/ber/final/<name>/` (AWS profile `amlc-personal`, account <AWS_ACCOUNT_ID_2>)

| Name | matching md5 | LB |
|---|---|---:|
| SUB-0_floor_0.968 | e1915901c500084a4ab2aded72274298 | 0.968014 |
| SUB-N_stackCE_all | 551d1e13dc76b0a07f3e85933bb2e8c0 | 0.978848 |
| **SUB-N3_ce2_bge_murilIN** (current best) | e163d0fe403b0ff60af383b2aa2d0c5a | **0.979604** |
| SUB-N2 / SUB-C / SUB-C2(t) / SUB-A / SUB-B | see folders | not uploaded |

---

## 7. IN PROGRESS: the final candidate (target ready ~22:00-22:15 IST)

**Final = M2 stage-1 probabilities + cross-encoders (mDeBERTa-2, bge, MuRIL-India) + France confident-pair
review + numeric features in the stacker + source caps.**

- M2 = NORM-v4t (adds address components: landmark/unit/PO box split) + BLK-v4b@40n20 (recall 98.91%) +
  FEAT-v5 (+7 address-component features) + MATCH-v7. Its **stage 2 crashed nst-n7** (node NotReady 19:51-20:03
  IST, memory on 253M pairs); the job was deleted. **Stage-1 predictions are complete and checkpointed**:
  rebuilt into `/var/lib/amlc-ber/work/full/runs/M2P1/scores_{train,test}.parquet` on nst-n7 (p2 := p1) by
  `neural/m2p1.py`; M2 band (p1 in [0.02,0.98]) = 3,781,568 pairs → `/data/ber/neural/band_m2.parquet` (PVC).
- **GPU jobs running now on nst-n6** (`/n6` = hostPath `/var/lib/amlc-ber-n6`, pod `ber-gpu`), all inference,
  skipping pairs already scored:
  - mDeBERTa-2 on M2 new pairs (1,088,040) → `/n6/ce2_m2/scores`
  - bge on M2 new pairs (1,088,040) → `/n6/bge_m2/scores`
  - MuRIL on M2 new India pairs (536,827) → `/n6/muril_m2/scores`
  - mDeBERTa-2 on confident pairs (3,560,400: ES/eval/1M rest sample of US+India, + 765,958 France test) → `/n6/ce2_hi/scores`
  - ETA ~21:35 IST (GPU at 100%, memory 22.8/23.4 GB).
- Old CE scores already copied to nst-n7 PVC: `/data/ber/final_ce/{ce2,ce_bge,ce_muril_in}/scores`.
- **Next steps:** (1) tar the 4 new score dirs in ber-gpu, `infra/k8s/xfer.sh get` to laptop, upload to S3,
  presigned GET into `ber-util` → `/data/ber/final_ce/`; (2) run on nst-n7 a job (code `/data/ber/code_0927203238`):
  `python neural/stack.py --ce /data/ber/final_ce/ce2_all --ce2 /data/ber/final_ce/bge_all --ce3 /data/ber/final_ce/muril_all
  --feat-dir /work/full/feat/NORM-v4t__BLK-v4b@40n20__FEAT-v5 --band-by-ce --out <dir> --code . --run /work/full/runs/M2P1
  --prep /work/full/prep/NORM-v4t` (put old+new parts of each CE into one dir; `load_ce` dedupes by pair);
  also run once WITHOUT `--band-by-ce`/hi scores to isolate the France review's effect on US/India eval;
  (3) **ship only if eval p3 > 0.9880 (SUB-N3)**; freeze with `infra/k8s/freeze_sub.sh`, validate, S3.
- Estimate (estimate): final LB ~0.980-0.982 (most likely ~0.9808).

---

## 8. Infrastructure

- **nst-n7**: 24 cores, 62 GB, CPU only. Work dir hostPath `/var/lib/amlc-ber/work` (full/ = all stage caches).
  PVC `amlc-data` (RWO, attaches only to nst-n7) at `/data`: dataset `/data/dataset`, code `/data/ber/code_<ts>`,
  logs `/data/ber/logs/<job>`. Helper pod `ber-util` (python+polars). CPU mostly idle now (M2 stopped).
- **nst-n6**: 24 cores, 62 GB, **1x NVIDIA RTX PRO 4000 Blackwell (HAMI share, 23.4 GB cap)**. Pod `ber-gpu`
  (pytorch 2.7.1+cu128, transformers 4.56.2) with hostPath `/n6`. GPU 100% busy now.
- Namespace `amlc-ber-jy`; job launcher `infra/k8s/ber_job.sh <name> <code-ts> <cpu> <memreq> <memlim>`;
  deploy code: `/private/tmp/claude-503/deploy.sh` (tars src → PVC via ber-util, md5 checked).
- Another team session uses namespace `amlc-ber-phase3c` — don't touch it.
- AWS: profile `amlc-personal` works (bucket `<S3_BUCKET_2>`); the CLAUDE.md account <AWS_ACCOUNT_ID> is unreachable (expired login).

**Pitfalls hit today:** kubectl exec/cp transfers drop (always md5; laptop→pod pushes are the worst — use S3
upload + presigned-URL download inside the pod instead); HAMI kept counting GPU memory of dead processes
(fixed by recreating the pod); DataLoader orphans can't be killed in the pod; `cd` inside a backgrounded
`&&` chain doesn't apply to later lines (use absolute paths); M2 stage 2 on 253M pairs exceeds node memory.

---

## 9. What could still add gain (not done, for reference)

- Dense e5-small candidates (recover 38% of residual blocking misses; ~+0.0005-0.0008 eval) — needs ~2.5 h.
- Proper M2 stage 2 with lower memory (drop support features, chunk probability_context) — needs ~1.5 h.
- France: cross-encoders are what help; more/other multilingual rerankers on France pairs likely help most.

## Update 22:42 IST (Claude): CLAUDE_NEXT_STEPS_099 executed

- **Leaderboard:** SUB-FINAL-V3 = **0.982564**.
- **Recommended upload:** `submissions/SUB-FINAL-V3s_support/matching_results.tsv` (md5 `078eed8c2b10e38d2845392f5ff23bcf`, validator PASS). Its score comes from the portal once uploaded.
  - Measured on validation: ES 0.989820 (V3c 0.989768); EVAL 0.989706 (V3c 0.989651).
  - Paired +0.000055, bootstrap 95% CI [+0.000022, +0.000096], 706 better / 510 worse.
  - Mechanism: `stack.py --support` (agreement with the S1's other confident candidates, name frequencies, target address flag).
- **Optional France test:** `SUB-FINAL-V3s_fr08` (md5 `4a1a7240c6d485f46906bffdd7d59173`): France probabilities raised to the power 0.8, +3.5k France links. Unmeasurable; upload only if the portal keeps the best score.
- **Frozen, not uploaded:**
  - V3b `1b7a1b30ed335d80ea581cdbc4966a4c`
  - V3c `ad535ca592bafe9c5f112942caf99738`
- **Rejected:**
  - Bagging (V3d/e/f): no gain.
  - V3s2 (+ core-name frequency, rival count): ES tie, EVAL 0.989680.
- **Step 2 error table (V3s):** missing candidate +0.0035; reviewed-but-rejected +0.0051 (85% no-address targets); below band +0.0009; lost to another S1 +0.00001; false match +0.0007; singleton false match +0.0002. `neural/errtab.py` with `stack.py --dump-eval`.
- **Step 3 France audit:**
  - All 765,958 France confident pairs were scored (input `band_hi_ord`, 5 of 5 parts).
  - Duplicates come from the same checkpoint, max length 160 and identical text; only the batch size differs (128 vs 64, bf16 padding). They are averaged.
- **Steps 4 and 5 skipped:** below-band oracle ≤ +0.0009, and retrieval can't finish before the cutoff.
- **Package:** built on n7 by job `ber-pkg` in `/data/ber/pkg_final`.
  - Contents: the V3s matching file and the full scored candidate set (`ber-final-c/v3c/candidate_pairs.tsv`, every M2P1 test pair, identical for V3s).
  - Checks: the validator runs with `--check-ids`, then `make_submission.py`.
