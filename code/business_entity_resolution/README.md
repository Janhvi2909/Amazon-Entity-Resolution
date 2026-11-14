# Business Entity Resolution

End-to-end pipeline for the Amazon ML Challenge 2026: for every Source 1 record, find the
Source 2 / Source 3 records of the same business. Stages: normalise → block (candidate
generation) → pairwise features → two-stage LightGBM → per-entity decision → the two output
files.

## Setup

Python 3.12.

```bash
pip install -r requirements.txt
```

Every dependency is open source; the model is LightGBM (MIT). No external data, APIs or
lookups are used — everything is learned from the training files.

## Data

Place the challenge dataset at `resources/student_resource/dataset/` (with `train/` and `test/`).
It is not in git: the files exceed GitHub's size limit.

## Run

From `src/`:

```bash
python run_pipeline.py \
  --data ../../../resources/student_resource/dataset \
  --work ../../../work \
  --out  ../../../output \
  --validator ../../../resources/student_resource/utils/validate_submission.py
```

This writes `output/matching_results.tsv` (the scored file) and `output/candidate_pairs.tsv`
(the exact candidate set the model scored), then runs the official validator on both. A full
run peaks at about 45 GB of RAM (during blocking); on 8 cores, normalisation takes about 6
minutes and blocking about 20. `--sample 0.01` runs the same pipeline end to end on 1% of the
data in about a minute.

Stages can be run separately (`--stages prep,block,features,train,predict`); each one reads
the previous stage's files from `--work`, so an interrupted run resumes where it stopped.

## Versions

Every component is a named version in `src/ber/versions.py`, using the IDs of the
experiment log (`method_result.md`). A run is one combination of the four:

| Component | Stage | Current |
|---|---|---|
| NORM — normalisation and learned aliases | prep | NORM-v2 |
| BLK — candidate generation | block | BLK-v4b@20 |
| FEAT — pairwise features | features | FEAT-v2 |
| MATCH — models and decision rule | train, predict | MATCH-v2 |

```bash
python run_pipeline.py --list                          # every version and preset
python run_pipeline.py ... --preset M-v4               # another end-to-end version
python run_pipeline.py ... --feat FEAT-v3 --stages features,train,predict   # swap one component
```

Each stage caches its output under the versions it depends on (for example
`work/block/NORM-v2__BLK-v4b@20/`), so swapping the matcher reuses the blocking, and a new
blocking version never reuses stale features. `work/runs/<versions>/metrics.json` records the
versions, the data scope and every stage's results. A registered version is never edited;
a change is a new version.

## Final submission (neural stack): SUB-FINAL-V3, public leaderboard 0.982564

The submitted `output/matching_results.tsv` (md5 `fffea65157b36546bbf566889ce23ca4`) is not a single
`run_pipeline.py` preset: it stacks two LightGBM runs and three cross-encoders with a LightGBM stacker,
**`src/neural/stack_v3.py`** (the exact stacker version that produced the file; `src/neural/stack.py` is a later
version with post-submission experiments, see the end of this section). Paths below are those of our cluster
(`/work/full` = work dir, `/data/dataset` = dataset); replace them with yours.

Models (all pretrained public checkpoints, fine-tuned **only on the challenge training files**; no
external data, APIs or lookups; every model is MIT / Apache-2.0 and far below 8B parameters):

| Model | Licence | Parameters | Use |
|---|---|---:|---|
| LightGBM (stage 1, stage 2, stacker) | MIT | — | pair classifier and stacker |
| microsoft/mdeberta-v3-base | MIT | 278M | cross-encoder, all countries |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 568M | cross-encoder, all countries |
| google/muril-base-cased | Apache-2.0 | ~237M | cross-encoder, **India pairs only** |

Hardware: CPU stages on a 24-core / 62 GB node (blocking and features are the memory peak);
cross-encoders on one NVIDIA RTX PRO 4000 Blackwell (24 GB, bf16). Neural packages:
`requirements-neural.txt`.

1. **Two LightGBM runs** (from `src/`):
   ```bash
   # M-v5: NORM-v3t + BLK-v4b@20 + FEAT-v4 + MATCH-v5 (stage 1 + stage 2) -- the "old" run
   python run_pipeline.py --data /data/dataset --work /work/full --out out_mv5 --preset M-v5
   # M2: NORM-v4t + BLK-v4b@40n20 + FEAT-v5 + MATCH-v7, stage 1 only (stage 2 on 253M pairs exceeds 62 GB)
   python run_pipeline.py --data /data/dataset --work /work/full --out out_m2 --preset M-v7 --stages prep,block,features,train
   python neural/m2p1.py /work/full/runs/M2P1        # score tables from M2's stage-1 predictions (p2 := p1)
   ```
   **Candidate set (blocking):** BLK-v4b@40n20 on NORM-v4t, i.e. every test pair in
   `/work/full/runs/M2P1/scores_test.parquet` (200,471,427 pairs, 115.7 per Source 1 record). The stacker scores
   every one of these pairs, so this is also the exact scored set.
2. **Pair bands with raw text** (`neural/band.py <out> [run] [prep]`, `neural/band_hi.py <out> [run] [prep] [lo] [hi]`):
   ```bash
   python neural/band.py band.parquet                                   # M-v5 pairs with p2 in [0.02, 0.98]
   python neural/band.py band_m2.parquet /work/full/runs/M2P1 /work/full/prep/NORM-v4t   # M2 pairs with p1 in [0.02, 0.98]
   python neural/band_hi.py band_hi.parquet /work/full/runs/NORM-v3t__BLK-v4b@20__FEAT-v4__MATCH-v5 /work/full/prep/NORM-v3t 0.98 1.0
   ```
   High-probability review band: `band_hi_sub.parquet` keeps every early-stop and eval row, 1,000,000 rest rows
   (`seed=7`) and every France test row (`neural/select_band.py` documents this selection). The band actually
   scored for the submission, `band_hi_ord.parquet` (2,024,643 rows), is drawn from it with polars:
   ```python
   b = pl.read_parquet("band_hi_sub.parquet")
   o = pl.concat([b.filter(pl.col("split") == "test"),                                        # all France test rows
                  b.filter((pl.col("split") == "train") & (pl.col("role") == 1)),             # all early-stop rows
                  b.filter((pl.col("split") == "train") & (pl.col("role") == 3)).sample(400000, seed=11),  # rest
                  b.filter((pl.col("split") == "train") & (pl.col("role") == 2)).sample(500000, seed=12)]) # eval
   o.write_parquet("band_hi_ord.parquet")
   ```
   The MuRIL bands keep `country == "India"` only.
3. **Cross-encoders** (`neural/ce.py`; train on fit entities, validate on early-stop entities, then score
   early-stop, eval, rest and test rows):
   ```bash
   python neural/ce.py --band band.parquet --out ce2 --model <mdeberta-v3-base> --bs 64 --maxlen 160 --epochs 2 --lr 2e-5
   python neural/ce.py --band band.parquet --out ce_bge --model <bge-reranker-v2-m3> --bs 32 --maxlen 128 --epochs 1 --lr 1e-5
   python neural/ce.py --band band_india.parquet --out ce_muril_in --model <muril-base-cased> --bs 64 --maxlen 160 --epochs 2 --lr 2e-5
   # score the M2 band's new pairs and the review band with the trained models (inference only)
   python neural/ce.py --band band_m2.parquet --out ce2_m2 --model ce2/model_final --model-dir ce2/model_final --stages infer --bs 64 --maxlen 160 --skip-scored 'ce2/scores/part-*.parquet'
   python neural/ce.py --band band_m2.parquet --out bge_m2 --model ce_bge/model_final --model-dir ce_bge/model_final --stages infer --skip-scored 'ce_bge/scores/part-*.parquet'
   python neural/ce.py --band band_m2_india.parquet --out muril_m2 --model ce_muril_in/model_final --model-dir ce_muril_in/model_final --stages infer --maxlen 160 --skip-scored 'ce_muril_in/scores/part-*.parquet'
   python neural/ce.py --band band_hi_ord.parquet --out ce2_hi --model ce2/model_final --model-dir ce2/model_final --stages infer --bs 128 --maxlen 160
   ```
   Use a fresh `--out` directory per run (cached parts are accepted on row count). Inference runs in bf16 on
   length-sorted batches, so a pair scored at a different batch size can differ by about 1e-3.
4. **Stacker and decision** (gather each cross-encoder's `scores/` dirs under one directory:
   `ce2_all/{old,m2,hi}` = `ce2/scores`, `ce2_m2/scores`, `ce2_hi/scores`; `bge_all/{old,m2}`; `muril_all/{old,m2}`):
   ```bash
   python neural/stack_v3.py --ce ce2_all --ce2 bge_all --ce3 muril_all \
     --feat-dir /work/full/feat/NORM-v4t__BLK-v4b@40n20__FEAT-v5 --code . \
     --run /work/full/runs/M2P1 --prep /work/full/prep/NORM-v4t \
     --old-run /work/full/runs/NORM-v3t__BLK-v4b@20__FEAT-v4__MATCH-v5 \
     --band-by-ce --out final
   ```
   The stacker (LightGBM, seed 42) is trained on the rest entities' CE-scored rows and early-stopped on the
   early-stop entities. The decision is exclusive assignment plus gated expected-F0.5, with the rule and gate chosen
   on the eval entities (gated expected-F, gate 0.55, eval F0.5 0.98965); on test, per-entity caps of 5 Source 2 /
   6 Source 3 links (the training maxima) are applied afterwards, keeping the most probable. A pair scored twice
   keeps the first score in sorted path order (`hi/` before `m2/`). It writes `final/matching_results.tsv`
   (5,751,431 links), `final/stack_eval.json` and a top-15 `final/candidate_pairs.tsv`. The packaged
   `candidate_pairs.tsv` is the full scored set of step 1, exported by `neural/stack.py ... --cand-full` (same
   run and pairs; md5 `ac4c44e927509b6809bb8c37e9c23800` in test-file row order). Then run the official
   validator on both files. To fit the 1 GB upload limit with standard zip deflate, the packaged copy lists the
   same 1,732,544 rows grouped by (country, last address token, normalised name) instead of test-file order, so
   records sharing candidates compress together (md5 `11e2f2e6e0007fd6b0ac527505046177`); the validator reads
   the file by Source 1 id, so row order does not matter.

**Post-submission experiments in `neural/stack.py`** (not part of the submitted file):
- `--protocol es` (gate chosen on early-stop entities) and duplicate CE scores averaged: V3c.
- `--support` (name agreement with the S1's other confident candidates, name frequencies): V3s, eval 0.98971.
- `--anchor` (target-to-target cross-encoder evidence): rejected, paired eval -0.000005.
- `--unseen-alphas` / `--unseen-gates` (France-only calibration variants), `--seeds` (bagging: no gain).
- `neural/errtab.py` (with `--dump-eval`) prints the eval error table.

## Validation

Train S1 entities are split once (seed 42): fit 30%, early stopping 5%, **evaluation 20%**,
rest 45%. All entities stay in the candidate search, so targets are contested as densely as
at test time. Aliases and region merges are learned without the evaluation entities' links,
and the models are cross-fitted, so evaluation rows are scored exactly like test rows. The
decision rule and its parameter are chosen on the evaluation slice (the submitted V3 stack; the
later `--protocol es` variants choose them on the early-stop slice).

## Layout

```
src/
  run_pipeline.py      the pipeline (stages above)
  sagemaker_entry.py   entry point when the pipeline runs as a SageMaker training job
  ber/                 normalize, aliases, blocking, features, model, metrics, data, versions
  experiments/         DEV-10 experiments behind the versions (see method_result.md)
  neural/              final-submission branch: bands, cross-encoders (ce.py), submitted stacker (stack_v3.py),
                       later stacker (stack.py), audits (errtab.py, validate_streaming.py)
```

`infra/aws/` (repository root) holds the AWS setup: SageMaker training-job launcher, EC2
runner bootstrap, quota and bucket scripts.
