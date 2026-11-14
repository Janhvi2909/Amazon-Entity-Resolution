"""Stack the cross-encoder score onto M-v5's probabilities for the uncertain band, evaluate, write test files.

Band rows (0.02 <= p2 <= 0.98) of the rest entities train a LightGBM on (p1, p2, ce, and the ce context:
rank among the S1's band candidates, best rival ce for the same target, band size); early-stopped on the
early-stop entities; applied to eval and test band rows. Outside the band p2 is kept. The CE never saw
rest, eval or test pairs (it was trained on the fit entities). Judged on the eval slice with the same
decision rule as M-v5 (exclusive assignment, gated expected-F, gate chosen on eval) against M-v5 itself.

    python stack.py --ce <dir of ce parts> --out <dir> --code <src dir>
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--ce", required=True); ap.add_argument("--ce2", default="");
ap.add_argument("--feat-dir", default="", help="feature parts dir (train/, test/): numeric/house/legal columns join the stacker")
ap.add_argument("--ce3", default="", help="a third CE score dir (e.g. India-only MuRIL; NaN elsewhere)")
ap.add_argument("--old-run", default="", help="old M-v5 run dir: its p1/p2 join band rows as fusion features p1_old/p2_old (NaN = pair absent)")
ap.add_argument("--band-by-ce", action="store_true", help="stack every pair that has a CE score (e.g. confident France pairs too)")
ap.add_argument("--out", required=True); ap.add_argument("--code", required=True)
ap.add_argument("--run", default="/work/full/runs/NORM-v3t__BLK-v4b@20__FEAT-v4__MATCH-v5")
ap.add_argument("--prep", default="/work/full/prep/NORM-v3t")
a = ap.parse_args()
sys.path.insert(0, a.code)
import lightgbm as lgb
from ber import model as M
from ber.metrics import macro_f05

t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
FIT, ES, EVAL, REST = 0, 1, 2, 3
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
def load_ce(d):
    """All score parts under d (recursively), one row per pair: parts from different runs can overlap."""
    parts = sorted(Path(d).rglob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"FATAL: no CE parts under {d}")
    return pl.concat([pl.read_parquet(f, columns=["split", "q_rid", "t_rid", "ce"]) for f in parts]).unique(["split", "q_rid", "t_rid"], keep="first")


CE = load_ce(a.ce)
if a.ce2:  # a second cross-encoder's score as an extra stacker input
    CE2 = load_ce(a.ce2).rename({"ce": "ce_b"})
    CE = CE.join(CE2.select("split", "q_rid", "t_rid", "ce_b"), on=["split", "q_rid", "t_rid"], how="left")
if a.ce3:
    CE3 = load_ce(a.ce3).rename({"ce": "ce_c"})
    CE = CE.join(CE3.select("split", "q_rid", "t_rid", "ce_c"), on=["split", "q_rid", "t_rid"], how="left")
log(f"ce rows {CE.height:,}")
role = pl.read_parquet(f"{a.prep}/split.parquet")
pairs = pl.read_parquet(f"{a.prep}/pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))


def ctx(S):
    """ce context within the band: rank in the S1's band list, best rival ce for the target, band size."""
    b = S.filter(pl.col("ce").is_not_null())
    b = b.with_columns(pl.col("ce").rank("ordinal", descending=True).over("q_rid").cast(pl.Float32).alias("ce_rank_q"),
                       pl.len().over("q_rid").cast(pl.Float32).alias("band_n_q"),
                       pl.when(pl.col("ce") == pl.col("ce").max().over("t_rid"))
                         .then(pl.col("ce").sort(descending=True).slice(1, 1).first().over("t_rid"))
                         .otherwise(pl.col("ce").max().over("t_rid")).fill_null(0).alias("ce_other_t"))
    return S.join(b.select("q_rid", "t_rid", "ce_rank_q", "band_n_q", "ce_other_t"), on=["q_rid", "t_rid"], how="left")


NUMF = ["h_gap", "h_loggap", "h_edit", "h_diffpos", "h_near", "h_same_key", "h_suf_conflict", "legal_equal", "legal_conflict",
        "num_x_edit", "num_x_loggap", "num_jacc", "a_tset", "core_tset", "lm_state", "unit_state"]


def band_feats(split, keys):
    """Numeric/house/legal feature columns for the band rows ``keys`` (q_rid, t_rid), streamed part by part."""
    out = []
    for f in sorted(Path(a.feat_dir, split).glob("part-*.parquet")):
        cols = [c for c in ["q_rid", "t_rid"] + NUMF if c in pl.read_parquet_schema(f)]
        out.append(pl.read_parquet(f, columns=cols).join(keys, on=["q_rid", "t_rid"], how="semi"))
    return pl.concat(out, how="diagonal")


FEATS = ["p1", "p2", "ce", "ce_rank_q", "band_n_q", "ce_other_t"] + (["ce_b"] if a.ce2 else []) + (["ce_c"] if a.ce3 else []) + (NUMF if a.feat_dir else []) + (["p1_old", "p2_old"] if a.old_run else [])
# ---------------- train side
S = pl.read_parquet(f"{a.run}/scores_train.parquet", columns=["q_rid", "t_rid", "p1", "p2"]).join(role, on="q_rid")
S = S.join(CE.filter(pl.col("split") == "train").select("q_rid", "t_rid", "ce", *(["ce_b"] if a.ce2 else []), *(["ce_c"] if a.ce3 else [])), on=["q_rid", "t_rid"], how="left")
S = S.join(pairs, on=["q_rid", "t_rid"], how="left").with_columns(pl.col("y").fill_null(0))
band = pl.col("ce").is_not_null() if a.band_by_ce else pl.col("p2").is_between(0.02, 0.98)
S = ctx(S)
def old_scores(split, keys):
    """M-v5 stage-1/stage-2 probabilities for the band pairs ``keys`` (row ids verified identical across NORM-v3t/v4t)."""
    O = pl.read_parquet(f"{a.old_run}/scores_{split}.parquet", columns=["q_rid", "t_rid", "p1", "p2"]).rename({"p1": "p1_old", "p2": "p2_old"})
    return O.join(keys, on=["q_rid", "t_rid"], how="semi")


if a.old_run:
    S = S.join(old_scores("train", S.filter(band).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
    log(f"old M-v5 scores joined on band rows: coverage {S.filter(band)['p2_old'].is_not_null().mean():.3f}")
if a.feat_dir:
    S = S.join(band_feats("train", S.filter(band & pl.col("role").is_in([REST, ES, EVAL])).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
    FEATS = [c for c in FEATS if c in S.columns]
    log(f"numeric features joined: {[c for c in NUMF if c in S.columns]}")
tr = S.filter(band & (pl.col("role") == REST)); es = S.filter(band & (pl.col("role") == ES))
log(f"stacker train {tr.height:,} (ce present {tr['ce'].is_not_null().mean():.3f}), early stop {es.height:,}")
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 42, "num_threads": 0}
m = lgb.train(params, lgb.Dataset(tr.select(FEATS).to_numpy(), label=tr["y"].to_numpy(), feature_name=FEATS), 3000,
              valid_sets=[lgb.Dataset(es.select(FEATS).to_numpy(), label=es["y"].to_numpy())],
              callbacks=[lgb.early_stopping(100, verbose=False)])
m.save_model(str(out / "stacker.txt"))
imp = dict(zip(FEATS, (m.feature_importance("gain") / m.feature_importance("gain").sum()).round(4).tolist()))
log(f"stacker best iteration {m.best_iteration}; gain {imp}")


def score(S):
    Sb = S.filter(band)
    p3 = m.predict(Sb.select(FEATS).to_numpy(), num_iteration=m.best_iteration)
    return S.join(Sb.select("q_rid", "t_rid").with_columns(pl.Series("p3", p3)), on=["q_rid", "t_rid"], how="left") \
            .with_columns(pl.coalesce("p3", "p2").alias("p3"))


ev = score(S.filter(pl.col("role") == EVAL))
q_eval = role.filter(pl.col("role") == EVAL)["q_rid"].to_numpy().astype(np.uint32)
truth = pairs.join(role.filter(pl.col("role") == EVAL), on="q_rid", how="semi").select("q_rid", "t_rid")
country = pl.read_parquet(f"{a.prep}/Q_train.parquet", columns=["rid", "country"])
res = {"stacker": {"best_iteration": m.best_iteration, "gain": imp}}
for col in ("p2", "p3"):
    ex = ev.select("q_rid", "t_rid", pl.col(col).alias("p")).filter(pl.col("p") == pl.col("p").max().over("t_rid"))
    best = None
    for rule, grid in (("gated_ef", np.round(np.arange(0.30, 0.96, 0.05), 3)), ("threshold", np.round(np.arange(0.20, 0.96, 0.025), 3))):
        for g in grid:
            pred = M.decide_gated(ex, gate=float(g), exclusive=False) if rule == "gated_ef" else ex.filter(pl.col("p") >= g).select("q_rid", "t_rid")
            r = macro_f05(pred, truth, q_eval)
            if best is None or r["f05"] > best["f05"]:
                best = {**r, "rule": rule, "param": float(g)}
                best_pred = pred
    per = {}
    for c in country["country"].unique().sort().to_list():
        qc = np.intersect1d(q_eval, country.filter(pl.col("country") == c)["rid"].to_numpy()).astype(np.uint32)
        per[c] = macro_f05(best_pred, truth, qc)["f05"]
    res[col] = {**best, "per_country": per}
    log(f"{col}: {best['rule']}={best['param']} F0.5={best['f05']:.4f} P={best['precision']:.4f} R={best['recall']:.4f} | " +
        ", ".join(f"{c} {v:.4f}" for c, v in per.items()))
(out / "stack_eval.json").write_text(json.dumps(res, indent=2, default=float))
del S, ev

# ---------------- test
tcols = [c for c in ("q_rid", "t_rid", "src", "p1", "p2") if c in pl.read_parquet_schema(f"{a.run}/scores_test.parquet")]
St = pl.read_parquet(f"{a.run}/scores_test.parquet", columns=tcols)
St = St.join(CE.filter(pl.col("split") == "test").select("q_rid", "t_rid", "ce", *(["ce_b"] if a.ce2 else []), *(["ce_c"] if a.ce3 else [])), on=["q_rid", "t_rid"], how="left")
St = ctx(St)
if a.old_run:
    St = St.join(old_scores("test", St.filter(band).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
if a.feat_dir:
    St = St.join(band_feats("test", St.filter(band).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
St = score(St)
r = res["p3"]
ex = St.select("q_rid", "t_rid", pl.col("p3").alias("p")).filter(pl.col("p") == pl.col("p").max().over("t_rid"))
links = M.decide_gated(ex, gate=r["param"], exclusive=False) if r["rule"] == "gated_ef" else ex.filter(pl.col("p") >= r["param"]).select("q_rid", "t_rid")
# per-S1 source caps seen in the training truth: at most 5 links from S2 and 6 from S3; keep the most probable
CAPS = {2: 5, 3: 6}
if "src" in St.columns:
    n0 = links.height
    links = (links.join(St.select("q_rid", "t_rid", "src", "p3"), on=["q_rid", "t_rid"])
             .with_columns(pl.col("p3").rank("ordinal", descending=True).over("q_rid", "src").alias("rk"))
             .filter(pl.col("rk") <= pl.col("src").cast(pl.Int32).replace_strict(CAPS, default=99))
             .select("q_rid", "t_rid"))
    log(f"source caps removed {n0 - links.height} links")
Q = pl.read_parquet(f"{a.prep}/Q_test.parquet", columns=["rid", "entity_id", "country"]); T = pl.read_parquet(f"{a.prep}/T_test.parquet", columns=["rid", "entity_id"])
top = St.with_columns(pl.col("p3").rank("ordinal", descending=True).over("q_rid").alias("rk")).filter(pl.col("rk") <= 15).select("q_rid", "t_rid")
forced = links.join(top, on=["q_rid", "t_rid"], how="anti")
cand = pl.concat([top, forced])


def write(L, col, path):
    ids = (L.join(T.select(pl.col("rid").alias("t_rid"), pl.col("entity_id").alias("tid")), on="t_rid").sort("q_rid", "tid")
           .group_by("q_rid", maintain_order=True).agg(pl.col("tid").unique(maintain_order=True).str.join(",").alias(col)))
    df = (Q.select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("source1_entity_id")).join(ids, on="q_rid", how="left", maintain_order="left")
          .with_columns(pl.col(col).fill_null("")).select("source1_entity_id", col))
    assert df.height == Q.height
    df.write_csv(path, separator="\t", quote_style="never")


write(links, "matched_entity_ids", out / "matching_results.tsv")
write(cand, "candidate_entity_ids", out / "candidate_pairs.tsv")
prof = links.join(Q.select(pl.col("rid").alias("q_rid"), "country"), on="q_rid").group_by("country").len()
res["test"] = {"links": links.height, "forced": forced.height, "cand_per_s1": cand.height / Q.height, "links_by_country": dict(prof.iter_rows())}
(out / "stack_eval.json").write_text(json.dumps(res, indent=2, default=float))
log(f"test: {res['test']}")
