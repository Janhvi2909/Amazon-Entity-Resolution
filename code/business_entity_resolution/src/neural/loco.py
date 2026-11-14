"""Leave-one-country-out test of the cross-encoder stacker: how does a country score when the stacker never
saw it (as France at test time), and which decision gate suits an unseen country?

For each held-out country H in {US, India}: train the stacker on the rest entities' band rows of the OTHER
country only (early stop on the other country's early-stop rows), apply it to H's eval rows, and sweep the
gated expected-F gate. Compare with the stacker trained on both countries (the setup used for France).
Only the stacker is held out; M-v5 and the cross-encoder were trained on both countries, so this bounds
what the stacker layer itself contributes to the unseen-country gap.

    python loco.py <stack dir with run/ prep/ ce2_scores.parquet> <src dir>
"""
import json, sys, time
from pathlib import Path
import numpy as np
import polars as pl

D = Path(sys.argv[1]); sys.path.insert(0, sys.argv[2])
import lightgbm as lgb
from ber import model as M
from ber.metrics import macro_f05

t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
FIT, ES, EVAL, REST = 0, 1, 2, 3
role = pl.read_parquet(D / "prep/split.parquet")
pairs = pl.read_parquet(D / "prep/pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
country = pl.read_parquet(D / "prep/Q_train.parquet", columns=["rid", "country"]).rename({"rid": "q_rid"})
CE = pl.read_parquet(D / "ce2_scores.parquet").filter(pl.col("split") == "train").select("q_rid", "t_rid", "ce")
need = role.filter(pl.col("role").is_in([ES, EVAL, REST]))
S = (pl.scan_parquet(D / "run/scores_train.parquet").join(need.lazy(), on="q_rid", how="semi").collect()
     .join(role, on="q_rid").join(country, on="q_rid"))
# rest/early-stop rows are only needed inside the band; eval rows all
S = S.filter((pl.col("role") == EVAL) | pl.col("p2").is_between(0.02, 0.98))
S = S.join(CE, on=["q_rid", "t_rid"], how="left").join(pairs, on=["q_rid", "t_rid"], how="left").with_columns(pl.col("y").fill_null(0))
b = S.filter(pl.col("ce").is_not_null())
b = b.with_columns(pl.col("ce").rank("ordinal", descending=True).over("q_rid").cast(pl.Float32).alias("ce_rank_q"),
                   pl.len().over("q_rid").cast(pl.Float32).alias("band_n_q"),
                   pl.when(pl.col("ce") == pl.col("ce").max().over("t_rid")).then(pl.col("ce").sort(descending=True).slice(1, 1).first().over("t_rid"))
                   .otherwise(pl.col("ce").max().over("t_rid")).fill_null(0).alias("ce_other_t"))
S = S.join(b.select("q_rid", "t_rid", "ce_rank_q", "band_n_q", "ce_other_t"), on=["q_rid", "t_rid"], how="left")
log(f"rows {S.height:,}")
FEATS = ["p1", "p2", "ce", "ce_rank_q", "band_n_q", "ce_other_t"]
band = pl.col("p2").is_between(0.02, 0.98)
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 42, "num_threads": 0}
truth = pairs.join(role.filter(pl.col("role") == EVAL), on="q_rid", how="semi").select("q_rid", "t_rid")
GATES = np.round(np.arange(0.3, 0.96, 0.05), 3)


def fit(train_countries):
    tr = S.filter(band & (pl.col("role") == REST) & pl.col("country").is_in(train_countries))
    es = S.filter(band & (pl.col("role") == ES) & pl.col("country").is_in(train_countries))
    return lgb.train(params, lgb.Dataset(tr.select(FEATS).to_numpy(), label=tr["y"].to_numpy()), 3000,
                     valid_sets=[lgb.Dataset(es.select(FEATS).to_numpy(), label=es["y"].to_numpy())], callbacks=[lgb.early_stopping(100, verbose=False)])


def evaluate(m, held):
    ev = S.filter((pl.col("role") == EVAL) & (pl.col("country") == held))
    evb = ev.filter(band)
    ev = ev.join(evb.select("q_rid", "t_rid").with_columns(pl.Series("p3", m.predict(evb.select(FEATS).to_numpy(), num_iteration=m.best_iteration))),
                 on=["q_rid", "t_rid"], how="left").with_columns(pl.coalesce("p3", "p2").alias("p"))
    ex = ev.select("q_rid", "t_rid", "p").filter(pl.col("p") == pl.col("p").max().over("t_rid"))
    ids = np.intersect1d(role.filter(pl.col("role") == EVAL)["q_rid"].to_numpy(), country.filter(pl.col("country") == held)["q_rid"].to_numpy()).astype(np.uint32)
    return {float(g): macro_f05(M.decide_gated(ex, gate=float(g), exclusive=False), truth, ids)["f05"] for g in GATES}


res = {}
both = fit(["US", "India"])
for held, other in (("India", "US"), ("US", "India")):
    seen = evaluate(both, held)
    unseen = evaluate(fit([other]), held)
    gs, gu = max(seen, key=seen.get), max(unseen, key=unseen.get)
    res[held] = {"seen": {"best_gate": gs, "f05": seen[gs], "at_0.6": seen[0.6]},
                 "unseen": {"best_gate": gu, "f05": unseen[gu], "at_0.6": unseen[0.6], "at_seen_best_gate": unseen[gs]},
                 "curve_seen": seen, "curve_unseen": unseen}
    log(f"{held}: stacker seen F0.5 {seen[gs]:.5f} (gate {gs}) | unseen {unseen[gu]:.5f} (gate {gu}); unseen at the seen gate {unseen[gs]:.5f}")
Path(D / "loco.json").write_text(json.dumps(res, indent=2, default=float))
log("done")
