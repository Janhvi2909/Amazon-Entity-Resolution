"""Decision-layer experiments on the stacked scores (eval only): per-country gates, contested-target handling,
and an entity-level "has any match" model. Everything is judged on the eval slice; the entity model is
cross-validated over two halves of the eval entities so it is never scored on the rows it was fitted on.

    python decision_lab.py --ce <ce parts> --run <scores dir> --prep <prep dir> --code <src>
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--ce", required=True); ap.add_argument("--run", required=True); ap.add_argument("--prep", required=True)
ap.add_argument("--code", required=True); ap.add_argument("--out", default="/n6/decision_lab.json")
a = ap.parse_args()
sys.path.insert(0, a.code)
import lightgbm as lgb
from ber import model as M
from ber.metrics import macro_f05

t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
FIT, ES, EVAL, REST = 0, 1, 2, 3
role = pl.read_parquet(f"{a.prep}/split.parquet")
pairs = pl.read_parquet(f"{a.prep}/pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
country = pl.read_parquet(f"{a.prep}/Q_train.parquet", columns=["rid", "country"]).rename({"rid": "q_rid"})
CE = pl.concat([pl.read_parquet(f) for f in sorted(Path(a.ce).glob("part-*.parquet"))]).filter(pl.col("split") == "train")
S = (pl.read_parquet(f"{a.run}/scores_train.parquet", columns=["q_rid", "t_rid", "p1", "p2"]).join(role, on="q_rid")
     .join(CE.select("q_rid", "t_rid", "ce"), on=["q_rid", "t_rid"], how="left")
     .join(pairs, on=["q_rid", "t_rid"], how="left").with_columns(pl.col("y").fill_null(0)))
band = pl.col("p2").is_between(0.02, 0.98)
b = S.filter(pl.col("ce").is_not_null())
b = b.with_columns(pl.col("ce").rank("ordinal", descending=True).over("q_rid").cast(pl.Float32).alias("ce_rank_q"),
                   pl.len().over("q_rid").cast(pl.Float32).alias("band_n_q"),
                   pl.when(pl.col("ce") == pl.col("ce").max().over("t_rid")).then(pl.col("ce").sort(descending=True).slice(1, 1).first().over("t_rid"))
                   .otherwise(pl.col("ce").max().over("t_rid")).fill_null(0).alias("ce_other_t"))
S = S.join(b.select("q_rid", "t_rid", "ce_rank_q", "band_n_q", "ce_other_t"), on=["q_rid", "t_rid"], how="left")
FEATS = ["p1", "p2", "ce", "ce_rank_q", "band_n_q", "ce_other_t"]
tr, es = S.filter(band & (pl.col("role") == REST)), S.filter(band & (pl.col("role") == ES))
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 42, "num_threads": 0}
m = lgb.train(params, lgb.Dataset(tr.select(FEATS).to_numpy(), label=tr["y"].to_numpy()), 3000,
              valid_sets=[lgb.Dataset(es.select(FEATS).to_numpy(), label=es["y"].to_numpy())], callbacks=[lgb.early_stopping(100, verbose=False)])
ev = S.filter(pl.col("role") == EVAL)
evb = ev.filter(band)
ev = ev.join(evb.select("q_rid", "t_rid").with_columns(pl.Series("p3", m.predict(evb.select(FEATS).to_numpy(), num_iteration=m.best_iteration))),
             on=["q_rid", "t_rid"], how="left").with_columns(pl.coalesce("p3", "p2").alias("p"))
ev = ev.join(country, on="q_rid")
q_eval = role.filter(pl.col("role") == EVAL)["q_rid"].to_numpy().astype(np.uint32)
truth = pairs.join(role.filter(pl.col("role") == EVAL), on="q_rid", how="semi").select("q_rid", "t_rid")
qc = {c: np.intersect1d(q_eval, country.filter(pl.col("country") == c)["q_rid"].to_numpy()).astype(np.uint32) for c in ("US", "India")}
ex = ev.filter(pl.col("p") == pl.col("p").max().over("t_rid"))
res = {}


def decide(E, gate):
    return M.decide_gated(E.select("q_rid", "t_rid", "p"), gate=gate, exclusive=False)


def score(pred, ids=q_eval):
    return macro_f05(pred, truth, ids)["f05"]


GATES = np.round(np.arange(0.3, 0.96, 0.05), 3)
base = {g: decide(ex, g) for g in GATES}
bg = max(GATES, key=lambda g: score(base[g]))
res["baseline"] = {"gate": float(bg), "f05": score(base[bg]), "US": score(base[bg], qc["US"]), "India": score(base[bg], qc["India"])}
log(f"baseline gate {bg}: {res['baseline']}")

# (6) per-country gates
best_c = {c: max(GATES, key=lambda g: score(base[g], qc[c])) for c in qc}
pc = pl.concat([base[best_c[c]].join(pl.DataFrame({"q_rid": qc[c]}), on="q_rid", how="semi") for c in qc])
res["per_country_gate"] = {"gates": {c: float(g) for c, g in best_c.items()}, "f05": score(pc)}
log(f"per-country gates {best_c}: {res['per_country_gate']['f05']:.5f}")

# (10) contested targets: drop a target when the runner-up S1's probability is within m of the winner's
run = ev.group_by("t_rid").agg(pl.col("p").sort(descending=True).alias("ps")).with_columns(
    (pl.col("ps").list.get(0) - pl.col("ps").list.get(1, null_on_oob=True).fill_null(0)).alias("margin")).select("t_rid", "margin")
exm = ex.join(run, on="t_rid")
res["contested_drop"] = {}
for mg in (0.02, 0.05, 0.1, 0.2):
    f = score(decide(exm.filter(pl.col("margin") >= mg), bg))
    res["contested_drop"][mg] = f
log(f"contested drop: {res['contested_drop']}")

# (12) entity model: P(S1 has any true link), from its score distribution; 2-fold over eval entities
g = ex.filter(pl.col("p") >= 0.01).sort("p", descending=True).group_by("q_rid").agg(
    pl.col("p").first().alias("top1"), pl.col("p").get(1, null_on_oob=True).fill_null(0).alias("top2"),
    pl.col("p").sum().alias("sum_p"), (pl.col("p") > 0.5).sum().alias("n05"), (pl.col("p") > 0.9).sum().alias("n09"),
    pl.len().alias("n_c"), pl.col("ce").max().fill_null(-1).alias("ce_max"), pl.col("p2").max().alias("p2_max"))
E = pl.DataFrame({"q_rid": q_eval}).join(g, on="q_rid", how="left").fill_null(0).join(
    truth.group_by("q_rid").len().rename({"len": "nt"}), on="q_rid", how="left").with_columns((pl.col("nt").fill_null(0) > 0).cast(pl.Int8).alias("has"))
EF = ["top1", "top2", "sum_p", "n05", "n09", "n_c", "ce_max", "p2_max"]
half = (E["q_rid"].to_numpy().astype(np.int64) * 2654435761 % 2**32) % 2
pe = np.zeros(E.height)
for h in (0, 1):
    trm, tem = half != h, half == h
    em = lgb.train({**params, "num_leaves": 31}, lgb.Dataset(E.filter(pl.Series(trm)).select(EF).to_numpy(), label=E.filter(pl.Series(trm))["has"].to_numpy()), 300)
    pe[tem] = em.predict(E.filter(pl.Series(tem)).select(EF).to_numpy())
E = E.with_columns(pl.Series("pe", pe))
res["entity_model"] = {}
for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
    keep = E.filter(pl.col("pe") >= tau).select("q_rid")
    for gate in (0.0, 0.2, bg):
        pr = decide(ex.join(keep, on="q_rid", how="semi"), gate)
        res["entity_model"][f"tau{tau}_gate{gate}"] = score(pr)
bk = max(res["entity_model"], key=res["entity_model"].get)
log(f"entity model best {bk}: {res['entity_model'][bk]:.5f} (baseline {res['baseline']['f05']:.5f})")
Path(a.out).write_text(json.dumps(res, indent=2, default=float))
log("done")
