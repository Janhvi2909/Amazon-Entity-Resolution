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
ap.add_argument("--protocol", default="eval", choices=["eval", "es"], help="es: choose rule/gate on ES entities, report EVAL (plan protocol)")
ap.add_argument("--cand-full", action="store_true", help="candidate file = every scored test pair (exact scored set), not top-15")
ap.add_argument("--band-by-ce", action="store_true", help="stack every pair that has a CE score (e.g. confident France pairs too)")
ap.add_argument("--seeds", type=int, default=1, help="bag the stacker over this many seeds (mean probability)")
ap.add_argument("--leaves", type=int, default=63); ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--unseen-gates", default="", help="extra matching files: countries absent from train (France) decided at these gates, e.g. 0.5,0.6")
ap.add_argument("--unseen-alphas", default="", help="extra files: unseen-country probabilities raised to alpha (<1 = more links), base gate")
ap.add_argument("--dump-eval", action="store_true", help="write eval pairs (p2, p3, reviewed flag) and the chosen predictions for error analysis")
ap.add_argument("--support", action="store_true", help="label-free evidence for band rows: agreement with the S1's other confident candidates, name frequencies, target address flag")
ap.add_argument("--support2", action="store_true", help="with --support: core-name frequencies and the target's rival count")
ap.add_argument("--anchor", default="", help="anchor CE scores (candidate target vs a reliable different predicted target of the S1): feature anc_ce, NaN = no anchor")
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
    D = pl.concat([pl.read_parquet(f, columns=["split", "q_rid", "t_rid", "ce"]) for f in parts])
    g = D.group_by("split", "q_rid", "t_rid").agg(pl.len().alias("n"), (pl.col("ce").max() - pl.col("ce").min()).alias("spread"))
    audit = {"dir": str(d), "parts": len(parts), "rows": D.height, "unique_pairs": g.height,
             "non_finite": int(D["ce"].is_nan().sum() + D["ce"].is_null().sum()), "min": float(D["ce"].min()), "max": float(D["ce"].max()),
             "dup_pairs": int((g["n"] > 1).sum()), "dup_disagree_gt_1e-3": int(((g["n"] > 1) & (g["spread"] > 1e-3)).sum())}
    log(f"CE audit: {audit}")
    if audit["non_finite"] or audit["min"] < 0 or audit["max"] > 1:
        raise SystemExit(f"FATAL: CE scores out of range / non-finite in {d}")
    # the only duplicates are one model scoring a pair in two batches (bf16 padding noise): average them
    return D.group_by("split", "q_rid", "t_rid").agg(pl.col("ce").mean())


CE = load_ce(a.ce)
if a.ce2:  # a second cross-encoder's score as an extra stacker input
    CE2 = load_ce(a.ce2).rename({"ce": "ce_b"})
    CE = CE.join(CE2.select("split", "q_rid", "t_rid", "ce_b"), on=["split", "q_rid", "t_rid"], how="left")
if a.ce3:
    CE3 = load_ce(a.ce3).rename({"ce": "ce_c"})
    CE = CE.join(CE3.select("split", "q_rid", "t_rid", "ce_c"), on=["split", "q_rid", "t_rid"], how="left")
if a.anchor:
    AN = load_ce(a.anchor).rename({"ce": "anc_ce"})
    CE = CE.join(AN.select("split", "q_rid", "t_rid", "anc_ce"), on=["split", "q_rid", "t_rid"], how="left")
    log(f"anchor scores joined: {CE['anc_ce'].is_not_null().sum():,} of {CE.height:,} CE rows")
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


SUPF = ["sup_name_eq", "sup_core_eq", "conf_n_q", "t_has_addr", "qname_freq", "tname_freq"]


def support(S, split):
    """For band rows: how many of the S1's OTHER confident (p2 >= 0.9) candidates share the target's name / core name,
    how many confident candidates the S1 has, the target's address flag, and how common the S1 / target names are."""
    Tn = pl.read_parquet(f"{a.prep}/T_{split}.parquet", columns=["rid", "country", "name_n", "name_core", "has_addr"]).rename({"rid": "t_rid"})
    Qn = pl.read_parquet(f"{a.prep}/Q_{split}.parquet", columns=["rid", "country", "name_n"]).rename({"rid": "q_rid"})
    Qn = Qn.with_columns(pl.len().over("country", "name_n").cast(pl.Float32).alias("qname_freq")).select("q_rid", "qname_freq")
    Tn = Tn.with_columns(pl.len().over("country", "name_n").cast(pl.Float32).alias("tname_freq"))
    conf = S.filter(pl.col("p2") >= 0.9).select("q_rid", "t_rid").join(Tn.select("t_rid", "name_n", "name_core"), on="t_rid")
    cn = conf.group_by("q_rid", "name_n").len().rename({"len": "cn"}); cc = conf.group_by("q_rid", "name_core").len().rename({"len": "cc"})
    nq = conf.group_by("q_rid").len().rename({"len": "conf_n_q"})
    B = (S.filter(band).select("q_rid", "t_rid", (pl.col("p2") >= 0.9).cast(pl.Int32).alias("self"))
         .join(Tn.select("t_rid", "name_n", "name_core", "has_addr", "tname_freq"), on="t_rid", how="left")
         .join(cn, on=["q_rid", "name_n"], how="left").join(cc, on=["q_rid", "name_core"], how="left").join(nq, on="q_rid", how="left").join(Qn, on="q_rid", how="left")
         .with_columns((pl.col("cn").fill_null(0) - pl.col("self")).clip(0).cast(pl.Float32).alias("sup_name_eq"),
                       (pl.col("cc").fill_null(0) - pl.col("self")).clip(0).cast(pl.Float32).alias("sup_core_eq"),
                       (pl.col("conf_n_q").fill_null(0) - pl.col("self")).clip(0).cast(pl.Float32).alias("conf_n_q"),
                       pl.col("has_addr").cast(pl.Float32).alias("t_has_addr")))
    B = B.select("q_rid", "t_rid", *SUPF)
    if a.support2:
        Qc = pl.read_parquet(f"{a.prep}/Q_{split}.parquet", columns=["rid", "country", "name_core"]).rename({"rid": "q_rid"})
        Qc = Qc.with_columns(pl.len().over("country", "name_core").cast(pl.Float32).alias("qcore_freq")).select("q_rid", "qcore_freq")
        Tc = pl.read_parquet(f"{a.prep}/T_{split}.parquet", columns=["rid", "country", "name_core"]).rename({"rid": "t_rid"})
        Tc = Tc.with_columns(pl.len().over("country", "name_core").cast(pl.Float32).alias("tcore_freq")).select("t_rid", "tcore_freq")
        rv = S.filter(pl.col("p2") >= 0.3).group_by("t_rid").len().rename({"len": "t_rivals"}).with_columns(pl.col("t_rivals").cast(pl.Float32))
        B = B.join(Qc, on="q_rid", how="left").join(Tc, on="t_rid", how="left").join(rv, on="t_rid", how="left").with_columns(pl.col("t_rivals").fill_null(0))
    return S.join(B, on=["q_rid", "t_rid"], how="left")


FEATS = ["p1", "p2", "ce", "ce_rank_q", "band_n_q", "ce_other_t"] + (["ce_b"] if a.ce2 else []) + (["ce_c"] if a.ce3 else []) + (NUMF if a.feat_dir else []) + (["p1_old", "p2_old", "has_old"] if a.old_run else []) + (SUPF if a.support else []) + (["anc_ce"] if a.anchor else []) + (["qcore_freq", "tcore_freq", "t_rivals"] if a.support2 else [])
# ---------------- train side
_trc = [c for c in ("q_rid", "t_rid", "src", "p1", "p2") if c in pl.read_parquet_schema(f"{a.run}/scores_train.parquet")]
S = pl.read_parquet(f"{a.run}/scores_train.parquet", columns=_trc).join(role, on="q_rid")
S = S.join(CE.filter(pl.col("split") == "train").select("q_rid", "t_rid", "ce", *(["ce_b"] if a.ce2 else []), *(["ce_c"] if a.ce3 else []), *(["anc_ce"] if a.anchor else [])), on=["q_rid", "t_rid"], how="left")
S = S.join(pairs, on=["q_rid", "t_rid"], how="left").with_columns(pl.col("y").fill_null(0))
band = pl.col("ce").is_not_null() if a.band_by_ce else pl.col("p2").is_between(0.02, 0.98)
S = ctx(S)
if a.support:
    S = support(S, "train"); log(f"support features joined: {SUPF}")
def old_scores(split, keys):
    """M-v5 stage-1/stage-2 probabilities for the band pairs ``keys`` (row ids verified identical across NORM-v3t/v4t)."""
    O = pl.read_parquet(f"{a.old_run}/scores_{split}.parquet", columns=["q_rid", "t_rid", "p1", "p2"]).rename({"p1": "p1_old", "p2": "p2_old"})
    return O.join(keys, on=["q_rid", "t_rid"], how="semi")


if a.old_run:
    S = S.join(old_scores("train", S.filter(band).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
    S = S.with_columns(pl.col("p2_old").is_not_null().cast(pl.Float32).alias("has_old"))
    log(f"old M-v5 scores joined on band rows: coverage {S.filter(band)['p2_old'].is_not_null().mean():.3f}")
if a.feat_dir:
    S = S.join(band_feats("train", S.filter(band & pl.col("role").is_in([REST, ES, EVAL])).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
    FEATS = [c for c in FEATS if c in S.columns]
    log(f"numeric features joined: {[c for c in NUMF if c in S.columns]}")
tr = S.filter(band & (pl.col("role") == REST)); es = S.filter(band & (pl.col("role") == ES))
log(f"stacker train {tr.height:,} (ce present {tr['ce'].is_not_null().mean():.3f}), early stop {es.height:,}")
params = {"objective": "binary", "learning_rate": a.lr, "num_leaves": a.leaves, "min_data_in_leaf": 200, "feature_fraction": 0.9,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 42, "num_threads": 0}
ms = [lgb.train({**params, "seed": 42 + i}, lgb.Dataset(tr.select(FEATS).to_numpy(), label=tr["y"].to_numpy(), feature_name=FEATS), 3000,
                valid_sets=[lgb.Dataset(es.select(FEATS).to_numpy(), label=es["y"].to_numpy())],
                callbacks=[lgb.early_stopping(100, verbose=False)]) for i in range(a.seeds)]
m = ms[0]
for i, mi in enumerate(ms):
    mi.save_model(str(out / ("stacker.txt" if i == 0 else f"stacker_s{i}.txt")))
imp = dict(zip(FEATS, (m.feature_importance("gain") / m.feature_importance("gain").sum()).round(4).tolist()))
log(f"stacker best iteration {m.best_iteration}; gain {imp}")


def score(S):
    Sb = S.filter(band)
    X = Sb.select(FEATS).to_numpy()
    p3 = np.mean([mi.predict(X, num_iteration=mi.best_iteration) for mi in ms], axis=0)
    return S.join(Sb.select("q_rid", "t_rid").with_columns(pl.Series("p3", p3)), on=["q_rid", "t_rid"], how="left") \
            .with_columns(pl.coalesce("p3", "p2").alias("p3"))


CAPS = {2: 5, 3: 6}


def capped(pred, Sx, col):
    """Per-S1 source caps (at most 5 S2 / 6 S3 links, most probable kept), applied identically in validation and test."""
    if "src" not in Sx.columns:
        return pred
    return (pred.join(Sx.select("q_rid", "t_rid", "src", pl.col(col).alias("pc")), on=["q_rid", "t_rid"])
            .with_columns(pl.col("pc").rank("ordinal", descending=True).over("q_rid", "src").alias("rk"))
            .filter(pl.col("rk") <= pl.col("src").cast(pl.Int32).replace_strict(CAPS, default=99)).select("q_rid", "t_rid"))


def decide_rule(E, col, rule, g):
    ex = E.select("q_rid", "t_rid", pl.col(col).alias("p")).filter(pl.col("p") == pl.col("p").max().over("t_rid"))
    pred = M.decide_gated(ex, gate=float(g), exclusive=False) if rule == "gated_ef" else ex.filter(pl.col("p") >= g).select("q_rid", "t_rid")
    return capped(pred, E, col)


ev = score(S.filter(pl.col("role") == EVAL))
sel_role = ES if a.protocol == "es" else EVAL
sel = score(S.filter(pl.col("role") == sel_role)) if a.protocol == "es" else ev
q_eval = role.filter(pl.col("role") == EVAL)["q_rid"].to_numpy().astype(np.uint32)
q_sel = role.filter(pl.col("role") == sel_role)["q_rid"].to_numpy().astype(np.uint32)
truth = pairs.join(role.filter(pl.col("role") == EVAL), on="q_rid", how="semi").select("q_rid", "t_rid")
truth_sel = pairs.join(role.filter(pl.col("role") == sel_role), on="q_rid", how="semi").select("q_rid", "t_rid")
country = pl.read_parquet(f"{a.prep}/Q_train.parquet", columns=["rid", "country"])
res = {"stacker": {"best_iteration": m.best_iteration, "gain": imp}, "protocol": a.protocol}
GRID = [("gated_ef", g) for g in np.round(np.arange(0.30, 0.96, 0.05), 3)] + [("threshold", g) for g in np.round(np.arange(0.20, 0.96, 0.025), 3)]
for col in ("p2", "p3"):
    scores_sel = {(r, float(g)): macro_f05(decide_rule(sel, col, r, g), truth_sel, q_sel)["f05"] for r, g in GRID}
    rule, g = max(scores_sel, key=scores_sel.get)
    best_pred = decide_rule(ev, col, rule, g)
    best = {**macro_f05(best_pred, truth, q_eval), "rule": rule, "param": g, "selected_on": "ES" if a.protocol == "es" else "EVAL",
            "f05_on_selection_set": scores_sel[(rule, g)]}
    per = {}
    for c in country["country"].unique().sort().to_list():
        qc = np.intersect1d(q_eval, country.filter(pl.col("country") == c)["rid"].to_numpy()).astype(np.uint32)
        per[c] = macro_f05(best_pred, truth, qc)["f05"]
    res[col] = {**best, "per_country": per}
    if col == "p3":  # per-entity F0.5 on EVAL for paired comparisons
        tp = best_pred.join(truth, on=["q_rid", "t_rid"]).group_by("q_rid").len().rename({"len": "tp"})
        E = (pl.DataFrame({"q_rid": q_eval}).join(best_pred.group_by("q_rid").len().rename({"len": "np"}), on="q_rid", how="left")
             .join(truth.group_by("q_rid").len().rename({"len": "nt"}), on="q_rid", how="left").join(tp, on="q_rid", how="left").fill_null(0))
        E.write_parquet(out / "eval_entity_counts.parquet")
        if a.dump_eval:
            ev.select("q_rid", "t_rid", *(["src"] if "src" in ev.columns else []), "p2", "p3", pl.col("ce").is_not_null().alias("reviewed")).write_parquet(out / "eval_pairs.parquet")
            best_pred.write_parquet(out / "eval_pred.parquet")
    log(f"{col}: {rule}={g} (chosen on {best['selected_on']}) EVAL F0.5={best['f05']:.4f} P={best['precision']:.4f} R={best['recall']:.4f} | " +
        ", ".join(f"{c} {v:.4f}" for c, v in per.items()))
(out / "stack_eval.json").write_text(json.dumps(res, indent=2, default=float))
del S, ev

# ---------------- test
tcols = [c for c in ("q_rid", "t_rid", "src", "p1", "p2") if c in pl.read_parquet_schema(f"{a.run}/scores_test.parquet")]
St = pl.read_parquet(f"{a.run}/scores_test.parquet", columns=tcols)
St = St.join(CE.filter(pl.col("split") == "test").select("q_rid", "t_rid", "ce", *(["ce_b"] if a.ce2 else []), *(["ce_c"] if a.ce3 else []), *(["anc_ce"] if a.anchor else [])), on=["q_rid", "t_rid"], how="left")
St = ctx(St)
if a.support:
    St = support(St, "test")
if a.old_run:
    St = St.join(old_scores("test", St.filter(band).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
    St = St.with_columns(pl.col("p2_old").is_not_null().cast(pl.Float32).alias("has_old"))
if a.feat_dir:
    St = St.join(band_feats("test", St.filter(band).select("q_rid", "t_rid")), on=["q_rid", "t_rid"], how="left")
St = score(St)
r = res["p3"]
links = decide_rule(St, "p3", r["rule"], r["param"])
Q = pl.read_parquet(f"{a.prep}/Q_test.parquet", columns=["rid", "entity_id", "country"]); T = pl.read_parquet(f"{a.prep}/T_test.parquet", columns=["rid", "entity_id"])
top = (St.select("q_rid", "t_rid") if a.cand_full else
       St.with_columns(pl.col("p3").rank("ordinal", descending=True).over("q_rid").alias("rk")).filter(pl.col("rk") <= 15).select("q_rid", "t_rid"))
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
if a.unseen_gates or a.unseen_alphas:  # the unseen country's gate cannot be tuned on labels: extra files for leaderboard checks
    seen = set(country["country"].unique().to_list())
    uq = Q.filter(~pl.col("country").is_in(list(seen))).select(pl.col("rid").alias("q_rid"))
    keep = links.join(uq, on="q_rid", how="anti")
    Su = St.join(uq, on="q_rid", how="semi")
    for gu in [float(x) for x in a.unseen_gates.split(",")] if a.unseen_gates else []:
        lu = decide_rule(Su, "p3", r["rule"], gu)
        write(pl.concat([keep, lu]), "matched_entity_ids", out / f"matching_results_unseen{gu}.tsv")
        res.setdefault("unseen_variants", {})[str(gu)] = {"unseen_links": lu.height, "base_unseen_links": links.height - keep.height}
        log(f"unseen-country gate {gu}: {lu.height:,} links (base {links.height - keep.height:,})")
    for al in [float(x) for x in a.unseen_alphas.split(",")] if a.unseen_alphas else []:
        lu = decide_rule(Su.with_columns(pl.col("p3").clip(0, 1).pow(al)), "p3", r["rule"], r["param"])
        write(pl.concat([keep, lu]), "matched_entity_ids", out / f"matching_results_alpha{al}.tsv")
        res.setdefault("unseen_variants", {})[f"alpha{al}"] = {"unseen_links": lu.height, "base_unseen_links": links.height - keep.height}
        log(f"unseen-country alpha {al}: {lu.height:,} links (base {links.height - keep.height:,})")
res["test"] = {"links": links.height, "forced": forced.height, "cand_per_s1": cand.height / Q.height, "links_by_country": dict(prof.iter_rows())}
(out / "stack_eval.json").write_text(json.dumps(res, indent=2, default=float))
log(f"test: {res['test']}")
