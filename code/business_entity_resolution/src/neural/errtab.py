"""Error table of the final fusion on EVAL (Codex step 2): every true link / predicted link falls in one group,
with country, source and address breakdowns, businesses affected and the oracle macro-F0.5 if the group were fixed.

    python errtab.py <stack out dir with eval_pairs/eval_pred> <prep dir>
"""
import json, sys
from pathlib import Path
import polars as pl

D, P = Path(sys.argv[1]), sys.argv[2]
EV = pl.read_parquet(D / "eval_pairs.parquet"); PR = pl.read_parquet(D / "eval_pred.parquet").select("q_rid", "t_rid")
role = pl.read_parquet(f"{P}/split.parquet"); qe = role.filter(pl.col("role") == 2).select("q_rid")
truth = pl.read_parquet(f"{P}/pairs.parquet").join(qe, on="q_rid", how="semi").select("q_rid", "t_rid")
Q = pl.read_parquet(f"{P}/Q_train.parquet"); T = pl.read_parquet(f"{P}/T_train.parquet")
acol = next(c for c in T.columns if "addr" in c.lower())
scol = next((c for c in T.columns if c in ("src", "source")), None)
Tm = T.select(pl.col("rid").alias("t_rid"), (pl.col(acol).is_null() | (pl.col(acol).cast(pl.Utf8).str.strip_chars() == "")).alias("t_noaddr"),
              *([pl.col(scol).cast(pl.Utf8).alias("t_src")] if scol else []))
Qm = Q.select(pl.col("rid").alias("q_rid"), "country")


def macro(pred):
    tp = pred.join(truth, on=["q_rid", "t_rid"]).group_by("q_rid").len().rename({"len": "tp"})
    E = (qe.join(pred.group_by("q_rid").len().rename({"len": "np"}), on="q_rid", how="left")
         .join(truth.group_by("q_rid").len().rename({"len": "nt"}), on="q_rid", how="left").join(tp, on="q_rid", how="left").fill_null(0))
    f = pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64)).otherwise(
        1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp"))))
    return float(E.select(f.fill_nan(0).mean()).item()), E


base, E = macro(PR)
owner = PR.rename({"q_rid": "owner"})
miss = truth.join(PR, on=["q_rid", "t_rid"], how="anti")
miss = (miss.join(EV.select("q_rid", "t_rid", "reviewed", "p3").with_columns(pl.lit(True).alias("cand")), on=["q_rid", "t_rid"], how="left")
        .join(owner, on="t_rid", how="left")
        .with_columns(pl.when(pl.col("cand").is_null()).then(pl.lit("missing_candidate"))
                      .when(pl.col("owner").is_not_null()).then(pl.lit("lost_to_other_s1"))
                      .when(~pl.col("reviewed")).then(pl.lit("below_band_unreviewed"))
                      .otherwise(pl.lit("reviewed_rejected")).alias("group")))
fp = PR.join(truth, on=["q_rid", "t_rid"], how="anti").join(E.select("q_rid", "nt"), on="q_rid")
fp = fp.with_columns(pl.when(pl.col("nt") == 0).then(pl.lit("singleton_false_match")).otherwise(pl.lit("false_match")).alias("group"))
rows = pl.concat([miss.select("q_rid", "t_rid", "group"), fp.select("q_rid", "t_rid", "group")]).join(Qm, on="q_rid").join(Tm, on="t_rid", how="left")
out = {"base_f05": base, "eval_s1": qe.height, "groups": {}}
for g in ["missing_candidate", "below_band_unreviewed", "reviewed_rejected", "lost_to_other_s1", "false_match", "singleton_false_match"]:
    R = rows.filter(pl.col("group") == g)
    fixed = pl.concat([PR, R.select("q_rid", "t_rid")]).unique() if g in ("missing_candidate", "below_band_unreviewed", "reviewed_rejected", "lost_to_other_s1") \
        else PR.join(R.select("q_rid", "t_rid"), on=["q_rid", "t_rid"], how="anti")
    of, _ = macro(fixed)
    br = lambda c: dict(R.group_by(c).len().sort(c).iter_rows()) if c in R.columns else {}
    out["groups"][g] = {"links": R.height, "businesses": R["q_rid"].n_unique(), "oracle_f05": of, "oracle_gain": of - base,
                        "by_country": br("country"), "by_source": br("t_src"), "target_no_address": int(R["t_noaddr"].sum() or 0)}
    print(f"{g:24s} links {R.height:7,} businesses {R['q_rid'].n_unique():7,} oracle gain {of - base:+.5f} | {out['groups'][g]['by_country']} | "
          f"src {out['groups'][g]['by_source']} | no-address {out['groups'][g]['target_no_address']:,}", flush=True)
# singleton decisions: entities with no true match that received links
sm = E.filter((pl.col("nt") == 0) & (pl.col("np") > 0)).height; em = E.filter((pl.col("nt") > 0) & (pl.col("np") == 0)).height
out["singleton_mistakes"] = {"no_match_but_predicted": sm, "has_match_but_empty": em}
print(f"base macro F0.5 {base:.6f}; singletons predicted non-empty {sm:,}; matched entities left empty {em:,}")
(D / "errtab.json").write_text(json.dumps(out, indent=2, default=float))
