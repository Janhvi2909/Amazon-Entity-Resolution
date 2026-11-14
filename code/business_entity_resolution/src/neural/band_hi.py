"""Extract M-v5 pairs with lo <= p2 <= hi (default the uncertain band [0.02, 0.98]) with raw text; hi band: lo=0.98, hi=1."""
import json, sys, time
import numpy as np
import polars as pl

# usage: band_hi.py <out.parquet> [<run dir> <prep dir> [<lo> <hi>]]; defaults are the M-v5 run
W = "/work/full"
OUT = sys.argv[1]
RUN = sys.argv[2] if len(sys.argv) > 2 else f"{W}/runs/NORM-v3t__BLK-v4b@20__FEAT-v4__MATCH-v5"
PREP = sys.argv[3] if len(sys.argv) > 3 else f"{W}/prep/NORM-v3t"
LO, HI = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02, float(sys.argv[5]) if len(sys.argv) > 5 else 0.98
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
nr = json.load(open(f"{PREP}/prep.json"))["train_targets_real"]
role = pl.read_parquet(f"{PREP}/split.parquet")
pairs = pl.read_parquet(f"{PREP}/pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
parts = []
for split in ("train", "test"):
    S = pl.read_parquet(f"{RUN}/scores_{split}.parquet", columns=["q_rid", "t_rid", "p1", "p2"])
    n_all = S.height
    S = S.filter(pl.col("p2").is_between(LO, HI))
    Q = pl.read_parquet(f"{PREP}/Q_{split}.parquet", columns=["rid", "entity_id", "country"])
    T = pl.read_parquet(f"{PREP}/T_{split}.parquet", columns=["rid", "entity_id"])
    if split == "train":
        S = (S.filter(pl.col("t_rid") < nr)  # synthetic clones have no raw text; the stacker sees them as missing
             .join(role, on="q_rid").join(pairs, on=["q_rid", "t_rid"], how="left").with_columns(pl.col("y").fill_null(0)))
    else:
        S = S.with_columns(pl.lit(-1, pl.Int8).alias("role"), pl.lit(-1, pl.Int8).alias("y"))
    S = (S.join(Q.select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("q_id"), "country"), on="q_rid")
          .join(T.select(pl.col("rid").alias("t_rid"), pl.col("entity_id").alias("t_id")), on="t_rid"))
    s1 = rd(f"/data/dataset/{split}/{split}_source1.tsv").select(pl.col("entity_id").alias("q_id"), pl.col("business_name").alias("qn"), pl.col("business_address").alias("qa"))
    tg = pl.concat([rd(f"/data/dataset/{split}/{split}_source{k}.tsv") for k in (2, 3)]).select(pl.col("entity_id").alias("t_id"), pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta"))
    S = S.join(s1, on="q_id").join(tg, on="t_id").with_columns(pl.lit(split).alias("split"))
    log(f"{split}: {S.height:,} band rows of {n_all:,} ({S.height / n_all:.3%}); by role {dict(S.group_by('role').len().iter_rows())}; "
        f"positives {int((S['y'] == 1).sum()):,}")
    parts.append(S.select("split", "role", "country", "q_rid", "t_rid", "q_id", "t_id", "p1", "p2", "y", "qn", "qa", "tn", "ta"))
B = pl.concat(parts)
tmp = OUT + ".tmp"
B.write_parquet(tmp, compression="zstd")
import os; os.replace(tmp, OUT)
log(f"written {B.height:,} rows, {os.path.getsize(OUT) / 2**20:.0f} MB")
