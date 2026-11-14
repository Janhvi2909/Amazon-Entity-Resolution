"""Which independent retrieval channel recovers the true links our blocker misses? (eval only, diagnostic)

For every eval S1 that has at least one M2 blocking miss, retrieve per target source:
  address-only : sparse cosine on the blocker's address features (tokens, prefixes, skeletons, number affixes)
  name-only    : sparse cosine on the blocker's name features (tokens, prefixes, skeletons, joined name)
  rare-addr    : targets sharing an address token whose document frequency is <= RARE (ranked by count)
  dense        : multilingual-e5-small cosine on "name, address" (optional, GPU)
and report how many of the missed true links each channel finds within its top 1/3/5/10, alone and combined.
IDF is computed per (country, source) over targets, as in the blocker.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, sys.argv[2])
from ber.blocking import HALF, NBITS, feature_frame
from sparse_dot_topn import sp_matmul_topn

D = Path(sys.argv[1]); DENSE = len(sys.argv) > 3 and sys.argv[3] == "dense"
KS = (1, 3, 5, 10); RARE = 10
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
M = pl.read_parquet(D / "misses.parquet").filter(pl.col("missed_by") == "M2")
Q = pl.read_parquet(D / "Q.parquet"); T = pl.read_parquet(D / "T.parquet")
res = {}


def mat(rows, cols, n, idf, lo, hi):
    keep = (cols >= lo) & (cols < hi) & (idf[cols] > 0)
    r, c = rows[keep], cols[keep]
    w = idf[c]
    ss = np.bincount(r, weights=w * w, minlength=n)
    w = w / np.sqrt(np.maximum(ss[r], 1e-12))
    return sp.csr_matrix((w.astype(np.float32), (r, c)), shape=(n, 1 << NBITS))


def topk(Qm, Tm, k):
    C = sp_matmul_topn(Qm, Tm.T.tocsr(), top_n=k, threshold=0.0, sort=True, n_threads=0).tocoo()
    df = pl.DataFrame({"qi": C.row.astype(np.int64), "ti": C.col.astype(np.int64), "s": C.data})
    return df.with_columns(pl.col("s").rank("ordinal", descending=True).over("qi").alias("r"))


found = []
for country in ("India", "US"):
    mc = M.filter(pl.col("country") == country)
    qs = mc["q_rid"].unique().sort()
    Qc = Q.filter(pl.col("rid").is_in(qs.implode())).sort("rid")
    q_ids = Qc["rid"].to_numpy()
    fq = feature_frame(Qc, char_grams=False, joined_name=True)
    qi = np.searchsorted(q_ids, fq["rid"].to_numpy()); qc_ = fq["fid"].to_numpy()
    for src in (2, 3):
        Tc = T.filter((pl.col("country") == country) & (pl.col("src") == src)).sort("rid")
        t_ids = Tc["rid"].to_numpy()
        ft = feature_frame(Tc, char_grams=False, joined_name=True)
        ti = np.searchsorted(t_ids, ft["rid"].to_numpy()); tc_ = ft["fid"].to_numpy()
        df = np.bincount(tc_, minlength=1 << NBITS)
        cap = max(1000, int(0.01 * len(t_ids)))
        idf = np.where((df > 0) & (df <= cap), np.log(len(t_ids) / np.maximum(df, 1)), 0.0)
        for ch, lo, hi in (("address", HALF, 2 * HALF), ("name", 0, HALF)):
            R = topk(mat(qi, qc_, len(q_ids), idf, lo, hi), mat(ti, tc_, len(t_ids), idf, lo, hi), max(KS))
            found.append(R.select(pl.Series("q_rid", q_ids[R["qi"].to_numpy()]), pl.Series("t_rid", t_ids[R["ti"].to_numpy()]), "r")
                         .with_columns(pl.lit(ch).alias("ch")))
        # rare address tokens (whole tokens, no variants)
        at = Tc.select("rid", pl.col("addr_n").str.split(" ").alias("tok")).explode("tok").filter(pl.col("tok").str.len_chars() >= 3).unique()
        dfa = at.group_by("tok").len().filter(pl.col("len") <= RARE)
        qa = Qc.select(pl.col("rid").alias("q_rid"), pl.col("addr_n").str.split(" ").alias("tok")).explode("tok").unique()
        rr = (qa.join(dfa.select("tok"), on="tok").join(at.rename({"rid": "t_rid"}), on="tok")
              .group_by("q_rid", "t_rid").len().with_columns(pl.col("len").rank("ordinal", descending=True).over("q_rid").alias("r"))
              .filter(pl.col("r") <= max(KS)).select("q_rid", "t_rid", "r").with_columns(pl.lit("rare_addr").alias("ch")))
        found.append(rr)
        log(f"{country} S{src}: {len(q_ids):,} S1 x {len(t_ids):,} targets done")
    if DENSE:
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = "intfloat/multilingual-e5-small"
        tok = AutoTokenizer.from_pretrained(name); mdl = AutoModel.from_pretrained(name).cuda().eval().half()
        def emb(texts, bs=1024):
            out = []
            for s in range(0, len(texts), bs):
                e = tok(["query: " + x for x in texts[s:s + bs]], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    h = mdl(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).half()
                v = (h * m).sum(1) / m.sum(1)
                out.append(torch.nn.functional.normalize(v.float(), dim=1).cpu())
            return torch.cat(out)
        txt = lambda df: [f"{a}, {b}" for a, b in zip(df["name_n"].to_list(), df["addr_n"].to_list())]
        qe = emb(txt(Qc)).cuda()
        for src in (2, 3):
            Tc = T.filter((pl.col("country") == country) & (pl.col("src") == src)).sort("rid")
            t_ids = Tc["rid"].to_numpy(); te = emb(txt(Tc))
            best_s, best_i = None, None
            for s in range(0, te.shape[0], 50_000):
                sc = qe @ te[s:s + 50_000].cuda().T
                v, i = sc.topk(max(KS), dim=1)
                i = i + s
                if best_s is None:
                    best_s, best_i = v, i
                else:
                    cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i], 1)
                    best_s, o = cs.topk(max(KS), dim=1); best_i = ci.gather(1, o)
            bi = best_i.cpu().numpy()
            found.append(pl.DataFrame({"q_rid": np.repeat(q_ids, max(KS)), "t_rid": t_ids[bi.ravel()],
                                       "r": np.tile(np.arange(1, max(KS) + 1), len(q_ids))}).with_columns(pl.lit("dense").alias("ch")))
            log(f"{country} S{src}: dense done ({len(t_ids):,} targets embedded)")

F = pl.concat([f.with_columns(pl.col("r").cast(pl.Int64), pl.col("q_rid").cast(pl.UInt32), pl.col("t_rid").cast(pl.UInt32)) for f in found])
Mx = M.select(pl.col("q_rid").cast(pl.UInt32), pl.col("t_rid").cast(pl.UInt32), "country")
hit = Mx.join(F, on=["q_rid", "t_rid"], how="left")
table = []
chans = sorted(F["ch"].unique().to_list())
combos = [[c] for c in chans] + [["address", "name"], ["address", "name", "rare_addr"]] + ([chans] if "dense" in chans else [])
for country in ("India", "US"):
    h = hit.filter(pl.col("country") == country); n = Mx.filter(pl.col("country") == country).height
    for combo in combos:
        row = {"country": country, "channels": "+".join(combo), "misses": n}
        for k in KS:
            got = h.filter(pl.col("ch").is_in(combo) & (pl.col("r") <= k)).select("q_rid", "t_rid").unique().height
            row[f"@{k}"] = got
        table.append(row)
out = pl.DataFrame(table)
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(out)
out.write_csv(D / "retr_diag.csv")
log("done")
