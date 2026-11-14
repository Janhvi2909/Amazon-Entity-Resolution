"""Pairwise features for (S1 record, candidate) pairs.

Three groups:
* string similarity of names and addresses (rapidfuzz, multi-threaded);
* house-number agreement (exact, truncated, conflicting);
* context: how this candidate ranks for its S1 record, and how strongly other
  S1 records compete for the same candidate. Every S2/S3 record belongs to at
  most one S1 entity, and 40% of S1 names are shared, so the competition
  features carry much of the precision.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

Q_COLS = ["name_n", "name_core", "name_nosp", "addr_n", "nums"]
T_COLS = ["name_n", "name_core", "name_nosp", "name_alt", "name_pre", "addr_n", "nums",
          "has_addr", "is_web", "has_alias", "non_ascii"]


def _pairwise(a: list, b: list, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=RF_WORKERS, dtype=np.float32)


def _number_features(na: list, nb: list) -> dict:
    n = len(na)
    jac = np.full(n, -1.0, np.float32)
    b_in_a = np.full(n, -1.0, np.float32)
    first = np.full(n, -1.0, np.float32)
    trunc = np.zeros(n, np.float32)
    for i in range(n):
        A, B = na[i], nb[i]
        if not A or not B:
            continue
        sa, sb = set(A), set(B)
        inter = len(sa & sb)
        jac[i] = inter / len(sa | sb)
        b_in_a[i] = inter / len(sb)
        first[i] = float(A[0] in sb)
        if inter < len(sb):  # a number of B missing from A: truncated/extended variant?
            trunc[i] = float(any(x != y and len(y) >= 2 and (x.startswith(y) or x.endswith(y)
                                                              or y.startswith(x) or y.endswith(x))
                                 for x in sa for y in sb))
    return {"num_jacc": jac, "num_b_in_a": b_in_a, "num_first_in_b": first, "num_trunc": trunc}


def number_gap_features(na: list, nb: list) -> dict:
    """The closest pair among numbers found on one side only: digit edit distance and numeric gap.

    Same-name distractors sit at a nearby house number (831 vs 835), while true matches carry
    digit typos (189 vs 889) and truncations, so "different but close" and "different and far"
    need separate signals. -1 when either side has no unmatched number.
    """
    n = len(na)
    edit = np.full(n, -1.0, np.float32)
    loggap = np.full(n, -1.0, np.float32)
    relgap = np.full(n, -1.0, np.float32)
    for i in range(n):
        A, B = na[i], nb[i]
        if not A or not B:
            continue
        sa, sb = set(A), set(B)
        ao, bo = sa - sb, sb - sa
        if not ao or not bo:
            continue
        best = min((Levenshtein.distance(x, y), abs(int(x) - int(y)), max(int(x), int(y))) for x in ao for y in bo)
        edit[i] = best[0]
        loggap[i] = np.log1p(float(best[1]))
        relgap[i] = best[1] / max(best[2], 1)
    return {"num_x_edit": edit, "num_x_loggap": loggap, "num_x_relgap": relgap}


TWIN_COLS = ["h_num", "h_suf", "h_key", "legal"]
NUMGAP_FEATS = ["num_x_edit", "num_x_loggap", "num_x_relgap"]
HOUSE_FEATS = ["h_both", "h_same_key", "h_gap", "h_loggap", "h_edit", "h_diffpos", "h_near", "h_suf_conflict",
               "legal_equal", "legal_conflict"]
ADDR_COLS = ["addr_core", "addr_lm", "addr_unit", "addr_pobox"]
ADDR_FEATS = ["core_ratio", "core_tset", "core_partial", "lm_state", "lm_tset", "unit_state", "pobox_state"]
RF_WORKERS = -1  # rapidfuzz threads per process (feature workers set it to their share of the cores)


def house_features(c: pl.DataFrame) -> dict:
    """FEAT-v4: parsed house numbers and legal forms of both sides (``c`` has a_/b_ TWIN_COLS).

    Planted distractors keep the name and street and move the number a few doors; typos
    change one digit anywhere. Gap, digit edit, the most significant differing digit and
    the same-street flag let the model tell them apart. Legal forms are identity-bearing
    where the data says so (France: "X SARL" and "X SAS" at one address are two records).
    """
    a, b = c["a_h_num"].to_numpy(), c["b_h_num"].to_numpy()
    both = ~(np.isnan(a) | np.isnan(b)) if a.dtype.kind == "f" else (c["a_h_num"].is_not_null() & c["b_h_num"].is_not_null()).to_numpy()
    a = np.nan_to_num(c["a_h_num"].fill_null(-1).to_numpy().astype(np.int64), nan=-1)
    b = np.nan_to_num(c["b_h_num"].fill_null(-1).to_numpy().astype(np.int64), nan=-1)
    same_key = (c["a_h_key"] == c["b_h_key"]).fill_null(False).to_numpy()
    gap = np.where(both, np.abs(a - b), -1).astype(np.float32)
    edit = _pairwise(c["a_h_num"].cast(pl.Utf8).fill_null("").to_list(), c["b_h_num"].cast(pl.Utf8).fill_null("").to_list(),
                     Levenshtein.distance)
    diffpos = np.full(len(a), -1.0, np.float32)
    aa, bb = np.abs(a), np.abs(b)
    for k in range(7):  # most significant differing digit, counted from the right (1 = units)
        d = (aa // 10**k) % 10 != (bb // 10**k) % 10
        diffpos = np.where(both & d, k + 1, diffpos)
    la, lb = c["a_legal"].fill_null(""), c["b_legal"].fill_null("")
    has = ((la != "") & (lb != "")).to_numpy()
    return {
        "h_both": both.astype(np.float32),
        "h_same_key": np.where(both, same_key, -1).astype(np.float32),
        "h_gap": gap,
        "h_loggap": np.where(both, np.log1p(np.maximum(gap, 0)), -1).astype(np.float32),
        "h_edit": np.where(both, edit, -1).astype(np.float32),
        "h_diffpos": diffpos,
        "h_near": (both & same_key & (gap >= 1) & (gap <= 30)).astype(np.float32),
        "h_suf_conflict": np.where(both, (c["a_h_suf"].fill_null("") != c["b_h_suf"].fill_null("")).to_numpy(), -1).astype(np.float32),
        "legal_equal": np.where(has, (la == lb).to_numpy(), -1).astype(np.float32),
        "legal_conflict": np.where(has, (la != lb).to_numpy(), -1).astype(np.float32),
    }


def support_features(S: pl.DataFrame, T: pl.DataFrame, anchor_p: float = 0.5, max_anchors: int = 4,
                     chunk: int = 2_000_000) -> pl.DataFrame:
    """How much each candidate resembles its S1 record's confident candidates (stage 2).

    S2/S3 hold several records of one business (up to 5 and 6 per S1 entity), so a weak
    candidate that nearly duplicates a confident one is probably a match, and one unlike
    all of them is suspect. ``S`` has q_rid, t_rid, src, p1; anchors are the q's top
    ``max_anchors`` candidates with p1 >= ``anchor_p``, other than the row's own target.
    Row order is preserved; rows without anchors get sup_n = 0 and -1 elsewhere.
    """
    anch = (S.filter(pl.col("p1") >= anchor_p)
            .with_columns(pl.col("p1").rank("ordinal", descending=True).over("q_rid").alias("r"))
            .filter(pl.col("r") <= max_anchors)
            .select("q_rid", pl.col("t_rid").alias("a_rid"), pl.col("p1").alias("a_p1"), pl.col("src").alias("a_src")))
    Ts = T.select("rid", "name_core", "name_nosp", "addr_n")
    rows = S.select("q_rid", "t_rid", "src").with_row_index("row")
    outs = []
    for s in range(0, rows.height, chunk):
        x = (rows.slice(s, chunk).join(anch, on="q_rid").filter(pl.col("t_rid") != pl.col("a_rid"))
             .join(Ts.rename({"rid": "t_rid", "name_core": "tn", "name_nosp": "tw", "addr_n": "ta"}), on="t_rid", how="left")
             .join(Ts.rename({"rid": "a_rid", "name_core": "an", "name_nosp": "aw", "addr_n": "aa"}), on="a_rid", how="left"))
        if x.height == 0:
            continue
        x = x.with_columns(pl.Series("sn", _pairwise(x["tn"].to_list(), x["an"].to_list(), fuzz.token_set_ratio)),
                           pl.Series("sw", _pairwise(x["tw"].to_list(), x["aw"].to_list(), fuzz.ratio)),
                           pl.Series("sa", _pairwise(x["ta"].to_list(), x["aa"].to_list(), fuzz.token_set_ratio)))
        # a missing address says nothing about the business
        x = x.with_columns(pl.when((pl.col("ta") == "") | (pl.col("aa") == "")).then(-1.0).otherwise(pl.col("sa")).alias("sa"))
        x = x.with_columns(pl.min_horizontal(pl.max_horizontal("sn", "sw"), pl.col("sa")).alias("joint"))
        outs.append(x.sort("joint", descending=True).group_by("row").agg(
            pl.len().alias("sup_n"),
            pl.col("sn").max().alias("sup_name_max"),
            pl.col("sw").max().alias("sup_nosp_max"),
            pl.col("sa").max().alias("sup_addr_max"),
            pl.col("joint").first().alias("sup_joint_max"),
            pl.col("a_p1").first().alias("sup_joint_p1"),
            (pl.col("a_src").first() == pl.col("src").first()).alias("sup_joint_same_src")))
    cols = ["sup_n", "sup_name_max", "sup_nosp_max", "sup_addr_max", "sup_joint_max", "sup_joint_p1", "sup_joint_same_src"]
    g = pl.concat(outs) if outs else pl.DataFrame(schema={"row": pl.UInt32, **{c: pl.Float32 for c in cols}})
    return (rows.select("row").join(g, on="row", how="left", maintain_order="left").select(cols)
            .with_columns(pl.col("sup_n").cast(pl.Float32).fill_null(0),
                          pl.all().exclude("sup_n").cast(pl.Float32).fill_null(-1)))


def _token_containment(a: list, b: list) -> tuple[np.ndarray, np.ndarray]:
    """Share of B's tokens found in A, and of A's tokens found in B."""
    ba = np.zeros(len(a), np.float32)
    ab = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx and sy:
            inter = len(sx & sy)
            ba[i] = inter / len(sy)
            ab[i] = inter / len(sx)
    return ba, ab


def _part_state(a: pl.Series, b: pl.Series) -> np.ndarray:
    """0 neither side, 1 S1 only, 2 candidate only, 3 both share a token, 4 both present and disjoint."""
    ha, hb = (a != "").to_numpy(), (b != "").to_numpy()
    share = np.array([bool(set(x.split()) & set(y.split())) for x, y in zip(a.to_list(), b.to_list())])
    return np.select([~ha & ~hb, ha & ~hb, ~ha & hb, share], [0, 1, 2, 3], 4).astype(np.float32)


def address_part_features(c: pl.DataFrame) -> dict:
    """FEAT-v5: street core compared on its own; landmark, unit and PO box as optional parts."""
    ac, bc = c["a_addr_core"].to_list(), c["b_addr_core"].to_list()
    hb = c["b_has_addr"].to_numpy()
    f = {"core_ratio": _pairwise(ac, bc, fuzz.ratio), "core_tset": _pairwise(ac, bc, fuzz.token_set_ratio),
         "core_partial": _pairwise(ac, bc, fuzz.partial_ratio)}
    for k in f:
        f[k] = np.where(hb, f[k], -1).astype(np.float32)
    f["lm_state"] = _part_state(c["a_addr_lm"], c["b_addr_lm"])
    lt = _pairwise(c["a_addr_lm"].to_list(), c["b_addr_lm"].to_list(), fuzz.token_set_ratio)
    f["lm_tset"] = np.where(f["lm_state"] >= 3, lt, -1).astype(np.float32)
    f["unit_state"] = _part_state(c["a_addr_unit"], c["b_addr_unit"])
    f["pobox_state"] = _part_state(c["a_addr_pobox"], c["b_addr_pobox"])
    return f


def string_features(cand: pl.DataFrame, Q: pl.DataFrame, T: pl.DataFrame, chunk: int = 2_000_000,
                    number_gap: bool = False, twins: bool = False, addr_parts: bool = False) -> pl.DataFrame:
    """Similarity features for every row of ``cand`` (q_rid, t_rid, ...); ``number_gap`` adds 3 (FEAT-v3),
    ``twins`` 10 house/legal features (FEAT-v4), ``addr_parts`` 7 address-component features (FEAT-v5)."""
    extra = (TWIN_COLS if twins else []) + (ADDR_COLS if addr_parts else [])
    qc, tc = Q_COLS + extra, T_COLS + extra
    Qs = Q.select("rid", *qc).rename({c: f"a_{c}" for c in qc})
    Ts = T.select("rid", *tc).rename({c: f"b_{c}" for c in tc})
    outs = []
    for s in range(0, cand.height, chunk):
        c = (cand.slice(s, chunk).select("q_rid", "t_rid")
             .join(Qs, left_on="q_rid", right_on="rid", how="left", maintain_order="left")
             .join(Ts, left_on="t_rid", right_on="rid", how="left", maintain_order="left"))
        an, bn = c["a_name_n"].to_list(), c["b_name_n"].to_list()
        ac, bc = c["a_name_core"].to_list(), c["b_name_core"].to_list()
        aw, bw = c["a_name_nosp"].to_list(), c["b_name_nosp"].to_list()
        aa, ba = c["a_addr_n"].to_list(), c["b_addr_n"].to_list()
        f = {
            "n_ratio": _pairwise(an, bn, fuzz.ratio),
            "n_tset": _pairwise(an, bn, fuzz.token_set_ratio),
            "n_tsort": _pairwise(an, bn, fuzz.token_sort_ratio),
            "c_ratio": _pairwise(ac, bc, fuzz.ratio),
            "c_tset": _pairwise(ac, bc, fuzz.token_set_ratio),
            "c_partial": _pairwise(ac, bc, fuzz.partial_ratio),
            "w_ratio": _pairwise(aw, bw, fuzz.ratio),
            "w_jw": _pairwise(aw, bw, JaroWinkler.normalized_similarity),
            "w_partial": _pairwise(aw, bw, fuzz.partial_ratio),
            "a_ratio": _pairwise(aa, ba, fuzz.ratio),
            "a_tset": _pairwise(aa, ba, fuzz.token_set_ratio),
            "a_tsort": _pairwise(aa, ba, fuzz.token_sort_ratio),
            "a_partial": _pairwise(aa, ba, fuzz.partial_ratio),
        }
        # alias parts ("X fka Y"): best match of the S1 core name against either part
        alt = _pairwise(ac, c["b_name_alt"].to_list(), fuzz.token_set_ratio)
        pre = _pairwise(ac, c["b_name_pre"].to_list(), fuzz.token_set_ratio)
        f["alias_best"] = np.where(c["b_has_alias"].to_numpy(), np.maximum(alt, pre), -1).astype(np.float32)
        f["n_b_in_a"], f["n_a_in_b"] = _token_containment(ac, bc)
        f["a_b_in_a"], f["a_a_in_b"] = _token_containment(aa, ba)
        a_nums, b_nums = c["a_nums"].to_list(), c["b_nums"].to_list()
        f.update(_number_features(a_nums, b_nums))
        if number_gap:
            f.update(number_gap_features(a_nums, b_nums))
        if twins:
            f.update(house_features(c))
        if addr_parts:
            f.update(address_part_features(c))
        f["a_name_len"] = c["a_name_core"].str.len_chars().to_numpy().astype(np.float32)
        f["b_name_len"] = c["b_name_core"].str.len_chars().to_numpy().astype(np.float32)
        f["a_addr_ntok"] = c["a_addr_n"].str.count_matches(" ").to_numpy().astype(np.float32) + 1
        f["b_addr_ntok"] = np.where(c["b_has_addr"].to_numpy(),
                                    c["b_addr_n"].str.count_matches(" ").to_numpy() + 1, 0).astype(np.float32)
        for col in ("has_addr", "is_web", "has_alias", "non_ascii"):
            f[f"b_{col}"] = c[f"b_{col}"].cast(pl.Float32).to_numpy()
        outs.append(pl.DataFrame(f))
    return pl.concat(outs)


def name_idf(Q: pl.DataFrame) -> pl.DataFrame:
    """IDF of core-name tokens over S1 records, per country: (country, tok, idf)."""
    toks = (Q.select("rid", "country", pl.col("name_core").str.split(" ").alias("tok")).explode("tok")
            .filter(pl.col("tok") != "").unique(["rid", "tok"]))
    n = Q.group_by("country").len().rename({"len": "n"})
    return (toks.group_by("country", "tok").len().join(n, on="country")
            .select("country", "tok", (pl.col("n") / pl.col("len")).log().cast(pl.Float32).alias("idf")))


def _side_stats(x: pl.DataFrame, pos: str, tok: str, prefix: str, miss_thr: float) -> pl.DataFrame:
    best = x.group_by("pid", pos).agg(pl.col("s").max(), pl.col(tok).first(), pl.col("idf_" + tok).first())
    miss = pl.col("s") < miss_thr
    return best.group_by("pid").agg(
        pl.col("s").mean().alias(f"{prefix}_me"),            # Monge-Elkan similarity
        pl.col("s").min().alias(f"{prefix}_worst"),
        miss.sum().cast(pl.Float32).alias(f"{prefix}_n_unaligned"),
        pl.col("idf_" + tok).filter(miss).max().fill_null(0).alias(f"{prefix}_unaligned_idf"),
    )


def align_features(cand: pl.DataFrame, Q: pl.DataFrame, T: pl.DataFrame, idf: pl.DataFrame,
                   chunk: int = 500_000, miss_thr: float = 0.85) -> pl.DataFrame:
    """Soft word alignment of core names (Jaro-Winkler), both directions.

    Separates noise (typos, added generic words) from a different business (a
    rare word replaced by another). Unaligned tokens carry their S1-side IDF;
    unseen tokens get the country's maximum IDF.
    """
    idf_max = idf.group_by("country").agg(pl.col("idf").max().alias("idf_max"))
    qa = Q.select(pl.col("rid").alias("q_rid"), "country", pl.col("name_core").alias("an"))
    tb = T.select(pl.col("rid").alias("t_rid"), pl.col("name_core").alias("bn"))
    outs = []
    for s in range(0, cand.height, chunk):
        c = (cand.slice(s, chunk).select("q_rid", "t_rid").with_row_index("pid")
             .join(qa, on="q_rid", how="left").join(tb, on="t_rid", how="left"))
        ea = (c.select("pid", "country", pl.col("an").str.split(" ").alias("ta"))
              .explode("ta").with_columns(pl.int_range(pl.len()).over("pid").alias("apos")))
        eb = (c.select("pid", pl.col("bn").str.split(" ").alias("tb"))
              .explode("tb").with_columns(pl.int_range(pl.len()).over("pid").alias("bpos")))
        x = ea.join(eb, on="pid")
        x = x.with_columns(pl.Series("s", _pairwise(x["ta"].to_list(), x["tb"].to_list(),
                                                    JaroWinkler.normalized_similarity)))
        x = (x.join(idf.rename({"tok": "ta", "idf": "idf_ta"}), on=["country", "ta"], how="left")
             .join(idf.rename({"tok": "tb", "idf": "idf_tb"}), on=["country", "tb"], how="left")
             .join(idf_max, on="country", how="left")
             .with_columns(pl.col("idf_ta").fill_null(pl.col("idf_max")), pl.col("idf_tb").fill_null(pl.col("idf_max"))))
        fa = _side_stats(x, "apos", "ta", "al_a", miss_thr)
        fb = _side_stats(x, "bpos", "tb", "al_b", miss_thr)
        ids = pl.DataFrame({"pid": np.arange(c.height, dtype=np.uint32)})
        outs.append(ids.join(fa, on="pid", how="left", maintain_order="left")
                    .join(fb, on="pid", how="left", maintain_order="left").drop("pid"))
    return pl.concat(outs).with_columns(pl.all().cast(pl.Float32).fill_null(-1))


def consensus_features(S: pl.DataFrame, HQ: pl.DataFrame, HT: pl.DataFrame, anchor_p: float = 0.5,
                       max_anchors: int = 6, chunk: int = 2_000_000) -> pl.DataFrame:
    """Stage 2 (MATCH-v5): does this candidate's house agree with the S1's confident candidates?

    Twins link to no S1 record, so nothing competes for them and exclusivity cannot remove
    them. But the S1's true records mostly sit at one number: if three confident candidates
    are at 423 and this one is at 422, it is the twin. ``S`` has q_rid, t_rid, p1 (row order
    kept); HQ/HT have rid, h_num, h_key. Anchors: the S1's top ``max_anchors`` other candidates
    with p1 >= anchor_p.
    """
    anch = (S.select("q_rid", "t_rid", "p1").filter(pl.col("p1") >= anchor_p)
            .with_columns(pl.col("p1").rank("ordinal", descending=True).over("q_rid").alias("r"))
            .filter(pl.col("r") <= max_anchors)
            .join(HT.select(pl.col("rid").alias("t_rid"), pl.col("h_num").alias("an"), pl.col("h_key").alias("ak")), on="t_rid", how="left")
            .select("q_rid", pl.col("t_rid").alias("a_rid"), "an", "ak"))
    hq = HQ.select(pl.col("rid").alias("q_rid"), pl.col("h_num").alias("qn"), pl.col("h_key").alias("qk"))
    ht = HT.select(pl.col("rid").alias("t_rid"), pl.col("h_num").alias("tn"), pl.col("h_key").alias("tk"))
    rows = S.select("q_rid", "t_rid").with_row_index("row")
    cols = ["cons_n", "cons_eq_me", "cons_eq_s1", "cons_me_eq_s1", "cons_twin"]
    outs = []
    for s in range(0, rows.height, chunk):
        x = (rows.slice(s, chunk).join(hq, on="q_rid", how="left").join(ht, on="t_rid", how="left")
             .join(anch, on="q_rid", how="inner").filter(pl.col("a_rid") != pl.col("t_rid"))
             .filter(pl.col("an").is_not_null()))
        if x.height == 0:
            continue
        g = x.group_by("row").agg(
            pl.len().cast(pl.Float32).alias("cons_n"),
            ((pl.col("an") == pl.col("tn")) & (pl.col("ak") == pl.col("tk"))).mean().cast(pl.Float32).alias("cons_eq_me"),
            ((pl.col("an") == pl.col("qn")) & (pl.col("ak") == pl.col("qk"))).mean().cast(pl.Float32).alias("cons_eq_s1"),
            pl.col("tn").first().alias("tn"), pl.col("qn").first().alias("qn"), pl.col("tk").first().alias("tk"), pl.col("qk").first().alias("qk"))
        outs.append(g)
    g = pl.concat(outs) if outs else pl.DataFrame(schema={"row": pl.UInt32, "cons_n": pl.Float32, "cons_eq_me": pl.Float32,
                                                        "cons_eq_s1": pl.Float32, "tn": pl.Int64, "qn": pl.Int64, "tk": pl.Utf8, "qk": pl.Utf8})
    me_eq = (pl.col("tn") == pl.col("qn")) & (pl.col("tk") == pl.col("qk"))
    g = g.with_columns(pl.when(pl.col("tn").is_null() | pl.col("qn").is_null()).then(-1.0).otherwise(me_eq.cast(pl.Float32)).alias("cons_me_eq_s1"),
                       ((pl.col("cons_eq_s1") >= 0.5) & (pl.col("tk") == pl.col("qk")) & (pl.col("tn") != pl.col("qn"))).cast(pl.Float32).alias("cons_twin"))
    return (rows.select("row").join(g.select("row", *cols), on="row", how="left", maintain_order="left").select(cols)
            .with_columns(pl.col("cons_n").fill_null(0), pl.all().exclude("cons_n").cast(pl.Float32).fill_null(-1)))


def context_features(cand: pl.DataFrame, Q: pl.DataFrame, T: pl.DataFrame) -> pl.DataFrame:
    """Rank/competition features from the blocking scores (row order preserved)."""
    c = cand.select("q_rid", "t_rid", "src", "bscore")
    name_freq = Q.group_by("country", "name_core").agg(pl.len().alias("s1_name_freq"))
    qf = Q.select("rid", "country", "name_core").join(name_freq, on=["country", "name_core"], how="left")
    tf = (T.select("rid", "country", "name_core").join(name_freq, on=["country", "name_core"], how="left")
          .with_columns(pl.col("s1_name_freq").fill_null(0)))
    c = (c.join(qf.select(pl.col("rid").alias("q_rid"), pl.col("s1_name_freq").alias("a_name_freq")),
                on="q_rid", how="left", maintain_order="left")
         .join(tf.select(pl.col("rid").alias("t_rid"), pl.col("s1_name_freq").alias("b_name_freq")),
               on="t_rid", how="left", maintain_order="left"))
    c = c.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over("q_rid", "src").cast(pl.Float32).alias("rank_in_src"),
        pl.col("bscore").rank("ordinal", descending=True).over("q_rid").cast(pl.Float32).alias("rank_in_q"),
        (pl.col("bscore") - pl.col("bscore").max().over("q_rid", "src")).alias("gap_to_best_src"),
        pl.len().over("q_rid").cast(pl.Float32).alias("n_cand_q"),
        # competition for the same target from other S1 records
        pl.len().over("t_rid").cast(pl.Float32).alias("t_indegree"),
        pl.col("bscore").rank("ordinal", descending=True).over("t_rid").cast(pl.Float32).alias("rank_for_t"),
        pl.col("bscore").max().over("t_rid").alias("t_best"),
    ).with_columns(
        # best score of any *other* S1 record for this target
        pl.when(pl.col("rank_for_t") == 1)
          .then(pl.col("bscore").sort(descending=True).slice(1, 1).first().over("t_rid"))
          .otherwise(pl.col("t_best")).fill_null(0).alias("t_best_other"),
    ).with_columns((pl.col("bscore") - pl.col("t_best_other")).alias("margin_vs_other_s1"))
    return c.select("a_name_freq", "b_name_freq", "rank_in_src", "rank_in_q", "gap_to_best_src", "n_cand_q",
                    "t_indegree", "rank_for_t", "margin_vs_other_s1").with_columns(pl.all().cast(pl.Float32))


def build_features(cand: pl.DataFrame, Q: pl.DataFrame, T: pl.DataFrame, align: bool = True,
                   number_gap: bool = False, twins: bool = False) -> pl.DataFrame:
    base = cand.select("q_rid", "t_rid", pl.col("src").cast(pl.Float32), "bscore", "cos_name", "cos_addr",
                       pl.col("noaddr_pass").cast(pl.Float32))
    parts = [base, string_features(cand, Q, T, number_gap=number_gap, twins=twins), context_features(cand, Q, T)]
    if align:
        parts.append(align_features(cand, Q, T, name_idf(Q)))
    return pl.concat(parts, how="horizontal")
