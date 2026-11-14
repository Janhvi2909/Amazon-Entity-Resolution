"""House numbers, streets and near-number "twins".

The data plants distractors that copy a business's name and street but move its house
number a few doors (831 vs 835 Winsor Pl). In train they link to no S1 record at all, so
exclusive assignment and the competition features never see a rival for them; only the
number itself separates them from true matches, which carry digit typos (189 vs 889) at
large numeric gaps. Measured on US train (same name, same street): gap 2-30 -> 2-11%
matches; one digit edited with a gap above 30 -> 82-97% matches.

``house_frame`` parses every normalised address into (number, suffix, street key);
``twin_links`` flags pairs whose street keys agree and whose numbers are close;
``augment_twins`` clones real twin distractors of the train pool to reach the twin density
measured on test (used to build a test-density evaluation and training set).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .normalize import ADDR_CANON, clean_addresses, clean_names, _fix_name_token, LEGAL

SUFFIXES = {"bis", "ter", "quater", "a", "b", "c"}
_STREET_TYPES = set(ADDR_CANON) | set(ADDR_CANON.values())
STOP = _STREET_TYPES | {"de", "des", "du", "la", "le", "les", "the", "and", "of", "et", "sur", "en", "aux", "au",
                        "d", "l", "st", "no", "nr", "opp"}


def parse_house(addr: str):
    """(number, suffix, street key) of a normalised address, or None.

    The house number is the first all-digit token followed by a non-digit token; digits
    right before it (Indian "3 4 114 12") form a prefix that must also agree. The street
    key is that prefix plus the first two content words after the number (street types and
    stop words skipped), so "18 rue des pins francs" and "18 r des pins francs" agree.
    """
    toks = addr.split()
    n = len(toks)
    for i, t in enumerate(toks):
        if t.isdigit() and i + 1 < n and not toks[i + 1].isdigit():
            if len(t) > 6:
                return None
            j = i
            while j > 0 and toks[j - 1].isdigit():
                j -= 1
            k, suf = i + 1, ""
            nxt = toks[k]
            if nxt in SUFFIXES:
                suf, k = nxt, k + 1
            elif nxt.startswith("h") and nxt[1:].rstrip("abcdefghijklmnopqrstuvwxyz") == t:  # "h5bis" marker
                suf, k = nxt[1 + len(t):], k + 1
            content = [x for x in toks[k:k + 8] if x.isalpha() and len(x) >= 3 and x not in STOP][:2]
            if not content:
                return None
            return int(t), suf, " ".join(toks[j:i]) + "|" + " ".join(content)
    return None


def house_frame(rid: pl.Series, addr: pl.Series) -> pl.DataFrame:
    """rid, h_num (Int64), h_suf, h_key for every address (nulls where unparsed)."""
    vals = addr.fill_null("").to_list()
    parsed = [parse_house(a) if a else None for a in vals]
    return pl.DataFrame({
        "rid": rid,
        "h_num": pl.Series([p[0] if p else None for p in parsed], dtype=pl.Int64),
        "h_suf": pl.Series([p[1] if p else None for p in parsed], dtype=pl.Utf8),
        "h_key": pl.Series([p[2] if p else None for p in parsed], dtype=pl.Utf8),
    })


def twin_links(links: pl.DataFrame, HQ: pl.DataFrame, HT: pl.DataFrame, lo: int = 2, hi: int = 30) -> pl.Series:
    """Boolean per row of ``links`` (q_rid, t_rid): same street key, house numbers lo..hi apart."""
    x = (links.select("q_rid", "t_rid")
         .join(HQ.select(pl.col("rid").alias("q_rid"), pl.col("h_num").alias("qa"), pl.col("h_key").alias("qk")),
               on="q_rid", how="left", maintain_order="left")
         .join(HT.select(pl.col("rid").alias("t_rid"), pl.col("h_num").alias("tb"), pl.col("h_key").alias("tk")),
               on="t_rid", how="left", maintain_order="left"))
    gap = (pl.col("qa") - pl.col("tb")).abs()
    return x.select((pl.col("qk") == pl.col("tk")) & gap.is_between(lo, hi)).to_series().fill_null(False)


# --------------------------------------------------------------- augmentation
def _raw_keys(df: pl.DataFrame) -> pl.DataFrame:
    """rid, country, core name, house parse of raw records (generic rules only, no learned aliases)."""
    names = clean_names(df)
    toks = names.select("rid", pl.col("c").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
    vocab = toks.select(pl.col("t").unique())
    vocab = vocab.with_columns(pl.col("t").map_elements(_fix_name_token, return_dtype=pl.Utf8).alias("m"))
    core = (toks.join(vocab, on="t").filter(~pl.col("m").is_in(list(LEGAL)))
            .group_by("rid", maintain_order=True).agg(pl.col("m").str.join(" ").alias("nm")))
    a = clean_addresses(df)
    a = a.with_columns(pl.col("a").str.split(" ").list.eval(pl.element().replace(ADDR_CANON)).list.join(" ").alias("a"))
    h = house_frame(a["rid"], a["a"])
    return df.select("rid", "country").join(core, on="rid", how="left").join(h, on="rid", how="left")


def twin_pairs(Q: pl.DataFrame, T: pl.DataFrame, lo: int = 1, hi: int = 30) -> pl.DataFrame:
    """(q_rid, t_rid, gap) with the same core name and street key and numbers lo..hi apart."""
    kq, kt = _raw_keys(Q), _raw_keys(T)
    kq = kq.filter(pl.col("h_num").is_not_null() & (pl.col("nm") != ""))
    kt = kt.filter(pl.col("h_num").is_not_null() & (pl.col("nm") != ""))
    j = (kq.select(pl.col("rid").alias("q_rid"), "country", "nm", "h_key", pl.col("h_num").alias("qa"))
         .join(kt.select(pl.col("rid").alias("t_rid"), "country", "nm", "h_key", pl.col("h_num").alias("tb")),
               on=["country", "nm", "h_key"]))
    return j.with_columns((pl.col("qa") - pl.col("tb")).abs().alias("gap")).filter(pl.col("gap").is_between(lo, hi))


def twin_rates(Q: pl.DataFrame, T: pl.DataFrame) -> dict:
    """Near-number twins per S1 record, per country."""
    tp = twin_pairs(Q, T)
    n = Q.group_by("country").len()
    c = tp.group_by("country").len().rename({"len": "twins"})
    return {r["country"]: r["twins"] / r["len"] for r in n.join(c, on="country", how="left").fill_null(0).iter_rows(named=True)}


def augment_twins(Q: pl.DataFrame, T: pl.DataFrame, pairs: pl.DataFrame, target_rates: dict, seed: int = 42):
    """Clone real twin distractors of the train pool until each country reaches ``target_rates`` twins per S1.

    Q, T are raw train records with rid; ``pairs`` the true links (q_rid, t_rid). Only
    targets that link to no S1 record are cloned (never a matched row). A clone keeps the
    distractor's name, source and address and gets a new house number: the anchor S1's
    number moved by a gap drawn from the real twins' gaps. Returns (new target rows, report).
    """
    rng = np.random.default_rng(seed)
    tp = twin_pairs(Q, T)
    tp = tp.join(pairs.select(pl.col("t_rid")).unique(), on="t_rid", how="anti")  # unmatched targets only
    now = twin_rates(Q, T)
    nq = dict(Q.group_by("country").len().iter_rows())
    rows, report = [], {}
    for country, rate in sorted(target_rates.items()):
        pool = tp.filter(pl.col("country") == country)
        need = int(round((rate - now.get(country, 0.0)) * nq.get(country, 0)))
        report[country] = {"train_rate": now.get(country, 0.0), "target_rate": rate, "pool": pool.height, "clones": max(need, 0)}
        if need <= 0 or pool.height == 0:
            continue
        pick = pool.sample(n=need, with_replacement=need > pool.height, seed=int(rng.integers(1 << 31)), shuffle=True)
        gaps = rng.choice(pool["gap"].to_numpy(), size=need) * rng.choice([-1, 1], size=need)
        new = pick["qa"].to_numpy() + gaps
        bad = (new <= 0) | (new == pick["tb"].to_numpy())
        new = np.where(bad, pick["qa"].to_numpy() + np.abs(gaps), new)
        bad = new == pick["tb"].to_numpy()
        new = np.where(bad, new + 1, new)
        rows.append(pick.select("t_rid", "tb").with_columns(pl.Series("new_num", new.astype(np.int64))))
    if not rows:
        return T.clear(), report
    R = pl.concat(rows).with_row_index("k")
    src = T.join(R.select(pl.col("t_rid").alias("rid"), "k", "tb", "new_num"), on="rid").sort("k")
    # replace the first standalone occurrence of the old number (leading zeros allowed) in the raw address
    found = [_find_num(a, int(b)) for a, b in zip(src["business_address"].to_list(), src["tb"].to_list())]
    addr = [a.replace(m, str(nn), 1) if m else None for a, m, nn in
            zip(src["business_address"].to_list(), found, src["new_num"].to_list())]
    out = src.with_columns(pl.Series("business_address", addr, dtype=pl.Utf8),
                           (pl.col("entity_id").str.slice(0, 3) + pl.lit("SYN") + pl.col("k").cast(pl.Utf8).str.zfill(8)).alias("entity_id"))
    out = out.filter(pl.col("business_address").is_not_null())  # number not found verbatim: no clone
    for country in report:
        report[country]["written"] = int((out["country"] == country).sum())
    return out.select([c for c in T.columns if c != "rid"]), report


def _find_num(addr: str, num: int):
    import re
    m = re.search(rf"(?<!\d)0*{num}(?!\d)", addr or "")
    return m.group(0) if m else None
