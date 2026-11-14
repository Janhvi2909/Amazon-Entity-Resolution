"""Canonicalisation learned from the provided files for countries that have no training links.

A country with training links gets its aliases and region merges learned from those links
(aliases.py, blocking.py). A country without any (France in test) gets none of that, so
three structural rules are learned here instead, from its S1/S2/S3 records alone:

* geography: S1 addresses end "..., city, region". City -> region is read from S1;
  departements (segments seen only in S2/S3, e.g. "Gironde") are mapped to the region of
  the cities they co-occur with. Every address is rewritten as street segments, then the
  city, then the region, whatever the order and whichever of region/departement it carried.
* digit-corrupted street types ("COUR2 du Chapeau Rouge"): a token of 3-5 letters plus one
  digit is mapped to the most frequent S1 street-type word starting with those letters.
* bis/ter/quater stay attached to the house number ("5 bis" != "5"), see normalize.py.

Legal forms: where stripping legal forms from names makes many distinct S1 records
identical (same name and address), the legal form is identity-bearing and is kept in the
core name. Measured: France 2 -> 59 duplicates, US 0 -> 6, India 0 -> 2.

No external data: only the challenge files are read.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import polars as pl
from anyascii import anyascii

from .normalize import ADDR_CANON, LEGAL, _fix_name_token, clean_addresses, clean_names

LEGAL_FORMS = frozenset("llc inc corp co ltd pvt pllc llp lp plc pa pc sa sas sasu sarl eurl sci snc scop selarl".split())
FALLBACK_STREET = {"cour2": "cours", "chem1": "chemin", "espl1": "esplanade", "chau1": "chaussee", "cite1": "cite"}
_CORRUPT = re.compile(r"^([a-z]{3,5})(\d)$")


def seg_key(s: str) -> str:
    """Order-free spelling key of an address segment: "St.-Nazaire" and "SAINT NAZAIRE" agree."""
    return "".join(ADDR_CANON.get(t, t) for t in re.findall(r"[a-z0-9]+", anyascii(s or "").lower()))


def _segments(addr: str) -> list[str]:
    return [p.strip() for p in (addr or "").split(",") if p.strip()]


def learn_keep_legal(s1: pl.DataFrame, min_dups: int = 20, min_ratio: float = 10.0) -> dict:
    """Countries where stripping legal forms merges distinct S1 records (same name and address)."""
    df = s1.select("entity_id", "country", "business_name", "business_address").with_row_index("rid")
    names = clean_names(df)
    toks = names.select("rid", pl.col("c").str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
    vocab = toks.select(pl.col("t").unique()).with_columns(pl.col("t").map_elements(_fix_name_token, return_dtype=pl.Utf8).alias("m"))
    toks = toks.join(vocab, on="t")
    full = toks.group_by("rid", maintain_order=True).agg(pl.col("m").str.join(" ").alias("full"))
    core = toks.filter(~pl.col("m").is_in(list(LEGAL))).group_by("rid", maintain_order=True).agg(pl.col("m").str.join(" ").alias("core"))
    x = (df.select("rid", "country").join(full, on="rid", how="left").join(core, on="rid", how="left")
         .join(clean_addresses(df), on="rid", how="left"))
    out = {}
    for c in x["country"].unique().sort().to_list():
        xc = x.filter(pl.col("country") == c)
        kept = int(xc.select("full", "a").is_duplicated().sum())
        stripped = int(xc.select("core", "a").is_duplicated().sum())
        out[c] = {"dups_kept": kept, "dups_stripped": stripped,
                  "keep_legal": stripped >= min_dups and stripped >= min_ratio * max(kept, 1)}
    return out


def learn_geo(s1: pl.DataFrame, targets: pl.DataFrame, min_share: float = 0.002, dominance: float = 0.9) -> dict:
    """{segment key: [kind, region key]} for one country; kind in city / region / dept."""
    pairs, last_n, prev_n = Counter(), Counter(), Counter()
    for a in s1["business_address"].to_list():
        s = [seg_key(x) for x in _segments(a)]
        if len(s) >= 2:
            last_n[s[-1]] += 1
            prev_n[s[-2]] += 1
            pairs[(s[-2], s[-1])] += 1
    n = s1.height
    # a region ends S1 addresses and (almost) never stands before another segment; a city does both
    regions = {r for r, k in last_n.items() if k >= min_share * n and prev_n[r] < 0.1 * k}
    city_tot, city_best = Counter(), {}
    by_city = defaultdict(Counter)
    for (c, r), k in pairs.items():
        if r in regions:
            by_city[c][r] += k
    geo = {r: ["region", r] for r in regions}
    for c, cnt in by_city.items():
        tot = sum(cnt.values())
        r, k = cnt.most_common(1)[0]
        if tot >= 50 and k >= dominance * tot and c not in regions:
            geo[c] = ["city", r]
    # departements: frequent target segments, not city/region, co-occurring with known cities
    seg_n, seg_reg = Counter(), defaultdict(Counter)
    for a in targets["business_address"].to_list():
        keys = [seg_key(s) for s in _segments(a)]
        city_regions = [geo[k][1] for k in keys if k in geo and geo[k][0] == "city"]
        for k in set(keys):
            if k in geo or not k or any(ch.isdigit() for ch in k):
                continue
            seg_n[k] += 1
            for r in city_regions:
                seg_reg[k][r] += 1
    for k, cnt in seg_n.items():
        if cnt < min_share * targets.height or not seg_reg[k]:
            continue
        r, kr = seg_reg[k].most_common(1)[0]
        if kr >= 0.5 * cnt and kr >= dominance * sum(seg_reg[k].values()):
            geo[k] = ["dept", r]
    return geo


def learn_street_fix(s1: pl.DataFrame, targets: pl.DataFrame, min_tok: int = 20, min_cand: int = 30) -> dict:
    """Digit-corrupted street-type tokens -> the S1 street-type word they abbreviate."""
    street = Counter()  # S1 words right after the house number
    for a in s1["business_address"].to_list():
        t = re.findall(r"[a-z0-9]+", anyascii(a or "").lower())
        for i in range(len(t) - 1):
            if t[i].isdigit() and t[i + 1].isalpha():
                street[t[i + 1]] += 1
                break
    bad = Counter()
    for a in targets["business_address"].to_list():
        for tok in re.findall(r"[a-z0-9]+", anyascii(a or "").lower()):
            if _CORRUPT.match(tok):
                bad[tok] += 1
    fix = {}
    for tok, k in bad.items():
        if k < min_tok:
            continue
        pre = _CORRUPT.match(tok).group(1)
        cands = [(n, w) for w, n in street.items() if w.startswith(pre) and n >= min_cand]
        if cands:
            fix[tok] = max(cands)[1]
    for tok, w in FALLBACK_STREET.items():
        if tok in bad and tok not in fix:
            fix[tok] = w
    return fix


def learn_canon(train_s1: pl.DataFrame, test_s1: pl.DataFrame, test_t: pl.DataFrame, link_countries: list[str],
                unseen_canon: bool, legal_learned: bool) -> dict:
    spec = {"keep_legal": [], "legal_report": {}, "unseen": [], "geo": {}, "street_fix": {}}
    if legal_learned:
        rep = learn_keep_legal(pl.concat([train_s1, test_s1]))
        spec["legal_report"] = rep
        spec["keep_legal"] = sorted(c for c, v in rep.items() if v["keep_legal"])
    if unseen_canon:
        unseen = sorted(set(test_s1["country"].unique().to_list()) - set(link_countries))
        spec["unseen"] = unseen
        for c in unseen:
            s1c, tc = test_s1.filter(pl.col("country") == c), test_t.filter(pl.col("country") == c)
            spec["geo"][c] = learn_geo(s1c, tc)
            spec["street_fix"][c] = learn_street_fix(s1c, tc)
    return spec


def canon_address(addr: str, geo: dict, fix: dict) -> str:
    """Street segments first, then city, then region; departements dropped; street types repaired."""
    if not addr:
        return addr
    street, city, region = [], None, None
    for s in _segments(addr):
        k = seg_key(s)
        g = geo.get(k)
        if g is None:
            if fix:
                s = re.sub(r"\b[A-Za-z]{3,5}\d\b", lambda m: fix.get(m.group(0).lower(), m.group(0)), s)
            street.append(s)
        elif g[0] == "city":
            city, region = k, g[1]
        elif region is None:
            region = g[1]
    out = street + ([city] if city else []) + ([region] if region else [])
    return ", ".join(out)


def apply_canon(df: pl.DataFrame, spec: dict) -> pl.DataFrame:
    """Rewrite business_address of rows in the spec's unseen countries (all sources alike)."""
    if not spec or not spec.get("unseen"):
        return df
    parts = []
    for c in spec["unseen"]:
        m = pl.col("country") == c
        sub = df.filter(m)
        if sub.height == 0:
            continue
        geo, fix = spec["geo"].get(c, {}), spec["street_fix"].get(c, {})
        new = [canon_address(a, geo, fix) for a in sub["business_address"].to_list()]
        parts.append(sub.with_columns(pl.Series("business_address", new, dtype=pl.Utf8)))
    rest = df.filter(~pl.col("country").is_in(spec["unseen"]))
    order = df.columns
    out = pl.concat([rest] + [p.select(order) for p in parts])
    return out.sort("rid") if "rid" in order else out
