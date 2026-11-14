"""Address components (NORM-v4t): street core, landmark clause, unit, PO box.

The sources add, drop and reword optional address parts, and the parts differ by country:
Indian addresses carry landmark clauses ("near SBI ATM", 13.5% of S1) and flat/shop numbers,
US ones suites and PO boxes/PMBs (injected into 1.6% of S2/S3, never in S1), French ones "appt".
Measured on true India links: when S1 has a landmark the other record dropped, address
token-set similarity falls from 91 to 79 and 26% of such true pairs score below 70 (5% otherwise).
Splitting the optional parts off lets the matcher compare the street core on its own and
treat each optional part as agree / conflict / present on one side only. The keywords are
generic (no country is named); a country simply uses the ones its data contains.
"""
from __future__ import annotations

import re

import polars as pl
from anyascii import anyascii

LANDMARK = re.compile(r"\b(near|nr|opp|opposite|behind|beside|besides|next to|in front of|infront of|adjacent to|adj to|close to)\b")
UNIT = re.compile(r"(?:\b(?:apt|apartment|suite|ste|unit|flat|room|rm|shop|office|appt|appartement|bureau|door)\b\.?\s*(?:no\.?|number|#)?\s*[:\-]?\s*([a-z]?\d[\w/\-]*)"
                  r"|#\s*([a-z]?\d[\w/\-]*))")
POBOX = re.compile(r"\b(?:p\.?\s?o\.?\s?box|pmb|post box|boite postale|bp)\s*[:#]?\s*(\d+)")


def split_address(addr: str) -> tuple[str, str, str, str]:
    """(core, landmark, units, pobox) of a raw address; units/pobox are sorted identifier strings."""
    if not addr:
        return "", "", "", ""
    s = anyascii(addr).lower()
    lm, core = [], []
    for seg in s.split(","):
        seg = seg.strip()
        if not seg:
            continue
        m = LANDMARK.search(seg)
        if m:  # the clause from the landmark keyword to the end of the segment is optional context
            lm.append(seg[m.start():])
            seg = seg[:m.start()].strip()
            if not seg:
                continue
        core.append(seg)
    core_s = ", ".join(core)
    box = sorted({m.group(1) for m in POBOX.finditer(core_s)})
    core_s = POBOX.sub(" ", core_s)
    units = sorted({(m.group(1) or m.group(2)).strip("-/") for m in UNIT.finditer(core_s)})
    core_s = UNIT.sub(" ", core_s)
    return core_s, " ".join(lm), " ".join(units), " ".join(box)


def address_components(df: pl.DataFrame) -> pl.DataFrame:
    """rid, a_core_raw, addr_lm, addr_unit, addr_pobox (raw core is normalised by the caller)."""
    parts = [split_address(a) for a in df["business_address"].fill_null("").to_list()]
    return pl.DataFrame({"rid": df["rid"],
                         "a_core_raw": [p[0] for p in parts],
                         "addr_lm": [re.sub(r"[^a-z0-9]+", " ", p[1]).strip() for p in parts],
                         "addr_unit": [p[2] for p in parts],
                         "addr_pobox": [p[3] for p in parts]})
