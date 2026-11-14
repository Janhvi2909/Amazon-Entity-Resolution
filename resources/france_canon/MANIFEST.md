# France canonicalisation (NORM-v3t): what was learned, and from what

Everything here was learned by `ber/canon.py` during the prep stage of the full run from the
provided challenge files only (test S1/S2/S3 for the geography and street types, train+test S1
for the legal-form rule). No external data. The full learned spec is `canon_spec.json`.

The rules apply to **countries without training links** (learned: France only), so US and India
keep their NORM-v2 behaviour; aliases and region merges learned from links do that job there.

## 1. Legal forms are identity-bearing (keep_legal)

Rule: keep legal forms in the core name of a country when stripping them makes at least 20
S1 records exact duplicates (same name and address) and at least 10x as many as with them kept.
S1 is deduplicated, so such records are distinct businesses.

| Country | Duplicates, legal forms kept | Duplicates, stripped | Keep legal |
|---|---:|---:|---|
| France | 2 | 59 | yes |
| India | 0 | 6 | no |
| US | 0 | 10 | no |

FEAT-v4 adds `legal_equal` / `legal_conflict` (legal forms SARL/SAS/SASU/EURL/SCI/SA/LLC/INC/... of both names).

## 2. Geography (city / region / departement)

S1 addresses end `..., city, region`. Learned: 3 regions, 15 cities (each mapped to its region),
4 departements seen only in S2/S3, mapped to the region of the cities they co-occur with.
Every French address is rewritten as street segments, then city, then region, so
`Nouvelle-Aquitaine, Bordeaux, 18 Rue ...` and `18 Rue ..., Bordeaux, Gironde` agree.

- Regions: hautsdefrance, nouvelleaquitaine, paysdelaloire
- Departements: gironde -> nouvelleaquitaine, loireatlantique -> paysdelaloire, nord -> hautsdefrance, pasdecalais -> hautsdefrance
- Cities: bordeaux, pessac, merignac, latestedebuch, legecapferret (Nouvelle-Aquitaine); nantes, stnazaire, pornic,
  labauleescoublac, stherblain (Pays de la Loire); lille, tourcoing, dunkerque, roubaix, calais (Hauts-de-France)

## 3. Digit-corrupted street types

A token of 3-5 letters plus one digit that S1 never uses is mapped to the most frequent S1
street-type word (the word right after a house number) starting with those letters. Learned:

| Token | Repaired to |
|---|---|
| cour2, cou2, cour1 | cours |
| chem1 | chemin |
| espl1 | esplanade |
| chau1 | chaussee |
| cite1 | cite |
| sent1 | sente |
| ruel1 | ruelle |
| peti3, peti6 | petite |
| parc1 | parc |

Fallback (only used if learning had missed them): cour2->cours, chem1->chemin, espl1->esplanade,
chau1->chaussee, cite1->cite. Learning found all five, so the fallback was not used.

## 4. bis / ter / quater

`5 bis` becomes tokens `5 h5bis`: the house number stays a number for the number features,
and the extra token makes `5 bis` a different address from `5`. The house parser records the
suffix and FEAT-v4 has `h_suf_conflict`.
