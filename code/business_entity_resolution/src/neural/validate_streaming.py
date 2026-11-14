"""Bounded-memory supplementary validation for outputs in test-S1 row order.

Checks coverage/order, list syntax, duplicates within lists and match subset.
Does not check target existence or exclusive ownership; use the official ID check
or a separate target-index audit for those. Never a replacement for that report.
"""
import argparse
import itertools
import json
from pathlib import Path


def validate(matching, candidate, source1):
    counts = {'rows': 0, 'matches': 0, 'candidates': 0, 'empty_matches': 0}
    with open(matching) as m, open(candidate) as c, open(source1) as q:
        if m.readline().rstrip('\r\n') != 'source1_entity_id\tmatched_entity_ids':
            raise ValueError('Wrong matching header')
        if c.readline().rstrip('\r\n') != 'source1_entity_id\tcandidate_entity_ids':
            raise ValueError('Wrong candidate header')
        if q.readline().split('\t')[0] != 'entity_id':
            raise ValueError('Wrong test S1 header')
        for row, values in enumerate(itertools.zip_longest(m, c, q), 2):
            if any(v is None for v in values):
                raise ValueError(f'Row count differs at line {row}')
            ml, cl, ql = values
            mc, cc = ml.rstrip('\r\n').split('\t'), cl.rstrip('\r\n').split('\t')
            if len(mc) != 2 or len(cc) != 2:
                raise ValueError(f'Wrong column count at line {row}')
            if mc[0] != cc[0] or mc[0] != ql.split('\t', 1)[0]:
                raise ValueError(f'S1 mismatch or output not in source order at line {row}')
            groups = []
            for field in [mc[1], cc[1]]:
                ids = field.split(',') if field else []
                if len(set(ids)) != len(ids):
                    raise ValueError(f'Duplicate list ID at line {row}')
                if any(not x.startswith(('S2-', 'S3-')) or x.strip() != x for x in ids):
                    raise ValueError(f'Invalid target ID syntax at line {row}')
                groups.append(set(ids))
            if not groups[0] <= groups[1]:
                raise ValueError(f'Match missing from candidates at line {row}')
            counts['rows'] += 1
            counts['matches'] += len(groups[0])
            counts['candidates'] += len(groups[1])
            counts['empty_matches'] += not groups[0]
    return {'status': 'PASS', 'checks': 'S1 order/coverage, headers, list syntax, per-list uniqueness, subset',
            'not_checked': ['S2/S3 existence', 'target exclusive ownership', 'scored-candidate provenance'], **counts}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['matching', 'candidate', 'source1']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    result = validate(a.matching, a.candidate, a.source1)
    text = json.dumps(result, indent=2)
    if a.out:
        Path(a.out).write_text(text + '\n')
    print(text)
