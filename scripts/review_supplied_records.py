"""Recheck existing text-tail prefixes against received records. Does not infer keys."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from parse_text_records import crc, scan_records


def review(counters, prefixes, mask, du_counters, phases, du, repeat_blocks=0):
    if any(a.dtype.kind not in 'iu' or a.ndim != 1 for a in (counters, du_counters, phases)):
        raise ValueError('expected 1D integer counter and phase fields')
    counters = counters.astype(np.int64)
    du_counters = du_counters.astype(np.int64)
    if (counters.ndim != 1 or not len(counters) or np.any(np.diff(counters) <= 0)
            or np.any((counters < 0) | (counters > 0xffffffff))):
        raise ValueError('prefix archive needs unique increasing 32-bit counters')
    if (prefixes.dtype != np.uint8 or mask.dtype != np.bool_
            or prefixes.shape != mask.shape or mask.shape != (len(counters), 16)):
        raise ValueError('expected a uint8 prefix and bool mask with 16 columns')
    if (phases.shape != du_counters.shape or np.any((phases < 0) | (phases > 2))
            or np.any((du_counters < 0) | (du_counters > 0xffffffff))
            or du.shape != (len(phases), 716) or du.dtype != np.uint8):
        raise ValueError('expected complete 716-byte data units tagged by counter and phase')
    span = int(counters[-1]-counters[0])+1
    if span > 100000:
        raise ValueError('use a single capture spanning at most 100,000 groups')
    if repeat_blocks < 0:
        raise ValueError('repeat_blocks must be nonnegative')
    observed = 3*du_counters + phases
    if len(np.unique(observed)) != len(observed):
        raise ValueError('duplicate counter and phase observations')
    if not np.all(du[:, 1] >> 4 == phases):
        raise ValueError('archive phase does not match data-unit header')
    for n, row in enumerate(du):
        if crc(bytes(row[:714])) != int.from_bytes(bytes(row[714:]), 'big'):
            raise ValueError(f'data-unit CRC mismatch at archive row {n}')
        if phases[n] == 0 and int.from_bytes(bytes(row[664:668]), 'big') != int(du_counters[n]):
            raise ValueError(f'counter mismatch at archive row {n}')

    first = int(counters[0])
    _, data_i, key_i = np.intersect1d(observed, (3*counters[:, None]+np.arange(3)).ravel(),
                                      return_indices=True)
    blocks = observed[data_i]-3*first
    plain = np.zeros((3*span, 16), dtype=np.uint8)
    known = np.zeros_like(plain, dtype=bool)
    row_keys = prefixes[key_i//3]
    row_mask = mask[key_i//3]
    plain[blocks] = np.where(row_mask, du[data_i, 698:714] ^ row_keys, 0)
    known[blocks] = row_mask
    records = list(scan_records(plain.tobytes(), known_bytes=known.tobytes()))

    def family(r):
        return (r.offset//16 % repeat_blocks, r.sequence, r.final, r.text)

    families = Counter(family(r) for r in records) if repeat_blocks else {}
    accepted = [r for r in records if not repeat_blocks or families[family(r)] >= 2]
    confirmed_bytes = np.zeros(plain.size, dtype=bool)
    for r in accepted:
        confirmed_bytes[r.offset:r.offset+r.size] = True
    checked = confirmed_bytes.reshape(span, 3, 16).any(axis=1)[counters-first] & mask
    rows = []
    for r in records:
        block = r.offset//16 + 3*first
        rows.append(dict(asdict(r), counter=block//3, phase=block%3,
                         matching_copies=families[family(r)] if repeat_blocks else None))
    return checked, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-units', required=True, type=Path,
                    help='data_units.npz from the CRC-checked packet salvage step')
    ap.add_argument('--keys', required=True, type=Path, help='existing supplied prefixes')
    ap.add_argument('--repeat-blocks', type=int, default=0,
                    help='require a matching record at another position in this 16-byte block cycle')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    with np.load(a.keys, allow_pickle=False) as k, np.load(a.data_units, allow_pickle=False) as d:
        counters, keys, mask = k['counters'], k['ks'], k['crc_supported']
        checked, rows = review(counters, keys, mask, d['counters'], d['phases'], d['du'],
                               a.repeat_blocks)
    a.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.out/'keys.npz', counters=counters, ks=np.where(checked,keys,0),
                        crc_supported=checked)
    (a.out/'records.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),encoding='utf-8')
    digest = dict(exported_full_prefixes=int(mask.all(axis=1).sum()),
                  strict_records=len(rows), final_records=sum(r['final'] for r in rows),
                  body_lengths=sorted({r['body_length'] for r in rows}),
                  selected_records=sum(not a.repeat_blocks or r['matching_copies'] >= 2 for r in rows),
                  selected_full_prefixes=int(checked.all(axis=1).sum()), repeat_blocks=a.repeat_blocks)
    (a.out/'report.json').write_text(json.dumps(digest,indent=2),encoding='utf-8')
    print(json.dumps(digest,indent=2))


if __name__=='__main__':
    main()
