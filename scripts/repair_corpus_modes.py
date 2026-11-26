"""Rebuild corpus modes from source DUs, checking payloads and March idle keys.

Never overwrites the supplied corpus. Any conflict, missing group, payload
mismatch, or unsupported key stops the export.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tail_fill_audit import crc, read_groups
from exact_frame_key_attack import STARTS, MODES


def repair(source, chunk_root, output):
    with np.load(source) as d:
        arrays = {k: d[k].copy() for k in d.files}
    paths = [Path(chunk_root) / str(name) / 'msc.bin' for name in np.unique(arrays['src_file'])]
    counters, du, _, conflicts, stats = read_groups(paths)
    if conflicts:
        raise ValueError(f'{len(conflicts)} conflicting counters in source')
    lookup = {int(c): i for i, c in enumerate(counters)}
    selected = du[[lookup[int(c)] for c in arrays['counter']]]
    raw = np.stack([selected[:, ph, s:s+32] for ph in range(3) for s in STARTS], axis=1)
    if not np.array_equal(raw, arrays['packets_raw']):
        raise ValueError('source payload differs from stored corpus')
    keys = selected[:, 0, 698:714] ^ 255
    if not np.array_equal(keys, arrays['ks0_15']):
        raise ValueError('source idle key differs from stored corpus')
    messages = np.concatenate((selected[:, 1, 698:714] ^ keys,
                               selected[:, 2, 698:714] ^ keys), axis=1)
    for message in messages:
        if (bytes(message[:27]) != b'\x68\xf0TextMessage is sending...' or
                crc(message[:27]) != int.from_bytes(bytes(message[27:29]), 'big')):
            raise ValueError('key fails exact known text plus CRC check')
    modes = np.stack([selected[:, ph, m] for ph in range(3) for m in MODES], axis=1)
    report = dict(groups=len(selected), source_files=len(paths),
                  payload_match=True, exact_idle_text_and_crc=True,
                  mode_changes_per_role=[int(np.sum(arrays['mode'][:, r::4] != modes[:, r::4]))
                                         for r in range(4)],
                  note='Only mode labels repaired. This does not identify the codec.')
    arrays['mode'] = modes
    arrays['mode_source_positions'] = np.array(MODES, dtype=np.int16)
    arrays['idle_text_crc_ok'] = np.ones(len(selected), dtype=bool)
    arrays['repair_report'] = np.array(json.dumps(report))
    with Path(output).open('xb') as f:
        np.savez_compressed(f, **arrays)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source')
    ap.add_argument('chunk_root')
    ap.add_argument('output')
    a = ap.parse_args()
    repair(a.source, a.chunk_root, a.output)
