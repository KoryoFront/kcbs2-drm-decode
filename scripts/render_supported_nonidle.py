"""Render non-idle excerpts from existing CRC-supported prefixes, not fill guesses."""
import argparse
import json
from pathlib import Path
import tempfile
import wave

import numpy as np

from kcbs_decode_fast import assemble_frames, decode_evs, group_runs, replay_nearby

RATE = 32000


def join_supported(groups, keys):
    with np.load(keys) as f:
        if 'crc_supported' not in f or f['crc_supported'].dtype != np.bool_:
            raise ValueError('expected a per-byte CRC-supported mask')
        mask, ks, key_c = f['crc_supported'], f['ks'], f['counters']
    if (mask.ndim != 2 or mask.shape != ks.shape or mask.shape[1] != 16
            or ks.dtype != np.uint8 or len(mask) != len(key_c)):
        raise ValueError('expected a 16-byte key and support mask for each counter')
    with np.load(groups) as f:
        counters, du = f['counters'].astype(np.int64), f['du']
    if np.unique(counters).size != len(counters) or du.shape != (len(counters), 3, 716):
        raise ValueError('expected one complete group per received counter')
    full = mask.all(axis=1)
    known_c = key_c[full].astype(np.int64)
    if len(np.unique(known_c)) != len(known_c):
        raise ValueError('ambiguous verified counters')
    shared, group_i, key_i = np.intersect1d(counters, known_c, return_indices=True)
    return shared, ks[full][key_i], du[group_i], counters


def render(groups, keys, out):
    counters, prefix, du, available = join_supported(groups, keys)
    if not len(counters):
        raise ValueError('no received groups with a full CRC-supported prefix')
    frame_samples = RATE//50
    offsets = counters - available.min()
    runs = group_runs(offsets, 0)
    pieces, parts = [], []
    cursor = 0
    out.mkdir(parents=True, exist_ok=False)
    path = out/'NONIDLE_CRC_PREFIX_UNVERIFIED_REPLAY_EXCERPTS_32k.wav'
    with tempfile.TemporaryDirectory(dir=out) as td:
        for lo, hi in runs:
            use = (offsets >= lo) & (offsets < hi)
            frames, supplied = assemble_frames(du[use], prefix[use], offsets[use]-lo, hi-lo)
            frames, supplied = replay_nearby(frames, supplied)
            pcm = decode_evs(frames, supplied, RATE, Path(td))
            ramp = np.linspace(0, 1, frame_samples)
            pcm[:frame_samples] = np.rint(pcm[:frame_samples]*ramp)
            pcm[-frame_samples:] = np.rint(pcm[-frame_samples:]*ramp[::-1])
            parts.append(dict(first_counter=int(available.min())+lo,
                              last_counter=int(available.min())+hi-1,
                              counter_span_seconds=round(lo*1.2, 3),
                              montage_seconds=round(cursor/RATE,3), duration_seconds=len(pcm)/RATE))
            pieces.extend((pcm, np.zeros(RATE//2, dtype='<i2')))
            cursor += len(pcm) + RATE//2
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        for p in pieces[:-1]:
            wav.writeframesraw(p.tobytes())
    report = dict(file=path.name, received_groups=len(available), supported_groups=len(counters),
                  supported_seconds=round(len(counters)*1.2,2),
                  separator_seconds=.5, playback_method='nearest_supplied', excerpts=parts,
                  note='CRC supports the 16-byte prefixes only, not codec identity or programme audio.')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for opt in ('groups', 'keys', 'out'):
        ap.add_argument('--'+opt, required=True, type=Path)
    a = ap.parse_args()
    render(a.groups, a.keys, a.out)
