"""Render November runs with each control-confirmed role-1 fill strategy.

Only strategies confirmed on held-out genuine controls by partial_frame_quality.py
are allowed. Role-0 and the 26 known role-1 bytes are untouched; the 6 unknown
role-1 keyed bytes are substitutes, and roles 2 and 3 stay explicit losses.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pfq', ROOT/'scripts/partial_frame_quality.py')
pfq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfq)

MODE_POSITIONS = [(2 + c) if role == 0 else
                  ((39 if role == 1 else 72 if role == 2 else 105) + 131*c)
                  for c in range(5) for role in range(4)]
PACKET_STARTS = [(7 if role == 0 else 40 if role == 1 else 73 if role == 2 else 106) + 131*c
                 for c in range(5) for role in range(4)]
KEYED = np.arange(0, 32, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--validation', default='work/nov25_partial_quality/report.json')
    ap.add_argument('--rate', type=int, default=32000)
    a = ap.parse_args()
    confirmed = json.loads((ROOT/a.validation).read_text(encoding='utf-8'))['summary']['confirmed_on_holdout']
    if not confirmed:
        raise SystemExit('no strategy passed the held-out controls')
    out = (ROOT/a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with np.load(ROOT/'work/nov25_repeated_text/keys.npz') as d:
        take = d['crc_supported'][:, :11].all(axis=1)
        counters = d['counters'][take].astype(np.int64)
        keys = d['ks'][take, :16].copy()
    with np.load(ROOT/'work/nov25_decode_salvaged/groups.npz') as d:
        ix = np.searchsorted(d['counters'], counters)
        assert np.array_equal(d['counters'][ix], counters)
        du = d['du'][ix].copy()
    runs = np.split(np.arange(len(counters)), np.flatnonzero(np.diff(counters) != 1) + 1)
    log, rendered = [], []
    rng = np.random.default_rng(31337)
    with tempfile.TemporaryDirectory(prefix='rv_', dir=out) as td:
        tmp = Path(td)
        for ids in runs:
            span = len(ids)*60
            frames = np.zeros((span, 33), np.uint8)
            flat = np.zeros((span, 32), np.uint8)
            slot = np.arange(span) % 4
            for k, g in enumerate(ids):
                for p in range(3):
                    row = du[g, p]
                    for j, (start, mode_byte) in enumerate(zip(PACKET_STARTS, MODE_POSITIONS)):
                        i = k*60 + p*20 + j
                        flat[i] = row[start:start+32]
                        frames[i, 0] = row[mode_byte]
                base = k*60
                rows0 = base + np.flatnonzero(slot[base:base+60] == 0)
                rows1 = base + np.flatnonzero(slot[base:base+60] == 1)
                flat[np.ix_(rows0, KEYED)] ^= keys[g, :11]
                flat[np.ix_(rows1, KEYED[:5])] ^= keys[g, 11:16]
            frames[:, 1:] = flat
            tag = str(counters[ids[0]]) + '-' + str(counters[ids[-1]])
            for strategy in confirmed:
                built, keep = pfq.apply_strategy(frames, strategy, rng, frames)
                pcm = pfq.decode(tmp, 'r_' + tag + '_' + strategy, built, keep, log, a.rate)
                name = 'NOV25_' + strategy + '_' + str(a.rate//1000) + 'k_' + tag
                path = out/(name + '.wav')
                with wave.open(str(path), 'wb') as w:
                    w.setparams((1, 2, a.rate, 0, 'NONE', 'not compressed'))
                    w.writeframes(pcm.tobytes())
                rendered.append(dict(file=path.name, strategy=strategy, counters=tag,
                                     frames_supplied=int(keep.sum()), frames_total=int(span),
                                     seconds=round(len(pcm)/a.rate, 2),
                                     rms=round(float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))), 5)))
                print(json.dumps(rendered[-1]), flush=True)
    (out/'report.json').write_text(json.dumps(dict(strategies=confirmed, rendered=rendered,
        note='Role-1 unknown bytes are substitutes validated only on genuine controls; '
             'November output remains unverified broadcast audio.'), indent=2), encoding='utf-8')
    (out/'calls.json').write_text(json.dumps(log, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
