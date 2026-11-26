"""Best-quality decode of one counter window.

The window is decoded as a single continuous stream so codec state carries across
the short holes. Keyed groups get role 0 exact and role 1 filled per group by the
continuity choice that scored best on held-out genuine controls (the six unknown
role-1 bytes are substitutes, not keys). Roles 2/3 are erased by default; --role23
donor fills the whole thing instead, which is louder and only sometimes clearer.

Usage:
  python scripts/render_window_best.py <salvage> <keys.npz> --out DIR \
      --from 43680098 --to 43680150 [--role23 donor] [--rate 32000]
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import wave

import numpy as np
from build_evs_gcc_independent import linux

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import role1_key_search as rks
from role1_smooth_fill import smooth_fill

_spec = importlib.util.spec_from_file_location('pfq', HERE/'partial_frame_quality.py')
pfq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pfq)

PACKET_STARTS = [(7 if r == 0 else 40 if r == 1 else 73 if r == 2 else 106) + 131*c
                 for c in range(5) for r in range(4)]
MODE_POSITIONS = [(2 + c) if r == 0 else
                  ((39 if r == 1 else 72 if r == 2 else 105) + 131*c)
                  for c in range(5) for r in range(4)]
KEYED = np.arange(0, 32, 3)
ROLE1_UNKNOWN = np.array([16, 19, 22, 25, 28, 31])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('salvage')
    ap.add_argument('keys')
    ap.add_argument('--out', required=True)
    ap.add_argument('--from', dest='lo', type=int, required=True)
    ap.add_argument('--to', dest='hi', type=int, required=True)
    ap.add_argument('--rate', type=int, default=32000)
    ap.add_argument('--role23', choices=('erase', 'donor'), default='erase')
    a = ap.parse_args()
    salvage = (ROOT/a.salvage).resolve()
    out = (ROOT/a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with np.load(salvage/'groups.npz') as d:
        counters = d['counters'].astype(np.int64)
        du = d['du'].copy()
    with np.load(ROOT/a.keys) as d:
        usable = d['candidate_mask'].all(axis=1)
        keys = {int(c): k.copy() for c, k in zip(d['counters'][usable], d['ks'][usable])}
    idx = {int(c): i for i, c in enumerate(counters)}

    span = a.hi - a.lo + 1
    frames = np.zeros((span*60, 33), np.uint8)
    supplied = np.zeros(span*60, bool)
    log, rng = [], np.random.default_rng(0)
    role1_rows = np.flatnonzero(np.arange(span*60) % 4 == 1)
    role1_groups = np.repeat(np.arange(span), 15)

    keyed_here = [g for g in range(a.lo, a.hi+1) if g in keys and g in idx]
    donor_src = None
    if keyed_here:
        # Donor pool: the plaintext keyed bytes of the nearest keyed group.
        def plain_keyed(g):
            base = g - a.lo
            flat = np.zeros((60, 32), np.uint8)
            for phase in range(3):
                for j, begin in enumerate(PACKET_STARTS):
                    flat[phase*20 + j] = du[idx[g], phase, begin:begin+32]
            slot = np.arange(60) % 4
            flat[np.ix_(np.flatnonzero(slot == 0), KEYED)] ^= keys[g][:11]
            flat[np.ix_(np.flatnonzero(slot == 1), KEYED[:5])] ^= keys[g][11:16]
            return flat
        donor_src = {g: plain_keyed(g) for g in keyed_here}

    with tempfile.TemporaryDirectory(prefix='win_', dir=out) as td:
        tmp = Path(td)
        rks.build_traced(log)
        for g in range(a.lo, a.hi+1):
            base = (g - a.lo)*60
            if g not in idx:
                continue
            for phase in range(3):
                for j, (begin, mode_byte) in enumerate(zip(PACKET_STARTS, MODE_POSITIONS)):
                    i = base + phase*20 + j
                    frames[i, 0] = du[idx[g], phase, mode_byte]
                    frames[i, 1:] = du[idx[g], phase, begin:begin+32]
            if g in keys:
                flat = np.zeros((60, 32), np.uint8)
                for phase in range(3):
                    for j, begin in enumerate(PACKET_STARTS):
                        flat[phase*20 + j] = du[idx[g], phase, begin:begin+32]
                slot = np.arange(60) % 4
                flat[np.ix_(np.flatnonzero(slot == 0), KEYED)] ^= keys[g][:11]
                flat[np.ix_(np.flatnonzero(slot == 1), KEYED[:5])] ^= keys[g][11:16]
                frames[base:base+60, 1:] = flat
                # Continuity choice for the six unknown role-1 bytes.
                part, part_keep, _ = smooth_fill(tmp, 'g%d' % g, frames[base:base+60], log, rng)
                frames[base:base+60] = part
                supplied[base:base+60] = part_keep
        # Roles 2/3 in keyed groups: erase (default) or fill from the nearest keyed group.
        if a.role23 == 'donor' and keyed_here:
            for g in keyed_here:
                base = (g - a.lo)*60
                near = min(keyed_here, key=lambda x: (abs(x - g), x))
                rows23 = base + np.flatnonzero(np.arange(60) % 4 > 1)
                src23 = np.flatnonzero(np.arange(60) % 4 > 1)
                frames[np.ix_(rows23, np.arange(1, 33))] = \
                    donor_src[near][np.ix_(src23, np.arange(0, 32))]
                supplied[rows23] = True

        stream, raw = out/'window.192', out/'window.raw'
        with stream.open('wb') as f:
            for bits, good in zip(np.unpackbits(frames, axis=1), supplied):
                if good:
                    f.write(np.array([0x6b21, 264], '<u2').tobytes())
                    f.write(np.where(bits, 0x81, 0x7f).astype('<u2').tobytes())
                else:
                    f.write(np.array([0x6b20, 0], '<u2').tobytes())
        command = ['wsl', '-d', 'Ubuntu', '--', linux(ROOT/'work/evs_gcc_independent/EVS_dec'),
                   '-q', '-no_delay_cmp', str(a.rate//1000), linux(stream), linux(raw)]
        p = subprocess.run(command, capture_output=True, timeout=3600)
        if p.returncode:
            raise SystemExit(p.stderr.decode(errors='replace')[-2000:])
        pcm = np.fromfile(raw, '<i2')
        assert len(pcm) == len(frames)*(a.rate//50)
    stream.unlink(); raw.unlink()
    tag = 'role23%s' % a.role23
    path = out/('KCBS2_WINDOW_%s_%dk_%d-%d.wav' % (tag, a.rate//1000, a.lo, a.hi))
    with wave.open(str(path), 'wb') as w:
        w.setparams((1, 2, a.rate, 0, 'NONE', 'not compressed'))
        w.writeframes(pcm.tobytes())
    report = dict(file=path.name, seconds=round(len(pcm)/a.rate, 2), counters=span,
                  keyed_groups=len(keyed_here),
                  supplied_frames=int(supplied.sum()), total_frames=int(len(frames)),
                  role23=a.role23,
                  clipped_fraction=round(float(np.mean(np.abs(pcm) >= 32767)), 5),
                  rms=round(float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))), 4))
    (out/('report_%s.json' % tag)).write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out/('calls_%s.json' % tag)).write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
