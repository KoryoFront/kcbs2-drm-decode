"""Fast decode for captures with the March-style idle tail.

The March phase-0 tail plaintext is sixteen 0xFF bytes, so the keystream falls out
directly: ks[0..15] = tail XOR 0xFF, and the known text payload plus its CRC-16
confirm it per group. There is nothing to search, so this path skips fill-crib
scanning, repeated-text matching and per-group decoding.

Speed measures: existing MSC dumps are reused when present (no re-demodulation),
keys come from one vectorised pass, and the whole capture is decoded in a single
codec call instead of one call per group.

Role 0 is exact and role 1 has five of eleven keyed bytes; the six unknown role-1
bytes are substitutes and roles 2/3 are erased.

Several captures of the same broadcast may be passed at once. Sequential recordings
extend the decoded span and simultaneous ones fill each other's gaps, because groups
merge by their 32-bit counter.

Usage:
  python scripts/kcbs_decode_fast.py <iq.wav> [more.wav ...] --out <dir>
"""
import argparse
import glob
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import wave

import numpy as np
from build_evs_gcc_independent import linux

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from tail_fill_audit import audit, crc
import render_continuous as rc

KNOWN = b'\x68\xf0TextMessage is sending...'


def find_dumps(source):
    """Existing dumps for this capture, so demodulation is not repeated."""
    stem = Path(source).stem.replace('_iq', '')
    found = sorted(glob.glob(str(ROOT/'dumps'/(stem + '*.msc.bin'))))
    found += sorted(glob.glob(str(ROOT/'work/drmradio/_chunks'/(stem + '__c*')/'msc.bin')))
    found += sorted(glob.glob(str(ROOT/'DRM Radio'/(stem + '*.msc.bin'))))
    return [Path(f) for f in found if Path(f).stat().st_size > 0]


def idle_keys(groups_npz):
    """Vectorised keystream recovery from the phase-0 idle tail."""
    with np.load(groups_npz) as d:
        tails = d['tails']
        counters = d['counters'].astype(np.int64)
        du = d['du']
    keys = tails[:, 0] ^ 0xFF
    message = np.concatenate((tails[:, 1] ^ keys, tails[:, 2] ^ keys), axis=1)
    ok = np.array([bytes(m[:27]) == KNOWN and
                   crc(m[:27]) == int.from_bytes(bytes(m[27:29]), 'big') for m in message])
    return counters[ok], keys[ok], du[ok], int(ok.sum()), len(counters)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', nargs='+',
                    help='one or more captures; sequential or simultaneous recordings of '
                         'the same broadcast merge by counter and raise coverage')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=int, default=32000)
    ap.add_argument('--fresh', action='store_true', help='ignore cached dumps')
    a = ap.parse_args()
    out = (ROOT/a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timings = {}

    dumps, need_demod = [], []
    for src in a.source:
        found = [] if a.fresh else find_dumps(src)
        if found:
            dumps += found
        else:
            need_demod.append(Path(src).resolve())
    if dumps:
        print('reusing %d cached dump(s)' % len(dumps), flush=True)
    if need_demod:
        from kcbs_decode_audio import stage_demod
        log = []
        dumps += stage_demod(need_demod, out, 0, 0, log)
    timings['dumps_seconds'] = round(time.monotonic() - started, 1)

    if not dumps:
        raise SystemExit('no dumps available')
    groups_npz = out/'salvage'/'groups.npz'
    if not groups_npz.exists():
        audit([str(p) for p in dumps], out/'salvage')
    stage = time.monotonic()
    counters, keys, du, verified, total = idle_keys(groups_npz)
    timings['keys_seconds'] = round(time.monotonic() - stage, 1)
    print('idle keys: %d of %d groups pass the known text + CRC (%d verified)'
          % (verified, total, verified), flush=True)
    if not verified:
        raise SystemExit('no idle-key groups; this capture is not March-style')

    # Contiguous stream over the keyed span; unkeyed groups stay erased.
    first, last = int(counters[0]), int(counters[-1])
    span = last - first + 1
    pos = {int(c): i for i, c in enumerate(counters)}
    frames = np.zeros((span*60, 33), np.uint8)
    supplied = np.zeros(span*60, bool)
    slot_by_frame = np.arange(span*60) % 4
    # Nearest available keyed group per offset, for role-1 donor bytes.
    present = np.zeros(span, bool)
    present[[g - first for g in pos]] = True
    where = np.flatnonzero(present)
    donor_at = np.zeros(span, np.int64)
    for lo, hi in zip(where, where[1:] if len(where) > 1 else where):
        donor_at[lo:hi] = lo
    if len(where):
        donor_at[where[-1]:] = where[-1]
    stage = time.monotonic()
    for off in range(span):
        g = first + off
        if g not in pos:
            continue
        base = off*60
        row_du = du[pos[g]]
        for phase in range(3):
            for j, (begin, mode_byte) in enumerate(zip(rc.PACKET_STARTS, rc.MODE_POSITIONS)):
                i = base + phase*20 + j
                frames[i, 0] = row_du[phase, mode_byte]
                frames[i, 1:] = row_du[phase, begin:begin+32]
        base_slot = slot_by_frame[base:base+60]
        rows0 = base + np.flatnonzero(base_slot == 0)
        rows1 = base + np.flatnonzero(base_slot == 1)
        frames[np.ix_(rows0, rc.KEYED + 1)] ^= keys[pos[g]][:11]
        frames[np.ix_(rows1, rc.KEYED[:5] + 1)] ^= keys[pos[g]][11:16]
        handy = base + np.flatnonzero(base_slot == 1)
        # Six unknown role-1 bytes: take them from the nearest keyed group instead of
        # zeroing, so the frame carries plausible values. Substitutes, not keys.
        donor_base = int(donor_at[off])*60
        frames[np.ix_(handy, rc.ROLE1_UNKNOWN + 1)] =             frames[np.ix_(donor_base + np.flatnonzero(base_slot == 1), rc.ROLE1_UNKNOWN + 1)]
        supplied[base:base+60] = base_slot < 2
    timings['frames_seconds'] = round(time.monotonic() - stage, 1)

    stream, raw = out/'stream.192', out/'stream.raw'
    stage = time.monotonic()
    with stream.open('wb') as f:
        for bits, good in zip(np.unpackbits(frames, axis=1), supplied):
            if good:
                f.write(np.array([0x6b21, 264], '<u2').tobytes())
                f.write(np.where(bits, 0x81, 0x7f).astype('<u2').tobytes())
            else:
                f.write(np.array([0x6b20, 0], '<u2').tobytes())
    command = ['wsl', '-d', 'Ubuntu', '--', linux(ROOT/'work/evs_gcc_independent/EVS_dec'),
               '-q', '-no_delay_cmp', str(a.rate//1000), linux(stream), linux(raw)]
    p = subprocess.run(command, capture_output=True, timeout=7200)
    if p.returncode:
        raise SystemExit(p.stderr.decode(errors='replace')[-2000:])
    pcm = np.fromfile(raw, '<i2')
    timings['decode_seconds'] = round(time.monotonic() - stage, 1)
    path = out/('KCBS2_MARCH_IDLEKEYS_%dk_%d-%d.wav' % (a.rate//1000, first, last))
    with wave.open(str(path), 'wb') as w:
        w.setparams((1, 2, a.rate, 0, 'NONE', 'not compressed'))
        w.writeframes(pcm.tobytes())
    stream.unlink(); raw.unlink()

    report = dict(source=list(a.source), dumps=len(dumps), file=path.name, seconds=round(len(pcm)/a.rate, 2),
                  keyed_groups=verified, groups_in_span=span,
                  coverage=round(verified/span, 3),
                  supplied_frames=int(supplied.sum()), total_frames=int(len(frames)),
                  rms=round(float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))), 4),
                  clipped_fraction=round(float(np.mean(np.abs(pcm) >= 32767)), 5),
                  timings=timings, wall_seconds=round(time.monotonic() - started, 1),
                  note='Idle-tail keys are exact and CRC-confirmed. Role-1 unknown bytes are '
                       'neighbour substitutes; roles 2/3 erased.')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in
                      ('file', 'seconds', 'keyed_groups', 'coverage', 'rms',
                       'clipped_fraction', 'timings', 'wall_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
