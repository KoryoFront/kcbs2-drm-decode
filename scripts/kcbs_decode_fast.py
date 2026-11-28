"""Fast decode for captures with the March-style idle tail.

The March phase-0 tail plaintext is sixteen 0xFF bytes, so the keystream falls out
directly: ks[0..15] = tail XOR 0xFF, and the known text payload plus its CRC-16
confirm it across phases. There is nothing to search, so this path skips fill-crib
scanning, repeated-text matching and per-group decoding.

Speed and continuity measures: existing MSC dumps are reused when present,
CRC-good 179-byte packets are salvaged across partial groups when at least two
phases confirm the same 16-byte idle-tail prefix, and the codec runs once per
stretch of usable groups instead of once per group.

The mask is known for all role-0 bytes and five of eleven role-1 bytes. The six
unknown role-1 bytes are substitutes. Roles 2/3 are erased by default, or replay
nearby supplied frames with --replay-nearby. The codec and audio are unverified.

Several captures of the same broadcast may be passed at once. Sequential recordings
extend the decoded span and simultaneous ones fill each other's gaps, because groups
merge by their 32-bit counter.

Usage:
  python scripts/kcbs_decode_fast.py <iq.wav> [more.wav ...] --out <dir>
"""
import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import subprocess
import tempfile
import time
import wave

import numpy as np
from build_evs_gcc_independent import linux

import render_continuous as rc
from salvage_msc_packets import records
from tail_fill_audit import audit, crc

ROOT = Path(__file__).resolve().parents[1]

KNOWN = b'\x68\xf0TextMessage is sending...'
PACKET_BYTES = 179
IDLE_TAILS = (
    b'\xff' * 16,
    KNOWN[:16],
    KNOWN[16:] + crc(KNOWN).to_bytes(2, 'big') + b'\x00\x00\x00',
)


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
                   crc(m[:27]) == int.from_bytes(bytes(m[27:29]), 'big') for m in message], dtype=bool)
    return counters[ok], keys[ok], du[ok], int(ok.sum()), len(counters)


def assemble_frames(du, keys, offsets, span):
    frames = np.zeros((span*60, 33), np.uint8)
    supplied = np.zeros(span*60, bool)
    rows0 = np.arange(0, 60, 4)
    rows1 = rows0 + 1
    # group_frames returns the mode separately, so use payload coordinates here.
    unknown = rc.KEYED[5:]
    for off, row_du, key in zip(offsets, du, keys):
        base = int(off)*60
        modes, payload = rc.group_frames(row_du, key)
        payload[np.ix_(rows1, unknown)] = payload[np.ix_(rows0, unknown)]
        frames[base:base+60, 0] = modes
        frames[base:base+60, 1:] = payload
        supplied[base + rows0] = supplied[base + rows1] = True
    return frames, supplied


def salvage_idle_packets(paths):
    """Assign (counter, phase) to synchronous MSC records and collect CRC-good 179-byte packets."""
    payloads = defaultdict(set)
    for path in paths:
        recs = list(records(path))
        barriers = {i for i, r in enumerate(recs) if r is None}
        raw_p0 = {i: r['counter'] for i, r in enumerate(recs)
                  if r and r['phase'] == 0 and r['packet_ok'][0] and r['packet_ok'][3]}
        anchors = {}
        for i, c in raw_p0.items():
            ks0 = bytes(a ^ b for a, b in zip(recs[i]['payload'][3][161:177], IDLE_TAILS[0]))
            tail_match = any(
                i + p < len(recs) and recs[i + p] and recs[i + p]['packet_ok'][0]
                and recs[i + p]['phase'] == p and recs[i + p]['packet_ok'][3]
                and not any(i < k <= i + p for k in barriers)
                and bytes(a ^ b for a, b in zip(recs[i + p]['payload'][3][161:177], IDLE_TAILS[p])) == ks0
                for p in (1, 2)
            )
            sync_match = any(
                j != i and (j - i) == 3 * (cj - c) and abs(cj - c) <= len(recs)
                and not any(min(i, j) < k < max(i, j) for k in barriers)
                for j, cj in raw_p0.items()
            )
            if tail_match or sync_match:
                anchors[i] = c

        assigned = defaultdict(set)
        for i, c in anchors.items():
            assigned[i].add((c, 0))
            for p in (1, 2):
                if i + p < len(recs) and recs[i + p] and recs[i + p]['packet_ok'][0] and recs[i + p]['phase'] == p:
                    if not any(i < k <= i + p for k in barriers):
                        assigned[i + p].add((c, p))
            for back, p in ((1, 2), (2, 1)):
                if c > 0 and i - back >= 0 and recs[i - back] and recs[i - back]['packet_ok'][0] and recs[i - back]['phase'] == p:
                    if not any(i - back <= k < i for k in barriers):
                        assigned[i - back].add((c - 1, p))

        # Dream wraps to the start of a file at EOF; only forward synchronous spans stay within one pass.
        a_idx = sorted(anchors)
        for a, b in zip(a_idx, a_idx[1:]):
            dc = anchors[b] - anchors[a]
            if dc > 0 and (b - a) == 3 * dc and not any(a < k < b for k in barriers):
                for k in range(a, b + 1):
                    r = recs[k]
                    if r and (not r['packet_ok'][0] or r['phase'] == (k - a) % 3):
                        assigned[k].add((anchors[a] + (k - a) // 3, (k - a) % 3))

        for k, candidates in assigned.items():
            if len(candidates) != 1:
                continue
            c, p = next(iter(candidates))
            r = recs[k]
            for q, ok in enumerate(r['packet_ok']):
                if ok:
                    payloads[(int(c), int(p), q)].add(r['payload'][q])
    return payloads


def assemble_salvaged_idle(payloads):
    """Unmask CRC-good packet frames in groups where >=2 phases confirm the 16-byte idle-tail key."""
    candidate_counters = sorted({c for c, _, _ in payloads})
    verified_keys = {}
    for c in candidate_counters:
        obs = []
        for p in range(3):
            if len(payloads.get((c, p, 3), set())) == 1:
                p3 = next(iter(payloads[(c, p, 3)]))
                obs.append(bytes(a ^ b for a, b in zip(p3[161:177], IDLE_TAILS[p])))
        if len(obs) >= 2 and len(set(obs)) == 1:
            verified_keys[c] = np.frombuffer(obs[0], np.uint8)

    if not verified_keys:
        return np.array([], np.int64), np.zeros((0, 33), np.uint8), np.zeros(0, bool), 0

    counters = np.array(sorted(verified_keys), dtype=np.int64)
    first, last = int(counters[0]), int(counters[-1])
    span = last - first + 1
    frames = np.zeros((span * 60, 33), np.uint8)
    exact_r0 = np.zeros(span * 60, bool)
    role1_ok = np.zeros(span * 60, bool)
    unknown = rc.KEYED[5:]

    for c in counters:
        key = verified_keys[int(c)]
        base = (int(c) - first) * 60
        unit_cache = {}
        for p in range(3):
            flags = np.array([len(payloads.get((int(c), p, q), set())) == 1 for q in range(4)], bool)
            if flags.all():
                du_bytes = b''.join(next(iter(payloads[(int(c), p, q)])) for q in range(4))
                if crc(du_bytes[:714]) != int.from_bytes(du_bytes[714:], 'big'):
                    continue
            unit = bytearray(716)
            for q in range(4):
                if flags[q]:
                    unit[q * PACKET_BYTES : (q + 1) * PACKET_BYTES] = next(iter(payloads[(int(c), p, q)]))
            unit_cache[p] = (flags, np.frombuffer(bytes(unit), np.uint8))

        # Unmask all CRC-good role-0 frames in this group first so role 1 can draw an in-group donor.
        for p, (flags, unit_arr) in unit_cache.items():
            for j in range(0, 20, 4):
                start, mode_pos = rc.PACKET_STARTS[j], rc.MODE_POSITIONS[j]
                if flags[mode_pos // PACKET_BYTES] and flags[start // PACKET_BYTES : (start + 31) // PACKET_BYTES + 1].all():
                    slot = base + p * 20 + j
                    pay = unit_arr[start : start + 32].copy()
                    pay[rc.KEYED] ^= key[:11]
                    frames[slot, 0] = unit_arr[mode_pos]
                    frames[slot, 1:] = pay
                    exact_r0[slot] = True

        group_r0 = np.flatnonzero(exact_r0[base : base + 60]) + base
        if not len(group_r0):
            continue
        for p, (flags, unit_arr) in unit_cache.items():
            for j in range(1, 20, 4):
                start, mode_pos = rc.PACKET_STARTS[j], rc.MODE_POSITIONS[j]
                if flags[mode_pos // PACKET_BYTES] and flags[start // PACKET_BYTES : (start + 31) // PACKET_BYTES + 1].all():
                    slot = base + p * 20 + j
                    donor = slot - 1 if exact_r0[slot - 1] else int(group_r0[np.argmin(np.abs(group_r0 - slot))])
                    pay = unit_arr[start : start + 32].copy()
                    pay[rc.KEYED[:5]] ^= key[11:16]
                    pay[unknown] = frames[donor, 1 + unknown]
                    frames[slot, 0] = unit_arr[mode_pos]
                    frames[slot, 1:] = pay
                    role1_ok[slot] = True

    return counters, frames, exact_r0 | role1_ok, int(exact_r0.sum())


def group_runs(offsets, merge_gap):
    """Half-open stretches of sorted, nonempty offsets, bridging up to merge_gap holes."""
    breaks = np.flatnonzero(np.diff(offsets) - 1 > merge_gap)
    return [(int(offsets[lo]), int(offsets[hi]) + 1)
            for lo, hi in zip(np.r_[0, breaks+1], np.r_[breaks, len(offsets)-1])]


def replay_nearby(frames, supplied):
    """Fill loss slots only inside received four-frame cycles; keep supplied bytes intact."""
    out, keep = frames.copy(), supplied.copy()
    rows = np.arange(len(frames))
    active = supplied[rows - rows % 4]
    for role, shift in ((2, -1), (3, 1), (3, -2)):
        targets = rows[active & (rows % 4 == role) & ~keep]
        targets = targets[(targets + shift >= 0) & (targets + shift < len(frames))]
        targets = targets[supplied[targets + shift]]
        out[targets] = frames[targets + shift]
        keep[targets] = True
    return out, keep


def attenuate(pcm, db):
    if not db:
        return pcm
    gain = np.float32(10**(-db/20))
    return np.rint(pcm.astype(np.float32)*gain).astype('<i2')


def decode_evs(part, keep, rate, temporary):
    stream, raw = temporary/'stream.192', temporary/'stream.raw'
    with stream.open('wb') as f:
        for bits, good in zip(np.unpackbits(part, axis=1), keep):
            if good:
                f.write(np.array([0x6b21, 264], '<u2').tobytes())
                f.write(np.where(bits, 0x81, 0x7f).astype('<u2').tobytes())
            else:
                f.write(np.array([0x6b20, 0], '<u2').tobytes())
    command = ['wsl', '-d', 'Ubuntu', '--', linux(ROOT/'work/evs_gcc_independent/EVS_dec'),
               '-q', '-no_delay_cmp', str(rate//1000), linux(stream), linux(raw)]
    proc = subprocess.run(command, capture_output=True, timeout=7200)
    if proc.returncode:
        raise SystemExit(proc.stderr.decode(errors='replace')[-2000:])
    expected_bytes = 2*len(part)*(rate//50)
    if not raw.is_file() or raw.stat().st_size != expected_bytes:
        raise SystemExit(f'EVS decoder did not write the expected {expected_bytes} PCM bytes')
    got = np.fromfile(raw, '<i2')
    raw.unlink()
    return got


def render_timeline(frames, supplied, runs, rate, decode):
    per_frame = rate//50
    pcm = np.zeros(len(frames)*per_frame, dtype='<i2')
    ramp = np.linspace(0, 1, per_frame)
    for r, (lo, hi) in enumerate(runs):
        start, stop = lo*60, hi*60
        part = pcm[start*per_frame:stop*per_frame]
        part[:] = decode(frames[start:stop], supplied[start:stop])
        if r:
            part[:per_frame] = np.rint(part[:per_frame]*ramp)
        if r+1 < len(runs):
            part[-per_frame:] = np.rint(part[-per_frame:]*ramp[::-1])
    return pcm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', nargs='+',
                    help='one or more captures; sequential or simultaneous recordings of '
                         'the same broadcast merge by counter and raise coverage')
    ap.add_argument('--out', required=True, help='new directory for the result and audit')
    ap.add_argument('--rate', type=int, default=32000, choices=(16000, 32000, 48000))
    ap.add_argument('--fresh', action='store_true', help='ignore cached dumps')
    ap.add_argument('--merge-gap', type=int, default=4,
                    help='bridge at most N missing groups with decoder concealment; '
                         'silence longer gaps (default: 4; 0 silences every gap)')
    ap.add_argument('--replay-nearby', action='store_true',
                    help='replay neighbouring supplied frames instead of concealing roles 2/3')
    ap.add_argument('--attenuate-db', type=float, default=0,
                    help='lower the final PCM level without compression or limiting')
    ap.add_argument('--complete-groups-only', action='store_true',
                    help='disable packet-level salvage of partial groups and require complete 3-phase DUs')
    ap.add_argument('--slice', type=float, nargs=2, metavar=('START_SEC', 'END_SEC'),
                    help='also write a lossless WAV excerpt [START_SEC, END_SEC] from the rendered timeline')
    a = ap.parse_args()
    if not 0 <= a.attenuate_db <= 60:
        ap.error('--attenuate-db must be between 0 and 60')
    if a.merge_gap < 0:
        ap.error('--merge-gap must be nonnegative')
    out = (ROOT/a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    timings = {}

    dumps, need_demod = [], []
    for src in a.source:
        found = [] if a.fresh else find_dumps(src)
        if found:
            dumps += found
        else:
            need_demod.append(Path(src).resolve())
    dumps = list(dict.fromkeys(dumps))
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
    stage = time.monotonic()
    audit([str(p) for p in dumps], out/'salvage')
    complete_counters, complete_keys, complete_du, complete_verified, total = idle_keys(groups_npz)
    if a.complete_groups_only:
        counters = complete_counters
        if not len(counters):
            raise SystemExit('no idle-key groups; this capture is not March-style')
        first, last = int(counters[0]), int(counters[-1])
        span = last - first + 1
        offsets = counters - first
        frames, supplied = assemble_frames(complete_du, complete_keys, offsets, span)
        exact_r0_count = int(complete_verified * 15)
    else:
        payloads = salvage_idle_packets(dumps)
        counters, frames, supplied, exact_r0_count = assemble_salvaged_idle(payloads)
        if not len(counters):
            raise SystemExit('no idle-key groups; this capture is not March-style')
        first, last = int(counters[0]), int(counters[-1])
        span = last - first + 1
        offsets = counters - first
    verified = len(counters)
    salvaged_partial_groups = int(verified - complete_verified)
    timings['audit_and_keys_seconds'] = round(time.monotonic() - stage, 1)
    print('idle keys: %d keyed groups (%d complete 3-phase + %d packet-salvaged partial groups)'
          % (verified, complete_verified, salvaged_partial_groups), flush=True)

    stage = time.monotonic()
    supplied_frames = int(supplied.sum())
    if a.replay_nearby:
        frames, supplied = replay_nearby(frames, supplied)
    replayed_frames = int(supplied.sum()) - supplied_frames
    runs = group_runs(offsets, a.merge_gap)
    timings['frames_seconds'] = round(time.monotonic() - stage, 1)

    stage = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='evs_', dir=out) as tmp:
        temporary = Path(tmp)
        pcm = render_timeline(frames, supplied, runs, a.rate,
                              lambda f, s: decode_evs(f, s, a.rate, temporary))
    timings['decode_seconds'] = round(time.monotonic() - stage, 1)
    pcm = attenuate(pcm, a.attenuate_db)
    tag = ('_REPLAY' if a.replay_nearby else '') + ('_QUIETER' if a.attenuate_db else '')
    path = out/f'KCBS2_MARCH_UNVERIFIED{tag}_{a.rate//1000}k_{first}-{last}.wav'
    with wave.open(str(path), 'wb') as w:
        w.setparams((1, 2, a.rate, 0, 'NONE', 'not compressed'))
        w.writeframes(pcm.tobytes())

    slice_file = None
    if a.slice:
        s0, s1 = a.slice
        i0, i1 = round(s0 * a.rate), round(s1 * a.rate)
        if not 0 <= i0 < i1 <= len(pcm):
            raise SystemExit(f'--slice {s0}..{s1} falls outside 0..{len(pcm)/a.rate:.3f}s')
        slice_path = out/f'excerpt_{s0:.2f}-{s1:.2f}s.wav'
        with wave.open(str(slice_path), 'wb') as w:
            w.setparams((1, 2, a.rate, 0, 'NONE', 'not compressed'))
            w.writeframes(pcm[i0:i1].tobytes())
        slice_file = slice_path.name

    silenced = [dict(start_seconds=round(hi*1.2, 2), duration_seconds=round((lo-hi)*1.2, 2))
                for (_, hi), (lo, _) in zip(runs, runs[1:])]
    peak = max(int(pcm.max()), -int(pcm.min()))
    clipped = int(np.count_nonzero((pcm == -32768) | (pcm == 32767)))
    norm = pcm.astype(np.float32) / 32768
    report = dict(source=list(a.source), dumps=len(dumps), file=path.name,
                  slice_file=slice_file,
                  seconds=round(len(pcm)/a.rate, 2), pcm_samples=len(pcm),
                  keyed_groups=verified, complete_keyed_groups=complete_verified,
                  salvaged_partial_groups=salvaged_partial_groups,
                  groups_in_span=span, coverage=round(verified/span, 3),
                  supplied_frames=supplied_frames, replayed_frames=replayed_frames,
                  attenuation_db=a.attenuate_db, fully_unmasked_frames=exact_r0_count,
                  total_frames=len(frames), rms=round(float(np.sqrt(np.mean(norm**2))), 4),
                  peak_pcm=peak, full_scale_samples=clipped, clipped_fraction=clipped/len(pcm),
                  merge_gap=a.merge_gap, decoder_calls=len(runs), silenced_gaps=silenced,
                  timings=timings, wall_seconds=round(time.monotonic() - started, 1),
                  note='The idle text and CRC support the 16-byte mask prefix, not codec identity '
                       'or an audio match. Partial groups require >=2 phases agreeing on the 16-byte '
                       'idle-tail prefix and per-packet CRC checks. Role 1 has six role-0 donor bytes; '
                       'replayed frames are copies.')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
