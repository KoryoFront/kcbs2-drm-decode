"""Does the decoded candidate carry speech structure, independent of alignment?

Three groups pass through the SAME validated GCC decoder and the same 75% loss
pattern:
  candidate       - CRC-supported keys, corrected modes
  wrong_key       - same modes and clear bytes, random group-reused XOR on keyed bytes
  genuine_loss75  - real reference speech encoded at 13.2 kbps, same loss pattern
  genuine_full    - the same encoded speech with no loss, as an upper anchor

Metrics describe speech structure (voicing, pitch continuity, harmonicity, syllabic
modulation). This tests for the presence of speech-like structure only. It does not
identify the codec, prove the content is KCBS-2, or measure intelligibility.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from build_evs_gcc_independent import linux

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
STARTS = [7 + 131*c for c in range(5)]


def run(argv, log, timeout=300):
    p = subprocess.run([str(x) for x in argv], capture_output=True, timeout=timeout)
    log.append(dict(command=[str(x) for x in argv], returncode=p.returncode,
                    stderr=p.stderr.decode(errors='replace')[-500:]))
    if p.returncode:
        raise RuntimeError(log[-1])


def gcc(tool, args, log):
    run(['wsl', '-d', 'Ubuntu', '--', linux(ROOT/'work/evs_gcc_independent'/tool)] +
        [linux(x) if isinstance(x, Path) else x for x in args], log)


def write_g192(path, frames, keep):
    with path.open('wb') as f:
        for b, good in zip(np.unpackbits(frames, axis=1), keep):
            if good:
                f.write(np.array([0x6b21, 264], '<u2').tobytes())
                f.write(np.where(b, 0x81, 0x7f).astype('<u2').tobytes())
            else:
                f.write(np.array([0x6b20, 0], '<u2').tobytes())


def decode(tmp, label, frames, keep, log):
    stream, raw = tmp/(label + '.192'), tmp/(label + '.raw')
    write_g192(stream, frames, keep)
    gcc('EVS_dec', ['-q', '-no_delay_cmp', '16', stream, raw], log)
    return np.fromfile(raw, '<i2').astype(np.float64)/32768


def features(x):
    """Speech-structure descriptors of a decoded waveform."""
    x = x[int(.24*RATE):]
    win, hop = 512, 160
    n = (len(x) - win)//hop
    if n < 50:
        raise ValueError('signal too short for analysis')
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    energy = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    active = energy > max(energy.max()*.02, 1e-6)
    f0 = np.zeros(n)
    voiced = np.zeros(n, bool)
    hnr = np.zeros(n)
    lo, hi = RATE//400, RATE//60
    for i in np.flatnonzero(active):
        seg = frames[i] - frames[i].mean()
        ac = np.correlate(seg, seg, 'full')[win-1:]
        if ac[0] <= 0:
            continue
        ac = ac/ac[0]
        k = lo + int(np.argmax(ac[lo:hi]))
        peak = float(ac[k])
        f0[i] = RATE/k
        voiced[i] = peak > .45
        hnr[i] = 10*np.log10(max(peak, 1e-6)/max(1 - peak, 1e-6))
    envelope = energy - energy.mean()
    spectrum = np.abs(np.fft.rfft(envelope*np.hanning(len(envelope))))**2
    freq = np.fft.rfftfreq(len(envelope), hop/RATE)
    syllabic = (freq >= 2) & (freq <= 8)
    total = (freq >= .5) & (freq <= 20)
    index = np.flatnonzero(voiced)
    groups = np.split(index, np.flatnonzero(np.diff(index) != 1) + 1) if len(index) else []
    jumps = [np.abs(12*np.log2(f0[g][1:]/f0[g][:-1])) for g in groups if len(g) > 1]
    steps = np.concatenate(jumps) if jumps else np.array([])
    return dict(
        voiced_fraction=float(voiced[active].mean()) if active.any() else 0.0,
        active_fraction=float(active.mean()),
        median_f0_hz=float(np.median(f0[voiced])) if voiced.any() else 0.0,
        pitch_jump_semitones=float(np.median(steps)) if len(steps) else float('nan'),
        voiced_run_frames=float(np.mean([len(g) for g in groups])) if groups else 0.0,
        hnr_db=float(np.median(hnr[voiced])) if voiced.any() else 0.0,
        syllabic_ratio=float(spectrum[syllabic].sum()/max(spectrum[total].sum(), 1e-12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--controls', type=int, default=6)
    a = ap.parse_args()
    out = (ROOT/a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    log, rows = [], []
    with np.load(ROOT/'work/nov25_repeated_text/keys.npz') as d:
        take = d['crc_supported'][:, :11].all(axis=1)
        counters = d['counters'][take].astype(np.int64)
        keys = d['ks'][take, :11].copy()
    with np.load(ROOT/'work/nov25_decode_salvaged/groups.npz') as d:
        ix = np.searchsorted(d['counters'], counters)
        assert np.array_equal(d['counters'][ix], counters)
        du = d['du'][ix].copy()
    payload = np.stack([du[:, p, s:s+32] for p in range(3) for s in STARTS], axis=1)
    payload[:, :, ::3] ^= keys[:, None, :]
    mode = np.stack([du[:, p, 2+c] for p in range(3) for c in range(5)], axis=1)
    runs = np.split(np.arange(len(counters)), np.flatnonzero(np.diff(counters) != 1) + 1)
    rng = np.random.default_rng(626)
    reference = ROOT/'work/live/session_20261125T054037Z/ref_20261125T054037Z.ts'
    with tempfile.TemporaryDirectory(prefix='speech_', dir=out) as td:
        tmp = Path(td)
        for ids in runs:
            span = len(ids)*60
            keep = np.arange(span) % 4 == 0
            frames = np.zeros((span, 33), np.uint8)
            frames[keep] = np.column_stack((mode[ids].reshape(-1), payload[ids].reshape(-1, 32)))
            tag = str(counters[ids[0]]) + '-' + str(counters[ids[-1]])
            rows.append(dict(group='candidate', label=tag,
                             **features(decode(tmp, 'cand_' + tag, frames, keep, log))))
            for t in range(2):
                wrong = payload[ids].copy()
                wrong[:, :, ::3] ^= rng.integers(0, 256, (len(ids), 1, 11), dtype=np.uint8)
                bad = frames.copy()
                bad[keep] = np.column_stack((mode[ids].reshape(-1), wrong.reshape(-1, 32)))
                rows.append(dict(group='wrong_key', label=tag + '_' + str(t),
                                 **features(decode(tmp, 'wrong_' + tag + '_' + str(t), bad, keep, log))))
        for i in range(a.controls):
            offset = 120 + i*110
            source = tmp/('ref' + str(i) + '.raw')
            run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(offset), '-i', str(reference),
                 '-map', '0:a:0', '-af', 'aresample=' + str(RATE) + ',atrim=end_sample=' + str(int(3.6*RATE)),
                 '-t', '5', '-ac', '1', '-ar', str(RATE), '-f', 's16le', str(source)], log, 120)
            encoded = tmp/('ref' + str(i) + '.192')
            gcc('EVS_cod', ['-q', '-no_delay_cmp', '-max_band', 'SWB', '13200', '16', source, encoded], log)
            words = np.fromfile(encoded, '<u2').reshape(-1, 266)
            frames = np.packbits(words[:, 2:] == 0x81, axis=1)
            keep = np.arange(len(frames)) % 4 == 0
            rows.append(dict(group='genuine_loss75', label='ref_' + str(offset) + 's',
                             **features(decode(tmp, 'gen_' + str(i), frames, keep, log))))
            rows.append(dict(group='genuine_full', label='ref_' + str(offset) + 's',
                             **features(decode(tmp, 'genfull_' + str(i), frames,
                                               np.ones(len(frames), bool), log))))
    names = ['voiced_fraction', 'median_f0_hz', 'pitch_jump_semitones', 'voiced_run_frames',
             'hnr_db', 'syllabic_ratio', 'active_fraction']
    summary = {}
    for group in ('candidate', 'wrong_key', 'genuine_loss75', 'genuine_full'):
        selected = [r for r in rows if r['group'] == group]
        summary[group] = dict(n=len(selected),
                              **{k: round(float(np.nanmean([r[k] for r in selected])), 4) for k in names})
        print(group, json.dumps(summary[group]), flush=True)
    (out/'report.json').write_text(json.dumps(dict(summary=summary, rows=rows,
        note='Speech-structure comparison only. A candidate resembling genuine_loss75 suggests real '
             'speech parameters are being decoded; it does not prove content, codec or intelligibility.'),
        indent=2), encoding='utf-8')
    (out/'calls.json').write_text(json.dumps(log, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
