"""Compare offline frame-loss handling on genuine EVS control encodes."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from pystoi import stoi

from kcbs_decode_fast import replay_nearby
import partial_frame_quality as evs

RATE = 32000
METHODS = ('donor_plc', 'future_edge', 'nearest_exact', 'nearest_supplied',
           'role0_replay', 'packed_atempo', 'packed_rubberband')


def make_stream(frames, supplied, method):
    if method == 'donor_plc':
        return frames, supplied
    if method == 'nearest_supplied':
        return replay_nearby(frames, supplied)
    positions = np.arange(len(frames))
    roles = positions % 4
    active = np.repeat(supplied[::4], 4)
    if method.startswith('packed_'):
        return frames[supplied], supplied[supplied]
    out, keep = frames.copy(), supplied.copy()
    if method == 'role0_replay':
        keep &= roles == 0
        choices = ((1, -1), (2, 2), (2, -2), (3, 1), (3, -3))
    elif method == 'future_edge':
        choices = ((3, 1),)
    elif method == 'nearest_exact':
        choices = ((2, 2), (2, -2), (3, 1), (3, -3))
    else:
        raise ValueError(method)
    donors = keep.copy()
    for role, shift in choices:
        rows = positions[active & (roles == role) & ~keep]
        rows = rows[(rows+shift >= 0) & (rows+shift < len(frames))]
        rows = rows[donors[rows+shift]]
        out[rows] = frames[rows+shift]
        keep[rows] = True
    return out, keep


def decode(tmp, tag, frames, supplied, method, log):
    bitstream, keep = make_stream(frames, supplied, method)
    got = evs.decode(tmp, tag, bitstream, keep, log, RATE)
    expected = len(frames)*(RATE//50)
    if method.startswith('packed_'):
        name = method.removeprefix('packed_')
        filt = 'atempo=0.5' if name == 'atempo' else 'rubberband=tempo=0.5:transients=smooth'
        cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-f', 's16le', '-ar', str(RATE),
               '-ac', '1', '-i', 'pipe:0', '-af', filt, '-f', 's16le', 'pipe:1']
        p = subprocess.run(cmd, input=got.tobytes(), capture_output=True, check=True, timeout=300)
        got = np.frombuffer(p.stdout, '<i2')
        if abs(len(got)-expected) > RATE//10:
            raise ValueError(f'{method} offset is larger than 100 ms')
        if len(got) < expected:
            got = np.pad(got, (0, expected-len(got)))
        got = got[:expected]
    if len(got) != expected:
        raise ValueError(f'{method}: expected {expected} samples, got {len(got)}')
    return got


def scores(ref, got, kind):
    rail = float(((got == -32768) | (got == 32767)).mean())
    truth, render = ref.astype(float)/32768, got.astype(float)/32768
    if kind == 'speech':
        return dict(stoi=float(stoi(truth, render, RATE)),
                    estoi=float(stoi(truth, render, RATE, extended=True)), rail=rail)
    def spectrum(x):
        n = RATE//10
        blocks = np.lib.stride_tricks.sliding_window_view(x, n)[::n//2]
        mag = np.abs(np.fft.rfft(blocks*np.hanning(n), axis=1))[:, 6:801]
        return np.log1p(mag*10)
    a, b = spectrum(truth), spectrum(render)
    cosine = (a*b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-8)
    rms_db = float(20*np.log10(max(np.linalg.norm(render), 1e-8)/np.linalg.norm(truth)))
    return dict(spectral_cosine=float(np.mean(cosine)), rms_change_db=rms_db, rail=rail)


def instrumental_control(seed):
    # Labelled stimuli for checking loss handling only; never used in the RF output.
    rng = np.random.default_rng(seed)
    mix = np.zeros(RATE*12)
    for i in range(24):
        f = 110*2**(int(rng.choice([0, 3, 5, 7, 10, 12, 15]))/12)
        start = i*RATE//2
        t = np.arange(min(RATE*2, len(mix)-start))/RATE
        env = np.minimum(t/.02, 1)*np.exp(-t*(1.8 if i % 3 else 4.5))
        env *= np.minimum((len(t)-np.arange(len(t)))/(RATE*.05), 1)
        for k, gain in ((1, 1), (2, .42), (3, .24), (5, .075)):
            mix[start:start+len(t)] += gain*env*np.sin(2*np.pi*k*f*t + rng.uniform(0, 2*np.pi))
    return np.rint(mix*.78*32767/np.abs(mix).max()).astype('<i2')


def evaluate(out, reference):
    rows, log = [], []
    cases = [('speech', 'selection' if i < 4 else 'holdout', start)
             for i, start in enumerate((40, 225, 565, 910, 80, 345, 705, 1050))]
    cases += [('instrumental', 'holdout', seed) for seed in (31, 113, 729)]
    with tempfile.TemporaryDirectory(prefix='controls_', dir=out) as td:
        tmp = Path(td)
        for kind, partition, start in cases:
            raw, packed = tmp/'control.raw', tmp/'control.192'
            if kind == 'instrumental':
                instrumental_control(start).tofile(raw)
            else:
                evs.run(['ffmpeg', '-y', '-nostdin', '-v', 'error', '-ss', str(start),
                         '-i', reference, '-map', '0:a:0', '-af',
                         f'aresample={RATE},atrim=end_sample={12*RATE}', '-t', '14',
                         '-ac', '1', '-ar', RATE, '-f', 's16le', raw], log, 120)
            packed.unlink(missing_ok=True)
            evs.gcc('EVS_cod', ['-q', '-no_delay_cmp', '-max_band', 'SWB', '13200', str(RATE//1000),
                                raw, packed], log)
            words = np.fromfile(packed, '<u2').reshape(-1, 266)
            truth = np.packbits(words[:, 2:] == 0x81, axis=1)
            reference_pcm = evs.decode(tmp, 'truth', truth, np.ones(len(truth), bool), log, RATE)
            test_frames, supplied = evs.apply_strategy(truth, 'role1_donor', None, truth)
            methods = {}
            metric = 'estoi' if kind == 'speech' else 'spectral_cosine'
            for method in METHODS:
                try:
                    got = decode(tmp, method, test_frames, supplied, method, log)
                except (ValueError, subprocess.CalledProcessError) as exc:
                    methods[method] = dict(error=str(exc))
                else:
                    methods[method] = scores(reference_pcm, got, kind)
            rows.append(dict(case=start, kind=kind, partition=partition, scores=methods))
            print(kind, partition, start,
                  {m: round(s[metric], 3) for m, s in methods.items() if metric in s}, flush=True)
    summary = {}
    for method in METHODS:
        values = [row['scores'][method] for row in rows]
        if any('error' in value for value in values):
            continue
        summary[method] = dict(select_estoi=float(np.mean([s['estoi'] for s in values[:4]])),
                               holdout_estoi=float(np.mean([s['estoi'] for s in values[4:8]])),
                               holdout_stoi=float(np.mean([s['stoi'] for s in values[4:8]])),
                               instrumental_similarity=float(np.mean([s['spectral_cosine'] for s in values[8:]])),
                               instrumental_rms_change_db=float(np.mean([s['rms_change_db'] for s in values[8:]])),
                               max_full_scale_fraction=max(s['rail'] for s in values))
    selected = max(summary, key=lambda m: summary[m]['select_estoi'])
    base = summary['donor_plc']
    found = summary[selected]
    better = (found['holdout_estoi'] >= base['holdout_estoi'] + .015
              and found['holdout_stoi'] >= base['holdout_stoi']
              and found['instrumental_similarity'] >= base['instrumental_similarity'] - .015
              and found['max_full_scale_fraction'] <= .001)
    report = dict(summary=summary, rows=rows, selected=selected, passed_holdout=bool(better))
    (out/'controls.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('selected', 'passed_holdout', 'summary')}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--speech', type=Path, required=True,
                    help='local speech audio, at least 18 minutes long; never uploaded')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    reference = a.speech.resolve(strict=True)
    if not reference.is_file():
        ap.error('--speech must be a local audio file')
    a.out.mkdir(parents=True, exist_ok=False)
    evaluate(a.out, reference)
