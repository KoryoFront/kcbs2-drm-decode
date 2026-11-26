"""Round-trip known speech through each EVS build, correcting codec delay.

Delay is found by best cross-correlation lag on the known source, which is valid
here because the input is known. Scores are reported at that lag. A correct
reference codec at 13.2 kbps must track its own input closely.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from pystoi import stoi
from build_evs_gcc_independent import linux
RATE = 16000


def codec(kind, build, args, log):
    if build == 'gcc':
        argv = ['wsl','-d','Ubuntu','--', linux(ROOT/'work/evs_gcc_independent'/kind)]
        argv += [linux(x) if isinstance(x, Path) else str(x) for x in args]
    else:
        argv = [str(ROOT/'work/ref/evs/build'/(kind+'.exe'))] + [str(x) for x in args]
    p = subprocess.run(argv, capture_output=True, timeout=300)
    log.append(dict(command=argv, returncode=p.returncode, stderr=p.stderr.decode(errors='replace')[-600:]))
    if p.returncode:
        raise RuntimeError(log[-1])


def best_lag(a, b, limit=1600):
    fa = np.fft.rfft(a, len(a)+limit*2)
    fb = np.fft.rfft(b, len(a)+limit*2)
    c = np.fft.irfft(fa*np.conj(fb))
    cand = np.concatenate((c[:limit+1], c[-limit:]))
    offs = np.concatenate((np.arange(limit+1), np.arange(-limit, 0)))
    return int(offs[int(np.argmax(cand))])


def score(src, test):
    lag = best_lag(src.astype(float), test.astype(float))
    t = np.roll(test.astype(float), lag)
    n = min(len(src), len(t)) - 2*RATE//10
    a = src[RATE//10:RATE//10+n].astype(float)/32768
    b = t[RATE//10:RATE//10+n]/32768
    snr = 10*np.log10(np.sum(a*a)/max(np.sum((a-b)**2), 1e-12))
    return dict(lag_samples=lag, stoi=round(float(stoi(a, b, RATE)), 4),
                estoi=round(float(stoi(a, b, RATE, extended=True)), 4),
                snr_db=round(float(snr), 2), correlation=round(float(np.corrcoef(a, b)[0, 1]), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = (ROOT/a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    log, rows = [], []
    ref = ROOT/'work/live/session_20261125T054037Z/ref_20261125T054037Z.ts'
    for name, offset in (('speech_a', 300), ('speech_b', 780)):
        source = out/(name+'.raw')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(offset),'-i',str(ref),'-map','0:a:0',
                        '-af',f'aresample={RATE},atrim=end_sample={10*RATE}','-t','12','-ac','1',
                        '-ar',str(RATE),'-f','s16le',str(source)], check=True, capture_output=True, timeout=120)
        original = np.fromfile(source, '<i2')
        for build in ('gcc','tcc'):
            stream = out/f'{name}_{build}.192'
            codec('EVS_cod', build, ['-q','-max_band','SWB',13200,RATE//1000,source,stream], log)
            decoded = out/f'{name}_{build}_{build}.raw'
            codec('EVS_dec', build, ['-q',RATE//1000,stream,decoded], log)
            row = dict(excerpt=name, build=build, **score(original, np.fromfile(decoded, '<i2')))
            rows.append(row); print(json.dumps(row), flush=True)
    (out/'report.json').write_text(json.dumps(dict(results=rows,
        note='Delay-compensated self round-trip. A sound build should show high STOI and positive SNR.'),
        indent=2), encoding='utf-8')
    (out/'calls.json').write_text(json.dumps(log, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
