"""Look for a repeated, shared amplitude pulse in two overlapping RF IQ recordings."""
import argparse
import json
from pathlib import Path
import wave

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import detrend, periodogram, welch

STEP = .02
BANDS = ((300, 4100), (5000, 5700))


def band_power(path, start, seconds):
    rows = []
    with wave.open(str(path), 'rb') as wav:
        rate = wav.getframerate()
        if wav.getnchannels() != 2 or wav.getsampwidth() != 2:
            raise ValueError(f'{path.name}: expected 16-bit IQ stereo')
        block = int(rate*STEP)
        if rate < 12000 or rate % 50:
            raise ValueError(f'{path.name}: expected at least 12 kHz with whole 20 ms sample blocks')
        start_sample = round(start*rate)
        available = min(round(seconds/STEP), (wav.getnframes()-start_sample)//block)
        if available < 60/STEP:
            raise ValueError(f'{path.name}: need at least 60 seconds within the recording')
        wav.setpos(start_sample)
        freq = abs(np.fft.fftfreq(block, 1/rate))
        masks = [(freq >= lo) & (freq < hi) for lo, hi in BANDS]
        window = np.hanning(block)
        while available:
            count = min(available, 5000)
            iq = np.frombuffer(wav.readframes(count*block), dtype='<i2').reshape(count, block, 2)
            spectrum = np.fft.fft((iq[:, :, 0] + 1j*iq[:, :, 1])*window, axis=1)
            power = np.abs(spectrum)**2
            rows.append(np.stack([power[:, mask].mean(axis=1) for mask in masks], axis=1))
            available -= count
    return 10*np.log10(np.maximum(np.concatenate(rows), 1))


def peaks(level):
    f, p = welch(level, fs=1/STEP, nperseg=5000, detrend='linear')
    indexes = np.flatnonzero((f >= .20) & (f <= 12) &
                             (p > np.roll(p, 1)) & (p > np.roll(p, -1)))
    floor = np.median(p[(f >= .20) & (f <= 12)])
    return [(round(float(f[i]), 3), round(float(p[i]/floor), 2))
            for i in sorted(indexes, key=lambda i: -p[i])[:8]]


def compare(x, y):
    def at_shift(lag):
        xc, yc = (x[:-lag],y[lag:]) if lag > 0 else ((x[-lag:],y[:lag]) if lag < 0 else (x,y))
        return float(np.corrcoef(xc,yc)[0,1])
    lag = max(range(-100, 101), key=lambda n: abs(at_shift(n)))
    return dict(seconds=round(lag*STEP, 2), correlation=round(at_shift(lag), 4))


def localized_pulses(x, y, seconds=40):
    """Require a common excess in two adjacent windows, not an isolated peak."""
    width = round(seconds/STEP)

    def bins(signal):
        freq, power = periodogram(signal, fs=1/STEP, window='hann', detrend='linear')
        floor = np.maximum(median_filter(power, size=31), 1e-20)
        relative = power/floor
        return np.flatnonzero((freq >= .2) & (freq <= 10) & (relative >= 10)
                              & (relative > np.roll(relative, 1))
                              & (relative > np.roll(relative, -1)))

    hits = []
    n = min(len(x), len(y))
    for start in range(0, n-width+1, width):
        first = bins(x[start:start+width])
        second = bins(y[start:start+width])
        hits.append([int(idx) for idx in first if np.any(np.abs(second-idx) <= 1)])
    hz = 1/(width*STEP)
    persistent = [dict(start_seconds=round(i*width*STEP, 2), hz=round(bin_id*hz, 3))
                  for i, (left, right) in enumerate(zip(hits,hits[1:]))
                  for bin_id in left if any(abs(bin_id-next_id) <= 1 for next_id in right)]
    return dict(window_seconds=seconds, windows=len(hits),
                shared_peaks_hz=[[round(idx*hz,3) for idx in row] for row in hits],
                adjacent_window_matches=persistent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('first', type=Path)
    ap.add_argument('second', type=Path)
    ap.add_argument('--first-skip', type=float, required=True)
    ap.add_argument('--second-skip', type=float, required=True)
    ap.add_argument('--seconds', type=float, default=500)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if not all(np.isfinite(x) and x >= 0 for x in (a.first_skip, a.second_skip, a.seconds)):
        ap.error('times must be finite and nonnegative')
    x = band_power(a.first, a.first_skip, a.seconds)
    y = band_power(a.second, a.second_skip, a.seconds)
    n = min(len(x),len(y))
    report = dict(seconds=round(n*STEP, 2), results={},
                  localized_inband=localized_pulses(x[:n,0], y[:n,0]),
                  note='RF power test, not the programme audio envelope.')
    for lab, s in [('early',slice(0,n//2)), ('late',slice(n//2,n))]:
        xs, ys = detrend(x[s],axis=0), detrend(y[s],axis=0)
        report['results'][lab] = dict(first_inband=peaks(xs[:,0]), second_inband=peaks(ys[:,0]),
                                      first_outer=peaks(xs[:,1]),second_outer=peaks(ys[:,1]),
                                      inband_correlation=compare(xs[:,0],ys[:,0]),
                                      power_ratio_correlation=compare(xs[:,0]-xs[:,1],ys[:,0]-ys[:,1]))
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print('Shared pulse candidates in adjacent windows:', report['localized_inband']['adjacent_window_matches'])
    for label,row in report['results'].items():
        print(label, 'in-band peaks (Hz, power/floor):',row['first_inband'][:4],row['second_inband'][:4],
              'cross-correlation:',row['inband_correlation'], row['power_ratio_correlation'])


if __name__ == '__main__':
    main()
