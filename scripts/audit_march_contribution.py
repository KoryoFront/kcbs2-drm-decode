"""Check the contributed March spectral observation against the raw RF dumps, not a labelled feed."""
import argparse
import json
from pathlib import Path
import tempfile
import wave

import numpy as np

from kcbs_decode_fast import assemble_frames, decode_evs, idle_keys
from tail_fill_audit import audit

RATE = 16000
KEYED = 1 + np.arange(0, 32, 3)


def band_summary(pcm, t0, t1):
    audio = pcm[int(t0*RATE):int(t1*RATE)].astype(float)
    if len(audio) < RATE:
        raise ValueError('window does not contain decoded samples')
    w = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio*w))**2
    bins = np.fft.rfftfreq(len(audio), 1/RATE)
    speech_band = (bins > 100) & (bins < 7000)
    narrow = (bins > 980) & (bins < 1020)
    energy = spectrum[speech_band].sum()
    dominant = bins[speech_band][np.argmax(spectrum[speech_band])] if energy else None
    return dict(rms_pcm=round(float(np.sqrt(np.mean(audio**2))),3),
                peak_hz=round(float(dominant),4) if dominant is not None else None,
                narrow_share=round(float(spectrum[narrow].sum()/energy),6) if energy else None,
                narrow_rms_pcm=round(float(np.sqrt(2*spectrum[narrow].sum()/(len(w)*(w*w).sum()))),3))


def audit_decodes(archives, out):
    arrays = {}
    results = {}
    for name, path in archives.items():
        counters, keys, du, _, received = idle_keys(path)
        arrays[name] = (counters, du)
        first = int(counters[0]); offsets = counters-first
        frames, keep = assemble_frames(du, keys, offsets, int(offsets[-1])+1)
        keep &= np.arange(len(frames)) % 4 == 0
        rows = np.flatnonzero(keep)
        phase_matched = frames.copy()
        for mode in np.unique(frames[rows,0]):
            ids = rows[frames[rows,0] == mode]
            positions = np.ix_(ids, KEYED)
            phase_matched[positions] = np.random.default_rng(38 + int(mode)).permuted(frames[positions], axis=0)
        wrong = frames.copy()
        wrong[np.ix_(rows,KEYED)] ^= np.repeat(keys[:, :11] ^ np.roll(keys[:, :11],-50,axis=0),15,axis=0)
        variants = {'candidate':frames, 'wrong_prefix_control':wrong, 'mode_matched_keyed_shuffle':phase_matched}
        result = dict(first_counter=first, last_counter=int(counters[-1]), supplied_role0_frames=len(rows),
                      complete_received_groups=received, seconds=len(frames)/50, spectra={})
        with tempfile.TemporaryDirectory(dir=out) as temp:
            for label, part in variants.items():
                pcm = decode_evs(part, keep, RATE, Path(temp))
                result['spectra'][label] = {f'{a}-{b}s':band_summary(pcm,a,b) for a,b in ((20,80),(100,220))}
                if label == 'candidate':
                    with wave.open(str(out/f'{name}_ROLE0_UNVERIFIED_16k.wav'),'wb') as w:
                        w.setparams((1,2,RATE,0,'NONE','not compressed'))
                        w.writeframes(pcm.tobytes())
        results[name] = result
    (a, x), (b, y) = arrays.values()
    common, i, j = np.intersect1d(a,b,return_indices=True)
    match = int(np.count_nonzero((x[i]==y[j]).all(axis=(1,2))))
    return dict(captures=results, shared_idle_supported_counters=len(common), byte_identical=match,
                first_only_count=int(len(np.setdiff1d(a,b))), second_only_count=int(len(np.setdiff1d(b,a))),
                note='Measures the RF-derived EVS candidate. No satellite audio, feed identity or correspondence check.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contribution', required=True,type=Path)
    ap.add_argument('--bundled-groups', required=True,type=Path)
    ap.add_argument('--out', required=True,type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    audit([str(a.contribution)],a.out/'contrib')
    report = audit_decodes({'contributed':a.out/'contrib'/'groups.npz','bundled':a.bundled_groups}, a.out)
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
