"""Counter-consistent November EVS/reference alignment with wrong-key controls.

The score is mean per-band correlation of log spectral shape at observed role-0
frame times, after removing each frame's across-band mean. All runs share ONE
reference offset determined by counter differences. Deliberately wrong keys keep
the same clear bytes, modes, erasures and group reuse. This is an exploratory
alignment test, not an intelligibility measurement or codec-identification proof.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.fft import rfft

STARTS = [7+131*c for c in range(5)]
KEYED = np.arange(0,32,3)


def invoke(command, log):
    p = subprocess.run([str(x) for x in command], capture_output=True, timeout=120)
    log.append(dict(command=[str(x) for x in command], returncode=p.returncode,
                    stderr=p.stderr.decode('utf-8', errors='replace')[-2000:]))
    if p.returncode:
        raise RuntimeError(log[-1])
    return p


def features(x, centers):
    edges = np.geomspace(200, 7600, 17)
    frequencies = np.fft.rfftfreq(1024, 1/16000)
    bands = [(frequencies >= a) & (frequencies < b) for a,b in zip(edges[:-1], edges[1:])]
    window = np.hanning(640).astype(np.float32)
    result = np.empty((len(centers), 16), np.float32)
    for lo in range(0, len(centers), 2048):
        c = centers[lo:lo+2048]
        indices = c[:, None]+np.arange(-320, 320)
        frames = x[np.clip(indices, 0, len(x)-1)]*window
        power = np.abs(rfft(frames, n=1024, axis=1))**2
        result[lo:lo+len(c)] = np.stack([np.log10(power[:, b].mean(axis=1)+1e-12) for b in bands], axis=1)
    result -= result.mean(axis=1, keepdims=True)
    return result


def score_offsets(query, reference, indices, max_offset):
    q = query-query.mean(axis=0)
    norm = np.sqrt((q*q).sum(axis=0))
    output = np.empty(max_offset+1, dtype=np.float32)
    n = len(query)
    for lo in range(0, len(output), 64):
        offsets = np.arange(lo, min(lo+64, len(output)))
        r = reference[indices[None, :]+offsets[:, None]]
        den = np.sqrt(np.maximum((r*r).sum(axis=1)-r.sum(axis=1)**2/n, 1e-12))*norm
        corr = (r*q).sum(axis=1)/np.maximum(den, 1e-12)
        output[lo:lo+len(offsets)] = corr.mean(axis=1)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--controls', type=int, default=20)
    ap.add_argument('--compiler', choices=('tcc','gcc'), default='tcc')
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (root/a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    log = []
    with np.load(root/'work/nov25_decode_salvaged/groups.npz') as d:
        du = d['du'].copy(); cs = d['counters'].astype(np.int64)
    with np.load(root/'work/nov25_repeated_text/keys.npz') as d:
        keep = d['crc_supported'][:, :11].all(axis=1) & (d['counters'] >= 43709059)
        counters, keys = d['counters'][keep].astype(np.int64), d['ks'][keep, :11].copy()
    ix = np.searchsorted(cs, counters)
    if not np.array_equal(cs[ix], counters):
        raise ValueError('key/source counters differ')
    du = du[ix]
    plain = np.stack([du[:, p, s:s+32] for p in range(3) for s in STARTS], axis=1)
    plain[:, :, KEYED] ^= keys[:, None, :]
    modes = np.stack([du[:, p, 2+c] for p in range(3) for c in range(5)], axis=1)
    runs = np.split(np.arange(len(counters)), np.flatnonzero(np.diff(counters) != 1)+1)
    decoder = root/'work/ref/evs/build/EVS_dec.exe'
    def decode_command(stream, output):
        if a.compiler == 'gcc':
            from build_evs_gcc_independent import linux
            return ['wsl','-d','Ubuntu','--',linux(root/'work/evs_gcc_independent/EVS_dec'),
                    '-q','-no_delay_cmp','16',linux(stream),linux(output)]
        return [decoder,'-q','-no_delay_cmp','16',stream,output]
    reference_path = root/'work/live/session_20261125T054037Z/ref_20261125T054037Z.ts'
    rng = np.random.default_rng(261127)
    with tempfile.TemporaryDirectory(prefix='alignment_', dir=out) as temp:
        temp = Path(temp)
        ref_file = temp/'reference.raw'
        invoke(['ffmpeg','-nostdin','-v','error','-i',reference_path,'-map','0:a:0',
                '-ac','1','-ar','16000','-f','f32le',ref_file], log)
        ref = np.fromfile(ref_file, '<f4')
        ref_features = features(ref, np.arange(len(ref)//160)*160)
        results, all_scores = [], []
        query_indices = []
        selected_slots = []
        for run in runs:
            # Ignore 240ms startup per independently initialized decoder run.
            slots = np.arange(12, len(run)*60, 4)
            centers = slots*320+160
            selected_slots.append(centers)
            query_indices.extend((int(counters[run[0]]-counters[0])*120+centers//160).tolist())
        indices = np.array(query_indices, dtype=np.int64)
        max_offset = len(ref_features)-1-int(indices.max())
        if max_offset <= 0:
            raise ValueError('reference too short for common-counter alignment')
        for trial in range(a.controls+1):
            p = plain.copy()
            if trial:
                p[:, :, KEYED] ^= rng.integers(0,256,(len(p),1,11),dtype=np.uint8)
            pieces = []
            for run, centers in zip(runs, selected_slots):
                frames = np.column_stack((modes[run].reshape(-1), p[run].reshape(-1,32)))
                bits = np.unpackbits(frames, axis=1)
                stream = temp/'frames.192'; pcm_path = temp/'out.raw'
                with stream.open('wb') as f:
                    for b in bits:
                        f.write(np.array([0x6b21,264],'<u2').tobytes())
                        f.write(np.where(b,0x81,0x7f).astype('<u2').tobytes())
                        f.write(np.array([0x6b20,0]*3,'<u2').tobytes())
                invoke(decode_command(stream, pcm_path), log)
                pcm = np.fromfile(pcm_path,'<i2').astype(np.float32)/32768
                if len(pcm) != len(run)*60*320:
                    raise ValueError('unexpected decoder sample count')
                pieces.append(features(pcm, centers))
                if not trial:
                    sf.write(out/f'UNVERIFIED_EVS_{counters[run[0]]}-{counters[run[-1]]}.wav',pcm,16000,subtype='PCM_16')
            query = np.concatenate(pieces)
            scores = score_offsets(query, ref_features, indices, max_offset)
            best = int(scores.argmax())
            row = dict(trial=trial, kind='candidate' if trial == 0 else 'wrong_key_control',
                       peak_shape_correlation=float(scores[best]), offset_seconds=best*.01)
            results.append(row); all_scores.append(scores)
            print(json.dumps(row), flush=True)
        np.savez_compressed(out/'scores.npz', scores=np.array(all_scores), offset_step_seconds=.01)
        best = results[0]['peak_shape_correlation']
        report = dict(compiler=a.compiler, reference=str(reference_path), reference_seconds=len(ref)/16000,
                      base_counter=int(counters[0]), groups=len(counters),
                      role0_query_frames=len(indices), offset_range_seconds=[0,max_offset*.01],
                      candidate=results[0], controls=results[1:],
                      controls_at_least_candidate=sum(r['peak_shape_correlation'] >= best for r in results[1:]),
                      note='Exploratory full-offset search; all runs share a common counter-based offset. '
                           'Not an intelligibility measurement. No positive alignment control yet.')
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (out/'calls.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
