"""Re-run the EVS error-flag discrimination test on a traced GCC build.

Builds the unmodified reference decoder with one added stderr trace line, then
compares November CRC-key-supported role-0 frames against same-mode wrong-key
controls and a genuine encoded-reference control. Decoder flags are a syntax
plausibility signal, not proof of codec identity or intelligibility.
"""
import argparse, json, re, shutil, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from build_evs_gcc_independent import linux

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'work/ref/evs/src/c-code'
STARTS = [7+131*c for c in range(5)]


def build_traced(tmp):
    src = (SRC/'lib_dec/decoder.c').read_bytes()
    anchor = b"            /* convert 'float' output data to 'short' */"
    assert src.count(anchor) == 1
    esc = bytes([92, 110])   # literal backslash-n inside the C string
    eol = bytes([10])        # end of the inserted C statement
    line = (b'fprintf(stderr,"TRACE,%ld,%d,%d,%d,%d' + esc + b'",frame,'
            b'(int)st->bfi,(int)st->BER_detect,(int)st->core,(int)st->codec_mode);' + eol)
    traced = src.replace(anchor, b'            ' + line + anchor)
    copy = tmp/'decoder_traced.c'
    copy.write_bytes(traced)
    files = [p for p in sorted((SRC/'lib_com').glob('*.c'))]
    files += [p for p in sorted((SRC/'lib_dec').glob('*.c')) if p.name != 'decoder.c']
    args = ['-O0','-std=gnu99','-fno-strict-aliasing']
    args += ['-I'+linux(SRC/s) for s in ('lib_com','lib_dec','lib_enc')]
    args += [linux(copy)]+[linux(p) for p in files]+['-lm','-o',linux(tmp/'EVS_dec_traced')]
    rsp = tmp/'traced.rsp'
    rsp.write_text('\n'.join('"'+a+'"' for a in args), encoding='utf-8')
    p = subprocess.run(['wsl','-d','Ubuntu','--','gcc','@'+linux(rsp)], capture_output=True, timeout=900)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
    return tmp/'EVS_dec_traced'


def write_g192(path, frames, keep):
    with path.open('wb') as f:
        for b, good in zip(np.unpackbits(frames, axis=1), keep):
            if good:
                f.write(np.array([0x6b21,264],'<u2').tobytes())
                f.write(np.where(b,0x81,0x7f).astype('<u2').tobytes())
            else:
                f.write(np.array([0x6b20,0],'<u2').tobytes())


def decode_flags(binary, tmp, label, frames, keep):
    stream, raw = tmp/(label+'.192'), tmp/(label+'.raw')
    write_g192(stream, frames, keep)
    p = subprocess.run(['wsl','-d','Ubuntu','--',linux(binary),'-q','-no_delay_cmp','16',
                        linux(stream), linux(raw)], capture_output=True, timeout=300)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-1500:])
    rows = [list(map(int, m.split(','))) for m in
            re.findall(r'TRACE,([^\n]*)', p.stderr.decode(errors='replace'))]
    data = np.array(rows, dtype=int)
    assert len(data) == len(keep), (len(data), len(keep))
    bfi, ber = data[:,1], data[:,2]
    flagged = ((bfi != 0) | (ber != 0)) & keep
    return dict(label=label, supplied=int(keep.sum()), flagged=int(flagged.sum()),
                bfi=int((bfi[keep] != 0).sum()), ber=int((ber[keep] != 0).sum()),
                core_hist={int(k): int(v) for k, v in Counter(data[keep,3]).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', required=True)
    ap.add_argument('--controls', type=int, default=8)
    a = ap.parse_args(); out = (ROOT/a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    with np.load(ROOT/'work/nov25_repeated_text/keys.npz') as d:
        take = d['crc_supported'][:,:11].all(axis=1)
        counters, keys = d['counters'][take].astype(np.int64), d['ks'][take,:11].copy()
    with np.load(ROOT/'work/nov25_decode_salvaged/groups.npz') as d:
        ix = np.searchsorted(d['counters'], counters); assert np.array_equal(d['counters'][ix], counters)
        du = d['du'][ix].copy()
    payload = np.stack([du[:,p,s:s+32] for p in range(3) for s in STARTS], axis=1)
    payload[:,:,::3] ^= keys[:,None,:]
    mode = np.stack([du[:,p,2+c] for p in range(3) for c in range(5)], axis=1)
    runs = np.split(np.arange(len(counters)), np.flatnonzero(np.diff(counters) != 1)+1)
    rng = np.random.default_rng(20261126)
    results = {'candidate': [], 'wrong_key': []}
    with tempfile.TemporaryDirectory(prefix='flags_', dir=out) as td:
        tmp = Path(td)
        binary = build_traced(tmp)
        print(json.dumps(dict(build='traced gcc decoder from unmodified source', ok=True)), flush=True)
        for ids in runs:
            span = len(ids)*60
            keep = np.arange(span) % 4 == 0
            frames = np.zeros((span,33), np.uint8)
            frames[keep] = np.column_stack((mode[ids].reshape(-1), payload[ids].reshape(-1,32)))
            tag = f'{counters[ids[0]]}-{counters[ids[-1]]}'
            results['candidate'].append(decode_flags(binary, tmp, 'candidate_'+tag, frames, keep))
            for trial in range(a.controls):
                wrong = payload[ids].copy()
                wrong[:,:,::3] ^= rng.integers(0,256,(len(ids),1,11),dtype=np.uint8)
                bad = frames.copy()
                bad[keep] = np.column_stack((mode[ids].reshape(-1), wrong.reshape(-1,32)))
                results['wrong_key'].append(decode_flags(binary, tmp, f'wrongkey_{tag}_{trial}', bad, keep))
        supplied = sum(r['supplied'] for r in results['candidate'])
        summary = dict(candidate_supplied=supplied,
                       candidate_flagged=sum(r['flagged'] for r in results['candidate']),
                       wrong_key_trials=a.controls,
                       wrong_key_flagged_mean=round(sum(r['flagged'] for r in results['wrong_key'])/a.controls, 2),
                       wrong_key_flagged_per_trial=[sum(r['flagged'] for r in results['wrong_key'][i::a.controls])
                                                    for i in range(a.controls)],
                       detail=results,
                       note='Flags are a syntax signal only. Equal flag counts would mean the test '
                            'does not discriminate the key hypothesis.')
        (out/'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in summary.items() if k != 'detail'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
