"""Can role 1's six unknown key bytes be recovered exactly?

Within a 1.2 s group one key is reused by all 15 role-1 frames, so a candidate byte
is scored on 15 frames at once. Two signals are used:
  flags      - decoder BFI/BER on role-1 frames (syntax validity)
  roughness  - log-mel frame-to-frame change around role-1 frames (continuity with
               the exact role-0 neighbours)

--pilot measures IDENTIFIABILITY on genuine encoded speech, where the true bytes are
known: every other unknown byte is set to its true value and all 256 values of one
position are ranked. A true value ranked 1 means that position is identifiable; a
poor rank means this oracle cannot recover it and no search over it is justified.

Candidates are batched into a single bitstream per decoder call. Decoder state
carries across concatenated segments, which biases all candidates equally and is
acceptable for ranking, but it is not a clean per-candidate decode.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from build_evs_gcc_independent import linux

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'work/ref/evs/src/c-code'
RATE = 16000
FRAME = RATE//50
TRACED = ROOT/'work/evs_gcc_independent/EVS_dec_traced'
ROLE1_UNKNOWN = np.array([16, 19, 22, 25, 28, 31])   # frame byte indices (mode byte is 0)


def build_traced(log):
    if TRACED.exists():
        return TRACED
    src = (SRC/'lib_dec/decoder.c').read_bytes()
    anchor = b"            /* convert 'float' output data to 'short' */"
    assert src.count(anchor) == 1
    esc = bytes([92, 110])
    eol = bytes([10])
    line = (b'fprintf(stderr,"TRACE,%ld,%d,%d' + esc + b'",frame,'
            b'(int)st->bfi,(int)st->BER_detect);' + eol)
    tmp = TRACED.parent/'traced_build'
    tmp.mkdir(exist_ok=True)
    copy = tmp/'decoder_traced.c'
    copy.write_bytes(src.replace(anchor, b'            ' + line + anchor))
    files = sorted((SRC/'lib_com').glob('*.c'))
    files += [p for p in sorted((SRC/'lib_dec').glob('*.c')) if p.name != 'decoder.c']
    args = ['-O1', '-std=gnu99', '-fno-strict-aliasing']
    args += ['-I' + linux(SRC/s) for s in ('lib_com', 'lib_dec', 'lib_enc')]
    args += [linux(copy)] + [linux(p) for p in files] + ['-lm', '-o', linux(TRACED)]
    rsp = tmp/'traced.rsp'
    rsp.write_text('\n'.join('"' + x + '"' for x in args), encoding='utf-8')
    p = subprocess.run(['wsl', '-d', 'Ubuntu', '--', 'gcc', '@' + linux(rsp)],
                       capture_output=True, timeout=1800)
    log.append(dict(step='build traced decoder', returncode=p.returncode))
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-2500:])
    return TRACED


def decode_batch(tmp, label, frames, keep, log):
    """Decode one concatenated stream; return per-frame (bfi, ber) and PCM."""
    stream, raw = tmp/(label + '.192'), tmp/(label + '.raw')
    with stream.open('wb') as f:
        for b, good in zip(np.unpackbits(frames, axis=1), keep):
            if good:
                f.write(np.array([0x6b21, 264], '<u2').tobytes())
                f.write(np.where(b, 0x81, 0x7f).astype('<u2').tobytes())
            else:
                f.write(np.array([0x6b20, 0], '<u2').tobytes())
    argv = ['wsl', '-d', 'Ubuntu', '--', linux(TRACED), '-q', '-no_delay_cmp', '16',
            linux(stream), linux(raw)]
    p = subprocess.run(argv, capture_output=True, timeout=1800)
    log.append(dict(step=label, returncode=p.returncode, frames=int(len(frames))))
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-1500:])
    rows = np.array([list(map(int, m.split(','))) for m in
                     re.findall(r'TRACE,([^\n]*)', p.stderr.decode(errors='replace'))], dtype=int)
    assert len(rows) == len(frames), (len(rows), len(frames))
    pcm = np.fromfile(raw, '<i2').astype(np.float64)/32768
    assert len(pcm) == len(frames)*FRAME
    return rows[:, 1:], pcm


def mel_basis(bands=20, nfft=512):
    def mel(f):
        return 2595*np.log10(1 + f/700)
    edges = 700*(10**(np.linspace(mel(100), mel(7800), bands + 2)/2595) - 1)
    freq = np.fft.rfftfreq(nfft, 1/RATE)
    basis = np.zeros((bands, len(freq)))
    for b in range(bands):
        lo, mid, hi = edges[b], edges[b+1], edges[b+2]
        up = (freq >= lo) & (freq <= mid)
        down = (freq > mid) & (freq <= hi)
        basis[b, up] = (freq[up] - lo)/max(mid - lo, 1e-9)
        basis[b, down] = (hi - freq[down])/max(hi - mid, 1e-9)
    return basis


BASIS = mel_basis()


def roughness(pcm, span, role1_rows):
    """Mean log-mel change across boundaries that involve a role-1 frame."""
    frames = pcm.reshape(-1, FRAME)
    window = np.hanning(FRAME)
    power = np.abs(np.fft.rfft(frames*window, n=512, axis=1))**2
    logmel = np.log(power @ BASIS.T + 1e-10)
    out = np.empty(len(frames)//span)
    for s in range(len(out)):
        block = logmel[s*span:(s+1)*span]
        cost = 0.0
        for r in role1_rows:
            if r > 0:
                cost += np.abs(block[r] - block[r-1]).mean()
            if r + 1 < span:
                cost += np.abs(block[r+1] - block[r]).mean()
        out[s] = cost/max(len(role1_rows), 1)
    return out


def genuine_group(tmp, offset, log):
    """One 60-frame block of genuine encoded speech, with its true bytes."""
    source = tmp/('g' + str(offset) + '.raw')
    encoded = tmp/('g' + str(offset) + '.192')
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-ss', str(offset),
                    '-i', str(ROOT/'work/live/session_20261125T054037Z/ref_20261125T054037Z.ts'),
                    '-map', '0:a:0', '-af', 'aresample=' + str(RATE) + ',atrim=end_sample=' + str(int(1.6*RATE)),
                    '-t', '3', '-ac', '1', '-ar', str(RATE), '-f', 's16le', str(source)],
                   check=True, capture_output=True, timeout=120)
    argv = ['wsl', '-d', 'Ubuntu', '--', linux(ROOT/'work/evs_gcc_independent/EVS_cod'),
            '-q', '-no_delay_cmp', '-max_band', 'SWB', '13200', '16', linux(source), linux(encoded)]
    p = subprocess.run(argv, capture_output=True, timeout=300)
    log.append(dict(step='encode genuine at ' + str(offset), returncode=p.returncode))
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-800:])
    words = np.fromfile(encoded, '<u2').reshape(-1, 266)
    frames = np.packbits(words[:, 2:] == 0x81, axis=1)
    return frames[:60]


def pilot(args):
    log, results = [], []
    build_traced(log)
    with tempfile.TemporaryDirectory(prefix='pilot_', dir=(ROOT/args.out)) as td:
        tmp = Path(td)
        for i in range(args.groups):
            offset = 150 + i*90
            truth = genuine_group(tmp, offset, log)
            span = len(truth)
            slot = np.arange(span) % 4
            role1 = np.flatnonzero(slot == 1)
            keep = np.isin(slot, (0, 1))          # roles 2/3 erased, as in the real case
            for position in ROLE1_UNKNOWN:
                batch = np.repeat(truth[None, :, :], 256, axis=0)
                batch[:, role1, position] = np.arange(256, dtype=np.uint8)[:, None]
                stacked = batch.reshape(-1, 33)
                flags, pcm = decode_batch(tmp, 'pilot_' + str(i) + '_' + str(position),
                                          stacked, np.tile(keep, 256), log)
                flagged = np.array([int(((flags[s*span:(s+1)*span][role1]) != 0).any(axis=1).sum())
                                    for s in range(256)])
                rough = roughness(pcm, span, role1)
                order = np.lexsort((rough, flagged))
                true_value = int(truth[role1[0], position])
                assert len(set(truth[role1, position].tolist())) >= 1
                rank = int(np.flatnonzero(order == true_value)[0]) + 1
                row = dict(excerpt='ref_' + str(offset) + 's', byte_position=int(position),
                           true_value=true_value, true_rank=rank,
                           true_flagged=int(flagged[true_value]),
                           candidates_with_zero_flags=int((flagged == 0).sum()),
                           best_rival_value=int(order[0]),
                           true_roughness=round(float(rough[true_value]), 5),
                           best_roughness=round(float(rough[order[0]]), 5))
                results.append(row)
                print(json.dumps(row), flush=True)
    identifiable = [r for r in results if r['true_rank'] == 1]
    summary = dict(tested=len(results), exact_rank1=len(identifiable),
                   median_rank=float(np.median([r['true_rank'] for r in results])),
                   mean_zero_flag_candidates=round(float(np.mean(
                       [r['candidates_with_zero_flags'] for r in results])), 1),
                   per_position={int(p): [r['true_rank'] for r in results if r['byte_position'] == p]
                                 for p in ROLE1_UNKNOWN},
                   note='Rank 1 means this oracle picks the true byte. Large median rank means '
                        'the byte is not identifiable this way and a search would invent values.')
    out = ROOT/args.out
    (out/'pilot.json').write_text(json.dumps(dict(summary=summary, results=results), indent=2),
                                  encoding='utf-8')
    (out/'calls.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--groups', type=int, default=2)
    a = ap.parse_args()
    (ROOT/a.out).mkdir(parents=True, exist_ok=False)
    if not a.pilot:
        raise SystemExit('run --pilot first: a search is only justified if bytes are identifiable')
    pilot(a)


if __name__ == '__main__':
    main()
