"""Test alternative packet-loss reconstruction, not recovery of missing keys.

Copies ONLY whole CRC-supported role-0 packets for unknown temporal slots. Uses
actual encoded reference controls with the same 75% loss pattern and STOI/eSTOI.
A method must beat native PLC on BOTH selection controls and a third held-out
control before a target WAV is retained.
Target audio remains unverified: original role-1/2/3 frames are not recovered.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


RATE = 32000
FRAME = RATE//50
DEC = ROOT/'work/ref/evs/build/EVS_dec.exe'
ENC = ROOT/'work/ref/evs/build/EVS_cod.exe'


def call(command, log):
    p = subprocess.run([str(x) for x in command], capture_output=True, timeout=90)
    log.append(dict(command=list(map(str, command)), returncode=p.returncode,
                    stderr=p.stderr.decode('utf-8', errors='replace')[-1600:]))
    if p.returncode:
        raise RuntimeError(log[-1])


def write_stream(path, frames, method):
    slots = np.arange(len(frames))
    if method == 'native_plc':
        sources = slots
        keep = slots % 4 == 0
    elif method == 'previous_packet':
        sources = slots//4*4
        keep = np.ones(len(frames), bool)
    elif method == 'nearest_packet':
        sources = np.minimum((slots+2)//4*4, (len(frames)-1)//4*4)
        keep = np.ones(len(frames), bool)
    else:
        raise ValueError(method)
    bits = np.unpackbits(frames[sources], axis=1)
    with path.open('wb') as f:
        for b,good in zip(bits,keep):
            if good:
                f.write(np.array([0x6b21,264],'<u2').tobytes())
                f.write(np.where(b,0x81,0x7f).astype('<u2').tobytes())
            else:
                f.write(np.array([0x6b20,0],'<u2').tobytes())


def decode(tmp, frames, method, log):
    stream, raw = tmp/'input.192', tmp/'output.raw'
    write_stream(stream, frames, method)
    call([DEC,'-q','-no_delay_cmp',32,stream,raw],log)
    pcm = np.fromfile(raw,'<i2')
    assert len(pcm) == len(frames)*FRAME
    return pcm


def control(tmp, seconds, log):
    source = ROOT/'work/live/session_20261125T054037Z/ref_20261125T054037Z.ts'
    raw, encoded, decoded = tmp/'reference.raw',tmp/'reference.192',tmp/'truth.raw'
    samples = 12*RATE
    call(['ffmpeg','-nostdin','-v','error','-y','-ss',seconds,'-i',source,
          '-map','0:a:0','-af',f'aresample={RATE},atrim=end_sample={samples}',
          '-t',14,'-ac',1,'-ar',RATE,'-f','s16le',raw],log)
    assert raw.stat().st_size == samples*2
    call([ENC,'-q','-no_delay_cmp','-max_band','SWB',13200,32,raw,encoded],log)
    words = np.fromfile(encoded,'<u2').reshape(-1,266)
    assert len(words) == 600 and np.all(words[:,0] == 0x6b21) and np.all(words[:,1] == 264)
    frames = np.packbits(words[:,2:] == 0x81,axis=1)
    call([DEC,'-q','-no_delay_cmp',32,encoded,decoded],log)
    truth = np.fromfile(decoded,'<i2')
    assert len(truth) == samples
    return frames,truth


def _stoi(*args, **kwargs):
    from pystoi import stoi          # imported on use: only scoring needs it
    return stoi(*args, **kwargs)


def metrics(pcm,truth):
    # Common startup/tail exclusion; no fitted lag or gain.
    x = truth[int(.24*RATE):-int(.12*RATE)].astype(float)/32768
    y = pcm[int(.24*RATE):-int(.12*RATE)].astype(float)/32768
    return dict(stoi=float(_stoi(x,y,RATE,extended=False)),
                estoi=float(_stoi(x,y,RATE,extended=True)),
                clipped_fraction=float(np.mean(np.abs(pcm.astype(float)) >= 32767)))


def target(start,end):
    counters = np.arange(start,end+1)
    with np.load(ROOT/'work/nov25_decode_salvaged/groups.npz') as d:
        ix = np.searchsorted(d['counters'],counters)
        assert np.array_equal(d['counters'][ix],counters)
        du = d['du'][ix].copy()
    with np.load(ROOT/'work/nov25_repeated_text/keys.npz') as d:
        ix = np.searchsorted(d['counters'],counters)
        assert np.array_equal(d['counters'][ix],counters)
        assert d['crc_supported'][ix,:11].all()
        keys = d['ks'][ix,:11].copy()
    p = np.stack([du[:,q,7+131*c:39+131*c] for q in range(3) for c in range(5)],axis=1)
    p[:,:,::3] ^= keys[:,None,:]
    modes = np.stack([du[:,q,2+c] for q in range(3) for c in range(5)],axis=1)
    f = np.zeros((len(counters)*60,33),np.uint8)
    f[::4] = np.column_stack((modes.reshape(-1),p.reshape(-1,32)))
    return f


def save_wav(path,pcm):
    with wave.open(str(path),'wb') as w:
        w.setparams((1,2,RATE,0,'NONE','not compressed'))
        w.writeframes(pcm.astype('<i2').tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    a = ap.parse_args()
    out = (ROOT/a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    log,results = [],[]
    methods = ('native_plc','previous_packet','nearest_packet')
    with tempfile.TemporaryDirectory(dir=out) as td:
        td = Path(td)
        for seconds in (300,780,500):
            frames,truth = control(td,seconds,log)
            scores = {m:metrics(decode(td,frames,m,log),truth) for m in methods}
            results.append(dict(reference_seconds=seconds,scores=scores))
            print(json.dumps(results[-1]),flush=True)
        def passes(method, control_results):
            return all(r['scores'][method]['estoi'] >= r['scores']['native_plc']['estoi']+.01 and
                       r['scores'][method]['stoi'] >= r['scores']['native_plc']['stoi'] and
                       r['scores'][method]['clipped_fraction'] <= .001 for r in control_results)
        selected = [m for m in methods[1:] if passes(m, results[:2])]
        passed = [m for m in selected if passes(m, results[2:])]
        target_outputs = []
        for start,end in ((43709920,43709921),(43709964,43709966)):
            frames = target(start,end)
            for method in passed:
                pcm = decode(td,frames,method,log)
                clipped = float(np.mean(np.abs(pcm.astype(float)) >= 32767))
                if clipped > .001:
                    target_outputs.append(dict(start=start,end=end,method=method,rejected='target_clipping'))
                    continue
                path = out/f'UNVERIFIED_RECONSTRUCTION_{method}_{start}-{end}.wav'
                save_wav(path,pcm)
                target_outputs.append(dict(start=start,end=end,method=method,file=path.name,
                                           seconds=len(pcm)/RATE,clipped_fraction=clipped))
        report = dict(controls=results,selection_offsets=[300,780],holdout_offset=500,
                      selected_before_holdout=selected,passed_methods=passed,target_outputs=target_outputs,
                      note='STOI/eSTOI are measured only against known encoded control audio. '
                           'No new keys recovered; copied packets replace unknown slots. '
                           'Target intelligibility and broadcast correspondence are unverified.')
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (out/'calls.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
