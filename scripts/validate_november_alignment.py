"""Check counter-based alignment on a known EVS/loss control and held-out runs."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('alignment', ROOT/'scripts/align_november_evs.py')
align = importlib.util.module_from_spec(spec); spec.loader.exec_module(align)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--indir', default='work/nov25_evs_alignment')
    ap.add_argument('--out', default='work/nov25_evs_alignment_validation')
    ap.add_argument('--compiler', choices=('gcc','tcc'), default='tcc')
    a = ap.parse_args()
    out = ROOT/a.out
    out.mkdir(parents=True, exist_ok=False)
    paths = sorted((ROOT/a.indir).glob('UNVERIFIED_EVS_*.wav'))
    query, indices, run_ids, runs = [], [], [], []
    base = 43709059
    log = []
    for number,p in enumerate(paths):
        start,end = map(int,p.stem.split('_')[-1].split('-'))
        pcm, rate = sf.read(p,dtype='float32')
        assert rate == 16000
        centers = np.arange(12, len(pcm)//320, 4)*320+160
        query.append(align.features(pcm,centers))
        indices.extend(((start-base)*120+centers//160).tolist())
        run_ids.extend([number]*len(centers))
        runs.append((start,end,len(pcm),centers))
    query = np.concatenate(query)
    indices, run_ids = np.array(indices), np.array(run_ids)
    with tempfile.TemporaryDirectory(dir=out) as td:
        td=Path(td); refpath=td/'ref.raw'
        refsource=ROOT/'work/live/session_20261125T054037Z/ref_20261125T054037Z.ts'
        align.invoke(['ffmpeg','-nostdin','-v','error','-i',refsource,'-map','0:a:0',
                      '-ac','1','-ar','16000','-f','f32le',refpath],log)
        ref=np.fromfile(refpath,'<f4')
        features=align.features(ref,np.arange(len(ref)//160)*160)
        maxoffset=len(features)-1-int(indices.max())
        per_run=[]
        for number,run in enumerate(runs):
            mask=run_ids==number
            scores=align.score_offsets(query[mask],features,indices[mask],maxoffset)
            per_run.append(dict(counter_start=run[0],counter_end=run[1],
                                best_offset_seconds=float(scores.argmax())*.01,
                                peak=float(scores.max()),at_joint_36_34=float(scores[3634])))
        train=run_ids<3; test=~train
        trainscores=align.score_offsets(query[train],features,indices[train],maxoffset)
        chosen=int(trainscores.argmax())
        testscores=align.score_offsets(query[test],features,indices[test],maxoffset)
        holdout=dict(train_runs=3,test_runs=2,train_chosen_offset=chosen*.01,
                     train_peak=float(trainscores[chosen]),test_at_train_offset=float(testscores[chosen]),
                     test_best_offset=float(testscores.argmax())*.01,test_peak=float(testscores.max()),
                     test_offset_fraction_at_least_selected=float(np.mean(testscores>=testscores[chosen])))
        # Known source alignment, actual encode/decode, same 75% erased slots.
        positive=[]; known_offset=50.0
        for start,end,n,centers in runs:
            sample=round((known_offset+(start-base)*1.2)*16000)
            wave=np.clip(np.rint(ref[sample:sample+n]*32768),-32768,32767).astype('<i2')
            assert len(wave)==n
            inp,encoded,lossy,decoded=[td/name for name in ('input.raw','encoded.192','lossy.192','decoded.raw')]
            wave.tofile(inp)
            if a.compiler == 'gcc':
                from build_evs_gcc_independent import linux
                align.invoke(['wsl','-d','Ubuntu','--',linux(ROOT/'work/evs_gcc_independent/EVS_cod'),
                              '-q','-no_delay_cmp','13200','16',linux(inp),linux(encoded)],log)
            else:
                align.invoke([ROOT/'work/ref/evs/build/EVS_cod.exe','-q','-no_delay_cmp','13200','16',inp,encoded],log)
            words=np.fromfile(encoded,'<u2').reshape(-1,266)
            assert len(words)==n//320 and np.all(words[:,1]==264)
            with lossy.open('wb') as f:
                for i,wordsrow in enumerate(words):
                    f.write(wordsrow.tobytes() if i%4==0 else np.array([0x6b20,0],'<u2').tobytes())
            if a.compiler == 'gcc':
                from build_evs_gcc_independent import linux
                align.invoke(['wsl','-d','Ubuntu','--',linux(ROOT/'work/evs_gcc_independent/EVS_dec'),
                              '-q','-no_delay_cmp','16',linux(lossy),linux(decoded)],log)
            else:
                align.invoke([ROOT/'work/ref/evs/build/EVS_dec.exe','-q','-no_delay_cmp','16',lossy,decoded],log)
            y=np.fromfile(decoded,'<i2').astype(np.float32)/32768
            assert len(y)==n
            positive.append(align.features(y,centers))
        pscores=align.score_offsets(np.concatenate(positive),features,indices,maxoffset)
        pos=dict(inserted_reference_offset=known_offset,recovered_offset=float(pscores.argmax())*.01,
                 peak=float(pscores.max()),passed=bool(abs(float(pscores.argmax())*.01-known_offset)<.15))
        report=dict(compiler=a.compiler,input_dir=a.indir,per_run=per_run,holdout=holdout,positive_control=pos,
                    note='Small exploratory dataset; no intelligibility claim. Held-out test uses a lag selected only on first three runs.')
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (out/'calls.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
