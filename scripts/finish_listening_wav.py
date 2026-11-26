"""Apply a modest gain and oversampled peak guard to a rendered listening WAV."""
import argparse
import json
from pathlib import Path
import subprocess
import wave

import numpy as np


def finish(source, report, dest, gain_db):
    with wave.open(str(source),'rb') as w:
        rate = w.getframerate()
        if (w.getnchannels(),w.getsampwidth()) != (1,2) or rate not in (16000,32000,48000):
            raise ValueError('expected a mono 16-bit PCM render at 16, 32, or 48 kHz')
        samples = w.getnframes()
    metadata = json.loads(report.read_text(encoding='utf-8'))
    if round(metadata['output_seconds']*rate) != samples or not metadata['timeline']:
        raise ValueError('report does not describe this counter timeline')
    ranges = [(round(g['start_seconds']*rate),round(g['duration_seconds']*rate))
              for g in metadata['silent_gaps']]
    if any(lo < 0 or size < 0 or lo+size > samples for lo, size in ranges):
        raise ValueError('silence interval falls outside the timeline')
    fx = (f'aformat=sample_fmts=flt,volume={gain_db:.2f}dB,aresample={4*rate},'
          f'alimiter=limit=0.73:attack=3:release=25:level=false:latency=true,aresample={rate}')
    output = subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(source.resolve()),
                             '-map','0:a:0','-af',fx,'-c:a','pcm_s16le','-f','s16le','pipe:1'],
                            check=True,capture_output=True,timeout=300)
    pcm = np.frombuffer(output.stdout,dtype='<i2').copy()
    if len(pcm) != samples:
        raise ValueError(f'peak processing changed the sample count: {samples} to {len(pcm)}')
    for lo,size in ranges:
        pcm[lo:lo+size] = 0
    with dest.open('xb') as file, wave.open(file,'wb') as wav:
        wav.setparams((1,2,rate,0,'NONE','not compressed'))
        wav.writeframes(pcm.tobytes())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--gain-db',type=float,default=2.2)
    a = ap.parse_args()
    if not -12 <= a.gain_db <= 12:
        ap.error('gain must be between -12 and 12 dB')
    finish(a.source,a.report,a.out,a.gain_db)
