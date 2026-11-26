"""Demodulate one bounded IQ window, preserving source sample provenance.

Does not join dumps or infer record timestamps. Dream may loop its input; the
raw dump is preserved for counter-based assembly. A timeout is expected and is
recorded explicitly, not treated as proof of successful DRM reception.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

import soundfile as sf
from scipy.signal import resample_poly


def wsl_path(path):
    p = str(Path(path).resolve()).replace('\\', '/')
    return '/mnt/' + p[0].lower() + p[2:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input')
    ap.add_argument('outdir')
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--seconds', type=float, default=240)
    args = ap.parse_args()
    if args.start < 0 or not 0 < args.seconds <= 600:
        ap.error('start must be nonnegative; seconds must be in (0, 600]')
    src = Path(args.input).resolve()
    info = sf.info(src)
    if info.channels != 2 or abs(info.samplerate - 12000) > 3:
        ap.error('expected two-channel approximately 12 kHz IQ')
    start = round(args.start * info.samplerate)
    if start >= info.frames:
        ap.error('window begins past end of input')
    count = min(round(args.seconds * info.samplerate), info.frames - start)
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=False)  # never reuse an incomplete cache
    meta = dict(source=str(src), source_bytes=src.stat().st_size,
                source_sample_rate=info.samplerate, source_frames=info.frames,
                start_sample=start, sample_count=count,
                start_seconds=start / info.samplerate,
                duration_seconds=count / info.samplerate,
                timing_note='Header sample rate used; physical clock calibration unknown. '
                            'Dump record index is NOT a source timestamp.',
                status='running')
    manifest = out / 'manifest.json'
    manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    wav = out / 'iq48.wav'
    try:
        with sf.SoundFile(src) as f:
            f.seek(start)
            x = f.read(count, dtype='float32', always_2d=True)
        g = math.gcd(info.samplerate, 48000)
        y = resample_poly(x, 48000 // g, info.samplerate // g, axis=0)
        sf.write(wav, y, 48000, subtype='PCM_16')
        del x, y
        dump = out / 'msc.bin'
        log = out / 'dream.log'
        seconds = math.ceil(meta['duration_seconds'] * 1.15 + 60)
        cmd = ('cd /root/kcbs/dream-src && '
               'QT_QPA_PLATFORM=offscreen KCBS_DUMP_MSC=' + shlex.quote(wsl_path(dump)) +
               f' timeout {seconds} ./dream -c 6 --sigsrate 48000 -m 1 -f ' +
               shlex.quote(wsl_path(wav)) + ' >' + shlex.quote(wsl_path(log)) + ' 2>&1')
        t0 = time.monotonic()
        proc = subprocess.run(['wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', cmd],
                              capture_output=True, text=True, errors='replace',
                              timeout=seconds + 60)
        meta.update(wall_seconds=time.monotonic() - t0, returncode=proc.returncode,
                    dump_bytes=dump.stat().st_size if dump.exists() else 0,
                    status='finished' if proc.returncode in (0, 124) else 'failed',
                    stderr=proc.stderr[-2000:])
        if meta['status'] == 'finished' and not meta['dump_bytes']:
            meta['status'] = 'no_msc_output'
        # Keep the raw dump/log; remove only this run's resampled intermediate.
        wav.unlink()
    except Exception as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2), flush=True)
    if meta['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
