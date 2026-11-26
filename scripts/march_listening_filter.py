"""Attenuate repetitive four-slot loudness variation in the March listening render."""
import argparse
import json
from pathlib import Path
import wave

import numpy as np


def slot_levels(cycles):
    rms = np.sqrt(np.mean(cycles**2, axis=2))
    db = 20*np.log10(np.maximum(rms, 1e-5))
    return db - db.mean(axis=1, keepdims=True), (rms**2).mean(axis=1) > 0.005**2


def reduce_flutter(pcm, rate):
    width = rate//50
    audio = pcm.astype(np.float32) / 32768
    cycles = audio.reshape(-1, 4, width)
    levels, active = slot_levels(cycles)
    even_groups = (np.arange(len(cycles))//15) % 2 == 0
    fit, check = active & even_groups, active & ~even_groups
    if min(fit.sum(), check.sum()) < 30:
        raise ValueError('not enough active audio to estimate and check a slot profile')
    bias = np.median(levels[fit], axis=0)
    reduction = np.clip(bias - bias.min(), 0, 4)
    centers = (np.arange(4) + 0.5)*width
    envelope = np.interp(np.arange(4*width), centers, -reduction, period=4*width)
    gain = (10**(envelope/20)).astype(np.float32).reshape(4, width)
    adjusted = cycles*gain
    before = np.median(levels[check], axis=0)
    after = np.median(slot_levels(adjusted)[0][check], axis=0)
    if np.ptp(after) >= np.ptp(before):
        raise ValueError('the correction did not reduce slot variation on held-out groups')
    peak = float(np.max(np.abs(adjusted)))
    adjusted *= min(1, 10**(-1/20)/peak)
    result = np.rint(adjusted.ravel()*32768).astype('<i2')
    return result, dict(held_out_active_cycles=int(check.sum()),
                        slot_gain_db=[round(-float(value), 2) for value in reduction],
                        held_out_slot_spread_db_before=round(float(np.ptp(before)), 2),
                        held_out_slot_spread_db_after=round(float(np.ptp(after)), 2),
                        output_peak_dbfs=round(float(20*np.log10(np.max(np.abs(result.astype(np.int32)))/32768)), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path, help='unfiltered March render from kcbs_decode_fast.py')
    ap.add_argument('--out', type=Path, required=True, help='new WAV path')
    a = ap.parse_args()
    with wave.open(str(a.source), 'rb') as wav:
        rate = wav.getframerate()
        if (wav.getnchannels(), wav.getsampwidth()) != (1, 2) or rate not in (16000, 32000, 48000):
            ap.error('expected mono 16-bit PCM at 16, 32, or 48 kHz')
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    if len(pcm) % (4*(rate//50)):
        ap.error('the input must start at slot zero and contain whole four-slot cycles')
    result, stats = reduce_flutter(pcm, rate)
    with a.out.open('xb') as dest, wave.open(dest, 'wb') as wav:
        wav.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        wav.writeframes(result.tobytes())
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
