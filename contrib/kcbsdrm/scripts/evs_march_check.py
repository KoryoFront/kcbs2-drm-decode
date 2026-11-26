"""Does the EVS reading of KCBS-2 reproduce the March satellite audio?  Read-only.

    python scripts/evs_march_check.py [--out DIR]

Decodes the March sign-on capture (idle-seed keys, CRC-checked) through the 3GPP EVS floating-point decoder,
reading each slot as [mode byte][32-byte payload] = 264 bits (the colleague's kcbs2-drm-decode framing).
Only role-0 frames (all 11 keyed bytes known) are supplied; every other slot is marked lost.

Controls, same loss pattern and same cleartext bytes:
  wrongkey  keyed bytes decrypted with the key of a group 50 groups away (keeps per-byte statistics roughly)
  randpay   all 32 payload bytes random, mode byte kept (header-leakage ablation)
Scores against IQ Files/satellite_aligned.wav: 1 kHz tone purity in the tone interval, and the correlation of
50 ms log-energy envelopes over a lag search, with a time-shifted null.
"""
import argparse, os, subprocess, tempfile, wave
import numpy as np
from pair_analyze import records, trim_loop, integrity, groups, crc16

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', 'reference')
EVS = os.environ.get('EVS_DEC', r'D:\KCBSDRM\SDR\3gpp-evs\EVS_dec.exe')
SAT = os.environ.get('KCBS_SAT', r'D:\KCBSDRM\IQ Files\satellite_aligned.wav')
FS = 16000
rng = np.random.default_rng(7)


def frame_starts():
    # 7,40,73,106 | 138,171,204,237 | ... (each 131-B group: 32,+1,32,+1,32,+1,32)
    return [7 + 131 * c + 33 * k for c in range(5) for k in range(4)]


def mode_pos(c, k, s):
    return 2 + c if k == 0 else s - 1


def load():
    r = trim_loop(records(os.path.join(REF, 'sf_signon_3205_2026-03-16.bin'))[0])[0]
    du, good = integrity(r)[:2]
    out = []
    for i in groups(du, good):
        ks = du[i, 698:714] ^ 0xFF
        s = bytes(du[i + 1, 698:714] ^ ks) + bytes(du[i + 2, 698:714] ^ ks)
        if crc16(s[:27]) == int.from_bytes(s[27:29], 'big'):
            out.append((i, ks))
    return du, out


def build(du, keyed, mode):
    """Frames (n_sf*20, 33) and supplied mask on the super-frame timeline."""
    n = du.shape[0]
    F = np.zeros((n * 20, 33), np.uint8)
    sup = np.zeros(n * 20, bool)
    st = frame_starts()
    keys = [k for _, k in keyed]
    for gi, (i, ks) in enumerate(keyed):
        if mode == 'wrongkey':
            ks = keys[(gi + 50) % len(keys)]
        for p in range(3):
            d = du[i + p]
            for f, s in enumerate(st):
                if f % 4:
                    continue
                c, k = divmod(f, 4)
                row = (i + p) * 20 + f
                F[row, 0] = d[mode_pos(c, k, s)]
                pay = d[s:s + 32].copy()
                pay[0::3] ^= ks[:11]
                if mode == 'randpay':
                    pay = rng.integers(0, 256, 32, dtype=np.uint8)
                F[row, 1:] = pay
                sup[row] = True
    return F, sup


def decode(F, sup, tmp):
    bs, raw = os.path.join(tmp, 's.192'), os.path.join(tmp, 's.raw')
    with open(bs, 'wb') as f:
        for bits, ok in zip(np.unpackbits(F, axis=1), sup):
            if ok:
                f.write(np.array([0x6b21, 264], '<u2').tobytes())
                f.write(np.where(bits, 0x81, 0x7f).astype('<u2').tobytes())
            else:
                f.write(np.array([0x6b20, 0], '<u2').tobytes())
    subprocess.run(EVS.split('|') + ['-q', '-no_delay_cmp', str(FS // 1000), bs, raw], check=True, capture_output=True)
    return np.fromfile(raw, '<i2').astype(np.float64)


def read_sat():
    with wave.open(SAT) as w:
        fs, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), '<i2').reshape(-1, ch).mean(1)
    return x, fs


def envelope(x, fs, hop=0.05):
    h = int(fs * hop)
    m = len(x) // h
    return np.log10(np.mean(x[:m * h].reshape(m, h) ** 2, 1) + 1.0)


def tone_purity(x, fs, t0, t1):
    seg = x[int(t0 * fs):int(t1 * fs)]
    if len(seg) < fs:
        return float('nan')
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / fs)
    band = (fr > 980) & (fr < 1020)
    return float(spec[band].sum() / (spec[(fr > 100) & (fr < 7000)].sum() + 1e-9))


def lagcorr(a, b, lags):
    out = []
    for L in lags:
        if L >= 0:
            x, y = a[:len(a) - L] if L else a, b[L:]
        else:
            x, y = a[-L:], b
        m = min(len(x), len(y))
        out.append(np.corrcoef(x[:m], y[:m])[0, 1] if m > 100 else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='write the decoded WAVs here')
    ap.add_argument('--evs', default=None, help="EVS_dec binary (default $EVS_DEC); for WSL use 'wsl|-d|Ubuntu|--|/path/EVS_dec'")
    ap.add_argument('--sat', default=None, help='satellite_aligned.wav (default $KCBS_SAT); envelope test skipped if absent')
    a = ap.parse_args()
    global EVS, SAT
    EVS, SAT = a.evs or EVS, a.sat or SAT
    du, keyed = load()
    print('super-frames %d, keyed groups %d' % (du.shape[0], len(keyed)))
    have_sat = os.path.isfile(SAT)
    if have_sat:
        sat, sfs = read_sat()
        es = envelope(sat, sfs)
        print('satellite: %.1f s; tone purity 10-130 s: %.3f' % (len(sat) / sfs, tone_purity(sat, sfs, 10, 130)))
    else:
        print('satellite file not found: tone/silence test only')
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ('real', 'wrongkey', 'randpay'):
            F, sup = build(du, keyed, mode)
            pcm = decode(F, sup, tmp)
            res[mode] = pcm
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                with wave.open(os.path.join(a.out, 'march_evs_role0_%s_16k.wav' % mode), 'wb') as w:
                    w.setparams((1, 2, FS, 0, 'NONE', ''))
                    w.writeframes(np.clip(pcm, -32768, 32767).astype('<i2').tobytes())
    # DRM time t (s from start of capture's first super-frame); tone->silence edge seen at SF 220 slot 14 = 88.28 s
    for mode, pcm in res.items():
        tp = [tone_purity(pcm, FS, t, t + 20) for t in (20, 40, 60, 100, 150, 200)]
        print('%-9s rms %7.1f  1kHz purity  t=20-40 %.3f 40-60 %.3f 60-80 %.3f | 100-120 %.3f 150-170 %.3f 200-220 %.3f'
              % (mode, pcm.std(), *tp))
    if not have_sat:
        return
    lags = np.arange(-int(70 / 0.05), int(70 / 0.05))  # satellite lag search -70..+70 s
    for mode, pcm in res.items():
        ed = envelope(pcm, FS)
        r = lagcorr(ed, es, lags)
        b = np.nanargmax(np.abs(r))
        far = np.abs(r[np.abs(lags * 0.05 - lags[b] * 0.05) > 10])
        print('%-9s best r %+.3f at sat lag %+.2f s; null(|r| >10 s away) max %.3f p95 %.3f'
              % (mode, r[b], lags[b] * 0.05, np.nanmax(far), np.nanpercentile(far, 95)))


if __name__ == '__main__':
    main()
