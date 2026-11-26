"""Segment framing of the November side channel and its check word.

Stream of UTF-16 units, absolute position = (counter - c0) * 24 + unit-in-group.
Keys: exact from fill where a group holds a fill tail, else the LM key from sidechannel_text_2026-11-25.txt.
Segment (found 2026-11-24): [hdr bytes 0x05/0x85, (seq<<4)|0x0C][46 text units][check word][8 x 0xBBBB].
Tests CRC-16 variants for the check word over the segment.

    PYTHONIOENCODING=utf-8 python pair_segments.py
"""
import re, binascii, collections
import numpy as np
from pair_text import load
from pair_fill import textlike, units

import os
# KCBS_SC_FILES: comma-separated sf_*.bin dumps of ONE capture (default: the 2026-11-25 04:48/04:57 KST pair)
# KCBS_SC_TAG:   output tag (default 2026-11-25); KCBS_SC_LMKEYS: initial LM-key file ('' = none, start from fill)
FILES = os.environ.get('KCBS_SC_FILES', '../reference/sf_2026-11-25_0448KST_3205_hik.bin,'
                       '../reference/sf_2026-11-25_0457KST_3205_hik.bin').split(',')
TAG = os.environ.get('KCBS_SC_TAG', '2026-11-25')
LMKEYS = os.environ.get('KCBS_SC_LMKEYS', '../reference/sidechannel_text_2026-11-25.txt')
G = load(FILES)
CS = sorted(G); C0 = CS[0]


def lm_keys():
    K = {c: np.zeros(8, int) for c in CS}
    if not LMKEYS:
        return K
    for l in open(LMKEYS, encoding='utf-8'):
        m = re.match(r'0x([0-9A-F]+)\s+(-?[\d.]+)\s+([0-9a-f]{32}) ', l)
        if m:
            K[int(m.group(1), 16)] = units(np.frombuffer(bytes.fromhex(m.group(3)), np.uint8))
    return K


def tails():
    return {c: units(np.array([G[c][0][G[c][1] + p, 698:714] for p in range(3)])) for c in CS}


def fill_keys(W):
    """Per group: 8 column keys and exact-mask from a fill tail (>=6 strictly text-consistent columns)."""
    out = {}
    for c, w in W.items():
        for a in range(3):
            k = w[a] ^ 0xBBBB
            o = [w[b] ^ k for b in range(3) if b != a]
            ok = textlike(o[0]) & textlike(o[1])
            if ok.sum() >= 6:
                out[c] = (k, w[a] ^ k == 0xBBBB)       # exact where the fill tail column really is fill
                break
    return out


def stream(K, W, F):
    n = (CS[-1] - C0 + 1) * 24
    u = np.full(n, -1, int); exact = np.zeros(n, bool)
    for c in CS:
        k = K[c].copy(); m = np.zeros(8, bool)
        if c in F:
            k = np.where(F[c][1], F[c][0], k); m = F[c][1]
        s = (c - C0) * 24
        u[s:s + 24] = (W[c] ^ k).ravel()
        exact[s:s + 24] = np.tile(m, 3)
    return u, exact


def crcs(b):
    return {'ccitt-ffff-xorffff': binascii.crc_hqx(b, 0xFFFF) ^ 0xFFFF,
            'ccitt-ffff': binascii.crc_hqx(b, 0xFFFF),
            'ccitt-0000 (xmodem)': binascii.crc_hqx(b, 0),
            'ccitt-1d0f': binascii.crc_hqx(b, 0x1D0F)}


def main():
    K, W = lm_keys(), tails()
    F = fill_keys(W)
    u, exact = stream(K, W, F)
    is_hdr = (u >= 0) & ((u & 0x0F7F) == 0x0C05)
    fillrun = np.convolve((u == 0xBBBB).astype(int), np.ones(8, int), 'valid') >= 6
    hdr = [i for i in np.flatnonzero(is_hdr) if i >= 9 and fillrun[i - 8]]     # header right after fill
    print('%d headers that follow a fill block' % len(hdr))
    print('spacing between headers:', collections.Counter(np.diff(hdr).tolist()).most_common(8))
    seqs = [(u[i] >> 12) & 0xF for i in hdr]
    print('seq nibbles of first 40:', ''.join('%X' % s for s in seqs[:40]))
    print('flag byte (05/85) of first 40:', ''.join('1' if u[i] & 0x80 else '0' for i in hdr[:40]))
    # check word = unit just before the fill block preceding each header
    tests = collections.Counter(); n = 0; full = 0
    for a, b in zip(hdr, hdr[1:]):
        if b - a != 56:
            continue
        seg = u[a:a + 47]; chk = u[a + 47]
        if (seg < 0).any() or chk < 0:
            continue
        n += 1
        full += bool(exact[a:a + 48].all())
        for order in ('le', 'be'):
            if order == 'le':
                data = b''.join(int(x).to_bytes(2, 'little') for x in seg); words = (chk, ((chk & 255) << 8) | (chk >> 8))
            else:
                data = b''.join(int(x).to_bytes(2, 'big') for x in seg); words = (chk, ((chk & 255) << 8) | (chk >> 8))
            for span, d in (('hdr+text', data), ('text', data[2:])):
                for name, v in crcs(d).items():
                    for wi, w in enumerate(words):
                        if v == w:
                            tests['%s %s %s word%s' % (name, span, order, 'swapped' if wi else '')] += 1
    print('%d complete 56-unit segments (%d with every key exact)' % (n, full))
    print('CRC variant passes (chance ~%.2f per variant):' % (n / 65536))
    for k, v in tests.most_common(10):
        print('   %4d  %s' % (v, k))


if __name__ == '__main__':
    main()
