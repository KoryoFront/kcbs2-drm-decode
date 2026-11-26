"""Exact keys from known-plaintext fill in the November side channel (no language model).

Known plaintext found 2026-11-24: idle fill is 16 x 0xBB per tail; record ends are padded with 0x00.
Per group and per 2-byte column j, hypothesis (tail a, value v in {BBBB, 0000}) gives key = tail_a[j] ^ v;
it is accepted when the code units it implies for the other two tails are both text-like.
Whole-tail fills (all 8 columns) are accepted only if all 16 implied units are text (false-accept ~1e-12).

Writes reference/sidechannel_keys_exact_2026-11-25.npz (key bytes + mask per counter).

    PYTHONIOENCODING=utf-8 python pair_fill.py
"""
import re
import numpy as np
from pair_text import load

KNOWN = (0xBBBB, 0x0000)


def textlike(u):
    u = np.asarray(u)
    return (((u >= 0xAC00) & (u < 0xD7A4)) | ((u >= 0x20) & (u < 0x7F)) | (u == 0x0D) | (u == 0x0A) |
            np.isin(u, [0x2103, 0x33A7, 0x300A, 0x300B, 0x201C, 0x201D, 0x2018, 0x2019, 0x3000, 0x00B7, 0x2026,
                        0xBBBB, 0x0000]))


def units(t):
    return t[..., 0::2].astype(int) | (t[..., 1::2].astype(int) << 8)


def main():
    G = load(['../reference/sf_2026-11-25_0448KST_3205_hik.bin', '../reference/sf_2026-11-25_0457KST_3205_hik.bin'])
    cs = sorted(G)
    keys = np.zeros((len(cs), 8), int); mask = np.zeros((len(cs), 8), bool); how = np.zeros((len(cs), 8), int)
    whole = 0
    for n, c in enumerate(cs):
        du, i = G[c]
        W = units(np.array([du[i + p, 698:714] for p in range(3)]))       # 3 x 8
        # 1) whole-tail fill
        for a in range(3):
            k = W[a] ^ 0xBBBB
            others = [W[b] ^ k for b in range(3) if b != a]
            if all(textlike(o).all() for o in others) and not all((o == 0xBBBB).all() for o in others):
                keys[n], mask[n], how[n] = k, True, 1
                whole += 1
                break
        if mask[n].all():
            continue
        # 2) per column: one unit known (fill or zero pad), unique accepted hypothesis only
        for j in range(8):
            acc = set()
            for a in range(3):
                for v in KNOWN:
                    k = W[a, j] ^ v
                    o = [W[b, j] ^ k for b in range(3) if b != a]
                    # need at least one other unit to be real text (not itself fill/zero) to be informative
                    if textlike(o).all() and any(x not in KNOWN for x in o):
                        acc.add(k)
            if len(acc) == 1:
                keys[n, j], mask[n, j], how[n, j] = acc.pop(), True, 2
    kb = np.zeros((len(cs), 16), np.uint8)
    kb[:, 0::2], kb[:, 1::2] = keys & 255, keys >> 8
    np.savez('../reference/sidechannel_keys_exact_2026-11-25.npz', counters=np.array(cs), keys=kb,
             mask=np.repeat(mask, 2, axis=1), how=np.repeat(how, 2, axis=1))
    print('%d groups; whole-tail fill found in %d (%.1f%%); key columns fixed: %d of %d (%.1f%%)'
          % (len(cs), whole, 100 * whole / len(cs), mask.sum(), mask.size, 100 * mask.mean()))
    print('   by column:', np.round(mask.mean(0), 2))

    # compare with the LM keys where both exist
    LMK = {}
    for l in open('../reference/sidechannel_text_2026-11-25.txt', encoding='utf-8'):
        m = re.match(r'0x([0-9A-F]+)\s+(-?[\d.]+)\s+([0-9a-f]{32}) ', l)
        if m:
            LMK[int(m.group(1), 16)] = np.frombuffer(bytes.fromhex(m.group(3)), np.uint8)
    lm = np.array([LMK[c] for c in cs])
    for h, name in ((1, 'whole-tail fill'), (2, 'per-column')):
        s = np.repeat(how == h, 2, axis=1)
        print('   %-16s agreement with LM key bytes: %.3f (n=%d)' % (name, (kb[s] == lm[s]).mean(), s.sum()))


if __name__ == '__main__':
    main()
