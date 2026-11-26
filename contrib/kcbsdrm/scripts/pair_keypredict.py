"""Can the side-channel key ks[0..15] be predicted?  Read-only.

    python scripts/pair_keypredict.py          (RUNBOOK 0c.7)

Main data: March sign-on, idle-seed keys that pass the segment CRC. November keys_records_*.npz (all 16 bytes
verified) are used only for the repeat search. Every test is also run on random keys (must find nothing) and on
a planted weakness (must find it).
"""
import collections, glob, os
import numpy as np
from pair_analyze import records, trim_loop, integrity, groups, counter, crc16

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference')
rng = np.random.default_rng(1)


def march():
    r = trim_loop(records(os.path.join(REF, 'sf_signon_3205_2026-03-16.bin'))[0])[0]
    du, good = integrity(r)[:2]
    C, K, T = [], [], []
    for i in groups(du, good):
        ks = du[i, 698:714] ^ 0xFF
        s = bytes(du[i + 1, 698:714] ^ ks) + bytes(du[i + 2, 698:714] ^ ks)
        if crc16(s[:27]) == int.from_bytes(s[27:29], 'big'):
            C.append(counter(du, i)); K.append(ks); T.append(du[i:i + 3, 662:698])
    return np.array(C), np.array(K, np.uint8), np.array(T)


def ints(bits):
    """Rows of a 0/1 matrix as Python integers (column j = bit j)."""
    return [int.from_bytes(np.packbits(row, bitorder='little').tobytes(), 'little') for row in bits]


def rank(rows):
    piv = {}
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h not in piv:
                piv[h] = v
                break
            v ^= piv[h]
    return len(piv)


def solvable(X, Y):
    """Per column of Y: is it an exact GF(2) affine function of the columns of X?"""
    A = np.hstack([X, np.ones((len(X), 1), np.uint8)])
    base = rank(ints(A))
    return base, [j for j in range(Y.shape[1]) if rank(ints(np.hstack([A, Y[:, j:j + 1]]))) == base]


def bm(bits):
    """Berlekamp-Massey linear complexity."""
    C, B, L, m, hist = 1, 1, 0, 1, 0
    for n, b in enumerate(bits):
        hist = (hist << 1) | int(b)                      # bit i = s[n-i]
        if bin(C & hist).count('1') & 1:
            old = C
            C ^= B << m
            if 2 * L <= n:
                L, B, m = n + 1 - L, old, 0
        m += 1
    return L


def longest_run(C):
    best, a = (0, 0), 0
    for i in range(1, len(C) + 1):
        if i == len(C) or C[i] != C[i - 1] + 1:
            if i - a > best[1] - best[0]:
                best = (a, i)
            a = i
    return best


def test_keys(name, C, K):
    n = len(K)
    bits = np.unpackbits(K, axis=1)
    print('\n== %s: %d keys' % (name, n))
    # repeats
    full = n - len({bytes(k) for k in K})
    half = sum(n - len({bytes(k[a:a + 8]) for k in K}) for a in (0, 8))
    print('  repeats: 16-byte %d, 8-byte halves %d' % (full, half))
    print('  bit P(1) range %.3f..%.3f (random: about %.3f..%.3f)'
          % (bits.mean(0).min(), bits.mean(0).max(), .5 - 1.4 / n ** .5, .5 + 1.4 / n ** .5))
    # affine in the counter
    cb = ((C[:, None] >> np.arange(32)) & 1).astype(np.uint8)
    base, hit = solvable(cb, bits)
    print('  affine in counter: independent counter terms %d; key bits solved %d / 128' % (base, len(hit)))
    for label, b in (('MSB first', bits), ('LSB first', np.unpackbits(K, axis=1, bitorder='little'))):
        L = np.array([bm(x) for x in b])
        print('  LFSR within a key, %s: L median %.0f, min %d, share L<=56 %.4f (random: 64)'
              % (label, np.median(L), L.min(), (L <= 56).mean()))
    a, z = longest_run(C)
    L = np.array([bm(bits[a:z, j]) for j in range(128)])
    print('  LFSR across groups: longest gap-free run %d; L median %.0f, min %d, max %d (random: %d)'
          % (z - a, np.median(L), L.min(), L.max(), (z - a) // 2))
    # linear relations among key bits, within a key and across consecutive keys
    for span in (1, 2, 3, 4):
        idx = [i for i in range(n - span + 1) if C[i + span - 1] - C[i] == span - 1]
        cols = 128 * span + 1
        if len(idx) < cols + 40:
            print('  linear relation over %d consecutive keys: too few rows (%d)' % (span, len(idx)))
            continue
        M = np.hstack([bits[[i + s for i in idx]] for s in range(span)] + [np.ones((len(idx), 1), np.uint8)])
        r = rank(ints(M))
        print('  linear relation over %d consecutive keys: rank %d of %d (%d rows)' % (span, r, cols, len(idx)))


def test_trailer(name, K, T):
    """T: (n, 3 phases, 36) = du[662:698]. Bytes 664..667 are the counter in phase 0 only."""
    bits = np.unpackbits(K, axis=1)
    cols = [[j for j in range(36) if p or not 2 <= j < 6] for p in range(3)]
    best = collections.Counter()
    for kind in ('xor', 'diff'):
        for p in range(3):
            for j in cols[p]:
                for k in range(16):
                    d = (K[:, k] ^ T[:, p, j]) if kind == 'xor' else (K[:, k] - T[:, p, j])
                    if len(np.unique(T[:, p, j])) > 16:          # a constant trailer byte cannot carry a key
                        best[kind] = max(best[kind], int(np.bincount(d, minlength=256).max()))
    print('  %s: most frequent (key byte op trailer byte) value, of %d keys: xor %d, difference %d'
          % (name, len(K), best['xor'], best['diff']))
    for p in range(3):
        X = np.unpackbits(T[:, p, cols[p]], axis=1)
        base, hit = solvable(X, bits)
        print('    affine in trailer bytes, phase %d: rank %d of %d possible; key bits solved %d / 128%s'
              % (p, base, min(len(K), X.shape[1] + 1), len(hit), '' if base < len(K) else '  (VACUOUS)'))
    X = np.unpackbits(np.hstack([T[:, p, cols[p]] for p in range(3)]), axis=1)
    print('    (all three phases together: %d unknowns for %d keys, not testable)' % (X.shape[1] + 1, len(K)))


def chi2(x):
    h = np.bincount(x, minlength=256)
    e = len(x) / 256
    return ((h - e) ** 2 / e).sum()


def region(T):
    """The 96 unexplained bytes per group, by phase: du[668:698], du[665:698], du[664:697]."""
    return [T[:, 0, 6:36], T[:, 1, 3:36], T[:, 2, 2:35]]


def cross_phase(name, R):
    """Key-free: XOR two phases' regions at every relative offset. A shared keystream would cancel."""
    for label, partner in (('same group', np.arange(len(R[0]))), ('null: other group', rng.permutation(len(R[0])))):
        top_chi = top_bit = 0
        for a, b in ((0, 1), (0, 2), (1, 2)):
            for d in range(-25, 26):                         # at least 5 overlapping bytes
                n = min(R[a].shape[1], R[b].shape[1] - d) - max(0, -d)
                x = (R[a][:, max(0, -d):][:, :n] ^ R[b][partner][:, max(0, d):][:, :n]).ravel()
                top_chi = max(top_chi, (chi2(x) - 255) / 510 ** .5)
                m = np.unpackbits(x[:, None], axis=1).mean(0)
                top_bit = max(top_bit, np.abs(m - .5).max() * 2 * len(x) ** .5)
        print('  cross-phase XOR, %-8s %-18s largest of 153 alignments: byte chi2 %+.1f sigma, bit bias %.1f sigma'
              % (name, label, top_chi, top_bit))


def under_key(name, K, T):
    """Is any trailer byte XOR any key byte non-uniform (biased content encrypted with ks[0..15])?"""
    top = max((chi2(T[:, p, j] ^ K[:, k]) - 255) / 510 ** .5
              for p in range(3) for j in range(36) if len(np.unique(T[:, p, j])) > 16 for k in range(16))
    print('  trailer byte XOR key byte, %-34s largest chi2 %+.1f sigma' % (name, top))


def planted(C, K, T):
    """Same counters, deliberately weak keys: each test above must fire."""
    n = int(C[-1] - C[0]) + 1
    s = list(rng.integers(0, 2, 41))
    s[0] = 1
    while len(s) < n * 352:                              # x^41 + x^3 + 1, 44 key bytes per group
        s.append(s[-38] ^ s[-41])
    s = np.array(s[:n * 352], np.uint8).reshape(n, 352)[:, :128]
    test_keys('CONTROL planted 41-bit LFSR, 16 of 44 bytes seen', C, np.packbits(s[C - C[0]], axis=1))
    M = rng.integers(0, 2, (32, 128))
    cb = ((C[:, None] >> np.arange(32)) & 1)
    test_keys('CONTROL planted affine-in-counter', C, np.packbits((cb @ M) & 1, axis=1))
    P = K.copy()
    P[300:] = K[:len(K) - 300]
    test_keys('CONTROL planted repeat', C, P)
    print('\n== CONTROL planted trailer dependence')
    Q = T.copy()
    Q[:, 0, 10:26] = rng.integers(0, 256, (len(K), 16))
    test_trailer('key = trailer XOR constant', Q[:, 0, 10:26] ^ np.arange(16, dtype=np.uint8), Q)


def november():
    seen = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(REF, 'keys_records_*.npz'))):
        z = np.load(f)
        for c, k, v in zip(z['counters'], z['ks'], z['verified']):
            if v.all():
                seen[int(c)].append((os.path.basename(f)[13:-4], bytes(k)))
    return seen


def repeats_all(C, K):
    seen = november()
    nov = sum(map(len, seen.values()))
    for c, k in zip(C, K):
        seen[int(c)].append(('March', bytes(k)))
    multi = [v for v in seen.values() if len(v) > 1]
    conflict = sum(len({k for _, k in v}) > 1 for v in multi)
    print('\n== repeat search over all captures: March %d + November %d fully verified keys' % (len(K), nov))
    print('  counters seen in two files: %d, with different keys: %d' % (len(multi), conflict))
    uniq = {c: v[0][1] for c, v in seen.items()}
    for width in (16, 8, 4):
        where = collections.defaultdict(set)
        for c, k in uniq.items():
            for a in range(17 - width):
                where[k[a:a + width]].add((c, a))
        pairs = sum(len({c for c, _ in s}) > 1 for s in where.values())
        windows = len(uniq) * (17 - width)
        print('  %2d-byte strings shared by different counters (any offset): %d  (random: about %.2g)'
              % (width, pairs, windows ** 2 / 2 / 256.0 ** width))


if __name__ == '__main__':
    C, K, T = march()
    print('March: counters %08X..%08X, span %d' % (C[0], C[-1], C[-1] - C[0] + 1))
    test_keys('March (real)', C, K)
    print('\n== trailer bytes 662..697 (March)')
    print('  distinct values per byte, phase 0: %s' % [len(np.unique(T[:, 0, j])) for j in range(36)])
    print('  distinct values per byte, phase 1: %s' % [len(np.unique(T[:, 1, j])) for j in range(36)])
    print('  distinct values per byte, phase 2: %s' % [len(np.unique(T[:, 2, j])) for j in range(36)])
    test_trailer('real', K, T)
    test_trailer('random keys', rng.integers(0, 256, K.shape).astype(np.uint8), T)
    for p in range(3):
        const = {662 + j: '0x%02X' % T[0, p, j] for j in range(36) if len(np.unique(T[:, p, j])) == 1}
        print('  constant bytes, phase %d: %s' % (p, const))
    print('\n== shared or known key on the 96-byte region')
    cross_phase('March', region(T))
    f = os.path.join(REF, 'sf_2026-11-26_1008KST_merged.bin')
    du, good = integrity(trim_loop(records(f)[0])[0])[:2]
    cross_phase('Nov', region(np.array([du[i:i + 3, 662:698] for i in groups(du, good)])))
    Q = [x.copy() for x in region(T)]                    # control: one key on phases 0 and 1, biased content
    pad = rng.integers(0, 256, (len(K), 30)).astype(np.uint8)
    for a in (0, 1):
        Q[a][:, :30] = pad ^ rng.integers(0, 64, (len(K), 30)).astype(np.uint8)
    cross_phase('CONTROL', Q)
    under_key('real keys', K, T)
    under_key('random keys', rng.integers(0, 256, K.shape).astype(np.uint8), T)
    Q = T.copy()
    Q[:, 0, 10] = K[:, 3] ^ rng.integers(0, 64, len(K)).astype(np.uint8)
    under_key('CONTROL planted biased content', K, Q)
    repeats_all(C, K)
    test_keys('CONTROL random keys', C, rng.integers(0, 256, K.shape).astype(np.uint8))
    planted(C, K, T)
