"""CRC-guided recovery of the November side channel.

Segment: [hdr 05|85, (seq<<4)|0C][46 text units][CRC-16-CCITT init FFFF xorout FFFF over hdr+text,
stored big-endian][8 x BBBB fill] = 56 units.  Keys are per (group, column); 3 units share each key.

Loop until nothing changes:
  1. fill units of every 56-spaced segment are BBBB  -> exact key for that column
  2. segments whose CRC passes                        -> every key they touch is exact
  3. carousel twin (+-646 groups, same plaintext)     -> copy plaintext, keep only if twin's CRC passes
  4. failing segments: try 1 or 2 key alternatives among non-exact keys; accept a UNIQUE fully
     text-like CRC match.

    PYTHONIOENCODING=utf-8 python pair_crcfix.py
"""
import binascii, itertools, collections
import numpy as np
from pair_text import prior
from pair_fill import textlike
from pair_segments import G, CS, C0, TAG, lm_keys, tails, fill_keys

TWIN = 646
L = prior()
CAND = np.arange(65536)


def swap(w):
    return ((w & 255) << 8) | (w >> 8)


def le(units):
    return b''.join(int(x).to_bytes(2, 'little') for x in units)


class State:
    def __init__(self):
        self.W = tails()
        K = lm_keys()
        self.key = {c: K[c].copy() for c in CS}
        self.exact = {c: np.zeros(8, bool) for c in CS}
        for c, (k, m) in fill_keys(self.W).items():
            self.key[c][m] = k[m]; self.exact[c][m] = True
        self.n = (CS[-1] - C0 + 1) * 24
        self.present = np.zeros(self.n, bool)
        for c in CS:
            self.present[(c - C0) * 24:(c - C0) * 24 + 24] = True
        self.cands = {}

    @staticmethod
    def loc(pos):
        return C0 + pos // 24, (pos % 24) // 8, pos % 8

    def unit(self, pos):
        c, p, j = self.loc(pos)
        return int(self.W[c][p, j] ^ self.key[c][j]) if c in self.W else -1

    def stream(self):
        return np.array([self.unit(i) for i in range(self.n)])

    def set_plain(self, pos, value):
        c, p, j = self.loc(pos)
        self.key[c][j] = self.W[c][p, j] ^ value

    def headers(self, u):
        h = []
        for i in np.flatnonzero((u >= 0) & ((u & 0x0F7F) == 0x0C05)):
            if i >= 8 and (u[i - 8:i] == 0xBBBB).sum() >= 5:
                h.append(int(i))
        return h

    def crc_ok(self, a, u=None):
        seg = [self.unit(i) for i in range(a, a + 48)] if u is None else list(u[a:a + 48])
        if min(seg) < 0:
            return None
        return binascii.crc_hqx(le(seg[:47]), 0xFFFF) ^ 0xFFFF == swap(seg[47])

    def keys_in(self, a, b):
        return sorted({self.loc(i)[::2] for i in range(a, b) if self.present[i]})   # (c, j)

    def candidates(self, c, j, k=12):
        if (c, j) not in self.cands:
            w = self.W[c][:, j].astype(int)
            sc = L[CAND] + L[CAND ^ w[0] ^ w[1]] + L[CAND ^ w[0] ^ w[2]]
            top = np.argpartition(sc, -k)[-k:]
            self.cands[(c, j)] = [int(w[0] ^ a) for a in top[np.argsort(sc[top])[::-1]]]
        return self.cands[(c, j)]

    def try_fix(self, a):
        """Search 1-2 key alternatives for segment at a. Return number of accepted keys or 0."""
        free = [(c, j) for c, j in self.keys_in(a, a + 48) if not self.exact[c][j]]
        if not free:
            return 0
        seg = [self.unit(i) for i in range(a, a + 48)]
        base = (binascii.crc_hqx(le(seg[:47]), 0xFFFF) ^ 0xFFFF) ^ swap(seg[47])
        # linear effect of changing key (c, j) by d
        posof = collections.defaultdict(list)
        for i in range(a, a + 48):
            if self.present[i]:
                c, p, j = self.loc(i)
                posof[(c, j)].append(i - a)
        alts = []
        for cj in free:
            cur = int(self.key[cj[0]][cj[1]])
            for kk in self.candidates(*cj)[:8]:
                d = kk ^ cur
                if d == 0:
                    continue
                e = [0] * 48
                for q in posof[cj]:
                    e[q] = d
                eff = binascii.crc_hqx(le(e[:47]), 0) ^ swap(e[47])
                alts.append((cj, kk, d, eff))
        sols = []
        for x in alts:
            if x[3] == base:
                sols.append((x,))
        for x, y in itertools.combinations(alts, 2):
            if x[0] != y[0] and x[3] ^ y[3] == base:
                sols.append((x, y))
        good = []
        for s in sols:
            ok = True
            for cj, kk, d, eff in s:
                for q in posof[cj]:
                    if q < 47 and not textlike([seg[q] ^ d])[0] and q != 0:
                        ok = False
            if ok:
                good.append(s)
        if len(good) != 1:
            return 0
        for cj, kk, d, eff in good[0]:
            self.key[cj[0]][cj[1]] = kk
        assert self.crc_ok(a)
        for c, j in self.keys_in(a, a + 48):
            self.exact[c][j] = True
        return len(good[0])


def run():
    S = State()
    for it in range(12):
        u = S.stream()
        H = S.headers(u)
        segs = [a for a, b in zip(H, H[1:]) if b - a == 56]
        forced = 0
        for a in segs:                                       # 1. fill is known plaintext
            for i in range(a + 48, a + 56):
                c, p, j = S.loc(i)
                if c in S.W and not S.exact[c][j]:
                    S.set_plain(i, 0xBBBB); S.exact[c][j] = True; forced += 1
        npass = 0
        for a in segs:                                       # 2. CRC-verified segments
            if S.crc_ok(a):
                npass += 1
                for c, j in S.keys_in(a, a + 48):
                    S.exact[c][j] = True
        twin = 0
        for a in segs:                                       # 3. carousel twin
            if not S.crc_ok(a):
                continue
            for t in (a + TWIN * 24, a - TWIN * 24):
                if t < 0 or t + 48 > S.n or not S.present[t:t + 48].all() or S.crc_ok(t):
                    continue
                saved = {cj: int(S.key[cj[0]][cj[1]]) for cj in S.keys_in(t, t + 48)}
                if any(S.exact[c][j] for c, j in saved):
                    # do not overwrite exact keys; only fill the free ones, then check
                    pass
                for i in range(48):
                    c, p, j = S.loc(t + i)
                    if not S.exact[c][j]:
                        S.set_plain(t + i, S.unit(a + i))
                if S.crc_ok(t):
                    twin += 1
                    for c, j in saved:
                        S.exact[c][j] = True
                else:
                    for (c, j), k in saved.items():
                        S.key[c][j] = k
        fixed = 0
        for a in segs:                                       # 4. CRC search
            if S.crc_ok(a) is False:
                fixed += bool(S.try_fix(a))
        u = S.stream()
        H = S.headers(u)
        segs = [a for a, b in zip(H, H[1:]) if b - a == 56]
        ok = sum(bool(S.crc_ok(a)) for a in segs)
        ex = np.mean([S.exact[c] for c in CS])
        print('iter %d: %d headers, %d 56-unit segments, CRC pass %d (%.1f%%); fill-forced %d, twin %d, '
              'searched-fixed %d; exact key columns %.1f%%' % (it, len(H), len(segs), ok, 100 * ok / max(len(segs), 1),
                                                              forced, twin, fixed, 100 * ex))
        if not (forced or twin or fixed):
            break
    return S


def write(S):
    u = S.stream()
    H = S.headers(u)
    ok = set()
    for a, b in zip(H, H[1:]):
        if b - a == 56 and S.crc_ok(a):
            ok.update(range(a, a + 56))
    kb = {c: np.stack([S.key[c] & 255, S.key[c] >> 8], 1).ravel().astype(np.uint8) for c in CS}
    np.savez('../reference/sidechannel_keys_crc_2026-11-25.npz', counters=np.array(CS),
             keys=np.array([kb[c] for c in CS]), exact=np.array([np.repeat(S.exact[c], 2) for c in CS]))

    def ch(x):
        if x == 0xBBBB: return ''
        if x in (0x0D,): return ''
        if x == 0x0A: return '\n'
        if 0xD800 <= x < 0xE000 or not textlike([x])[0] or x == 0: return '�'
        return chr(x)
    with open('../reference/sidechannel_text_2026-11-25_crc.txt', 'w', encoding='utf-8') as f:
        f.write('# November side channel, CRC-guided decode. One line per 56-unit segment.\n'
                '# [OK] = segment CRC-16 verified (exact).  [??] = not verified (best guess).\n'
                '# position = absolute unit index from counter 0x%08X\n' % C0)
        for a, b in zip(H, H[1:] + [S.n]):
            v = '[OK]' if a in ok else '[??]'
            seq = (u[a] >> 12) & 15
            text = ''.join(ch(x) for x in u[a + 1:min(b, a + 47)] if x >= 0).replace('\n', ' / ')
            f.write('%s pos %6d seq %X len %3d  %s\n' % (v, a, seq, b - a, text))
    print('wrote reference/sidechannel_text_2026-11-25_crc.txt and _keys_crc npz')


if __name__ == '__main__':
    write(run())
