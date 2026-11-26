"""Structure-aware, CRC-verified decode of the November side channel (second stage after pair_crcfix.py).

Every unit gets a role from the segment grid  [hdr][46 text][crc][8 fill]:
  fill  -> plaintext BBBB, key exact
  hdr   -> bytes (05|85, (seq<<4)|0C) with seq predicted from the previous header
  crc   -> unconstrained (not scored by the language model)
  text  -> language model trained on CRC-VERIFIED segments only (+ a digit-free seed)
Key candidates per (group, column) also use the keyed audio bytes of the group and of its carousel twin
(+-646 groups, identical plaintext, different key).  The CRC is the only acceptance test:
a segment's text is called verified only if its CRC passes.

    PYTHONIOENCODING=utf-8 python -u pair_crcfix2.py
"""
import binascii, itertools, collections, re, math, sys, time
import numpy as np
from pair_text import prior
from pair_fill import textlike
from pair_text_lm import LM, AudioCheck, SEED
import pair_crcfix as P1
from pair_crcfix import State, swap, le, TWIN

TEXT, HDR, CRC, FILL = 0, 1, 2, 3
A = np.arange(65536)
AUDIO_W = 0.7 * math.log(2)          # bits -> nats, discounted (model trained on March programme)
SEED_NODIGIT = re.sub('[0-9]', '', SEED)
# KCBS_SC_CORPUS: earlier *_crc.txt files whose [OK] (CRC-verified) segments train the language model
EXTRA = []
for _f in filter(None, __import__('os').environ.get('KCBS_SC_CORPUS', '').split(',')):
    EXTRA += [l.split('  ', 1)[1].rstrip('\n').replace(' / ', '\n') for l in open(_f, encoding='utf-8')
              if l.startswith('[OK]') and '  ' in l]


class S2:
    def __init__(self, S):
        self.S = S
        self.cs = P1.CS
        self.audio = AudioCheck(P1.G, self.cs)
        self.AC = {c: x for c, x in zip(self.cs, self.audio.C)}
        self.M = self.audio.M

    # ---------- grid ----------
    def grid(self):
        u = self.S.stream()
        H = self.S.headers(u)
        segs = {}
        for h, h2 in zip(H, H[1:] + [self.S.n]):
            s0 = (u[h] >> 12) & 15
            k = 0
            while h + 56 * (k + 1) <= h2 or (k == 0 and h2 == self.S.n and h + 56 <= self.S.n):
                segs[h + 56 * k] = (s0 + k) & 15
                k += 1
        role = np.full(self.S.n, TEXT)
        expect = {}
        for a, seq in segs.items():
            role[a] = HDR; expect[a] = seq
            role[a + 47] = CRC
            role[a + 48:a + 56] = FILL
        self.segs, self.role, self.expect = sorted(segs), role, expect
        return u

    # ---------- per-column candidate scoring ----------
    def colscore(self, c, j, L):
        """Score of every plaintext a of unit (c,0,j), i.e. key = W0 ^ a. Returns (scores over a, W0)."""
        W = self.S.W[c][:, j].astype(int)
        base = (c - P1.C0) * 24
        sc = np.zeros(65536)
        for p in range(3):
            pos = base + p * 8 + j
            v = A ^ W[0] ^ W[p]
            r = self.role[pos]
            if r == TEXT:
                sc += L[v]
            elif r == FILL:
                sc += np.where(v == 0xBBBB, 0.0, -40.0)
            elif r == HDR:
                exp = (self.expect[pos] << 12) | 0x0C00
                ok = ((v & 0xFF7F) == (exp | 0x05))
                sc += np.where(ok, math.log(0.5), np.where((v & 0x0F7F) == 0x0C05, -6.0, -40.0))
            # CRC: no constraint
        return sc, W[0]

    def audio_bits(self, c, j, keys):
        if c not in self.AC:
            return np.zeros(len(keys))
        x, cls = self.AC[c]
        lo, hi = keys & 255, keys >> 8
        s = self.M[2 * j][cls[None, :], x[2 * j][None, :] ^ lo[:, None]].sum(1)
        s += self.M[2 * j + 1][cls[None, :], x[2 * j + 1][None, :] ^ hi[:, None]].sum(1)
        return s

    def candidates(self, c, j, L, n=40, pre=400):
        sc, w0 = self.colscore(c, j, L)
        top = np.argpartition(sc, -pre)[-pre:]
        keys = (w0 ^ top).astype(int)
        tot = sc[top] + AUDIO_W * self.audio_bits(c, j, keys)
        for t in (c + TWIN, c - TWIN):
            if t in self.S.W:
                tk = keys ^ int(self.S.W[c][0, j]) ^ int(self.S.W[t][0, j])
                tot += AUDIO_W * self.audio_bits(t, j, tk)
        o = np.argsort(tot)[::-1][:n]
        return keys[o], tot[o]

    # ---------- beam decode of one group ----------
    def decode_group(self, c, lm, L, beam=40, ncand=30):
        S = self.S
        W = S.W[c]
        base = (c - P1.C0) * 24
        beams = [(0.0, ('', '', ''), ())]
        for j in range(8):
            if S.exact[c][j]:
                ks, ex = np.array([int(S.key[c][j])]), np.array([0.0])
            else:
                ks, ex = self.candidates(c, j, L, n=ncand)
            new = []
            for s, seg, kk in beams:
                for k, e in zip(ks.tolist(), ex.tolist()):
                    v = s + (e if not S.exact[c][j] else 0.0)
                    segn = list(seg)
                    for p in range(3):
                        u = int(W[p, j]) ^ k
                        if self.role[base + p * 8 + j] == TEXT and not (0xD800 <= u < 0xE000):
                            ch = chr(u)
                            v += lm.lp(segn[p][-2:], ch) - L[u]       # LM replaces the unigram part
                        else:
                            ch = '\x01'
                        segn[p] += ch
                    new.append((v, tuple(segn), kk + (k,)))
            new.sort(key=lambda x: -x[0])
            beams = new[:beam]
        return np.array(beams[0][2])

    # ---------- CRC search ----------
    def crc_search(self, a, L, nalt=8, maxchg=3):
        S = self.S
        cols = S.keys_in(a, a + 48)
        free = [cj for cj in cols if not S.exact[cj[0]][cj[1]]]
        if not free:
            return None
        seg = [S.unit(i) for i in range(a, a + 48)]
        if min(seg) < 0:
            return None
        target = (binascii.crc_hqx(le(seg[:47]), 0xFFFF) ^ 0xFFFF) ^ swap(seg[47])
        posof = collections.defaultdict(list)
        for i in range(a, a + 48):
            if S.present[i]:
                c, p, j = S.loc(i); posof[(c, j)].append(i - a)
        alts = []
        for cj in free:
            cur = int(S.key[cj[0]][cj[1]])
            ks, sc = self.candidates(cj[0], cj[1], L, n=nalt + 1)
            cs_ = dict(zip(ks.tolist(), sc.tolist()))
            curscore = cs_.get(cur, sc.min() - 5)
            for k, s in cs_.items():
                if k == cur:
                    continue
                d = k ^ cur
                e = [0] * 48
                for q in posof[cj]:
                    e[q] = d
                alts.append((cj, k, s - curscore, binascii.crc_hqx(le(e[:47]), 0) ^ swap(e[47])))
        sols = [((x,), x[2]) for x in alts if x[3] == target]
        pair = collections.defaultdict(list)
        for x, y in itertools.combinations(alts, 2):
            if x[0] != y[0]:
                pair[x[3] ^ y[3]].append((x, y))
        for x, y in pair.get(target, []):
            sols.append(((x, y), x[2] + y[2]))
        if maxchg >= 3:
            for z in alts:
                for x, y in pair.get(target ^ z[3], []):
                    if z[0] != x[0] and z[0] != y[0]:
                        sols.append(((x, y, z), x[2] + y[2] + z[2]))
        if not sols:
            return None
        sols.sort(key=lambda s: -s[1])
        return sols, len(alts)

    def apply(self, sol):
        for cj, k, ds, eff in sol:
            self.S.key[cj[0]][cj[1]] = k


def verified_text(S2o, u):
    out = []
    for a in S2o.segs:
        if a + 48 <= S2o.S.n and S2o.S.crc_ok(a, u):
            out.append(''.join(chr(x) for x in u[a + 1:a + 47] if 0 <= x and not 0xD800 <= x < 0xE000))
    return out


def lm_from(texts):
    texts = texts + EXTRA
    lm = LM([(SEED_NODIGIT, 1.0)] + [(t, 1.0) for t in texts])
    L = np.log(0.6 * np.exp(prior()) + 0.4 * np.bincount([ord(ch) for t in texts for ch in t], minlength=65536)
               / max(sum(map(len, texts)), 1) + 1e-12)
    return lm, L


def report(S2o, label):
    u = S2o.grid()
    ok = [a for a in S2o.segs if a + 48 <= S2o.S.n and S2o.S.crc_ok(a, u)]
    full = [a for a in S2o.segs if a + 48 <= S2o.S.n and S2o.S.present[a:a + 48].all()]
    ex = np.mean([S2o.S.exact[c] for c in P1.CS])
    print('%s: grid segments %d (complete %d), CRC verified %d (%.1f%% of complete); exact key columns %.1f%%'
          % (label, len(S2o.segs), len(full), len(ok), 100 * len(ok) / max(len(full), 1), 100 * ex), flush=True)
    return u, ok


def selftest(S2o, L, n=40, rng=np.random.default_rng(7)):
    """Corrupt 1-3 keys of CRC-verified segments to their alternatives; does the search restore them?"""
    S = S2o.S
    u = S.stream()
    ok = [a for a in S2o.segs if a + 48 <= S.n and S.crc_ok(a, u)]
    right = wrong = none = 0
    for a in rng.choice(ok, min(n, len(ok)), replace=False):
        cols = S.keys_in(a, a + 48)
        saved = {cj: int(S.key[cj[0]][cj[1]]) for cj in cols}
        savedx = {cj: bool(S.exact[cj[0]][cj[1]]) for cj in cols}
        m = int(rng.integers(1, 4))
        pick = [cols[i] for i in rng.choice(len(cols), m, replace=False)]
        for cj in cols:
            S.exact[cj[0]][cj[1]] = False
        for cj in pick:
            ks, _ = S2o.candidates(cj[0], cj[1], L, n=6)
            alt = [k for k in ks.tolist() if k != saved[cj]]
            S.key[cj[0]][cj[1]] = alt[int(rng.integers(0, len(alt)))]
        for cj in cols:                      # other columns of the segment stay exact, as in real use
            if cj not in pick:
                S.exact[cj[0]][cj[1]] = True
        r = S2o.crc_search(a, L)
        if r:
            S2o.apply(r[0][0][0])
            if all(int(S.key[cj[0]][cj[1]]) == saved[cj] for cj in cols):
                right += 1
            else:
                wrong += 1
        else:
            none += 1
        for cj in cols:
            S.key[cj[0]][cj[1]] = saved[cj]; S.exact[cj[0]][cj[1]] = savedx[cj]
    print('selftest (corrupt 1-3 keys of a verified segment, all others exact): restored %d, wrong %d, '
          'no solution %d' % (right, wrong, none), flush=True)


def main():
    t0 = time.time()
    S = P1.run()
    S2o = S2(S)
    u, ok = report(S2o, 'start')
    for it in range(4):
        lm, L = lm_from(verified_text(S2o, u))
        if it == 0:
            selftest(S2o, L)
        # 1. structure: fill is known plaintext on every grid segment
        for a in S2o.segs:
            for i in range(a + 48, min(a + 56, S.n)):
                c, p, j = S.loc(i)
                if c in S.W and not S.exact[c][j]:
                    S.set_plain(i, 0xBBBB); S.exact[c][j] = True
        # 2. beam re-decode of groups that still have free columns
        todo = [c for c in P1.CS if not S.exact[c].all()]
        for c in todo:
            k = S2o.decode_group(c, lm, L)
            for j in range(8):
                if not S.exact[c][j]:
                    S.key[c][j] = k[j]
        u, ok = report(S2o, 'iter %d after re-decode (%d groups, %.0f s)' % (it, len(todo), time.time() - t0))
        for a in ok:
            for c, j in S.keys_in(a, a + 48):
                S.exact[c][j] = True
        # 3. CRC search on failing segments: accept only the clearly best solution
        fixed = 0
        for a in S2o.segs:
            if a + 48 > S.n or S.crc_ok(a) is not False:
                continue
            r = S2o.crc_search(a, L)
            if not r:
                continue
            sols, nalt = r
            margin = sols[0][1] - (sols[1][1] if len(sols) > 1 else -1e9)
            if margin >= 4.0:
                S2o.apply(sols[0][0])
                if S.crc_ok(a):
                    fixed += 1
                    for c, j in S.keys_in(a, a + 48):
                        S.exact[c][j] = True
        u, ok = report(S2o, 'iter %d after CRC search (+%d, %.0f s)' % (it, fixed, time.time() - t0))
        for a in ok:
            for c, j in S.keys_in(a, a + 48):
                S.exact[c][j] = True
    S2o.grid()
    write(S2o)


def write(S2o):
    S = S2o.S
    u = S.stream()
    kb = {c: np.stack([S.key[c] & 255, S.key[c] >> 8], 1).ravel().astype(np.uint8) for c in P1.CS}
    np.savez('../reference/sidechannel_keys_crc_%s.npz' % P1.TAG, counters=np.array(P1.CS),
             keys=np.array([kb[c] for c in P1.CS]), exact=np.array([np.repeat(S.exact[c], 2) for c in P1.CS]))

    def ch(x):
        if x < 0: return '_'
        if x in (0x0D, 0): return ''
        if x == 0x0A: return ' / '
        if 0xD800 <= x < 0xE000 or not textlike([x])[0]: return '�'
        return chr(x)
    with open('../reference/sidechannel_text_%s_crc.txt' % P1.TAG, 'w', encoding='utf-8') as f:
        f.write('# November 2026-11-25 KST side channel, structure-aware CRC-verified decode (pair_crcfix2.py).\n'
                '# One line per 56-unit segment: [hdr][46 chars][CRC-16][fill].\n'
                '# [OK] = CRC-16 passes: every character exact.  [??] = CRC fails: best guess, do not quote.\n'
                '# [--] = segment not fully received.  pos = unit index from counter 0x%08X\n' % P1.C0)
        for a in S2o.segs:
            if a + 48 > S.n:
                continue
            st = '[--]' if not S.present[a:a + 48].all() else ('[OK]' if S.crc_ok(a, u) else '[??]')
            f.write('%s pos %6d seq %X  %s\n' % (st, a, S2o.expect[a], ''.join(ch(x) for x in u[a + 1:a + 47])))
    print('wrote reference/sidechannel_text_%s_crc.txt and sidechannel_keys_crc_%s.npz' % (P1.TAG, P1.TAG))


if __name__ == '__main__':
    main()
