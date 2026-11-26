"""Record-structured decode of the November side channel (replaces the fixed 56-unit grid of pair_crcfix2).

Stream model (verified 2026-11-25 on the 0448/0457 KST capture, see RUNBOOK s.0c):
  * each super-frame tail (16 B) is ONE block of 8 UTF-16 units; the stream is a sequence of blocks.
  * a RECORD starts on a block boundary with a header unit  [b0][b1]:
        b0 bit7  toggle, alternates item to item
        b0 bit5  last record of an item
        b0 &7, b1 &0x0F   text length in bytes = ((b0&7)<<4) | (b1&0x0F)   (full record: 0x5C = 92)
        b1 >>4   sequence number within the item (1, 2, ...)
    then the text (len/2 units), then CRC-16 (crc_hqx init FFFF, xorout FFFF, over hdr+text, stored
    big-endian), then 0000 padding up to the next block boundary.
  * a non-final record is followed by exactly one fill block (8 x BBBB).
  * between the last record of an item and the first record of the next lies item METADATA (binary,
    blocks such as FCx5 05xx 0008 C9xx 7EAB ..., ECxx BExx BB82 ... 071E, and a title record). It is
    NOT scored as text.

A record counts as VERIFIED only if its CRC passes AND its structure checks out (padding all 0000 and,
for a non-final record, the next block all BBBB). CRC alone is not enough: the key search can satisfy a
16-bit CRC by accident, and the carousel then replicates the error.

    KCBS_SC_FILES=.. KCBS_SC_TAG=.. KCBS_SC_LMKEYS= [KCBS_SC_INIT=keys.npz] python -u pair_records.py
Writes reference/sidechannel_items_<TAG>.txt, reference/keys_records_<TAG>.npz,
       reference/stream_census_<TAG>.txt
"""
import os, binascii, itertools, collections, math, time
import numpy as np
import pair_crcfix as P1
from pair_crcfix import State, swap, le
from pair_crcfix2 import S2, lm_from, AUDIO_W, TWIN

TEXT, HDR, CRC, FILL, PAD, META = 0, 1, 2, 3, 4, 5
RNAME = ['text', 'header', 'crc', 'fill', 'pad', 'metadata']
A = np.arange(65536)


def hdr_len(x):
    """Text length in units for a header value x, or None if x is not a valid record header."""
    b0, b1 = x & 255, x >> 8
    t = b0 & 0x7F
    n = ((b0 & 7) << 4) | (b1 & 15)
    if t == 0x05 and n == 92:
        return 46
    if 0x20 <= t <= 0x27 and 2 <= n <= 92 and n % 2 == 0:
        return n // 2
    return None


def ceil8(x):
    return -(-x // 8) * 8


class Rec(S2):
    def __init__(self, S):
        super().__init__(S)
        self.recs = {}
        self.seed = {}

    def add_seeds(self):
        """KCBS_SC_SEEDS: earlier *_crc.txt ([OK] pos lines; header read from the warm-start keys) and/or
        items_*.json (verified records; header rebuilt from seq, toggle, last, text length)."""
        import re, json
        u = self.S.stream()
        for f in filter(None, os.environ.get('KCBS_SC_SEEDS', '').split(',')):
            if f.endswith('.json'):
                for it in json.load(open(f, encoding='utf-8'))['items']:
                    for r in it:
                        if r['ok']:
                            nb = 2 * len(r['text'])
                            b0 = r['tog'] | (0x20 if r['last'] else 0) | ((nb >> 4) & 7)
                            if not r['last']:
                                b0 = r['tog'] | 0x05
                            x = b0 | (((r['seq'] << 4) | (nb & 15)) << 8)
                            if hdr_len(x):
                                self.seed[r['pos']] = x
            else:
                for l in open(f, encoding='utf-8'):
                    m = re.match(r'\[OK\] pos\s+(\d+)', l)
                    if m and int(m.group(1)) < self.S.n and u[int(m.group(1))] >= 0 \
                            and hdr_len(int(u[int(m.group(1))])):
                        self.seed[int(m.group(1))] = int(u[int(m.group(1))])
        print('seeded %d record headers' % len(self.seed), flush=True)

    # ---------- parse ----------
    def parse(self):
        S = self.S
        u = S.stream()
        n = S.n
        isfill = lambda a: a >= 8 and (u[a - 8:a] == 0xBBBB).sum() >= 5
        known = {a: int(u[a]) for a in range(8, n, 8) if u[a] >= 0 and hdr_len(int(u[a])) and isfill(a)}
        for a, x in self.seed.items():               # headers verified in an earlier run: always records
            if a < n and S.present[a]:
                known[a] = int(u[a]) if (u[a] >= 0 and hdr_len(int(u[a])) and (int(u[a]) >> 12) == (x >> 12)) else x
        role = np.full(n, META)
        recs, expect = {}, {}
        kset = sorted(known)
        for ki, a in enumerate(kset):
            nxt_known = kset[ki + 1] if ki + 1 < len(kset) else n
            x = known[a]
            seq, tog, last, L = x >> 12, x & 0x80, bool(x & 0x20), hdr_len(x)
            extrap = False
            while a < nxt_known and a + L + 1 <= n:
                end = a + L + 2                     # hdr + L text + crc
                recs[a] = dict(L=L + 1, seq=seq, tog=tog, last=last, extrap=extrap)
                role[a] = HDR; role[a + 1:a + 1 + L] = TEXT; role[a + 1 + L] = CRC
                role[end:ceil8(end)] = PAD
                expect[a] = (seq << 12) | 0x0C00 | 0x05 | tog if extrap else x
                nb = ceil8(end)
                if last:
                    break
                role[nb:nb + 8] = FILL
                a = nb + 8
                if a >= nxt_known:
                    break
                seq = (seq + 1) & 15
                y = int(u[a]) if a < n and u[a] >= 0 else -1
                if y >= 0 and hdr_len(y) and (y >> 12) == seq:
                    L, last, extrap = hdr_len(y), bool(y & 0x20), False
                else:
                    L, last, extrap = 46, False, True
        role[~S.present] = META
        self.recs, self.role, self.expect = recs, role, expect
        self.segs = sorted(recs)
        return u

    # ---------- per record ----------
    def span(self, a):
        L = self.recs[a]['L']
        return a, a + L + 1, ceil8(a + L + 1)          # start, after-crc, after-pad

    def crc_ok(self, a, u):
        s, e, _ = self.span(a)
        if e > self.S.n or not self.S.present[s:e].all():
            return None
        seg = [int(x) for x in u[s:e]]
        return binascii.crc_hqx(le(seg[:-1]), 0xFFFF) ^ 0xFFFF == swap(seg[-1])

    def struct_ok(self, a, u):
        s, e, p = self.span(a)
        if p + (0 if self.recs[a]['last'] else 8) > self.S.n:
            return False
        if not (u[e:p] == 0).all():
            return False
        if self.recs[a]['last']:
            return True
        return bool((u[p:p + 8] == 0xBBBB).all())

    def text_ok(self, a, u):
        """Every text unit must be a character DPRK text actually uses (Hangul, ASCII, CJK/fullwidth
        punctuation, letterlike and unit symbols such as ℃ ㎧). A false CRC solution changes several units
        to random values, and a random value passes this with probability ~0.2 per unit."""
        s, e, _ = self.span(a)
        for x in u[s + 1:e - 1]:
            o = int(x)
            if not (0xAC00 <= o <= 0xD7A3 or 0x20 <= o < 0x7F or o in (0x0A, 0x0D, 0xB0, 0xB7, 0xD7)
                    or 0x3000 <= o <= 0x303F or 0x2000 <= o <= 0x206F or 0xFF01 <= o <= 0xFF5E
                    or 0x3130 <= o <= 0x318F or 0x2100 <= o <= 0x218F or 0x3380 <= o <= 0x33FF):
                return False
        return True

    def verified(self, u):
        return [a for a in self.segs if self.crc_ok(a, u) and self.struct_ok(a, u) and self.text_ok(a, u)]

    def cols(self, a, with_fill=True):
        s, e, p = self.span(a)
        b = p + (8 if with_fill and not self.recs[a]['last'] else 0)
        return self.S.keys_in(s, min(b, self.S.n))

    # ---------- scoring ----------
    def colscore(self, c, j, L):
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
                sc += np.where(v == 0xBBBB, 0.0, -12.0)
            elif r == PAD:
                sc += np.where(v == 0, 0.0, -12.0)
            elif r == HDR:
                exp = self.expect.get(pos, -1)
                hdrish = ((v >> 12) == (exp >> 12)) & ((v & 0x5F) >= 0x20) & ((v & 0x5F) <= 0x27)
                sc += np.where(v == exp, math.log(0.7), np.where(hdrish, -3.0, -12.0))
        return sc, W[0]

    # ---------- CRC search over a record of any length ----------
    def crc_search(self, a, L, nalt=8, maxchg=3):
        S = self.S
        s, e, _ = self.span(a)
        cols = S.keys_in(s, e)
        free = [cj for cj in cols if not S.exact[cj[0]][cj[1]]]
        if not free:
            return None
        seg = [S.unit(i) for i in range(s, e)]
        if min(seg) < 0:
            return None
        m = e - s
        target = (binascii.crc_hqx(le(seg[:-1]), 0xFFFF) ^ 0xFFFF) ^ swap(seg[-1])
        posof = collections.defaultdict(list)
        for i in range(s, e):
            c, p, j = S.loc(i); posof[(c, j)].append(i - s)
        alts = []
        for cj in free:
            cur = int(S.key[cj[0]][cj[1]])
            ks, sc = self.candidates(cj[0], cj[1], L, n=nalt + 1)
            cs_ = dict(zip(ks.tolist(), sc.tolist()))
            curscore = cs_.get(cur, sc.min() - 5)
            for k, sv in cs_.items():
                if k == cur:
                    continue
                d = k ^ cur
                ev = [0] * m
                for q in posof[cj]:
                    ev[q] = d
                alts.append((cj, k, sv - curscore, binascii.crc_hqx(le(ev[:-1]), 0) ^ swap(ev[-1])))
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
        sols.sort(key=lambda t: -t[1])
        return sols, len(alts)


def report(R, label):
    u = R.parse()
    ok = R.verified(u)
    full = [a for a in R.segs if R.crc_ok(a, u) is not None]
    ex = np.mean([R.S.exact[c] for c in P1.CS])
    print('%s: records %d (complete %d, extrapolated %d), VERIFIED %d (%.1f%%); exact key columns %.1f%%'
          % (label, len(R.segs), len(full), sum(r['extrap'] for r in R.recs.values()), len(ok),
             100 * len(ok) / max(len(full), 1), 100 * ex), flush=True)
    return u, ok


def mark_exact(R, ok):
    for a in ok:
        for c, j in R.cols(a):
            R.S.exact[c][j] = True


def is_text_cu(x):
    return (0xAC00 <= x <= 0xD7A3 or x in (0x0D, 0x0A, 0x20) or 0x21 <= x <= 0x7E or 0x3130 <= x <= 0x318F
            or x in (0x2103, 0x00B0, 0x33A7, 0x339D, 0x338F))


def fill_inject(R):
    """Fill crib (as in the colleague's decode_drm_text.py): for every group that is not fully exact,
    hypothesise that one of its 3 tails is the 0xBB fill block -> key = tail ^ BBBB. Pick the phase whose
    other two blocks then look most like a header / text, apply it to the non-exact columns, and keep it
    ONLY where a record then verifies (CRC + structure). Everything else is reverted."""
    S = R.S
    u = R.parse()
    ok = set(R.verified(u))
    kept = 0
    for a in R.segs:
        if a in ok or R.crc_ok(a, u) is None:
            continue
        s, e, p = R.span(a)
        b = min(S.n, p + (0 if R.recs[a]['last'] else 8))
        gs = sorted({S.loc(i)[0] for i in range(s, b)} - {c for c in P1.CS if S.exact[c].all()})
        gs = [c for c in gs if c in S.W]
        if not gs or len(gs) > 3:
            continue
        opts = {}
        for c in gs:
            o = [None]
            for ph in range(3):
                kv = S.W[c][ph] ^ 0xBBBB
                d = [int(x) for q in range(3) if q != ph for x in S.W[c][q] ^ kv]
                if sum(is_text_cu(x) for x in d) >= 8 or any(hdr_len(x) for x in d):
                    o.append(kv)
            opts[c] = o
        saved = {c: S.key[c].copy() for c in gs}
        win = None
        for combo in itertools.product(*[opts[c] for c in gs]):
            if all(k is None for k in combo):
                continue
            for c, kv in zip(gs, combo):
                S.key[c] = saved[c].copy()
                if kv is not None:
                    S.key[c][~S.exact[c]] = kv[~S.exact[c]]
                base = (c - P1.C0) * 24
                for i in range(base, base + 24):
                    u[i] = S.unit(i)
            if R.crc_ok(a, u) and R.struct_ok(a, u) and R.text_ok(a, u):
                win = combo
                break
        if win is None:
            for c in gs:
                S.key[c] = saved[c]
                base = (c - P1.C0) * 24
                for i in range(base, base + 24):
                    u[i] = S.unit(i)
        else:
            kept += 1
            for c, j in R.cols(a):
                S.exact[c][j] = True
    u = R.parse()
    ok = R.verified(u)
    mark_exact(R, ok)
    print('fill crib (per record): %d records newly verified' % kept, flush=True)
    return u, ok


def main():
    t0 = time.time()
    S = State()
    init = os.environ.get('KCBS_SC_INIT', '../reference/keys_verified_%s.npz' % P1.TAG)
    if os.path.exists(init):
        z = np.load(init)
        for c, kb in zip(z['counters'].tolist(), z['ks'] if 'ks' in z else z['keys']):
            if c in S.key:
                S.key[c] = kb[0::2].astype(int) | (kb[1::2].astype(int) << 8)
        print('warm start from', init)
    for c in P1.CS:
        S.exact[c][:] = False                       # exactness is re-earned under the record model
    R = Rec(S)
    R.add_seeds()
    u, ok = report(R, 'start')
    mark_exact(R, ok)
    for it in range(5):
        u, ok = fill_inject(R)
        lm, L = lm_from([''.join(chr(x) for x in u[a + 1:a + R.recs[a]['L']] if not 0xD800 <= x < 0xE000)
                         for a in ok])
        # never let the beam rewrite a group that is all metadata: nothing scores it, so its keys would be
        # replaced by audio-driven guesses, which breaks the fill / header detection next to it.
        todo = [c for c in P1.CS if not S.exact[c].all()
                and (R.role[(c - P1.C0) * 24:(c - P1.C0) * 24 + 24] != META).any()]
        for c in todo:
            k = R.decode_group(c, lm, L)
            for j in range(8):
                if not S.exact[c][j]:
                    S.key[c][j] = k[j]
        u, ok = report(R, 'iter %d re-decode (%d groups, %.0f s)' % (it, len(todo), time.time() - t0))
        mark_exact(R, ok)
        fixed = 0
        for a in R.segs:
            if R.crc_ok(a, u) is None or (R.crc_ok(a, u) and R.struct_ok(a, u) and R.text_ok(a, u)):
                continue
            r = R.crc_search(a, L)
            if not r:
                continue
            sols, _ = r
            margin = sols[0][1] - (sols[1][1] if len(sols) > 1 else -1e9)
            if margin < 4.0:
                continue
            saved = {cj: int(S.key[cj[0]][cj[1]]) for cj, *_ in sols[0][0]}
            R.apply(sols[0][0])
            u2 = S.stream()
            if R.crc_ok(a, u2) and R.struct_ok(a, u2) and R.text_ok(a, u2):
                fixed += 1
                for c, j in R.cols(a):
                    S.exact[c][j] = True
            else:
                for (c, j), k in saved.items():
                    S.key[c][j] = k
        u, ok = report(R, 'iter %d CRC search (+%d, %.0f s)' % (it, fixed, time.time() - t0))
        mark_exact(R, ok)
        if not fixed and it >= 2:
            break
    write(R, u, ok)


def write(R, u, ok):
    S = R.S
    oks = set(ok)
    tag = P1.TAG
    cs = P1.CS
    ver = {c: np.zeros(8, bool) for c in cs}
    for a in ok:
        for c, j in R.cols(a):
            ver[c][j] = True
    ks = np.array([np.stack([S.key[c] & 255, S.key[c] >> 8], 1).ravel() for c in cs], np.uint8)
    np.savez('../reference/keys_records_%s.npz' % tag, counters=np.array(cs), ks=ks,
             verified=np.array([np.repeat(ver[c], 2) for c in cs]))

    # census of the whole stream
    pres = S.present
    vmask = np.zeros(S.n, bool)
    for a in ok:
        s, e, p = R.span(a)
        vmask[s:p + (0 if R.recs[a]['last'] else 8)] = True
    with open('../reference/stream_census_%s.txt' % tag, 'w', encoding='utf-8') as f:
        tot = pres.sum()
        f.write('# side-channel stream census, %s: %d units received (%d blocks)\n' % (tag, tot, tot // 8))
        f.write('# role        units   share   in verified records\n')
        for r in range(6):
            m = (R.role == r) & pres
            f.write('%-10s %8d  %5.1f%%   %5.1f%%\n' % (RNAME[r], m.sum(), 100 * m.sum() / tot,
                                                     100 * (m & vmask).sum() / max(m.sum(), 1)))
    print(open('../reference/stream_census_%s.txt' % tag, encoding='utf-8').read())

    # item-structured text
    def ch(x):
        if x < 0: return '_'
        if x == 0x0D: return ''
        if x == 0x0A: return '\n'
        if 0xD800 <= x < 0xE000 or x < 0x20: return '�'
        return chr(x)
    items, cur = [], None
    for a in R.segs:
        r = R.recs[a]
        if (cur is None or r['seq'] == 1 or r['tog'] != cur['tog']
                or r['seq'] <= R.recs[cur['recs'][-1]]['seq']):
            cur = dict(start=a, tog=r['tog'], recs=[]); items.append(cur)
        cur['recs'].append(a)
        if r['last']:
            cur = None
    with open('../reference/sidechannel_items_%s.txt' % tag, 'w', encoding='utf-8') as f:
        f.write('# %s side channel, record-structured decode (pair_records.py).\n'
                '# Text in verified records (CRC + structure) is exact. Text of unverified records is shown\n'
                '# inside ⟦ ⟧ and is a best guess; "_" = not received.\n' % tag)
        for k, it in enumerate(items):
            nv = sum(a in oks for a in it['recs'])
            c0 = P1.C0 + it['start'] // 24
            f.write('\n=== item %d  pos %d  counter 0x%08X  records %d  verified %d%s ===\n'
                    % (k, it['start'], c0, len(it['recs']), nv,
                       '' if R.recs[it['recs'][-1]]['last'] else '  (no final record seen)'))
            out = []
            for a in it['recs']:
                s, e, _ = R.span(a)
                t = ''.join(ch(int(x)) for x in u[s + 1:e - 1])
                out.append(t if a in oks else '⟦' + t + '⟧')
            f.write(''.join(out) + '\n')
    import json
    js = []
    for it in items:
        rr = []
        for a in it['recs']:
            s, e, _ = R.span(a)
            rr.append(dict(pos=int(a), counter=int(P1.C0 + a // 24), seq=int(R.recs[a]['seq']),
                           last=bool(R.recs[a]['last']), tog=int(R.recs[a]['tog']), ok=a in oks,
                           complete=bool(S.present[s:e].all()),
                           text=''.join(ch(int(x)) for x in u[s + 1:e - 1])))
        js.append(rr)
    json.dump(dict(tag=tag, items=js), open('../reference/items_%s.json' % tag, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)
    print('wrote reference/sidechannel_items_%s.txt, keys_records_%s.npz, stream_census_%s.txt' % (tag, tag, tag))


if __name__ == '__main__':
    main()
