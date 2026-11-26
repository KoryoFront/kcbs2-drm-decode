"""Joint decode of the item-opening region (META A, META B, title record) over every broadcast copy.

    python pair_title_joint.py            # all 2026-11-25 captures

The region between an item's previous final record and its seq-1 record has the same plaintext in every copy
(carousel loops, other captures) apart from per-copy fields (META A toggle/serial/clock, META B fields, the
title header's toggle, the title check word). Each copy sits at its own key-group alignment.
Unknowns: plaintext P[r] per region offset r (per copy for per-copy fields) and one key per (capture, group,
column). Every received unit gives an exact XOR equation  P[r] ^ K = tail.  Anchors: keys verified in
pair_records (CRC-verified records) and the block just before the seq-1 header (structurally fill).
Union-find with XOR offsets propagates them across copies. Offsets whose equations still contradict in >= 2
copies become per-copy fields; a single contradiction is a channel error and that observation is dropped.
Components with no anchor get a character bigram model (ICM over the shared sequence), margin reported.
Output: reference/title_joint_2026-11-25.txt ; main() returns the per-item solution for pair_titles.py.
"""
import collections, os
import numpy as np
from pair_text import load
from pair_fill import units
from pair_bulletin import merge, load as load_items

REF = '../reference/'
CAPS = [('2026-11-25', ['sf_2026-11-25_0448KST_3205_hik.bin', 'sf_2026-11-25_0457KST_3205_hik.bin']),
        ('2026-11-25_1246KST_merged', ['sf_2026-11-25_1246KST_6140_merged.bin']),
        ('2026-11-25_2242KST_3205_hl5ntr', ['sf_2026-11-25_2242KST_3205_hl5ntr.bin'])]
# other captures: KCBS_TITLE_DATE=2026-11-26 KCBS_TITLE_CAPS="tag:sf.bin[,sf.bin][;tag:...]"
DATE = os.environ.get('KCBS_TITLE_DATE', '2026-11-25')
if os.environ.get('KCBS_TITLE_CAPS'):
    CAPS = [(c.split(':')[0], c.split(':')[1].split(',')) for c in os.environ['KCBS_TITLE_CAPS'].split(';')]
CAT = ([0xAE30, 0xC0C1, 0xC815, 0xBCF4], [0xBB38, 0xD654, 0xC815, 0xBCF4])     # 기상정보, 문화정보


def ceil8(x):
    return -(-x // 8) * 8


class Cap:
    def __init__(self, tag, files):
        G = load([REF + f for f in files])
        self.tag = tag
        self.CS = sorted(G); self.C0 = self.CS[0]
        self.W = {c: units(np.array([G[c][0][G[c][1] + p, 698:714] for p in range(3)])) for c in self.CS}
        z = np.load(REF + 'keys_records_%s.npz' % tag)
        self.K, self.V = {}, {}
        for c, kb, vb in zip(z['counters'].tolist(), z['ks'], z['verified']):
            self.K[c] = kb[0::2].astype(int) | (kb[1::2].astype(int) << 8)
            self.V[c] = vb[0::2].astype(bool)

    def loc(self, i):
        return self.C0 + i // 24, (i % 24) // 8, i % 8

    def tail(self, i):
        c, p, j = self.loc(i)
        return int(self.W[c][p, j]) if c in self.W else None

    def cur(self, i):
        c, p, j = self.loc(i)
        return int(self.W[c][p, j] ^ self.K[c][j]) if c in self.W and c in self.K else None


class UF:
    def __init__(self):
        self.par, self.off = {}, {}            # value(x) = value(par) ^ off

    def find(self, x):
        if x not in self.par:
            self.par[x], self.off[x] = x, 0
            return x, 0
        o, y = 0, x
        while self.par[y] != y:
            o ^= self.off[y]; y = self.par[y]
        z, oz = x, o
        while self.par[z] != z:
            nz, noz = self.par[z], oz ^ self.off[z]
            self.par[z], self.off[z] = y, oz
            z, oz = nz, noz
        return y, o

    def union(self, a, b, w):
        """assert value(a) ^ value(b) == w; False on contradiction"""
        ra, oa = self.find(a); rb, ob = self.find(b)
        if ra == rb:
            return (oa ^ ob) == w
        self.par[rb] = ra; self.off[rb] = oa ^ ob ^ w
        return True


def show(v):
    if 0xAC00 <= v <= 0xD7A3 or 0x20 <= v < 0x7F or 0x2000 <= v <= 0x33FF:
        return chr(v)
    return '⏎' if v in (0x0A, 0x0D) else '·'


def declared_varying(cps, caps, n):
    var = {n - 9}                                   # title check word
    for b in range(0, n, 8):
        vA = vH = 0
        for t, s, a in cps:
            u = [caps[t].cur(s + b + j) for j in range(8)]
            if None in u:
                continue
            vA += ((u[2] == 0x0008) + (u[4] == 0x7EAB) + (u[7] == 0x0051)) >= 2
            vH += (u[0] >> 8) == 0xAB and u[4:8] in CAT
        if vA:
            var |= {b + j for j in (0, 1, 3, 5, 6)}
        if vH:
            var |= {b}
    return var


def solved_varying(val, n, ncp):
    """Same structural patterns, recognised in the SOLVED plaintext of any copy."""
    var = set()
    for b in range(0, n - 7, 8):
        for ci in range(ncp):
            u = [val.get(('P', b + j), val.get(('P', b + j, ci))) for j in range(8)]
            if None in u:
                continue
            if ((u[2] == 0x0008) + (u[4] == 0x7EAB) + (u[7] == 0x0051)) >= 2:
                var |= {b + j for j in (0, 1, 3, 5, 6)}
            if (u[0] >> 8) == 0xAB and u[4:8] in CAT:
                var |= {b}
    return var


TEMPLATE = {-56 + 2: 0x0008, -56 + 4: 0x7EAB, -56 + 7: 0x0051,      # META A constant columns
            -48 + 1: 0x007E, -48 + 3: 0x002E,                         # title header '~' N '.'
            -40 + 0: 0x0000, -40 + 1: 0x0000, -40 + 2: 0x0000}        # three zero units before the title
TVAR = [-56 + j for j in (0, 1, 3, 5, 6)] + [-48]                     # per-copy: toggle/serial/clock, title hdr


def build(cps, caps, varying, anchors=None):
    uf, bad = UF(), collections.Counter()
    uf.find('ZERO')
    for r, v in (anchors or {}).items():
        uf.union('ZERO', ('P', r), v)
    for ci, (t, s, a) in enumerate(cps):
        C = caps[t]
        b = a - 8
        if sum(C.cur(i) == 0xBBBB for i in range(b, b + 8)) >= 6:
            for i in range(b, b + 8):
                r = i - s
                uf.union('ZERO', ('P', r, ci) if r in varying else ('P', r), 0xBBBB)
        for i in range(s, a):
            tl = C.tail(i)
            if tl is None:
                continue
            c, p, j = C.loc(i)
            kv = ('K', t, c, j)
            if c in C.V and C.V[c][j]:
                uf.union('ZERO', kv, int(C.K[c][j]))
            r = i - s
            if not uf.union(('P', r, ci) if r in varying else ('P', r), kv, tl):
                bad[r] += 1
    return uf, bad


def lm(texts):
    cnt = collections.defaultdict(collections.Counter)
    uni = collections.Counter()
    for t in texts:
        t = t.replace('\n', '\r')
        uni.update(ord(x) for x in t)
        for x, y in zip(t, t[1:]):
            cnt[ord(x)][ord(y)] += 1
    tot = sum(uni.values())
    pu = np.full(65536, 0.02 / 65536)
    for k, v in uni.items():
        pu[k] += 0.98 * v / tot
    pu[0] += 0.01
    rev = collections.defaultdict(dict)
    for x, c in cnt.items():
        for y, v in c.items():
            rev[y][x] = v
    nb = np.zeros(65536)
    for x, c in cnt.items():
        nb[x] = sum(c.values())
    return dict(pu=pu, L=np.log(pu), cnt=cnt, rev=rev, nb=nb)


def resolve(uf, M, n, ncp, init=None):
    """Best of two ICM runs: unigram init, and init from the current (per-copy LM) keys."""
    runs = [_resolve(uf, M, n, ncp, None)]
    if init:
        runs.append(_resolve(uf, M, n, ncp, init))
    return max(runs, key=lambda t: t[2])[:2]


def total_score(val, M, n):
    alpha, pu, nb, cnt = 2.0, M['pu'], M['nb'], M['cnt']
    norm = lambda v: 0x0D if v == 0x0A else v
    sc = 0.0
    for r in range(n - 1):
        a, b = val.get(('P', r)), val.get(('P', r + 1))
        if a is None or b is None:
            continue
        a, b = norm(a), norm(b)
        sc += np.log((cnt.get(a, {}).get(b, 0) + alpha * pu[b]) / (nb[a] + alpha))
    return sc


def _resolve(uf, M, n, ncp, init):
    A = np.arange(65536)
    comp = collections.defaultdict(list)
    for x in list(uf.par):
        rt, o = uf.find(x)
        comp[rt].append((x, o))
    rz, oz = uf.find('ZERO')
    val, conf, free = {}, {}, []
    for rt, mem in comp.items():
        if rt == rz:
            for x, o in mem:
                val[x] = o ^ oz; conf[x] = 'exact'
        elif any(x[0] == 'P' for x, _ in mem):
            free.append(mem)
    for mem in free:
        best = None
        if init:
            for x, o in mem:
                if x[0] == 'P' and len(x) == 2 and x[1] in init:
                    best = init[x[1]] ^ o; break
        if best is None:
            sc = sum((M['L'][A ^ o] for x, o in mem if x[0] == 'P' and len(x) == 2), np.zeros(65536))
            best = int(np.argmax(sc))
        for x, o in mem:
            val[x] = best ^ o; conf[x] = 0.0
    alpha, pu, nb, cnt, rev = 2.0, M['pu'], M['nb'], M['cnt'], M['rev']
    norm = lambda v: 0x0D if v == 0x0A else v

    def pv(r, ci):
        x = ('P', r)                      # per-copy (binary) fields are not text: no LM context
        return norm(val[x]) if x in val else None
    for it in range(6):
        for mem in free:
            sc = np.zeros(65536); has = False
            for x, o in mem:
                if x[0] != 'P' or len(x) > 2:
                    continue
                r, ci = x[1], (x[2] if len(x) > 2 else 0)
                b = A ^ o
                pr = pv(r - 1, ci) if r > 0 else None
                nx = pv(r + 1, ci) if r + 1 < n else None
                if pr is not None:
                    row = np.log(alpha * pu[b] / (nb[pr] + alpha))
                    for y, cy in cnt.get(pr, {}).items():
                        row[y ^ o] = np.log((cy + alpha * pu[y]) / (nb[pr] + alpha))
                    sc += row; has = True
                if nx is not None:
                    col = np.log(alpha * pu[nx] / (nb[b] + alpha))
                    for y, cy in rev.get(nx, {}).items():
                        col[y ^ o] = np.log((cy + alpha * pu[nx]) / (nb[y] + alpha))
                    sc += col; has = True
            if not has:
                continue
            order = np.argpartition(sc, -2)[-2:]
            best, second = (order[1], order[0]) if sc[order[1]] >= sc[order[0]] else (order[0], order[1])
            for x, o in mem:
                val[x] = int(best) ^ o; conf[x] = float(sc[best] - sc[second])
    return val, conf, total_score(val, M, n)


def pick(val, conf, r, varying, ncp):
    cands = [('P', r)] if r not in varying else [('P', r, ci) for ci in range(ncp)]
    key = next((x for x in cands if conf.get(x) == 'exact'), next((x for x in cands if x in val), None))
    if key is None:
        return None, None
    return val[key], conf[key]


def main(write=True):
    caps = {t: Cap(t, f) for t, f in CAPS}
    datasets = [load_items(REF + 'items_%s.json' % t) for t, _ in CAPS]
    merged, where = merge(datasets)
    body = [r['text'] for m in merged for r in m['recs'].values() if r['ok']]
    # pass 1: LM from verified body text; pass 2: plus the exactly solved title regions of pass 1
    res1, _ = solve_all(caps, datasets, merged, where, lm(body), write=False)
    extra = []
    for k, R in res1.items():
        if R['ncontra'] == 0:
            run = ''
            for r, (v, cf) in enumerate(R['seq']):
                if cf == 'exact' and r not in R['varying'] and v is not None:
                    run += chr(0x0D if v == 0x0A else v)
                else:
                    if len(run) > 1: extra.append(run)
                    run = ''
            if len(run) > 1: extra.append(run)
    print('pass 2: LM adds %d exact title runs (%d chars)' % (len(extra), sum(map(len, extra))))
    res, merged = solve_all(caps, datasets, merged, where, lm(body + extra * 5), write=write)
    export(res, merged)
    return res, merged


def export(res, merged):
    """reference/titles_2026-11-25.json: per item, the verified (seq, text) keys that identify it, the category
    and the opening text (title record after its header, up to the check word) with a per-character grade
    '=' exact, '~' LM margin >= 6, '?' weak."""
    out = []
    for k, R in res.items():
        n, seq = R['n'], R['seq']
        if R['ncontra'] or n < 56:
            continue
        g = lambda r: seq[r] if 0 <= r < n else (None, None)
        hdr = [g(n - 48 + j)[0] for j in range(8)]
        if None in hdr or hdr[1] != 0x007E or hdr[3] != 0x002E:
            continue
        def grade(cf):
            return '=' if cf == 'exact' else ('~' if cf is not None and cf >= 6 else '?')
        cat = ''.join(chr(v) for v in hdr[4:8])
        text, marks = '', ''
        for r in range(n - 37, n - 9):                # after the three zero units, up to the check word
            v, cf = g(r)
            if v is None:
                text += '�'; marks += '?'; continue
            if v == 0:
                continue
            if v == 0x0D:                             # CR LF -> one newline; a lone CR is a line break too
                if g(r + 1)[0] == 0x0A:
                    continue
                v = 0x0A
            text += chr(v); marks += grade(cf)
        keys = sorted([s, t] for s, t in merged[k]['vkeys'])
        out.append(dict(keys=keys, catno=chr(hdr[2]), category=cat, opening=text, grade=marks,
                        catgrade=''.join(grade(g(n - 48 + j)[1]) for j in range(4, 8))))
    import json
    json.dump(out, open(REF + 'titles_%s.json' % DATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote reference/titles_%s.json: %d item openings' % (DATE, len(out)))


def solve_item(cps, caps, n, M, varying, anchors):
    varying = set(varying)
    for rnd in range(8):
        uf, bad = build(cps, caps, varying, anchors)
        init = {}
        for r in range(n):
            for t, s0, a in cps:
                v = caps[t].cur(s0 + r)
                if v is not None:
                    init[r] = v; break
        val, conf = resolve(uf, M, n, len(cps), init)
        nv = ({r for r, x in bad.items() if x >= 2} | solved_varying(val, n, len(cps))) - varying - set(anchors)
        if not nv:
            break
        varying |= nv
    return val, conf, varying, sum(bad.values())


def solve_all(caps, datasets, merged, where, M, write=True):
    copies = collections.defaultdict(list)
    for d in datasets:
        items = d['items']
        for ii, it in enumerate(items):
            s1 = [r for r in it if r['seq'] == 1]
            if not s1:
                continue
            a = s1[0]['pos']
            prev = items[ii - 1] if ii else None
            s = None
            if prev and prev[-1]['last'] and prev[-1]['ok']:
                f = prev[-1]
                s = ceil8(f['pos'] + len(f['text']) + 2)
            copies[where[(d['tag'], ii)]].append((d['tag'], s, a))
    results = {}
    lines = []
    for k in sorted(copies):
        cps = copies[k]
        lens = collections.Counter(a - s for _, s, a in cps if s is not None and 0 < a - s <= 400)
        n = lens.most_common(1)[0][0] if lens else 64     # standard opening: fill, A, H, title (5 blocks), fill
        cps = [(t, a - n, a) for t, s, a in cps if s is None or a - s == n]
        best = None
        for use_t in (True, False):
            anchors = {n + d: v for d, v in TEMPLATE.items() if 0 <= n + d < n} if use_t else {}
            varying = declared_varying(cps, caps, n) - set(anchors)
            if use_t:
                varying |= {n + d for d in TVAR if 0 <= n + d}
            sol = solve_item(cps, caps, n, M, varying, anchors)
            if best is None or sol[3] < best[3]:
                best = sol
            if use_t and sol[3] == 0:
                break
        val, conf, varying, ncontra = best
        results[k] = dict(cps=cps, n=n, varying=varying, ncontra=ncontra,
                          seq=[pick(val, conf, r, varying, len(cps)) for r in range(n)])
        first = merged[k]['recs'][min(merged[k]['recs'])]['text'][:20].replace('\n', '⏎')
        lines.append('\n=== item opening before "%s"  (%d copies: %s; region %d units; %d contradictions; '
                     'per-copy offsets: %s)'
                     % (first, len(cps), ', '.join('%s@%d' % (t[11:] or 'am', a) for t, s, a in cps), n,
                        ncontra, sorted(varying)))
        for b in range(0, n, 8):
            row, txt = [], ''
            for r in range(b, min(b + 8, n)):
                v, cf = results[k]['seq'][r]
                if v is None:
                    row.append('?????'); txt += '?'; continue
                mark = ('*' if r in varying else '=') if cf == 'exact' else ('~' if cf >= 6 else '?')
                row.append('%04X%s' % (v, mark)); txt += show(v)
            lines.append('  %4d  %s  %s' % (b, ' '.join(row), txt))
    lines.append('\n# = exact (verified keys / structural fill, propagated by XOR chains across copies)\n'
                 '# * exact, per-copy field (value from one copy)   ~ bigram LM, margin >= 6 nats   ? weak LM')
    if write:
        open(REF + 'title_joint_%s.txt' % DATE, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
    return results, merged


if __name__ == '__main__':
    main()
