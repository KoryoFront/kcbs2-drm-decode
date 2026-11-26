"""Cross-copy confirmation of "verified" side-channel records within ONE capture (key-free).

    python pair_confirm.py TAG sf.bin [--apply]

CRC + structure can pass on garbled text (the CRC search can hit a 16-bit CRC by accident). The carousel
repeats each record at a different key-group alignment, so: plant a candidate's exact units (text, CRC,
0000 padding, BBBB fill) at another copy's position; units sharing a (group, column) must imply one key.
True text gives 0 contradictions, garbled text does not. No decoded key of the target enters the test.
A candidate is CONFIRMED if some other copy gives >= MINCHK checks with 0 contradictions; the confirmed
candidate with most checks wins. Where none is, the record is solved jointly from all copies (see solve()). --apply rewrites reference/items_TAG.json (original kept in
reference/_unconfirmed/): matching copies become ok with the confirmed text; a verified copy that differs from
the confirmed text, or that was tested elsewhere and failed, is demoted. Items left without any ok or
complete record other than single-record fragments are dropped.
"""
import sys, os, json, glob, shutil, collections, binascii
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from pair_bulletin import merge, load
from pair_text import load as loadsf
from pair_fill import units
from pair_title_joint import UF
from pair_crcfix import swap, le

REF = '../reference/'
MINCHK = 6


def main():
    tag, sf, apply = sys.argv[1], sys.argv[2], '--apply' in sys.argv
    src = REF + '_unconfirmed/items_%s.json' % tag
    fn = REF + 'items_%s.json' % tag
    if not os.path.exists(src):
        src = fn
    d = load(src)
    merged, where = merge([d])                      # copies are compared within this capture only
    G = loadsf([REF + sf]); C0 = min(G)
    W = {c: units(np.array([G[c][0][G[c][1] + p, 698:714] for p in range(3)])) for c in G}
    z = np.load(REF + 'keys_records_%s.npz' % tag)
    K = {c: kb[0::2].astype(int) | (kb[1::2].astype(int) << 8) for c, kb in zip(z['counters'].tolist(), z['ks'])}

    def unit(i):
        c = C0 + i // 24
        return int(W[c][(i % 24) // 8, i % 8] ^ K[c][i % 8]) if c in W else None

    def raw(r):
        """text+CRC units of a copy as decoded, or None"""
        n = len(r['text'])
        for m in range(n, n + 8):                          # json text drops CR
            seg = [unit(r['pos'] + 1 + i) for i in range(m + 1)]
            if None in seg:
                return None
            h = unit(r['pos'])
            if ''.join(chr(x) for x in seg[:-1] if x != 13) == r['text'] and (((h & 7) << 4) | ((h >> 8) & 15)) // 2 == m:
                return seg
        return None

    def score(seg, last, seq, pos):
        n = len(seg) - 1
        a = pos
        plant = {a + 1 + i: v for i, v in enumerate(seg)}
        end = a + n + 2; pad = -(-end // 8) * 8
        for i in range(end, pad): plant[i] = 0
        if not last:
            for i in range(pad, pad + 8): plant[i] = 0xBBBB
        if seq > 1:
            for i in range(a - 8, a): plant[i] = 0xBBBB
        imp = collections.defaultdict(list)
        for i, v in plant.items():
            c = C0 + i // 24
            if i >= 0 and c in W:
                imp[(c, i % 8)].append(int(W[c][(i % 24) // 8, i % 8]) ^ v)
        chk = sum(len(l) - 1 for l in imp.values())
        bad = sum(len(l) - collections.Counter(l).most_common(1)[0][1] for l in imp.values())
        return chk, bad

    def solve(cps):
        """exact joint solve from all copies, no language model: one plaintext, a key per (copy, group, column),
        anchored on the BBBB fill / 0000 padding; accept only if every unit is determined and the CRC passes"""
        pos = sorted({r['pos'] for r in cps})
        if len(pos) < 2 or any(r['seq'] <= 1 or r['last'] for r in cps):
            return None
        for n in sorted({min(46, len(r['text'])) for r in cps} | {46}, reverse=True):
            end = n + 2; pad = -(-(pos[0] + end) // 8) * 8 - pos[0]
            uf = UF(); uf.find('Z'); bad = 0
            for ci, a in enumerate(pos):
                for k in range(-8, pad + 8):
                    i = a + k; c = C0 + i // 24
                    if c not in W:
                        continue
                    if k < 0 or k >= pad: bad += not uf.union('Z', ('P', k), 0xBBBB)
                    if end <= k < pad: bad += not uf.union('Z', ('P', k), 0)
                    bad += not uf.union(('P', k), ('K', ci, c, i % 8), int(W[c][(i % 24) // 8, i % 8]))
            rz, oz = uf.find('Z')
            val = {k: uf.find(('P', k))[1] ^ oz for k in range(end) if uf.find(('P', k))[0] == rz}
            if bad or len(val) < end:
                continue
            seg = [val[k] for k in range(end)]
            if binascii.crc_hqx(le(seg[:-1]), 0xFFFF) ^ 0xFFFF == swap(seg[-1]):
                return ''.join(chr(x) for x in seg[1:-1] if x != 13)
        return None

    groups = collections.defaultdict(list)
    for ii, it in enumerate(d['items']):
        for r in it:
            groups[(where[(tag, ii)], r['seq'])].append(r)
    out = collections.Counter()
    for key, cps in sorted(groups.items()):
        cands = []
        for r in cps:
            s = raw(r) if r['complete'] else None
            if s is not None:
                cands.append((r, s))
        best = None
        for r, s in cands:
            tests = [score(s, r['last'], r['seq'], o['pos']) for o in cps
                     if o is not r and o['pos'] != r['pos']]
            good = [c for c, b in tests if b == 0 and c >= MINCHK]
            if good:
                q = (len(good), sum(good), r['ok'])
                if best is None or q > best[0]:
                    best = (q, r, s, tests)
        nver = sum(r['ok'] for r in cps)
        if best is None:
            text = solve(cps)
            if text is not None:
                out['position solved exactly from several copies (CRC passes)'] += 1
                print('SOLVED', key, repr(text))
                for r in cps:
                    r.update(ok=True, text=text, last=False, complete=True, confirmed=True, replaced=True)
                continue
            if nver:
                out['position verified but unconfirmable (single usable copy or no clean cross-test)'] += 1
                print('UNCONFIRMED', key, 'copies', len(cps), 'verified', nver,
                      repr(next(r['text'] for r in cps if r['ok'])[:50]),
                      [score(s, r['last'], r['seq'], o['pos']) for r, s in cands for o in cps if o['pos'] != r['pos']][:4])
                disagree = len({x['text'] for x in cps if x['ok']}) > 1
                for r in cps:
                    r['confirmed'] = False
                    if disagree:                       # copies differ and none is provable
                        r['ok'] = False
            else:
                out['position unverified'] += 1
            continue
        _, br, bs, _ = best
        out['position confirmed'] += 1
        if not nver:
            out['  of which newly recovered'] += 1
        for r in cps:
            same = r['text'] == br['text']
            if not same:
                # does the other text also confirm? it must not
                if r['ok']:
                    out['garbled verified copies demoted'] += 1
                    print('DEMOTE', key, 'pos', r['pos'], repr(r['text'][:50]))
            r.update(ok=True, text=br['text'], last=br['last'], complete=True, confirmed=True,
                     **({} if same else {'replaced': True}))
    for k, v in out.items():
        print('%4d %s' % (v, k))
    if apply:
        if src == fn:
            os.makedirs(REF + '_unconfirmed', exist_ok=True)
            shutil.copy2(fn, REF + '_unconfirmed/items_%s.json' % tag)
        raw_d = json.load(open(src, encoding='utf-8'))
        by = {r['pos']: r for it in d['items'] for r in it}
        keep = []
        for it in raw_d['items']:
            it = [by.get(r['pos'], r) for r in it]
            if len(it) == 1 and not it[0]['ok']:
                print('drop single unverified fragment at pos', it[0]['pos'], 'seq', it[0]['seq'])
                continue
            keep.append(it)
        raw_d['items'] = keep
        json.dump(raw_d, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
        print('wrote', fn)


if __name__ == '__main__':
    main()
