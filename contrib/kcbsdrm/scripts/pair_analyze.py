"""Load / trim / integrity / clock counter / keystream / decrypted mode-bit test for sf_*.bin dumps.

    python scripts/pair_analyze.py reference/sf_XXXX.bin [more.bin ...]

HANDOFF_2026-11-25 steps 2-5. The keystream model is the outside team's claim; this script tests it.
"""
import sys, collections, binascii, struct
import numpy as np

FRAME_STARTS = [7, 40, 73, 106, 138, 171, 204, 237, 269, 302,
                335, 368, 400, 433, 466, 499, 531, 564, 597, 630]


def crc16(b):
    """CRC-16 poly 0x1021, init 0xFFFF, final complement."""
    return binascii.crc_hqx(bytes(b), 0xFFFF) ^ 0xFFFF


def records(path):
    """Walk the dump by each record's 4-byte length prefix."""
    d = open(path, 'rb').read()
    out, odd, i = [], collections.Counter(), 0
    while i + 4 <= len(d):
        n = struct.unpack_from('<I', d, i)[0]
        if n == 0 or n > 8192:
            raise ValueError('bad length %d at offset %d' % (n, i))
        if i + 4 + n > len(d):
            break
        if n == 728:
            out.append(d[i + 4:i + 4 + n])
        else:
            odd[n] += 1
        i += 4 + n
    return np.frombuffer(b''.join(out), np.uint8).reshape(-1, 728), odd


def trim_loop(r):
    """Cut at the modal exact-repeat offset (Dream re-reads its input file)."""
    seen, offs = {}, collections.Counter()
    for i, x in enumerate(map(bytes, r)):
        if x in seen:
            offs[i - seen[x]] += 1
        seen[x] = i
    if offs:
        off, cnt = offs.most_common(1)[0]
        if cnt >= 20:
            return r[:off], off, cnt
    return r, None, 0


def integrity(r):
    pk = np.array([[crc16(x[p * 182:p * 182 + 180]) == int.from_bytes(bytes(x[p * 182 + 180:p * 182 + 182]), 'big')
                    for p in range(4)] for x in r])
    du = np.concatenate([r[:, p * 182 + 1:p * 182 + 180] for p in range(4)], axis=1)
    dg = np.array([crc16(x[:714]) == int.from_bytes(bytes(x[714:716]), 'big') for x in du])
    return du, pk.all(1), pk, dg


def groups(du, good):
    """Index of every phase-0 SF followed by phases 1, 2, all CRC-good."""
    return [i for i in range(len(du) - 2)
            if good[i:i + 3].all() and tuple(du[i:i + 3, 1]) == (0x00, 0x10, 0x20)]


def counter(du, i):
    return int.from_bytes(bytes(du[i, 664:668]), 'big')


def frames_of(du, i):
    """60 x 32 frames of the group starting at SF i."""
    return np.stack([du[i + p, s:s + 32] for p in range(3) for s in FRAME_STARTS])


def analyze(path):
    print('\n######## %s' % path)
    r, odd = records(path)
    print('records: %d of 728 B; other sizes: %s' % (len(r), dict(odd) or 'none'))
    r, off, cnt = trim_loop(r)
    print('loop trim: %s' % ('offset %d (%.1f s), %d repeats -> %d records kept' % (off, off * 0.4, cnt, len(r))
                             if off else 'no loop repeat found'))
    du, good, pk, dg = integrity(r)
    print('packet CRC good: %.2f%% of packets, %.2f%% of super-frames; data-group CRC good: %.2f%%'
          % (100 * pk.mean(), 100 * good.mean(), 100 * dg.mean()))
    seq = du[:, 1]
    gi = np.flatnonzero(good)
    brk = [(a, b) for a, b in zip(gi, gi[1:]) if (int(seq[b]) // 16 - int(seq[a]) // 16 - (b - a)) % 3]
    print('mod-3 phase breaks between good SFs: %d %s' % (len(brk), brk[:5]))

    # ---- clock counter
    g = groups(du, good)
    p0 = [i for i in range(len(du)) if good[i] and seq[i] == 0]
    c = np.array([counter(du, i) for i in p0], dtype=np.int64)
    steps = collections.Counter((int(c[k + 1] - c[k]), int(p0[k + 1] - p0[k])) for k in range(len(c) - 1))
    print('counter: %d phase-0 SFs, first 0x%08X (SF %d), last 0x%08X (SF %d)'
          % (len(c), c[0], p0[0], c[-1], p0[-1]))
    print('   (delta counter, delta SF) most common: %s' % steps.most_common(6))

    # ---- mode-byte timeline (header copy M0), 10 s bins, good SFs only
    print('mode-byte high nibble per 10 s (count of 5/7/F/other over 5 group flags x 25 SF):')
    for b in range(0, len(du), 250):
        sel = []
        for k in range(b, min(b + 250, len(du)), 25):
            m = du[k:min(k + 25, len(du))][good[k:k + 25], 2:7].ravel() >> 4
            n = collections.Counter(m.tolist())
            t = max(len(m), 1)
            sel.append('.' if not len(m) else max(n, key=n.get).__format__('X'))
        print('   SF %5d  t=%6.1f s  %s' % (b, b * 0.4, ''.join(sel)))

    # ---- keystream: idle seed
    res = []
    for i in g:
        t0, t1, t2 = (du[i + p, 698:714] for p in range(3))
        ks = t0 ^ 0xFF
        s = bytes(t1 ^ ks) + bytes(t2 ^ ks)
        ok = crc16(s[:27]) == int.from_bytes(s[27:29], 'big')
        res.append((i, ks, s, ok, s[:2] == b'\x68\xf0', s[29:] == b'\0\0\0'))
    n = len(res)
    npass = sum(x[3] for x in res)
    print('keystream idle seed: %d complete groups; segment CRC pass %d (%.2f%%, chance 0.0015%%); '
          'header 68F0 %d; padding 000000 %d'
          % (n, npass, 100 * npass / max(n, 1), sum(x[4] for x in res), sum(x[5] for x in res)))
    for b in range(0, len(du), 250):
        s = [x for x in res if b <= x[0] < b + 250]
        if s:
            print('   t=%6.1f s  groups %3d  pass %3d' % (b * 0.4, len(s), sum(x[3] for x in s)))
    keys = {i: ks for i, ks, s, ok, h, z in res if ok}
    if keys:
        for i, ks, s, ok, h, z in [x for x in res if x[3]][:3]:
            print('   sample segment @SF %d: %s' % (i, s.hex()))
        validate(du, keys)
        modebits(du, keys)
    return du, good, keys


KEYED = [(role, j, role, 3 * j, 11 * role + j) for role in (0, 1) for j in range(11) if 11 * role + j < 16]


def validate(du, keys):
    """Equality of same-role keyed bytes within a group vs across adjacent groups, raw and decrypted."""
    starts = sorted(keys)
    within = [0, 0]
    raw = [0, 0]
    dec = [0, 0]
    wrong = [0, 0]
    rng = np.random.default_rng(1)
    for a, b in zip(starts, starts[1:]):
        if b - a != 3:
            continue
        fa, fb = frames_of(du, a), frames_of(du, b)
        for role, j, _, col, k in KEYED:
            x, y = fa[role::4, col], fb[role::4, col]
            within[0] += int((x[1:] == x[:-1]).sum()); within[1] += len(x) - 1
            # adjacent frames of the same role across the boundary distance (lag 1 in role index)
            xl, yf = x[-1:], y[:1]
            raw[0] += int((xl == yf).sum()); raw[1] += 1
            dec[0] += int(((xl ^ keys[a][k]) == (yf ^ keys[b][k])).sum()); dec[1] += 1
            r1, r2 = rng.choice(len(starts), 2, replace=False)      # keys of unrelated groups
            wrong[0] += int(((xl ^ keys[starts[r1]][k]) == (yf ^ keys[starts[r2]][k])).sum()); wrong[1] += 1
    f = lambda v: '%.4f (n=%d)' % (v[0] / max(v[1], 1), v[1])
    print('key validation, same-role keyed bytes, consecutive same-role frames (chance 0.0039):')
    print('   within group        %s' % f(within))
    print('   across groups raw   %s' % f(raw))
    print('   across decrypted    %s' % f(dec))
    print('   across wrong keys   %s' % f(wrong))


def legal_pair(a, b):
    """AMR-WB+ modes of frames 0,1 (or 2,3) of a superframe."""
    return np.where((a >= 2) | (b >= 2), a == b, True)


def modebits(du, keys):
    b0, b1, fl = [], [], []
    for i, ks in keys.items():
        f = frames_of(du, i)
        b0.extend(f[0::4, 0] ^ ks[0])
        b1.extend(f[1::4, 0] ^ ks[11])
        fl.extend(np.repeat(du[i:i + 3, 2] >> 4, 5))     # per-SF mode class
    b0, b1, fl = np.array(b0, np.uint8), np.array(b1, np.uint8), np.array(fl)
    n = len(b0)
    A, B = np.unpackbits(b0[:, None], axis=1).astype(int), np.unpackbits(b1[:, None], axis=1).astype(int)
    m0, m1 = A[:, 0] * 2 + A[:, 1], B[:, 0] * 2 + B[:, 1]
    print('decrypted mode-bit test (byte 0 of roles 0,1; %d pairs)' % n)
    print('   top two bits: legal %.3f (independent random 0.375, real FT25 1.000); (3,3) %.3f (real ~0.8)'
          % (legal_pair(m0, m1).mean(), ((m0 == 3) & (m1 == 3)).mean()))
    t = np.zeros((4, 4), int)
    np.add.at(t, (m0, m1), 1)
    print('   joint (m0 rows, m1 cols):\n%s' % t)
    # control: role1 of one 4-group vs role0 of the NEXT 4-group (never in one superframe)
    s0 = m0.reshape(-1, 15)[:, 1:].ravel(); s1 = m1.reshape(-1, 15)[:, :-1].ravel()
    print('   control pair straddling superframes (role1 of m, role0 of m+1): legal %.3f'
          % legal_pair(s1, s0).mean())
    for c in np.unique(fl):
        s = fl == c
        print('   mode class %X: %5d pairs, legal %.3f, (m0 dist) %s'
              % (c, s.sum(), legal_pair(m0[s], m1[s]).mean(), np.round(np.bincount(m0[s], minlength=4) / s.sum(), 3)))
    out = []
    for x in range(8):
        for y in range(8):
            if x != y:
                a, b = A[:, x] * 2 + A[:, y], B[:, x] * 2 + B[:, y]
                leg = legal_pair(a, b).mean()
                null = legal_pair(a, np.roll(b, 97)).mean()
                out.append((leg - null, leg, null, ((a >= 2) & (b >= 2)).mean(), x, y))
    out.sort(reverse=True)
    print('   best bit pairs by legality excess over shifted null: (excess, legal, null, both>=2, msb, lsb)')
    for row in out[:4]:
        print('      %+.3f %.3f %.3f %.3f  bit%d bit%d' % row)
    print('   byte0 bit P(1) role0: %s' % np.round(A.mean(0), 3))
    print('   byte0 bit P(1) role1: %s' % np.round(B.mean(0), 3))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyze(p)
