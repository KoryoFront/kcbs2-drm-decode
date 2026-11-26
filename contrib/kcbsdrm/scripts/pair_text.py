"""November side channel: UTF-16LE Korean text, three 16-byte tails per group under one 16-byte key.

For each of the 8 character columns of a group the key is one 16-bit value, shared by phases 0,1,2, so
choose c0 maximising P(c0) P(c0^d1) P(c0^d2) under a crude Korean character prior.
The keyed AUDIO bytes are NOT used by the decoder; they are the independent check:
mean log-likelihood (March-verified model) of the audio decrypted with the recovered key, against the
same score with keys taken from unrelated groups.

    python scripts/pair_text.py   (run from scripts/)
"""
import sys
import numpy as np
from pair_analyze import *

# rough syllable ranking for newspaper-style Korean, DPRK vocabulary included
COMMON = ('이다의는에하고한지가을로기서리인사자대아도수시어그적해일원상게나부정주화국전은여과를들제있'
          '것보장위동우선조성라면만오스내치합진영습니었없되었할당혁명김은령도자께서인민군력건설발'
          '높이여새운세계올통신보도방송중앙평양미괴남반투쟁승리청년학교농업공장생산경제문제요소모'
          '든할데대한위한바더욱열차비상무관회의결과단결강련합작품년월날말씀하시였으며였다고였던')


def prior():
    p = np.full(65536, 1e-9)
    p[0xAC00:0xD7A4] = 0.10 / (0xD7A4 - 0xAC00)
    seen = []
    for ch in COMMON:
        if ch not in seen:
            seen.append(ch)
    w = 1.0 / (np.arange(len(seen)) + 8.0)
    for ch, v in zip(seen, w / w.sum() * 0.55):
        p[ord(ch)] += v
    p[0x20] = 0.22
    for ch, v in (('.', .015), (',', .012), ('\r', .002), ('\n', .004), ('"', .002), ("'", .002), ('(', .002),
                  (')', .002), ('《', .003), ('》', .003), ('“', .002), ('”', .002), ('‘', .001), ('’', .001)):
        p[ord(ch)] += v
    for ch in '0123456789':
        p[ord(ch)] += .004
    return np.log(p)


def load(paths):
    G = {}
    for path in paths:
        r, _ = records(path); r, _, _ = trim_loop(r); du, good, _, _ = integrity(r)
        for i in groups(du, good):
            G[counter(du, i)] = (du, i)
    return G


def words(t):
    return t.astype('<u1').tobytes().decode('utf-16-le', errors='replace')


def main():
    L = prior()
    G = load(['../reference/sf_2026-11-25_0448KST_3205_hik.bin', '../reference/sf_2026-11-25_0457KST_3205_hik.bin'])
    cs = sorted(G)
    cand = np.arange(65536)
    keys, text, margin = {}, {}, {}
    for c in cs:
        du, i = G[c]
        t = np.array([du[i + p, 698:714] for p in range(3)])
        w = t[:, 0::2].astype(int) | (t[:, 1::2].astype(int) << 8)          # 3 x 8 code units
        key = np.zeros(8, int); mg = np.zeros(8)
        for j in range(8):
            sc = L[cand] + L[cand ^ w[0, j] ^ w[1, j]] + L[cand ^ w[0, j] ^ w[2, j]]
            a, b = np.argsort(sc)[::-1][:2]
            key[j] = w[0, j] ^ a; mg[j] = sc[a] - sc[b]
        keys[c] = np.array([[k & 255, k >> 8] for k in key], np.uint8).ravel()
        margin[c] = np.repeat(mg, 2)
        text[c] = ''.join(chr(v) if not 0xD800 <= v < 0xE000 else 'Fffd' for v in (w ^ key).ravel())

    # ---- independent check on the audio
    r, _ = records('../reference/sf_signon_3205_2026-03-16.bin'); du, good, _, _ = integrity(r)
    M = np.ones((16, 16, 256)) * 0.5
    for i in groups(du, good):
        ks = du[i, 698:714] ^ 0xFF; F = frames_of(du, i)
        for k in range(16):
            cls = np.repeat([du[i + q, 2] >> 4 for q in range(3)], 5)
            np.add.at(M[k], (cls, F[k // 11::4, 3 * (k % 11)] ^ ks[k]), 1)
    M = np.log2(M / M.sum(2, keepdims=True) * 256)
    rng = np.random.default_rng(1)
    C = {}
    for c in cs:
        du, i = G[c]; F = frames_of(du, i)
        cls = np.repeat([du[i + q, 2] >> 4 for q in range(3)], 5)
        C[c] = (np.stack([F[k // 11::4, 3 * (k % 11)] for k in range(16)]), cls)

    def score(keyof):
        s = np.zeros((len(cs), 16))
        for n, c in enumerate(cs):
            x, cls = C[c]
            for k in range(16):
                s[n, k] = M[k][cls, x[k] ^ keyof(n, c)[k]].mean()
        return s
    true = score(lambda n, c: keys[c])
    null = np.stack([score(lambda n, c, o=o: keys[cs[(n + o) % len(cs)]])
                     for o in rng.integers(20, len(cs) - 20, 8)])
    print('audio check, bits/sample over uniform (March-model); recovered key vs keys of unrelated groups')
    for k in range(16):
        m = null.mean(1)[:, k]
        z = (true[:, k].mean() - m.mean()) / (true[:, k].std() / np.sqrt(len(cs)))
        print('  k=%2d col %2d role %d   %+.3f   null %+.3f   z=%+.1f' % (k, 3 * (k % 11), k // 11,
                                                                       true[:, k].mean(), m.mean(), z))
    # high-confidence columns only
    mg = np.stack([margin[c] for c in cs])
    hi = mg > np.median(mg)
    for k in (0, 1, 10, 11):
        d = true[:, k] - null.mean(0)[:, k]
        print('  k=%2d gain: high-margin %+.3f  low-margin %+.3f' % (k, d[hi[:, k]].mean(), d[~hi[:, k]].mean()))
    with open('../reference/sidechannel_text_2026-11-25_unigram.txt', 'w', encoding='utf-8') as f:
        for c in cs:
            f.write('0x%08X  %s\n' % (c, text[c].replace('\r', '\\r').replace('\n', '\\n')))
    for c in cs[:60]:
        print(c - cs[0], repr(text[c]))


if __name__ == '__main__':
    main()
