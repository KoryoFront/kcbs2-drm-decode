"""Second-pass decoder for the November side-channel text: beam search over the 8 key columns of each
group with a character trigram model (hand-written forecast-style seed text + self-training).
Audio bytes remain unused by the decoder and are the independent check.

    PYTHONIOENCODING=utf-8 python pair_text_lm.py       (run from scripts/)
"""
import collections, math
import numpy as np
from pair_analyze import *
from pair_text import prior, load

SEED = """
1부 기상정보를 알려드리겠습니다.
11월 25일과 26일에 예견되는 날씨입니다.
25일에는 평양, 평성, 신의주, 해주, 사리원, 개성시에서 주로 맑겠으며 오후 한때 흐리겠습니다.
강계시에서는 새벽과 오전에 때때로 흐리다가 대체로 개이겠습니다.
원산, 함흥, 청진, 라선시에서 주로 맑겠으며 혜산시에서는 때때로 비나 눈이 내리겠으며
오후부터 점차 개이겠습니다.
평양의 낮최고기온은 3℃정도로 예견됩니다.
26일에는 전반적지방에서 대체로 흐리겠습니다. 오전부터 주로 개이겠습니다.
다음은 해상날씨입니다. 28일 전반해상에서 북서풍이 7~10㎧로 불고 물결높이는 0.5~1.0m로 일겠습니다.
29일 전반해상에서 남서풍이 10~15㎧로 세게 불고 물결은 1~2m로 일것으로 예견됩니다.
조선서해해상에서는 남서풍이 불다가 북서풍이 7~10㎧로 불겠습니다.
조선동해해상에서는 남풍이 불다가 북동풍이 12~15㎧로 세게 불겠습니다.
조선동해 먼바다에서는 센바람주의경보가 내렸습니다. 바다에서 주의해야 하겠습니다.
12월 1일부터 12월 6일까지의 날씨전망입니다. 아침최저기온 낮최고기온 비가 내리겠습니다. 눈이 내리겠습니다.
중앙기상수문국에서 알려드립니다. 농업부문에서는 대책을 세워야 하겠습니다.
"""
BOS = '\0'


class LM:
    def __init__(self, texts):
        self.u, self.b, self.t = collections.Counter(), collections.Counter(), collections.Counter()
        self.bc, self.tc = collections.Counter(), collections.Counter()
        for text, w in texts:
            s = ' '.join(text.split(' '))
            for i in range(len(s)):
                self.u[s[i]] += w
                if i:
                    self.b[s[i - 1:i + 1]] += w; self.bc[s[i - 1]] += w
                if i > 1:
                    self.t[s[i - 2:i + 1]] += w; self.tc[s[i - 2:i]] += w
        self.n = sum(self.u.values())
        self.base = np.exp(prior())
        self.cache = {}

    def lp(self, h, ch):
        k = (h, ch)
        v = self.cache.get(k)
        if v is None:
            p = 0.7 * self.base[ord(ch)] + 0.3 * self.u[ch] / self.n
            a = h[-1:]
            if self.bc[a]:
                p = (self.b[a + ch] + 1.5 * p) / (self.bc[a] + 1.5)
            if len(h) == 2 and self.tc[h]:
                p = (self.t[h + ch] + 1.0 * p) / (self.tc[h] + 1.0)
            v = self.cache[k] = math.log(max(p, 1e-12))
        return v


def decode(G, cs, lm, beam=60, ncand=60):
    L = prior()
    extra = np.zeros(65536)
    for ch, n in lm.u.items():
        extra[ord(ch)] = n / lm.n
    L = np.log(0.7 * np.exp(L) + 0.3 * extra + 1e-12)
    cand = np.arange(65536)
    keys, text, conf = {}, {}, {}
    prev = None
    for c in cs:
        du, i = G[c]
        t = np.array([du[i + p, 698:714] for p in range(3)])
        w = t[:, 0::2].astype(int) | (t[:, 1::2].astype(int) << 8)
        hist = text[c - 1][-2:] if c - 1 in text else BOS * 2
        beams = [(0.0, ('', '', ''), ())]
        for j in range(8):
            d1, d2 = int(w[0, j] ^ w[1, j]), int(w[0, j] ^ w[2, j])
            sc = L[cand] + L[cand ^ d1] + L[cand ^ d2]
            top = np.argpartition(sc, -ncand)[-ncand:]
            new = []
            for s, seg, ks in beams:
                for a in top.tolist():
                    tri = (a, a ^ d1, a ^ d2)
                    if any(0xD800 <= v < 0xE000 for v in tri):
                        continue
                    chs = [chr(v) for v in tri]
                    h = [hist if (p == 0 and j < 2) else '' for p in range(3)]
                    v = s
                    for p in range(3):
                        v += lm.lp((h[p] + seg[p])[-2:], chs[p])
                    new.append((v, tuple(seg[p] + chs[p] for p in range(3)), ks + (int(w[0, j]) ^ a,)))
            new.sort(key=lambda x: -x[0])
            beams = new[:beam]
        # score the two internal phase boundaries
        full = []
        for s, seg, ks in beams:
            x = ''.join(seg)
            for q in (8, 9, 16, 17):
                s += lm.lp(x[q - 2:q], x[q]) - lm.lp(x[q // 8 * 8:q][-2:], x[q])
            full.append((s, x, ks))
        full.sort(key=lambda x: -x[0])
        s, x, ks = full[0]
        keys[c] = np.array([[k & 255, k >> 8] for k in ks], np.uint8).ravel()
        text[c], conf[c] = x, s / 24
    return keys, text, conf


class AudioCheck:
    def __init__(self, G, cs):
        r, _ = records('../reference/sf_signon_3205_2026-03-16.bin'); du, good, _, _ = integrity(r)
        M = np.ones((16, 16, 256)) * 0.5
        for i in groups(du, good):
            ks = du[i, 698:714] ^ 0xFF; F = frames_of(du, i)
            cls = np.repeat([du[i + q, 2] >> 4 for q in range(3)], 5)
            for k in range(16):
                np.add.at(M[k], (cls, F[k // 11::4, 3 * (k % 11)] ^ ks[k]), 1)
        self.M = np.log2(M / M.sum(2, keepdims=True) * 256)
        self.cs = cs
        self.C = []
        for c in cs:
            du, i = G[c]; F = frames_of(du, i)
            self.C.append((np.stack([F[k // 11::4, 3 * (k % 11)] for k in range(16)]),
                           np.repeat([du[i + q, 2] >> 4 for q in range(3)], 5)))

    def score(self, keys, offset=0):
        n = len(self.cs)
        return np.array([[self.M[k][cls, x[k] ^ keys[self.cs[(m + offset) % n]][k]].mean() for k in range(16)]
                         for m, (x, cls) in enumerate(self.C)])

    def report(self, keys, label):
        t = self.score(keys)
        z = np.mean([self.score(keys, o) for o in (101, 233, 377, 520, 797)], axis=0)
        gain = (t - z).mean(0)
        print('%s: audio gain over unrelated-group keys, bits/sample: k0 %.3f k1 %.3f k2 %.3f k10 %.3f '
              'k11 %.3f k12 %.3f k13 %.3f | sum16 %.3f' % ((label,) + tuple(gain[[0, 1, 2, 10, 11, 12, 13]]) + (gain.sum(),)))


def main():
    G = load(['../reference/sf_2026-11-25_0448KST_3205_hik.bin', '../reference/sf_2026-11-25_0457KST_3205_hik.bin'])
    cs = sorted(G)
    chk = AudioCheck(G, cs)
    corpus = [(SEED, 4.0)]
    for it in range(4):
        keys, text, conf = decode(G, cs, LM(corpus))
        chk.report(keys, 'pass %d' % (it + 1))
        # agreement between the two carousel repetitions is automatic (same XORs); self-train on the best half
        med = np.median(list(conf.values()))
        corpus = [(SEED, 4.0)] + [(text[c], 0.6) for c in cs if conf[c] > med]
    with open('../reference/sidechannel_text_2026-11-25.txt', 'w', encoding='utf-8') as f:
        for c in cs:
            f.write('0x%08X %6.2f %s %s\n' % (c, conf[c], keys[c].tobytes().hex(),
                                             text[c].replace('\r', '\\r').replace('\n', '\\n')))
    for c in cs[:90]:
        print(c - cs[0], '%.1f' % conf[c], repr(text[c]))


if __name__ == '__main__':
    main()
