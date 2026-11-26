"""kcbs_tail_key.py -- recover part of the KCBS-2 audio keystream from the side channel's text message
and decrypt the keyed bytes it covers (2026-11-24, see KCBS_ALL_FINDINGS.md section 52).

Model (from the side-channel and cross-day analyses):
  - A 1.2 s group = 3 super-frames (du[1] phase 0,1,2) = 60 packets of 32 bytes.
  - Packet bytes 3j (j = 0..10) are XORed with keystream byte ks[11*r + j], r = packet index mod 4;
    ks is fixed for the whole group.
  - The 16-byte tail du[698:714] of each super-frame = ks[0..15] XOR text-stream plaintext.
  - March 2026-03-17: phase-0 tail plaintext is 16 x 0xFF; phases 1-2 carry the DRM-style segment
    68 f0 "TextMessage is sending..." + CRC-16-CCITT (init/xorout 0xFFFF) 9d ce + 00 00 00.
So ks[0..15] = tail(phase 0) XOR 0xFF, covering role 0 (all 11 keyed bytes) and role 1 (bytes 0..12).

Usage: python scripts/kcbs_tail_key.py <msc dump .bin> [out.npz]
Prints a validation: equality of decrypted bytes between packets in DIFFERENT groups (same role),
before vs after decryption, on the same pairs. Correct keys raise cross-group agreement to the
within-group level; wrong keys leave it at chance (~0.4%).
"""
import sys, os, struct
import numpy as np

ST = [s + 131*c for c in range(5) for s in (7, 40, 73, 106)]   # packet starts in the 716-byte du

def crc16(d, init=0xFFFF, xo=0xFFFF):
    c = init
    for b in d:
        c ^= b << 8
        for _ in range(8): c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c ^ xo

def load_du(path):
    recs = []
    with open(path, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4: break
            n = struct.unpack("<i", h)[0]; d = f.read(n)
            if len(d) < n: break
            if n == 728: recs.append(d)
    seen = {}                                   # Dream loops its input: keep the first pass
    for i, r in enumerate(recs):
        if r in seen and i - seen[r] > 20: recs = recs[:i]; break
        seen.setdefault(r, i)
    sf = np.frombuffer(b"".join(recs), np.uint8).reshape(len(recs), 728)
    du = np.concatenate([sf[:, q*182+1:q*182+180] for q in range(4)], 1)
    ok = np.array([all(crc16(sf[i, q*182:q*182+180].tobytes(), xo=0xFFFF) == (int(sf[i, q*182+180]) << 8 | int(sf[i, q*182+181]))
                       for q in range(4)) and crc16(du[i, :714].tobytes()) == (int(du[i, 714]) << 8 | int(du[i, 715]))
                   for i in range(len(du))])
    return du, ok

def groups(du, ok):
    ph = du[:, 1] >> 4
    return np.array([i for i in range(len(du) - 2)
                     if ph[i] == 0 and ph[i+1] == 1 and ph[i+2] == 2 and ok[i] and ok[i+1] and ok[i+2]])

def main():
    du, ok = load_du(sys.argv[1]); g = groups(du, ok); n = len(g)
    P = np.stack([np.stack([du[g + p, s:s+32] for s in ST], 1) for p in range(3)], 1).reshape(n, 60, 32)
    T = du[g][:, [698 + k for k in range(16)]]
    # text check: tails of phases 1,2 decrypted with the phase-0 key
    ks = T ^ 0xFF
    t12 = np.concatenate([du[g+1, 698:714] ^ ks, du[g+2, 698:714] ^ ks], 1)
    crc_ok = np.array([crc16(r[:27].tobytes()) == (int(r[27]) << 8 | int(r[28])) for r in t12])
    print(f"{n} groups; text segment CRC valid under 0xFF phase-0 hypothesis: {crc_ok.mean():.1%}")
    if crc_ok.any():
        print("  e.g.", t12[np.argmax(crc_ok)][:29].tobytes())
    D = P.copy()
    for r, js in ((0, range(11)), (1, range(5))):
        for j in js:
            D[:, r::4, 3*j] ^= ks[:, 11*r + j][:, None]
    # validation: same role, adjacent groups (packet slot s in group i vs slot s in group i+1)
    print("byte  role  equal-across-groups raw -> decrypted   (within-group, k vs k+4, raw)")
    for r, js in ((0, range(11)), (1, range(5))):
        for j in js:
            b = 3*j; sl = np.arange(r, 60, 4)
            raw = np.mean(P[:-1, sl, b] == P[1:, sl, b]); dec = np.mean(D[:-1, sl, b] == D[1:, sl, b])
            win = np.mean(P[:, sl[:-1], b] == P[:, sl[1:], b])
            print(f"{b:4d}  {r:4d}  {raw:8.3f} -> {dec:6.3f}            ({win:.3f})")
    if len(sys.argv) > 2:
        np.savez_compressed(sys.argv[2], packets_raw=P, packets_dec=D, keystream0_15=ks, group_du_index=g, text_crc_ok=crc_ok)
        print("wrote", sys.argv[2])

if __name__ == "__main__":
    main()
