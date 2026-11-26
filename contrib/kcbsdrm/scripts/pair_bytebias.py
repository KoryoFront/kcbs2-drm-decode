"""Per-bit P(1) of every frame byte, raw and decrypted, on the March sign-on (idle-seed keys, CRC-checked).

    python scripts/pair_bytebias.py        (RUNBOOK 0c.6; read-only)
"""
import os, sys, numpy as np
from pair_analyze import *
r,_ = records(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference', 'sf_signon_3205_2026-03-16.bin'))
r = trim_loop(r)[0]
du, good, pk, dg = integrity(r)
raw=[[],[],[],[]]; dec=[[],[]]; n=0
for i in groups(du, good):
    t0 = du[i,698:714]; ks = t0 ^ 0xFF
    s = bytes(du[i+1,698:714]^ks)+bytes(du[i+2,698:714]^ks)
    if crc16(s[:27]) != int.from_bytes(s[27:29],'big'): continue
    n+=1; f = frames_of(du,i)
    for role in range(4): raw[role].append(f[role::4])
    for role in (0,1):
        x = f[role::4].copy()
        for j in range(11):
            k = 11*role+j
            if k<16: x[:,3*j] ^= ks[k]
        dec[role].append(x)
print('verified groups', n)
for role in (0,1):
    A=np.unpackbits(np.concatenate(raw[role]),axis=1).mean(0).reshape(32,8)
    B=np.unpackbits(np.concatenate(dec[role]),axis=1).mean(0).reshape(32,8)
    print('role',role,'frames',len(np.concatenate(raw[role])))
    print('byte keyed | raw P(1) per bit | decrypted P(1) per bit (only keyed bytes with a known key)')
    for b in range(32):
        keyed = b%3==0; known = keyed and 11*role+b//3<16
        print('%2d %s | %s | %s'%(b,'K' if keyed else ' ',' '.join('%.2f'%v for v in A[b]), ' '.join('%.2f'%v for v in B[b]) if known else ''))
