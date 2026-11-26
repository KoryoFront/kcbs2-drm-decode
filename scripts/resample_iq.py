import sys, numpy as np, soundfile as sf
from scipy.signal import resample_poly
inp, out = sys.argv[1], sys.argv[2]
target = 48000
x, fs = sf.read(inp, always_2d=True)              # keeps I and Q
from math import gcd
g = gcd(int(fs), target)
y = resample_poly(x, target // g, int(fs) // g, axis=0)
sf.write(out, y, target, subtype="PCM_16")
print(f"{fs} Hz -> {target} Hz, {y.shape[0]} samples, wrote {out}")