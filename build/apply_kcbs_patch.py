#!/usr/bin/env python3
"""
Insert a raw-MSC dump into a Dream source tree.

Writes, at both the data-path and audio-path entry points, every logical frame the
receiver hands the decoder, in the format the KCBSDRM tooling already parses:
    <uint32 little-endian length> <length bytes> ...
i.e. exactly the layout of reference/sf_3205.bin.

Output paths come from the environment so nothing is hard-coded:
    KCBS_DUMP_MSC   data path  (CDataDecoder)        default ./kcbs_msc.bin
    KCBS_DUMP_AUD   audio path (CAudioSourceDecoder) default ./kcbs_aud.bin

KCBS-2 signals a *data* service, so the data path is the one that fires; the audio
hook is there in case a receiver build routes it differently.

Usage: python3 apply_kcbs_patch.py <dream-source-dir>
"""
import re, sys, os

SNIPPET = r'''
    /* ---- KCBS raw dump (inserted by apply_kcbs_patch.py) ---- */
    {
        static FILE* _kcbs_f = NULL;
        static int   _kcbs_init = 0;
        if (!_kcbs_init) {
            _kcbs_init = 1;
            const char* _p = getenv("%ENV%");
            _kcbs_f = fopen(_p ? _p : "%DEF%", "wb");
            fprintf(stderr, "[KCBS] dump %s -> %%s\n", _kcbs_f ? (_p ? _p : "%DEF%") : "FAILED");
        }
        if (_kcbs_f && pvecInputData) {
            int _nb = pvecInputData->Size() / 8;
            unsigned int _sz = (unsigned int) _nb;
            fwrite(&_sz, 4, 1, _kcbs_f);
            pvecInputData->ResetBitAccess();
            for (int _i = 0; _i < _nb; _i++) {
                unsigned char _b = 0;
                for (int _j = 0; _j < 8; _j++)
                    _b = (unsigned char)((_b << 1) | (((unsigned char) pvecInputData->Separate(1)) & 1));
                fwrite(&_b, 1, 1, _kcbs_f);
            }
            fflush(_kcbs_f);
            pvecInputData->ResetBitAccess();
        }
    }
    /* ---- end KCBS raw dump ---- */
'''

TARGETS = [
    ("src/datadecoding/DataDecoder.cpp",        "CDataDecoder::ProcessDataInternal",        "KCBS_DUMP_MSC", "kcbs_msc.bin"),
    ("src/sourcedecoders/AudioSourceDecoder.cpp","CAudioSourceDecoder::ProcessDataInternal", "KCBS_DUMP_AUD", "kcbs_aud.bin"),
]


def patch(path, func, env, default):
    src = open(path, encoding="utf-8", errors="surrogateescape").read()
    if "KCBS raw dump" in src:
        print(f"  {path}: already patched, skipping"); return True
    i = src.find(func)
    if i < 0:
        print(f"  {path}: FUNCTION {func} NOT FOUND"); return False
    # insert just after the DoNotProcessData guard that follows the function head
    m = re.compile(r"if\s*\(\s*DoNotProcessData[^)]*\)\s*(\{[^}]*?return\s*;[^}]*?\}|return\s*;)",
                   re.S).search(src, i)
    if not m:
        print(f"  {path}: DoNotProcessData guard not found after {func}"); return False
    at = m.end()
    body = SNIPPET.replace("%ENV%", env).replace("%DEF%", default)
    out = src[:at] + "\n" + body + src[at:]
    if "#include <cstdio>" not in out and "#include <stdio.h>" not in out:
        out = "#include <stdio.h>\n#include <stdlib.h>\n" + out
    open(path, "w", encoding="utf-8", errors="surrogateescape").write(out)
    print(f"  {path}: patched at offset {at} (env {env}, default {default})")
    return True


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "dream-src"
    ok = True
    print(f"patching {root}")
    for rel, func, env, default in TARGETS:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print(f"  {rel}: MISSING"); ok = False; continue
        ok &= patch(p, func, env, default)
    print("OK" if ok else "SOME TARGETS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
