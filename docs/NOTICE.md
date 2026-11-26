# Third-party notices

## FAAD2 (patches/faad2/, reference/faad2_stock.dll)

`patches/faad2/` contains modified source files from **FAAD2 2.8.8**
(https://www.audiocoding.com/faad2.html), licensed under the **GNU General
Public License v2** (see `COPYING.FAAD2`, copied verbatim from the FAAD2
distribution).

- `syntax.c.pristine` is the unmodified 2.8.8 source, kept as a diff base.
- `syntax.c.diff` is the patch actually applied (conformance checks bypassed
  to let FAAD2 attempt non-conformant DRM AAC — see RUNBOOK.md §4.4/§6 for
  why).
- `drm_dec.c`, `lt_predict.c`, `decoder.c` carry the same kind of
  conformance-bypass edits; no clean pristine copy of these was kept
  alongside the project, so they are shipped as full patched files rather
  than diffs.
- `reference/faad2_stock.dll` is a **conformance-checking** build (patches
  reverted) used as the validated "does this look like real AAC" oracle —
  see RUNBOOK.md §on the oracle test. It is a compiled FAAD2 binary and is
  therefore itself GPLv2-licensed; this NOTICE and `COPYING.FAAD2` must
  travel with it.

Any redistribution of these files, or of binaries built from them
(`faad2_drm.dll`, `faad2_stock.dll`), must preserve this notice and the GPLv2
license text.

## Dream DRM receiver (patches/dream/)

`patches/dream/` contains modified source files from the **Dream DRM
receiver** project (https://sourceforge.net/projects/drm/), also licensed
under GPLv2. The same terms in `COPYING.FAAD2` apply (Dream ships an
identical GPLv2 license); no separate copy is included here to avoid
duplication.
