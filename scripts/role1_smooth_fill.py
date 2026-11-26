"""Choose role-1 substitute bytes for smoothness, then check it on genuine speech.

The pilot showed the true role-1 bytes are not identifiable: every value passes the
decoder and continuity does not favour the truth. So this does NOT recover keys. It
picks, per group, the six substitute bytes that minimise log-mel discontinuity around
role-1 frames - aimed at the audible frame-rate stutter.

Legitimacy check: on genuine encoded speech, where truth is known, does smoothing
also improve intelligibility (STOI/eSTOI) versus the confirmed role1_donor fill, or
is it merely smoother? Four excerpts choose, four confirm.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from build_evs_gcc_independent import linux
import role1_key_search as rks

_spec = importlib.util.spec_from_file_location('pfq', HERE/'partial_frame_quality.py')
pfq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pfq)

RATE = 16000
POSITIONS = rks.ROLE1_UNKNOWN


def smooth_fill(tmp, label, frames, log, rng):
    """Greedy, bit-order choice of the six role-1 bytes minimising roughness."""
    span = len(frames)
    slot = np.arange(span) % 4
    role1 = np.flatnonzero(slot == 1)
    keep = np.isin(slot, (0, 1))
    current = frames.copy()
    current[np.ix_(role1, POSITIONS)] = 0
    chosen = {}
    for position in POSITIONS:
        batch = np.repeat(current[None, :, :], 256, axis=0)
        batch[:, role1, position] = np.arange(256, dtype=np.uint8)[:, None]
        flags, pcm = rks.decode_batch(tmp, label + '_p' + str(position),
                                      batch.reshape(-1, 33), np.tile(keep, 256), log)
        rough = rks.roughness(pcm, span, role1)
        flagged = np.array([int((flags[s*span:(s+1)*span][role1] != 0).any(axis=1).sum())
                            for s in range(256)])
        best = int(np.lexsort((rough, flagged))[0])
        current[np.ix_(role1, [position])] = best
        chosen[int(position)] = best
    return current, keep, chosen


def _stoi(*args, **kwargs):
    from pystoi import stoi          # imported on use: only scoring needs it
    return stoi(*args, **kwargs)


def quality(truth, test):
    n = min(len(truth), len(test))
    a = truth[:n].astype(float)/32768
    b = test[:n].astype(float)/32768
    return dict(stoi=round(float(_stoi(a, b, RATE)), 4),
                estoi=round(float(_stoi(a, b, RATE, extended=True)), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--select', type=int, default=4)
    ap.add_argument('--holdout', type=int, default=4)
    a = ap.parse_args()
    out = (ROOT/a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    log, rows = [], []
    rng = np.random.default_rng(5150)
    rks.build_traced(log)
    with tempfile.TemporaryDirectory(prefix='smooth_', dir=out) as td:
        tmp = Path(td)
        for i in range(a.select + a.holdout):
            offset = 160 + i*85
            frames = rks.genuine_group(tmp, offset, log)
            span = len(frames)
            truth = pfq.decode(tmp, 'truth' + str(i), frames, np.ones(span, bool), log)
            row = dict(excerpt='ref_' + str(offset) + 's',
                       role='select' if i < a.select else 'holdout', scores={})
            for name in ('erase_all', 'role1_donor'):
                built, keep = pfq.apply_strategy(frames, name, rng, frames)
                row['scores'][name] = quality(truth, pfq.decode(tmp, 'c%d_%s' % (i, name),
                                                                built, keep, log))
            built, keep, chosen = smooth_fill(tmp, 'sm%d' % i, frames, log, rng)
            row['scores']['role1_smooth'] = quality(truth, pfq.decode(tmp, 'c%d_smooth' % i,
                                                                      built, keep, log))
            row['chosen_bytes'] = chosen
            rows.append(row)
            print(json.dumps(row), flush=True)

    def mean(group, name, metric):
        return float(np.mean([r['scores'][name][metric] for r in rows if r['role'] == group]))

    summary = {name: dict(select_stoi=round(mean('select', name, 'stoi'), 4),
                          select_estoi=round(mean('select', name, 'estoi'), 4),
                          holdout_stoi=round(mean('holdout', name, 'stoi'), 4),
                          holdout_estoi=round(mean('holdout', name, 'estoi'), 4))
               for name in ('erase_all', 'role1_donor', 'role1_smooth')}
    better = (summary['role1_smooth']['holdout_estoi'] > summary['role1_donor']['holdout_estoi'] and
              summary['role1_smooth']['holdout_stoi'] >= summary['role1_donor']['holdout_stoi'])
    summary['smooth_beats_donor_on_holdout'] = bool(better)
    summary['note'] = ('Smoothing does not recover keys. If it does not beat role1_donor on '
                       'intelligibility it may still reduce stutter, which is a listening choice.')
    (out/'report.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=2),
                                   encoding='utf-8')
    (out/'calls.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
