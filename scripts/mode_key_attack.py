"""Evaluate mode-conditioned byte likelihoods on a time-ordered holdout.

Learns only from role-0 plaintext supported by exact March placeholder text and
CRC. Predicts XOR keys from fifteen frames/group. Byte predictions are not
verified keys; held-out role-0 and known role-1 prefix accuracy are reported.
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYED = np.arange(0, 32, 3)


def fit(modes, plain):
    table = np.full((256, 11, 256), 0.5, dtype=np.float64)
    for j in range(11):
        np.add.at(table[:, j, :], (modes.ravel(), plain[:, :, j].ravel()), 1)
    seen = np.bincount(modes.ravel(), minlength=256)
    global_counts = table.sum(axis=0)
    table[seen < 30] = global_counts
    return np.log(table / table.sum(axis=2, keepdims=True))


def predict(table, modes, cipher):
    result = np.empty((len(modes), 11), dtype=np.uint8)
    margins = np.empty((len(modes), 11), dtype=np.float32)
    candidates = np.arange(256, dtype=np.uint8)
    for lo in range(0, len(modes), 64):
        hi = lo + 64
        for j in range(11):
            pl = cipher[lo:hi, :, j, None] ^ candidates
            scores = table[modes[lo:hi, :, None], j, pl].sum(axis=1)
            result[lo:hi, j] = scores.argmax(axis=1)
            best = np.partition(scores, -2, axis=1)[:, -2:]
            margins[lo:hi, j] = best[:, 1] - best[:, 0]
    return result, margins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('corrected_corpus')
    ap.add_argument('--out', required=True)
    ap.add_argument('--old-corpus')
    args = ap.parse_args()
    d = np.load(args.corrected_corpus)
    if not d['idle_text_crc_ok'].all():
        ap.error('all input keys must pass idle text CRC')
    raw, plain, mode = d['packets_raw'], d['packets_dec'], d['mode']
    keys = d['ks0_15']
    split = int(0.7 * len(raw))
    raw0 = raw[:, 0::4][:, :, KEYED]
    plain0 = plain[:, 0::4][:, :, KEYED]
    table = fit(mode[:split, 0::4], plain0[:split])
    pred, margins = predict(table, mode[split:, 0::4], raw0[split:])
    eq = pred == keys[split:, :11]
    role1, _ = predict(table, mode[split:, 1::4], raw[split:, 1::4][:, :, KEYED])
    eq1 = role1[:, :5] == keys[split:, 11:16]
    report = dict(training_groups=split, heldout_groups=len(raw)-split,
                  split='first 70% counters train, final 30% test; no shuffle',
                  byte_positions=KEYED.tolist(),
                  role0_key_byte_accuracy=eq.mean(axis=0).tolist(),
                  role0_full_key_exact=int(eq.all(axis=1).sum()),
                  role1_known_prefix_byte_accuracy=eq1.mean(axis=0).tolist(),
                  role1_known_prefix_exact=int(eq1.all(axis=1).sum()))
    if args.old_corpus:
        old = np.load(args.old_corpus)
        if not np.array_equal(old['counter'], d['counter']):
            ap.error('old corpus counter order differs')
        om = old['mode']
        ot = fit(om[:split, 0::4], plain0[:split])
        op, _ = predict(ot, om[split:, 0::4], raw0[split:])
        report['old_bad_mode_role0_accuracy'] = (op == keys[split:, :11]).mean(axis=0).tolist()
    # Proposals for all groups use the same training-only model, not a refit.
    candidate_keys = np.zeros((len(raw), 44), np.uint8)
    verified = np.zeros_like(candidate_keys, dtype=bool)
    candidate_keys[:, :16] = keys
    verified[:, :16] = True  # March exact text + CRC; audio mapping remains a model.
    candidate_margin = np.zeros_like(candidate_keys, dtype=np.float32)
    for role in (1, 2, 3):
        pr, ma = predict(table, mode[:, role::4], raw[:, role::4][:, :, KEYED])
        start = 5 if role == 1 else 0
        candidate_keys[:, 11*role+start:11*role+11] = pr[:, start:]
        candidate_margin[:, 11*role+start:11*role+11] = ma[:, start:]
    report['note'] = 'Unknown-key outputs are statistical proposals, not recovered or verified keys.'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'candidate_keys.npz', counters=d['counter'],
                        ks=candidate_keys, verified=verified, likelihood_margin=candidate_margin)
    (out / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
