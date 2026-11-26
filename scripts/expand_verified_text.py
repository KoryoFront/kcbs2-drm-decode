"""Expand CRC-supported text cribs across the November data-unit corpus.

Uses only previously CRC-valid 92-byte text bodies as cribs. Requires at least
24 redundant text bytes to agree under the repeated group key before testing a
candidate. Headers and CRCs are not guessed or searched. Only complete 96-byte
segments passing the header and CRC checks contribute new supported key bytes.
Neither CRC16 nor this model authenticates a message or identifies the codec.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tail_fill_audit import crc


def key_id(tail, column):
    return tail//3, column


def propose(start, body, tails, known):
    derived = {}
    text_observations = 0
    text_nodes = set()
    # Prior and following fill may anchor columns absent from the text fragment.
    observations = []
    for offset, value in enumerate(body, 2):
        tail, col = start+offset//16, offset % 16
        if tail in tails:
            observations.append((tail, col, value, True))
    for tail in (start-1, start+6):
        if tail in tails:
            observations.extend((tail, col, 0xbb, False) for col in range(16))
    for tail, col, value, is_text in observations:
        node = key_id(tail, col)
        kv = int(tails[tail][col]) ^ value
        if node in derived and derived[node] != kv:
            return None
        if node in known and known[node] != kv:
            return None
        derived[node] = kv
        if is_text:
            text_observations += 1
            text_nodes.add(node)
    redundancy = text_observations-len(text_nodes)
    if redundancy < 24:
        return None
    # Never substitute the crib for missing captured bytes, including CRC bytes.
    if not all(start+j in tails for j in range(6)):
        return dict(start=start, status='missing_data', redundancy=redundancy)
    plain = bytearray()
    for offset in range(96):
        tail, col = start+offset//16, offset % 16
        node = key_id(tail, col)
        value = derived.get(node, known.get(node))
        if value is None:
            return dict(start=start, status='missing_key', redundancy=redundancy)
        plain.append(int(tails[tail][col]) ^ value)
    if plain[0] not in (5, 0x85) or plain[1] & 15 != 12:
        return dict(start=start, status='bad_header', redundancy=redundancy)
    if crc(plain[:94]) != int.from_bytes(plain[94:96], 'big'):
        return dict(start=start, status='bad_crc', redundancy=redundancy)
    if bytes(plain[2:94]) != body:
        raise AssertionError('reconstruction disagrees with crib')
    supported = {}
    for offset in range(96):
        tail, col = start+offset//16, offset % 16
        supported[key_id(tail, col)] = int(tails[tail][col]) ^ plain[offset]
    return dict(start=start, status='crc_supported', redundancy=redundancy,
                plaintext_hex=plain.hex(), text=body.decode('utf-16le', errors='replace'),
                supported=supported)


def expand(units_path, keys_path, segments_path, out):
    with np.load(units_path) as data:
        tails = {3*int(c)+int(p): du[698:714].copy()
                 for c,p,du in zip(data['counters'], data['phases'], data['du'])}
    with np.load(keys_path) as data:
        known = {(int(c), j): int(k[j]) for c,k,m in zip(data['counters'], data['ks'], data['crc_supported'])
                 for j in range(16) if m[j]}
    initial = len(known)
    rows = json.loads(Path(segments_path).read_text(encoding='utf-8'))
    bodies = set()
    for row in rows:
        plain = bytes.fromhex(row['plaintext_hex'])
        if crc(plain[:94]) != int.from_bytes(plain[94:96], 'big'):
            raise ValueError('input crib does not pass CRC')
        bodies.add(plain[2:94])
    # Every possible tail-aligned start touching at least one recorded text tail.
    starts = sorted({t-j for t in tails for j in range(6)})
    accepted = {}
    stats = {}
    iterations = 0
    for iteration in range(3):
        iterations += 1
        before = len(known)
        for start in starts:
            if start in accepted:
                continue
            for body in sorted(bodies):
                proposal = propose(start, body, tails, known)
                if proposal is None:
                    continue
                stats[proposal['status']] = stats.get(proposal['status'], 0)+1
                if proposal['status'] != 'crc_supported':
                    continue
                supported = proposal.pop('supported')
                if any(node in known and known[node] != value for node,value in supported.items()):
                    raise AssertionError('CRC-supported key conflict')
                known.update(supported)
                accepted[start] = proposal
        if len(known) == before:
            break
    counters = np.array(sorted({t//3 for t in tails}), np.uint32)
    ks = np.zeros((len(counters), 16), np.uint8)
    mask = np.zeros_like(ks, bool)
    for i,c in enumerate(counters):
        for j in range(16):
            if (int(c), j) in known:
                ks[i,j] = known[int(c), j]
                mask[i,j] = True
    report = dict(crib_text_bodies=len(bodies), candidate_starts=len(starts), iterations=iterations,
                  initial_crc_supported_key_bytes=initial, crc_supported_key_bytes=int(mask.sum()),
                  complete_key_groups=int(mask.all(axis=1).sum()),
                  role0_key_groups=int(mask[:, :11].all(axis=1).sum()),
                  crc_supported_segment_instances=len(accepted), tested_status_counts=stats,
                  note='No header brute force, CRC fitting, or language-model completion. '
                       'Only captured complete CRC-valid segments extend supported keys.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'keys.npz', counters=counters, ks=ks, crc_supported=mask, verified=mask)
    (out/'segments.json').write_text(json.dumps(list(accepted.values()), ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('units_npz'); ap.add_argument('keys_npz'); ap.add_argument('segments_json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    expand(a.units_npz, a.keys_npz, a.segments_json, a.out)
