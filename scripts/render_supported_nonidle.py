"""Render non-idle excerpts or a counter-aligned timeline from supplied prefixes."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import wave

import numpy as np

import render_continuous as rc
from kcbs_decode_fast import (assemble_frames, attenuate, decode_evs, group_runs,
                              render_timeline, replay_nearby)

RATE = 32000


def load_keymap(keys):
    """Confirmed 16-byte prefixes from a keys.npz, keyed by counter."""
    with np.load(keys) as f:
        if 'crc_supported' not in f or f['crc_supported'].dtype != np.bool_:
            raise ValueError('expected a per-byte CRC-supported mask')
        mask, ks, key_c = f['crc_supported'], f['ks'], f['counters'].astype(np.int64)
    if (mask.ndim != 2 or mask.shape != ks.shape or mask.shape[1] != 16
            or ks.dtype != np.uint8 or len(mask) != len(key_c)):
        raise ValueError('expected a 16-byte key and support mask for each counter')
    full = mask.all(axis=1)
    if len(np.unique(key_c[full])) != int(full.sum()):
        raise ValueError('ambiguous verified counters')
    return {int(c): k for c, k, m in zip(key_c, ks, full) if m}


def assemble_partial(data_units, keymap, first, span):
    """Role-0 (and donor role-1) frames from whatever phases each keyed group received.

    Unlike the complete-group path, a group contributes the role-0 audio of every phase
    it received even when the full three-phase data unit is incomplete; the missing
    phases' frames stay unsupplied for the loss policy to handle. Returns the frame
    array, the supplied mask, and the counters that contributed at least one phase.
    """
    with np.load(data_units) as d:
        du_c = d['counters'].astype(np.int64)
        du_p = d['phases'].astype(np.int64)
        du = d['du']
    if du.ndim != 2 or du.shape[1] != 716 or du.dtype != np.uint8:
        raise ValueError('expected per-phase 716-byte data units')
    by_counter = {}
    for c, p, row in zip(du_c, du_p, du):
        by_counter.setdefault(int(c), {})[int(p)] = row
    frames = np.zeros((span * 60, 33), np.uint8)
    supplied = np.zeros(span * 60, bool)
    rows0 = np.arange(0, 60, 4)
    rows1 = rows0 + 1
    unknown = rc.KEYED[5:]
    filled = []
    for c, phases in by_counter.items():
        if c not in keymap or not (0 <= c - first < span):
            continue
        base = (c - first) * 60
        du_row = np.zeros((3, 716), np.uint8)
        for p, row in phases.items():
            du_row[p] = row
        modes, payload = rc.group_frames(du_row, keymap[c])
        payload[np.ix_(rows1, unknown)] = payload[np.ix_(rows0, unknown)]
        frames[base:base + 60, 0] = modes
        frames[base:base + 60, 1:] = payload
        for p in phases:
            in_phase = (rows0 >= p * 20) & (rows0 < p * 20 + 20)
            supplied[base + rows0[in_phase]] = True
            supplied[base + rows1[in_phase]] = True
        filled.append(c)
    return frames, supplied, np.array(sorted(filled), np.int64)


def join_supported(groups, keys):
    with np.load(keys) as f:
        if 'crc_supported' not in f or f['crc_supported'].dtype != np.bool_:
            raise ValueError('expected a per-byte CRC-supported mask')
        mask, ks, key_c = f['crc_supported'], f['ks'], f['counters']
    if (mask.ndim != 2 or mask.shape != ks.shape or mask.shape[1] != 16
            or ks.dtype != np.uint8 or len(mask) != len(key_c)):
        raise ValueError('expected a 16-byte key and support mask for each counter')
    with np.load(groups) as f:
        counters, du = f['counters'].astype(np.int64), f['du']
    if np.unique(counters).size != len(counters) or du.shape != (len(counters), 3, 716):
        raise ValueError('expected one complete group per received counter')
    full = mask.all(axis=1)
    known_c = key_c[full].astype(np.int64)
    if len(np.unique(known_c)) != len(known_c):
        raise ValueError('ambiguous verified counters')
    shared, group_i, key_i = np.intersect1d(counters, known_c, return_indices=True)
    return shared, ks[full][key_i], du[group_i], counters


def with_loss_policy(frames, supplied, policy):
    if policy == 'plc':
        return frames, supplied
    if policy == 'replay':
        return replay_nearby(frames, supplied)
    if policy != 'future':
        raise ValueError(f'unknown loss policy: {policy}')
    out, keep = frames.copy(), supplied.copy()
    ends = np.arange(3,len(frames)-1,4)
    ends = ends[supplied[ends-3] & supplied[ends+1] & ~supplied[ends]]
    out[ends] = frames[ends+1]
    keep[ends] = True
    return out, keep


def render(groups, keys, out, *, timeline=False, merge_gap=4, policy='replay', attenuation=0,
           start_counter=None, end_counter=None, data_units=None):
    started = time.monotonic()
    frame_samples = RATE//50
    prebuilt = None
    if data_units is not None:
        if not timeline:
            raise ValueError('--data-units partial rendering requires --timeline')
        keymap = load_keymap(keys)
        with np.load(data_units) as d:
            received_all = np.unique(d['counters'].astype(np.int64))
        if not len(received_all):
            raise ValueError('no received groups in the data units')
        low, high = int(received_all.min()), int(received_all.max())
        first = low if start_counter is None else start_counter
        last = high if end_counter is None else end_counter
        if not low <= first <= last <= high:
            raise ValueError('counter slice must be within the recorded span')
        span = last-first+1
        if span*60*frame_samples*2 > 512*1024**2:
            raise ValueError('split this capture into shorter outputs to stay below 512 MiB of PCM')
        frames, supplied, filled = assemble_partial(data_units, keymap, first, span)
        available = received_all[(received_all >= first) & (received_all <= last)]
        counters = filled[(filled >= first) & (filled <= last)]
        if not len(counters):
            raise ValueError('no keyed groups with a received phase in selected span')
        offsets = counters-first
        prebuilt = (frames, supplied)
    else:
        counters, prefix, du, available = join_supported(groups, keys)
        if not len(available):
            raise ValueError('no complete received groups')
        low, high = int(available.min()), int(available.max())
        first = low if start_counter is None else start_counter
        last = high if end_counter is None else end_counter
        if not low <= first <= last <= high:
            raise ValueError('counter slice must be within the recorded span')
        use = (counters >= first) & (counters <= last)
        counters, prefix, du = counters[use], prefix[use], du[use]
        available = available[(available >= first) & (available <= last)]
        if not len(counters):
            raise ValueError('no received groups with a full CRC-supported prefix in selected span')
        span = last-first+1
        output_groups = span if timeline else len(counters)
        if output_groups*60*frame_samples*2 > 512*1024**2:
            raise ValueError('split this capture into shorter outputs to stay below 512 MiB of PCM')
        offsets = counters-first
    runs = group_runs(offsets, merge_gap if timeline else 0)
    parts, pieces = [], []
    cursor = 0
    out.mkdir(parents=True, exist_ok=False)
    kind = 'TIMELINE' if timeline else 'EXCERPTS'
    path = out/f'NONIDLE_CRC_PREFIX_UNVERIFIED_{policy.upper()}_{kind}_32k.wav'
    with tempfile.TemporaryDirectory(dir=out) as td:
        tmp = Path(td)
        if timeline:
            if prebuilt is None:
                frames, supplied = assemble_frames(du, prefix, offsets, span)
            else:
                frames, supplied = prebuilt
            frames, supplied = with_loss_policy(frames, supplied, policy)
            pcm = render_timeline(frames, supplied, runs, RATE,
                                  lambda f, s: decode_evs(f, s, RATE, tmp))
            pieces = [pcm]
        else:
            ramp = np.linspace(0,1,frame_samples)
            separator = np.zeros(RATE//2, dtype='<i2')
            for lo, hi in runs:
                use = (offsets >= lo) & (offsets < hi)
                frames, supplied = assemble_frames(du[use], prefix[use], offsets[use]-lo, hi-lo)
                frames, supplied = with_loss_policy(frames, supplied, policy)
                pcm = decode_evs(frames, supplied, RATE, tmp)
                pcm[:frame_samples] = np.rint(pcm[:frame_samples]*ramp)
                pcm[-frame_samples:] = np.rint(pcm[-frame_samples:]*ramp[::-1])
                parts.append(dict(first_counter=first+lo, last_counter=first+hi-1,
                                  counter_span_seconds=round(lo*1.2,3),
                                  montage_seconds=round(cursor/RATE,3), duration_seconds=len(pcm)/RATE))
                pieces.extend((pcm, separator))
                cursor += len(pcm)+len(separator)
            pieces.pop()
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1,2,RATE,0,'NONE','not compressed'))
        for piece in pieces:
            wav.writeframesraw(attenuate(piece, attenuation).tobytes())
    silences = []
    if timeline:
        boundaries = [(0,runs[0][0]), *[(hi,lo) for (_,hi),(lo,_) in zip(runs,runs[1:])],
                      (runs[-1][1],span)]
        silences = [dict(start_seconds=round(lo*1.2,3), duration_seconds=round((hi-lo)*1.2,3))
                    for lo,hi in boundaries if hi > lo]
    report = dict(file=path.name, first_counter=first, last_counter=last,
                  timeline=timeline, received_groups=len(available), supported_groups=len(counters),
                  supported_seconds=round(len(counters)*1.2,2),
                  output_seconds=sum(len(p) for p in pieces)/RATE, playback_method=policy,
                  attenuation_db=attenuation, decoder_calls=len(runs),
                  wall_seconds=round(time.monotonic()-started,2), silent_gaps=silences,
                  partial_phase=data_units is not None,
                  separator_seconds=0 if timeline else .5, excerpts=parts,
                  note='CRC supports text-tail prefixes, not audio identity. Missing slots use loss concealment or replay.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('silent_gaps','excerpts')},indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for opt in ('keys', 'out'):
        ap.add_argument('--'+opt, required=True, type=Path)
    ap.add_argument('--groups', type=Path,
                    help='complete-group groups.npz (omit when using --data-units)')
    ap.add_argument('--data-units', type=Path,
                    help='per-phase data_units.npz: render role-0 audio from every received '
                         'phase of a keyed group, not only complete three-phase groups (timeline only)')
    ap.add_argument('--timeline', action='store_true', help='preserve the entire capture clock')
    ap.add_argument('--merge-gap', type=int, default=4, help='timeline: conceal up to N missing groups')
    ap.add_argument('--loss-policy', choices=('replay','future','plc'), default='replay')
    ap.add_argument('--attenuate-db', type=float, default=0)
    ap.add_argument('--start-counter', type=int, help='include this counter as the first slot of the render')
    ap.add_argument('--end-counter', type=int, help='include this counter as the last slot of the render')
    a = ap.parse_args()
    if a.merge_gap < 0 or not 0 <= a.attenuate_db <= 60:
        ap.error('provide nonnegative merge-gap and attenuation between 0 and 60')
    if (a.groups is None) == (a.data_units is None):
        ap.error('provide exactly one of --groups or --data-units')
    render(a.groups, a.keys, a.out, timeline=a.timeline, merge_gap=a.merge_gap,
           policy=a.loss_policy, attenuation=a.attenuate_db,
           start_counter=a.start_counter, end_counter=a.end_counter, data_units=a.data_units)
