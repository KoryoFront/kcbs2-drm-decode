"""Report phase/trailer structure and gap-aware mode edges in complete CRC-good groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from render_continuous import MODE_POSITIONS


FRAME_RATE = 50
GROUP_FRAMES = 60


def inspect(counters, du, from_mode, to_mode, min_lead_frames):
    counters = np.asarray(counters, dtype=np.int64)
    if not len(counters) or du.shape != (len(counters), 3, 716) or du.dtype != np.uint8:
        raise ValueError('expected complete (group, phase, byte) data units')
    if np.any(np.diff(counters) <= 0):
        raise ValueError('expected increasing unique counters')
    counter_fields = np.ascontiguousarray(du[:, 0, 664:668]).view('>u4').ravel()
    zero_fields = np.concatenate((du[:,0,662:664], du[:,1,662:665],
                                  du[:,2,662:664], du[:,2,697:698]), axis=1)
    trailer = np.concatenate((du[:,0,668:698], du[:,1,665:698], du[:,2,664:697]), axis=1)
    modes = du[:, :, MODE_POSITIONS].reshape(-1, GROUP_FRAMES)
    edges = []
    for run in np.split(np.arange(len(counters)), np.flatnonzero(np.diff(counters) != 1)+1):
        sequence = modes[run].ravel()
        transitions = np.flatnonzero(sequence[1:] != sequence[:-1])+1
        for start, stop in zip(np.r_[0, transitions], transitions):
            if sequence[start] == from_mode and sequence[stop] == to_mode and stop-start >= min_lead_frames:
                group, slot = divmod(int(stop), GROUP_FRAMES)
                counter = int(counters[run[group]])
                edges.append(dict(counter=counter, phase=slot//20, slot=slot%20,
                                  relative_seconds=round(((counter-int(counters[0]))*GROUP_FRAMES+slot)/FRAME_RATE, 2),
                                  lead_seconds=round(float((stop-start)/FRAME_RATE), 2)))
    return dict(first_counter=int(counters[0]), last_counter=int(counters[-1]), groups=len(counters),
                counter_fields_matching=int(np.count_nonzero(counter_fields == counters)),
                phase_layout_matching=int(np.count_nonzero(np.all(du[:,:,1] >> 4 == [0,1,2], axis=1))),
                zero_layout_matching=int(np.count_nonzero(~zero_fields.any(axis=1))),
                distinct_trailers=len(np.unique(trailer, axis=0)),
                trailer_one_fraction=float(np.unpackbits(trailer).mean()),
                mode_transition=[from_mode, to_mode], edges=edges,
                note='Frame markers and trailer statistics do not identify audio or a cipher.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('groups', type=Path)
    ap.add_argument('--from-mode', type=lambda s: int(s,0), default=0xfe)
    ap.add_argument('--to-mode', type=lambda s: int(s,0), default=0xf8)
    ap.add_argument('--min-lead-seconds', type=float, default=5)
    a = ap.parse_args()
    if not (0 <= a.from_mode <= 255 and 0 <= a.to_mode <= 255
            and np.isfinite(a.min_lead_seconds) and a.min_lead_seconds > 0):
        ap.error('use byte values for modes and a positive lead duration')
    with np.load(a.groups) as archive:
        result = inspect(archive['counters'], archive['du'], a.from_mode, a.to_mode,
                         int(np.ceil(a.min_lead_seconds*FRAME_RATE)))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
