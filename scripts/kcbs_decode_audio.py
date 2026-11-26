"""End to end: KiwiSDR IQ .wav -> KCBS-2 side-channel text keys -> decoded audio.

Stages, each resumable and logged:
  1 demod    IQ .wav -> DRM MSC dumps (patched Dream in WSL, bounded windows)
  2 salvage  packet/DU CRC checked groups and data units, deduplicated by counter
  3 keys     fill cribs + repeated-text XOR propagation, kept only when a full
             112-byte segment passes its header, fill and CRC checks
  4 text     the Korean side-channel text those keys decrypt, written to a .txt:
             CRC-valid segments first, then candidates that failed a check
  5 audio    role 0 (11 keyed bytes known) and role 1 (5 of 11 known) frames decoded
             as EVS 13.2 at 20 ms spacing; roles 2 and 3 stay explicit losses

What the output is NOT: role 1's remaining 6 keyed bytes are substitutes chosen by a
strategy validated on genuine speech, not recovered key material, and roles 2/3 are
concealed by the decoder. The EVS reading is a hypothesis supported by decoder syntax
checks, not a confirmed codec identification, so the audio is UNVERIFIED.

Requires: WSL Ubuntu with the patched Dream at /root/kcbs/dream-src, and the EVS
build from scripts/build_evs_gcc_independent.py.

Usage:
  python scripts/kcbs_decode_audio.py capture_iq.wav --out work/my_decode
  python scripts/kcbs_decode_audio.py a_iq.wav b_iq.wav --out work/my_decode --seconds 600
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from build_evs_gcc_independent import linux
from salvage_msc_packets import salvage
from repeated_text_keys import recover
import role1_key_search as rks
from role1_smooth_fill import smooth_fill
from kcbs_decode_fast import assemble_frames, decode_evs, group_runs, render_timeline
from render_supported_nonidle import join_supported

_spec = importlib.util.spec_from_file_location('pfq', HERE/'partial_frame_quality.py')
pfq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pfq)

MODE_POSITIONS = [(2 + c) if role == 0 else
                  ((39 if role == 1 else 72 if role == 2 else 105) + 131*c)
                  for c in range(5) for role in range(4)]
PACKET_STARTS = [(7 if role == 0 else 40 if role == 1 else 73 if role == 2 else 106) + 131*c
                 for c in range(5) for role in range(4)]
KEYED = np.arange(0, 32, 3)
EVS_DECODER = ROOT/'work/evs_gcc_independent/EVS_dec'


def check_tools():
    missing = []
    if not EVS_DECODER.exists():
        missing.append('EVS build missing: run scripts/build_evs_gcc_independent.py '
                       '--out work/evs_gcc_independent')
    probe = subprocess.run(['wsl', '-d', 'Ubuntu', '--', 'test', '-x',
                            '/root/kcbs/dream-src/dream'], capture_output=True)
    if probe.returncode:
        missing.append('patched Dream not executable at /root/kcbs/dream-src/dream in WSL Ubuntu')
    return missing


def stage_demod(inputs, out, seconds, start, log):
    """IQ -> MSC dumps in bounded windows, preserving source sample offsets."""
    dumps = []
    for number, path in enumerate(inputs):
        info = sf.info(path)
        if info.channels != 2 or abs(info.samplerate - 12000) > 3:
            raise SystemExit('expected two-channel ~12 kHz IQ: ' + str(path))
        total = min(seconds, info.duration - start) if seconds else info.duration - start
        if total <= 0:
            raise SystemExit('requested window is past the end of ' + str(path))
        window, overlap = 600.0, 30.0
        offset, index = float(start), 0
        while offset < start + total:
            length = min(window, start + total - offset)
            directory = out/('demod/source%02d_window%03d' % (number, index))
            marker = directory/'manifest.json'
            cached = None
            if marker.exists():
                cached = json.loads(marker.read_text(encoding='utf-8'))
                # A 'running' manifest is from an interrupted run, not a result.
                if cached.get('status') == 'running':
                    import shutil
                    shutil.rmtree(directory, ignore_errors=True)
                    cached = None
            if cached is not None:
                state = cached
            else:
                command = [sys.executable, str(HERE/'demod_iq_window.py'), str(path), str(directory),
                           '--start', str(offset), '--seconds', str(length)]
                began = time.monotonic()
                proc = subprocess.run(command, capture_output=True, text=True,
                                      encoding='utf-8', errors='replace')
                log.append(dict(stage='demod', command=command, returncode=proc.returncode,
                                wall_seconds=round(time.monotonic() - began, 1)))
                if not marker.exists():
                    raise RuntimeError('demod produced no manifest: ' + (proc.stderr or '')[-800:])
                state = json.loads(marker.read_text(encoding='utf-8'))
            dump = directory/'msc.bin'
            if state.get('dump_bytes', 0) > 0 and dump.exists():
                dumps.append(dump)
            print('  demod %s window %d: %s (%d bytes)'
                  % (Path(path).name, index, state.get('status'), state.get('dump_bytes', 0)),
                  flush=True)
            if length < window:
                break
            offset += window - overlap
            index += 1
    return dumps


def candidate_text(salvage_dir):
    """Decrypt tails using fill-crib candidate keys. Not CRC-verified."""
    groups = salvage_dir/'groups.npz'
    if not groups.exists():
        return []
    rows = []
    with np.load(groups) as d:
        if 'candidate_mask' not in d.files:
            return []
        usable = d['candidate_mask'].all(axis=1)
        for counter, tails, key in zip(d['counters'][usable], d['tails'][usable],
                                       d['candidate_ks'][usable]):
            body = (np.asarray(tails, np.uint8) ^ key).tobytes().decode('utf-16le', errors='replace')
            hangul = sum(1 for ch in body if '가' <= ch <= '힣')
            rows.append((int(counter), body, hangul))
    rows.sort(key=lambda r: r[0])
    return rows


def _one_line(text):
    """Keep one segment per line; the text itself contains CR/LF breaks."""
    return text.replace(chr(13), '').replace(chr(10), ' / ')


def stage_text(keys_dir, salvage_dir, out, log):
    """Write the decrypted side-channel text, separating CRC-valid from candidates."""
    text_dir = out/'text'
    text_dir.mkdir(parents=True, exist_ok=True)
    def load(name):
        path = keys_dir/name
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    verified = sorted(load('crc_supported_text.json'), key=lambda r: r['start'])
    tested = sorted(load('segments.json'), key=lambda r: r['start'])
    failed = [r for r in tested
              if not (r.get('header_ok') and r.get('fill_ok') and r.get('crc_ok'))]
    lines = ['KCBS-2 side-channel text (DRM text-message channel)',
             '',
             'Segments are 112 bytes: header, 46 UTF-16LE characters, CRC-16, then 0xBB fill.',
             'A counter is one 1.2 s group, so counter differences give elapsed time.',
             'The CRC-valid section below passed header, fill and CRC checks with no CRC',
             'fitting or language-model guessing. The candidate section did not pass and may',
             'contain wrong characters. A line break inside a segment is shown as " / ".',
             '',
             'CRC-VALID SEGMENTS (%d)' % len(verified),
             '=' * 72]
    for row in verified:
        counter, phase = divmod(int(row['start']), 3)
        lines.append('[counter %d phase %d] %s'
                     % (counter, phase, _one_line(row.get('text', ''))))
    seen, unique = set(), []
    for row in verified:
        body = _one_line(row.get('text', ''))
        if body not in seen:
            seen.add(body)
            unique.append(body)
    lines += ['', 'DISTINCT CRC-VALID TEXT BODIES (%d, first-seen order)' % len(unique),
              '=' * 72]
    lines += list(unique)
    lines += ['', 'CANDIDATE SEGMENTS THAT FAILED A CHECK (%d) - NOT VERIFIED' % len(failed),
              '=' * 72]
    for row in failed:
        counter, phase = divmod(int(row['start']), 3)
        checks = 'header=%s fill=%s crc=%s' % (row.get('header_ok'), row.get('fill_ok'),
                                               row.get('crc_ok'))
        raw = row.get('plaintext_hex')
        body = (bytes.fromhex(raw)[2:94].decode('utf-16le', errors='replace')
                if raw else '')
        lines.append('[counter %d phase %d] (%s) %s'
                     % (counter, phase, checks, _one_line(body)))
    validated = hangul_validated(salvage_dir/'groups.npz')
    validated_rows = []
    if validated is not None:
        all_counters, all_keys, good, scores = validated
        with np.load(salvage_dir/'groups.npz') as d:
            du = d['du']
        for i in np.flatnonzero(good):
            text = ''
            for p in range(3):
                plain = du[i, p, 698:714] ^ all_keys[i]
                text += plain.tobytes().decode('utf-16le', errors='replace')
            validated_rows.append((int(all_counters[i]), text, int(scores[i])))
    lines += ['', 'KOREAN-TEXT-VALIDATED GROUPS (%d) - keys confirmed by text quality' % len(validated_rows),
              '=' * 72,
              'Each key was checked against a random-key control: genuine keys decode 13-23 of',
              '24 characters as Hangul, random keys at most 12. Control measured on the same',
              'data, not assumed. Missing counters are groups without a recovered key.']
    for counter, text, score in validated_rows:
        lines.append('[counter %d] (%d/24) %s' % (counter, score, _one_line(text)))

    candidates = candidate_text(salvage_dir)
    lines += ['', 'TEXT FROM FILL-CRIB CANDIDATE KEYS (%d groups) - NOT VERIFIED' % len(candidates),
              '=' * 72,
              'These keys passed header and Korean-character gates only. Readable Korean here is',
              'suggestive but unproven; wrong characters are expected.']
    for counter, body, hangul in candidates:
        lines.append('[counter %d] (%d hangul) %s' % (counter, hangul, _one_line(body)))
    path = text_dir/'kcbs2_text.txt'
    path.write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
    log.append(dict(stage='text', file=str(path), crc_valid=len(verified), candidates=len(failed)))
    print('  text %s: %d CRC-valid segments, %d distinct bodies, %d unverified candidates'
          % (path.name, len(verified), len(unique), len(failed)), flush=True)
    return dict(text_file=str(path.relative_to(out)), crc_valid_text_segments=len(verified),
                distinct_text_bodies=len(unique), unverified_text_candidates=len(failed),
                candidate_key_text_groups=len(candidates),
                korean_validated_text_groups=len(validated_rows))


def hangul_validated(groups_npz):
    """Keys whose decrypted group text is Korean, against a measured random control.

    A wrong key yields uniform 16-bit values, which land in the Hangul syllable block
    with probability 0.171, so a 24-unit group scores about 4. Measured on this code
    path: genuine fill keys score 12-23 (mean 17.6), random keys 0-12 (mean 4.1).
    The threshold is placed above the random maximum.
    """
    if not Path(groups_npz).exists():
        return None
    with np.load(groups_npz) as d:
        if 'candidate_mask' not in d.files:
            return None
        counters = d['counters'].astype(np.int64)
        du = d['du']
        seeds = d['candidate_ks'].copy()
        usable = d['candidate_mask'].all(axis=1)
    tails = {(i, p): du[i, p, 698:714] for i in range(len(counters)) for p in range(3)}
    good = usable.copy()
    scores = np.zeros(len(counters), dtype=np.int16)
    for i in range(len(counters)):
        if not usable[i]:
            scores[i] = -1
            continue
        count = 0
        for p in range(3):
            plain = tails[(i, p)] ^ seeds[i]
            for m in range(8):
                unit = int(plain[2*m]) | int(plain[2*m+1]) << 8
                if 0xAC00 <= unit <= 0xD7A3:
                    count += 1
        scores[i] = count
        good[i] = count >= 13
    return counters, seeds, good, scores


def stage_audio(groups_npz, keys_npz, out, rate, fill, log, allow_candidates=False,
                montage=False):
    with np.load(keys_npz) as d:
        usable = d['crc_supported'][:, :16].all(axis=1)
        counters = d['counters'][usable].astype(np.int64)
        keys = d['ks'][usable, :16].copy()
    verified_groups = int(len(counters))
    key_source = 'crc_verified'
    if not verified_groups and allow_candidates:
        validated = hangul_validated(groups_npz)
        if validated is not None:
            all_counters, all_keys, good, scores = validated
            counters = all_counters[good]
            keys = all_keys[good]
            key_source = 'hangul_validated'
            print('  %d of %d candidate groups pass the Korean-text rule (>=13 of 24 '
                  'characters); using those for audio' % (len(counters),
                                                          int((scores >= 0).sum())), flush=True)
    if not len(counters):
        return [], dict(audio_key_groups=0, crc_verified_key_groups=0, key_source='none')
    with np.load(groups_npz) as d:
        index = np.searchsorted(d['counters'], counters)
        found = index < len(d['counters'])
        index = index[found]
        counters, keys = counters[found], keys[found]
        keep_rows = d['counters'][index] == counters
        index, counters, keys = index[keep_rows], counters[keep_rows], keys[keep_rows]
        du = d['du'][index].copy()
    runs = np.split(np.arange(len(counters)), np.flatnonzero(np.diff(counters) != 1) + 1)
    write_runs = len(runs) <= 12 or not montage
    rendered = []
    decoded = {}
    rng = np.random.default_rng(0)
    audio_dir = out/'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    for ids in runs:
        span = len(ids)*60
        frames = np.zeros((span, 33), np.uint8)
        flat = np.zeros((span, 32), np.uint8)
        slot = np.arange(span) % 4
        for k, g in enumerate(ids):
            for phase in range(3):
                row = du[g, phase]
                for j, (begin, mode_byte) in enumerate(zip(PACKET_STARTS, MODE_POSITIONS)):
                    i = k*60 + phase*20 + j
                    flat[i] = row[begin:begin+32]
                    frames[i, 0] = row[mode_byte]
            base = k*60
            rows0 = base + np.flatnonzero(slot[base:base+60] == 0)
            rows1 = base + np.flatnonzero(slot[base:base+60] == 1)
            flat[np.ix_(rows0, KEYED)] ^= keys[k, :11]
            flat[np.ix_(rows1, KEYED[:5])] ^= keys[k, 11:16]
        frames[:, 1:] = flat
        if fill == 'role1_smooth' and len(counters) > 60:
            # Per-group byte search costs 6 decoder passes per group; on long runs
            # fall back to the donor fill, which is one pass and nearly as good.
            print('  %d groups: using role1_donor instead of role1_smooth (cost)'
                  % len(counters), flush=True)
            built, supplied = pfq.apply_strategy(frames, 'role1_donor', rng, frames)
            effective_fill = 'role1_donor'
        elif fill == 'role1_smooth':
            # Substitutes are chosen per group, matching the per-group role-1 key.
            blocks, flags_keep = [], []
            with tempfile.TemporaryDirectory(prefix='smooth_', dir=out) as smooth_tmp:
                rks.build_traced(log)
                for k, g in enumerate(ids):
                    part, part_keep, _ = smooth_fill(Path(smooth_tmp), 'g%d' % counters[k],
                                                     frames[k*60:(k+1)*60], log, rng)
                    blocks.append(part)
                    flags_keep.append(part_keep)
            built = np.concatenate(blocks)
            supplied = np.concatenate(flags_keep)
            effective_fill = 'role1_smooth'
        elif fill != 'role1_smooth' or len(counters) <= 60:
            built, supplied = pfq.apply_strategy(frames, fill, rng, frames)
            effective_fill = fill
        counter_start_byte = int(counters[ids[0]])
        tag = str(counters[ids[0]]) + '-' + str(counters[ids[-1]])
        stream = audio_dir/('run_' + tag + '.192')
        raw = audio_dir/('run_' + tag + '.raw')
        with stream.open('wb') as f:
            for bits, good in zip(np.unpackbits(built, axis=1), supplied):
                if good:
                    f.write(np.array([0x6b21, 264], '<u2').tobytes())
                    f.write(np.where(bits, 0x81, 0x7f).astype('<u2').tobytes())
                else:
                    f.write(np.array([0x6b20, 0], '<u2').tobytes())
        command = ['wsl', '-d', 'Ubuntu', '--', linux(EVS_DECODER), '-q', '-no_delay_cmp',
                   str(rate//1000), linux(stream), linux(raw)]
        proc = subprocess.run(command, capture_output=True, timeout=1800)
        log.append(dict(stage='audio', command=command, returncode=proc.returncode))
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode(errors='replace')[-1200:])
        pcm = np.fromfile(raw, '<i2')
        label = 'KCBS2_' + ('UNVERIFIED_HANGULKEYS_' if key_source == 'hangul_validated' else
                            'UNVERIFIED_CANDIDATEKEYS_' if key_source != 'crc_verified' else
                            'UNVERIFIED_')
        path = audio_dir/(label + effective_fill + '_' + str(rate//1000) + 'k_' + tag + '.wav')
        decoded[counter_start_byte] = pcm
        if write_runs:
            with wave.open(str(path), 'wb') as w:
                w.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                w.writeframes(pcm.tobytes())
        rendered.append(dict(file=(str(path.relative_to(out)) if write_runs else None),
                             fill=effective_fill, counters=tag,
                             seconds=round(len(pcm)/rate, 2),
                             frames_supplied=int(supplied.sum()), frames_total=int(span),
                             rms=round(float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))), 5)))
        print('  run %s: %.1f s, %d/%d frames supplied%s'
              % (tag, len(pcm)/rate, supplied.sum(), span,
                 '' if write_runs else ' (montage only)'), flush=True)

    if montage and decoded:
        # One file in true counter order. Groups with no decoded audio become
        # silence of the correct length, so the time axis stays honest; the result
        # is NOT a continuous broadcast and gaps are real missing reception.
        first, last = min(decoded), max(decoded)
        block = rate*60//50*1  # 60 frames per group at 20 ms
        group_samples = 60*(rate//50)
        total = (last - first + 1)*group_samples
        canvas = np.zeros(total, dtype=np.int16)
        filled = 0
        for counter, pcm in decoded.items():
            begin = (counter - first)*group_samples
            canvas[begin:begin+len(pcm)] = pcm
            filled += 1
        prefix = ('KCBS2_HANGULKEYS_' if key_source == 'hangul_validated' else
                  'KCBS2_UNVERIFIED_' if key_source == 'crc_verified' else
                  'KCBS2_UNVERIFIED_CANDIDATEKEYS_')
        path = audio_dir/(prefix + 'MONTAGE_' + str(rate//1000) + 'k_%d-%d.wav' % (first, last))
        with wave.open(str(path), 'wb') as w:
            w.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
            w.writeframes(canvas.tobytes())
        rendered.append(dict(file=str(path.relative_to(out)), kind='montage',
                             counter_start=first, counter_end=last,
                             groups_total=last - first + 1, groups_decoded=filled,
                             seconds=round(total/rate, 2),
                             note='Decoded groups placed at their true counter time; '
                                  'other groups are silence because nothing decoded.'))
        print('  montage %s: %.1f s, %d of %d groups decoded'
              % (path.name, total/rate, filled, last - first + 1), flush=True)
    return rendered, dict(crc_verified_key_groups=verified_groups,
                          key_source=key_source, audio_key_groups=int(len(counters)))


def stage_montage(groups_npz, keys_npz, out, rate, merge_gap, allow_candidates):
    counters, keys, du, available = join_supported(groups_npz, keys_npz)
    verified = len(counters)
    source = 'crc_supported'
    if not verified and allow_candidates:
        candidate = hangul_validated(groups_npz)
        if candidate is not None:
            all_counters, all_keys, good, _ = candidate
            counters, keys, source = all_counters[good], all_keys[good], 'hangul_candidate'
            with np.load(groups_npz) as data:
                du = data['du'][good]
    if not len(counters):
        return [], dict(audio_key_groups=0, crc_verified_key_groups=0, key_source='none')
    first, last = int(available[0]), int(available[-1])
    offsets = counters - first
    frames, supplied = assemble_frames(du, keys, offsets, last-first+1)
    runs = group_runs(offsets, merge_gap)
    audio_dir = out/'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='evs_', dir=audio_dir) as tmp:
        pcm = render_timeline(frames, supplied, runs, rate,
                              lambda f, s: decode_evs(f, s, rate, Path(tmp)))
    label = 'CRC_PREFIX' if source == 'crc_supported' else 'HANGUL_CANDIDATE'
    path = audio_dir/f'KCBS2_UNVERIFIED_{label}_montage_{rate//1000}k_{first}-{last}.wav'
    with wave.open(str(path),'wb') as wav:
        wav.setparams((1,2,rate,0,'NONE','not compressed'))
        wav.writeframes(pcm.tobytes())
    report = dict(file=path.name, seconds=round(len(pcm)/rate,2),
                  audio_key_groups=len(counters), crc_verified_key_groups=verified,
                  key_source=source, decoder_calls=len(runs), merge_gap=merge_gap)
    (audio_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return [dict(file=str(path.relative_to(out)), kind='montage', seconds=report['seconds'])], report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='KiwiSDR IQ .wav captures of the DRM signal')
    ap.add_argument('--out', required=True, help='output directory (created)')
    ap.add_argument('--seconds', type=float, default=0, help='limit per input, 0 = whole file')
    ap.add_argument('--start', type=float, default=0, help='skip this many seconds of each input')
    ap.add_argument('--rate', type=int, choices=(16000, 32000, 48000), default=32000)
    ap.add_argument('--merge-gap', type=int, default=4,
                    help='with --montage, bridge up to N unkeyed groups inside one decoder '
                         'call; larger values give longer stretches but more concealment')
    ap.add_argument('--montage', action='store_true',
                    help='write one counter-aligned timeline instead of separate runs')
    ap.add_argument('--allow-candidate-keys', action='store_true',
                    help='if no key is CRC-verified, decode audio with fill-crib candidate '
                         'keys instead; output is labelled CANDIDATEKEYS and is weaker evidence')
    ap.add_argument('--fill', choices=pfq.STRATEGIES + ['role1_smooth'], default='role1_smooth',
                    help='role-1 substitute strategy for separate-run output; '
                         'montage always uses the adjacent role-0 donor')
    a = ap.parse_args()
    if a.merge_gap < 0:
        ap.error('--merge-gap must be nonnegative')
    missing = check_tools()
    if missing:
        raise SystemExit('missing prerequisites:\n  ' + '\n  '.join(missing))
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = []
    print('[1/5] demodulating IQ', flush=True)
    dumps = stage_demod([Path(p).resolve() for p in a.inputs], out, a.seconds, a.start, log)
    if not dumps:
        raise SystemExit('no DRM data decoded from these captures (check the signal was on air)')
    print('[2/5] salvaging CRC-good packets from %d dump(s)' % len(dumps), flush=True)
    salvage_dir = out/'salvage'
    if not (salvage_dir/'groups.npz').exists():
        salvage(dumps, salvage_dir)
    salvage_report = json.loads((salvage_dir/'report.json').read_text(encoding='utf-8'))
    print('[3/5] recovering side-channel text keys', flush=True)
    keys_dir = out/'keys'
    if not (keys_dir/'keys.npz').exists():
        recover(salvage_dir/'groups.npz', salvage_dir/'data_units.npz', keys_dir)
    keys_report = json.loads((keys_dir/'report.json').read_text(encoding='utf-8'))
    print('[4/5] writing decoded text', flush=True)
    text_report = stage_text(keys_dir, salvage_dir, out, log)
    if a.montage:
        print('[5/5] decoding audio continuously (%d Hz)' % a.rate, flush=True)
        rendered, audio_report = stage_montage(salvage_dir/'groups.npz', keys_dir/'keys.npz',
                                               out, a.rate, a.merge_gap, a.allow_candidate_keys)
    else:
        print('[5/5] decoding audio (%s, %d Hz)' % (a.fill, a.rate), flush=True)
        rendered, audio_report = stage_audio(salvage_dir/'groups.npz', keys_dir/'keys.npz',
                                             out, a.rate, a.fill, log, a.allow_candidate_keys,
                                             False)
    manifest = dict(
        inputs=[str(Path(p).resolve()) for p in a.inputs],
        complete_groups=salvage_report['salvaged_complete_groups'],
        crc_good_data_units=salvage_report['crc_good_data_units'],
        crc_verified_key_bytes=keys_report['crc_supported_key_bytes'],
        **text_report,
        **audio_report,
        fill_strategy='role1_donor' if a.montage else a.fill, sample_rate=a.rate, audio=rendered,
        content_seconds=round(sum(r['seconds'] for r in rendered
                                  if r.get('kind') != 'montage'), 2),
        montage_seconds=round(sum(r['seconds'] for r in rendered
                                  if r.get('kind') == 'montage'), 2),
        status=('audio produced' if rendered else
                'no CRC-verified keys: text segments did not repeat enough in this capture; '
                'rerun with --allow-candidate-keys for unverified audio'),
        caveats=['Role-1 unknown bytes are substitutes, not recovered keys.',
                 'Roles 2 and 3 are concealed by the decoder.',
                 'EVS 13.2 is an unconfirmed codec hypothesis.',
                 'Separate runs are different times; do not treat them as continuous audio.',
                 'Hangul candidate audio is heuristic, not CRC-supported.'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out/'calls.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in
                      ('complete_groups', 'key_source', 'audio_key_groups',
                       'korean_validated_text_groups', 'crc_verified_key_groups',
                       'content_seconds', 'montage_seconds', 'status')}, indent=2),
          flush=True)
    print('text in ' + str(out/'text/kcbs2_text.txt'), flush=True)
    if rendered:
        print('audio in ' + str(out/'audio'), flush=True)


if __name__ == '__main__':
    main()
