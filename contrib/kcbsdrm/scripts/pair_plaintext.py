"""FINAL STAGE: plain, readable text of one full carousel cycle per capture, side-channel signalling removed.

    python pair_plaintext.py                      # the known 2026-11-25 captures (table below)
    python pair_plaintext.py TAG IQFILE [IQFILE ...]   # any capture already run through pair_records.py

Input: reference/items_<TAG>.json (pair_records.py). Every reference/items_*.json is pooled so a record that
failed in this capture is filled from a verified copy of the same item in any other capture (pair_bulletin.merge).
Output: D:\\KCBSDRM\\KCBS2_text_<date>_<HHMM>KST_<freq>kHz_<sdr>.txt (+ a copy in reference/).

Kept: text records only, one cycle, in broadcast order, rotated to the start of the bulletin.
Removed: headers, CRCs, fill, padding, item metadata, counters, sequence numbers, stub/duplicate items.
Unverified records: kept as [? ... ?] when they read as normal text, otherwise replaced by [...].
"""
import sys, os, re, glob, json, wave, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair_bulletin import merge, load, pool, same_item

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference')
OUT = os.path.normpath(os.path.join(REF, '..', '..'))      # the project root
IQ = os.path.join(OUT, 'IQ Files')
KST = datetime.timedelta(hours=9)

CAPTURES = [
    ('2026-11-25', ['SND.hik.iptime.org_2026-11-24T19_48_00Z_3205.00_iq.wav',
                    'SND.hik.iptime.org_2026-11-24T19_57_46Z_3205.00_iq.wav']),
    ('2026-11-25_1246KST_merged', ['SND.hik.iptime.org_2026-11-25T03_46_41Z_6140.00_iq.wav',
                                   'SND.sdrsapporo.ddns.net_2026-11-25T03_47_04Z_6140.00_iq.wav']),
    ('2026-11-25_2242KST_3205_hl5ntr', ['SND.hl5ntr-sdr.ddns.net_2026-11-25T13_42_17Z_3205.000_iq.wav']),
    ('2026-11-26_0500KST_3205_hl5ntr', ['SND.hl5ntr-sdr.ddns.net_2026-11-25T20_00_18Z_3205.000_iq.wav']),
    ('2026-11-26_0642KST_merged', ['SND.hl5ntr.ddns.net_2026-11-25T21_42_22Z_3205.000_iq.wav',
                                   'SND.sdrsapporo.ddns.net_2026-11-25T21_46_08Z_6140.00_iq.wav']),
    ('2026-11-26_1008KST_merged', ['SND.hik.iptime.org_2026-11-26T01_08_39Z_6140.00_iq.wav',
                                   'SND.22052.proxy.kiwisdr.com_2026-11-26T01_10_15Z_6140.00_iq.wav']),
]

# Where a bulletin cycle begins (the medium-range outlook): first item whose opening matches one of these.
START_CUES = [r'^\s*1\)', r'예견되는 바다날씨를 알려드리겠습니다']

FNRE = re.compile(r'SND\.(?P<sdr>.+?)_(?P<t>\d{4}-\d\d-\d\dT\d\d_\d\d_\d\dZ)_(?P<f>[\d.]+?)_iq')


def parse_name(fn):
    m = FNRE.search(os.path.basename(fn))
    t = datetime.datetime.strptime(m['t'], '%Y-%m-%dT%H_%M_%SZ')
    dur = None
    p = fn if os.path.isabs(fn) else os.path.join(IQ, fn)
    try:
        with wave.open(p) as w:
            dur = w.getnframes() / w.getframerate()
    except Exception:
        pass
    return dict(sdr=m['sdr'], t=t, f=float(m['f']), dur=dur, name=os.path.basename(fn))


def plausible(text, charset):
    cs = [c for c in text if not c.isspace()]
    return bool(cs) and sum(c in charset for c in cs) / len(cs) >= 0.85


def shingles(t, n=4):
    t = re.sub(r'\s+', '', t)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def render(m, charset, other_sh):
    seqs = sorted(m['recs'])
    out, prev, nv, nguess, ndrop = [], None, 0, 0, 0
    for s in seqs:
        r = m['recs'][s]
        if prev is not None and s != prev + 1:
            out.append(' [...] ')
        if r['ok']:
            out.append(r['text']); nv += len(r['text'])
        else:
            sh = shingles(r['text'])
            dup = sh and len(sh & other_sh) / len(sh) >= 0.3       # spill of another item's text / title
            if plausible(r['text'], charset) and not dup and '기상정보' not in r['text']:
                out.append('[?' + r['text'] + '?]'); nguess += len(r['text'])
            else:
                out.append(' [...] '); ndrop += len(r['text'])
        prev = s
    if not m['recs'][seqs[-1]]['last']:
        out.append(' [...]')
    txt = ''.join(out)
    txt = re.sub(r'(\s*\[\.\.\.\]\s*)+', ' [...] ', txt).strip()
    return txt, nv, nguess, ndrop


def loop_groups(starts, order):
    """Loop period in key groups: median of the single-loop gaps between appearances of the same item."""
    d = [b - a for k in order for a, b in zip(starts[k], starts[k][1:]) if b - a > 100]
    if not d:
        return None
    one = sorted(x for x in d if x < 1.5 * min(d))
    return one[len(one) // 2]


def cycle_of(tag, d, merged, where):
    """Distinct merged items of this capture in broadcast order, one loop, rotated to the bulletin start."""
    order = []
    for ii, it in enumerate(d['items']):
        k = where[(tag, ii)]
        if not any(r['ok'] for r in merged[k]['recs'].values()):
            continue                                   # stub: nothing verified anywhere (metadata spill)
        if k in order:
            continue                                   # a later loop; with fades an item may first appear there
        order.append(k)
    # broadcast order = position within the loop (arrival order is wrong when fades hide items in early loops)
    starts = {}
    for ii, it in enumerate(d['items']):
        r = it[0]                                      # item start, back-projected: one record = 56/24 groups
        starts.setdefault(where[(tag, ii)], []).append(r['counter'] - (r['seq'] - 1) * 56 / 24)
    P = loop_groups(starts, order)
    if P:
        base = starts[order[0]][0]
        order.sort(key=lambda k: (starts[k][0] - base) % P)
    for cue in START_CUES:
        for j, k in enumerate(order):
            first = merged[k]['recs'][min(merged[k]['recs'])]
            if min(merged[k]['recs']) == 1 and re.search(cue, first['text']):
                return order[j:] + order[:j]
    return order


def write_capture(tag, iqfiles, datasets, merged, where):
    d = next(x for x in datasets if x['tag'] == tag)
    caps = [parse_name(f) for f in iqfiles]
    charset = {c for m in merged for r in m['recs'].values() if r['ok'] for c in r['text']}
    order = cycle_of(tag, d, merged, where)
    t0 = min(c['t'] for c in caps)
    t1 = max(c['t'] + datetime.timedelta(seconds=c['dur'] or 0) for c in caps)
    freq = caps[0]['f']
    sdrs = sorted({c['sdr'] for c in caps})
    fname = 'KCBS2_text_%s_%sKST_%dkHz_%s.txt' % ((t0 + KST).strftime('%Y-%m-%d'), (t0 + KST).strftime('%H%M'),
                                                 round(freq), '+'.join(sdrs))
    part = re.search(r'_part(\d)$', tag)                # a recording split at a bulletin change (auto_decode.py)
    if part:
        fname = fname[:-4] + '_part%s.txt' % part.group(1)
    # loop length from this capture's own item counters
    firsts = {}
    for ii, it in enumerate(d['items']):
        firsts.setdefault(where[(tag, ii)], []).append(it[0]['counter'])
    per = [b - a for k in order for a, b in zip(firsts[k], firsts[k][1:])]
    loop_min = (sorted(per)[len(per) // 2] * 1.2 / 60) if per else None

    titles = []                                             # this capture's own headlines first
    own_titles = os.path.join(REF, 'titles_%s.json' % re.sub(r'_part\d$', '', tag))
    for fn in sorted(glob.glob(os.path.join(REF, 'titles_*.json')), key=lambda f: f != own_titles):
        titles += [dict(t, own=os.path.normpath(fn) == os.path.normpath(own_titles))
                   for t in json.load(open(fn, encoding='utf-8'))]
    bodies, tv, tg, td, filled = [], 0, 0, 0, 0
    oe = ol = ow = 0                        # opening characters: exact / LM / weak
    for k in order:
        other = set()
        for j, m in enumerate(merged):
            if j != k:
                for r in m['recs'].values():
                    if r['ok']:
                        other |= shingles(r['text'])
        txt, nv, ng, nd = render(merged[k], charset, other)
        tv += nv; tg += ng; td += nd
        own = {(r['seq'], r['text']) for it in d['items'] for r in it if r['ok']}
        filled += sum(1 for r in merged[k]['recs'].values() if r['ok'] and (r['seq'], r['text']) not in own)
        best, t = (False, 0), None
        for c in titles:
            keys = {}
            for s, x in c['keys']:
                keys.setdefault(s, set()).add(x)
            n = same_item(keys, merged[k]['texts'], c['own'])
            if n and (c['own'], n) > best:
                best, t = (c['own'], n), c
        if t is None or min(merged[k]['recs']) != 1:
            bodies.append('[...] ' + txt)
        else:
            op = ''
            for i, (ch_, g) in enumerate(zip(t['opening'], t['grade'])):
                if g == '?' and (i == 0 or t['grade'][i - 1] != '?'):
                    op += '[?'
                op += ch_
                if g == '?' and (i + 1 == len(t['grade']) or t['grade'][i + 1] != '?'):
                    op += '?]'
            oe += t['grade'].count('='); ol += t['grade'].count('~'); ow += t['grade'].count('?')
            sep = '\n' if (op and '가' <= op[-1] <= '힣'
                           and (txt[:1].isdigit() or (len(op) >= 4 and txt.startswith(op.split()[-1])))) else ''
            bodies.append('[%s] %s%s%s' % (t['category'], op, sep, txt))   # headline filled the record
    tot = max(tv + tg + td, 1)

    H = []
    H.append('KCBS-2 (조선중앙방송 제2방송) — text carried in the DRM side channel')
    H.append('Capture start : %s UTC  (%s KST)' % (t0.strftime('%Y-%m-%d %H:%M:%S'),
                                                (t0 + KST).strftime('%Y-%m-%d %H:%M:%S')))
    H.append('Capture end   : %s UTC  (%s KST)' % (t1.strftime('%Y-%m-%d %H:%M:%S'),
                                                (t1 + KST).strftime('%Y-%m-%d %H:%M:%S')))
    H.append('Frequency     : %s kHz' % ', '.join('%g' % f for f in sorted({c['f'] for c in caps})))
    H.append('SDR           : %s' % ', '.join(sdrs))
    H.append('Source file%s  : %s' % ('s' if len(caps) > 1 else ' ', ('\n' + ' ' * 16).join(c['name'] for c in caps)))
    H.append('Content       : one full cycle, %d items%s, in broadcast order (starting at the bulletin start)'
             % (len(order), ', repeating every %.1f min' % loop_min if loop_min else ''))
    H.append('Accuracy      : %.1f%% of characters CRC-verified exact, %.1f%% best guess, %.1f%% unreadable'
             % (100 * tv / tot, 100 * tg / tot, 100 * td / tot))
    if filled:
        H.append('                (%d records filled from verified copies in other captures of the same item)'
                 % filled)
    if oe + ol + ow:
        H.append('Item openings : category, date line / headline and first words from each item title record:')
        H.append('                %d characters, %.0f%% exact (XOR chain from verified keys across copies), %.0f%%'
                 % (oe + ol + ow, 100 * oe / (oe + ol + ow), 100 * ol / (oe + ol + ow)))
        H.append('                language-model choice with a clear margin, %.0f%% weak (marked [? ?]).'
                 % (100 * ow / (oe + ol + ow)))
        H.append('                The title record carries no CRC, so openings are not CRC-verified.')
    H.append('Markers       : [기상정보] etc.  category of the item (기상정보 weather, 경제정보 economy, 문화정보 culture)')
    H.append('                [...]       text not decoded (an item opening not solved, or a lost record)')
    H.append('                [? text ?]  best guess, not verified; may contain wrong characters')
    H.append('=' * 78)
    text = '\n'.join(H) + '\n\n' + ('\n\n' + '-' * 40 + '\n\n').join(bodies) + '\n'
    for p in (os.path.join(OUT, fname), os.path.join(REF, fname)):
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
    print('%s: %d items, %.1f%% verified, %.1f%% guess, %.1f%% unreadable, loop %s min'
          % (fname, len(order), 100 * tv / tot, 100 * tg / tot, 100 * td / tot,
             '%.2f' % loop_min if loop_min else '?'))


def main():
    caps = CAPTURES if len(sys.argv) < 3 else [(sys.argv[1], sys.argv[2:])]
    files = sorted(glob.glob(os.path.join(REF, 'items_*.json')))
    everything = [load(fn) for fn in files]
    for tag, iq in caps:
        current = next(d for d in everything if d['tag'] == tag)
        datasets = pool(current, everything)                # only captures of the same bulletin fill gaps
        datasets.sort(key=lambda d: d['tag'])
        merged, where = merge(datasets)
        write_capture(tag, iq, datasets, merged, where)


if __name__ == '__main__':
    main()
