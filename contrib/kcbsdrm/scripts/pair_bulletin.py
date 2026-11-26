"""Merge the item lists of one or more captures (reference/items_<TAG>.json from pair_records.py) into the
most complete text: every distinct item once, each record taken from a copy where it VERIFIED if any copy
did (carousel repeats within a capture, re-broadcasts across captures).

    python pair_bulletin.py out.txt items_A.json [items_B.json ...]

Two item copies are the same item when the records verified in both (same seq) mostly agree. One shared phrase
is not enough: short phrases recur from one day's bulletin to the next. Pool only captures of the same bulletin
(pool()) before merging across captures.
Output per item: verified text plain; best-guess text of records verified in no copy inside ⟦ ⟧.
merge() is also used by pair_plaintext.py (the plain-text, per-capture output).
"""
import sys, json


def fold_split_finals(items):
    """An item that is only the verified final record of the previous item (the item splitter cut there
    because a garbled record in between broke the seq chain) is folded back into that item."""
    out = []
    for it in items:
        if out:
            prev = out[-1]
            prev_final = any(r['ok'] and r['last'] for r in prev)
            if (not prev_final and it[0]['seq'] != 1 and it[0]['tog'] == prev[-1]['tog']
                    and prev[-1]['seq'] <= it[0]['seq'] <= prev[-1]['seq'] + 1
                    and all(r['ok'] for r in it)):
                out[-1] = [r for r in prev if r['seq'] < it[0]['seq']] + list(it)
                continue
        out.append(list(it))
    return out


LONG = 20          # a single shared record identifies an item only if it is at least this many characters
MOSTLY = 0.7       # share of the records verified in both that must agree (the rest: garbled "verified" copies)


def same_item(a, b, same_capture=False):
    """a, b: {seq: set of verified texts}. Returns the number of agreeing records, or 0 if not the same item."""
    common = [s for s in a if s in b]
    agree = [s for s in common if a[s] & b[s]]
    if not agree or len(agree) < MOSTLY * len(common):
        return 0
    if len(agree) == 1 and not same_capture and max(len(t) for t in a[agree[0]] & b[agree[0]]) < LONG:
        return 0
    return len(agree)


def pool(current, others):
    """current plus those datasets that carry the same bulletin: at least half of the long verified records of the
    smaller of the two are found in the other (same bulletin: 0.9 or more seen; different: 0)."""
    def long_records(d):
        return {(r['seq'], r['text']) for it in d['items'] for r in it if r['ok'] and len(r['text']) >= LONG}
    cur = long_records(current)
    out = [current]
    for d in others:
        if d['tag'] != current['tag']:
            x = long_records(d)
            n = len(cur & x)
            if n >= 3 and n >= 0.5 * min(len(cur), len(x)):
                out.append(d)
    return out


def _absorb(g, texts, tags, members):
    for s, t in texts.items():
        g['texts'].setdefault(s, set()).update(t)
    g['tags'] |= tags
    g['members'] += members


def merge(datasets):
    """datasets: list of dicts {'tag', 'items'}. Returns (merged, where) where merged is a list of
    dict(recs={seq: rec}, vkeys, seen) and where[(tag, item_index)] = merged index."""
    # 1. which copies are the same item. Sparse copies (few verified records) may only turn out to belong
    #    together once a fuller copy has been seen, so groups are compared again until nothing changes.
    groups = []                                             # dict(texts, tags, members=[(n, d, ii, it)])
    n = 0
    for d in datasets:
        for ii, it in enumerate(d['items']):
            texts = {}
            for r in it:
                if r['ok']:
                    texts.setdefault(r['seq'], set()).add(r['text'])
            home, best = None, 0
            for g in groups:
                k = same_item(texts, g['texts'], d['tag'] in g['tags'])
                if k > best:
                    home, best = g, k
            if home is None:
                home = dict(texts={}, tags=set(), members=[]); groups.append(home)
            _absorb(home, texts, {d['tag']}, [(n, d, ii, it)])
            n += 1
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(groups):
            for h in groups[i + 1:]:
                if h['members'] and g['members'] and same_item(g['texts'], h['texts'], bool(g['tags'] & h['tags'])):
                    _absorb(g, h['texts'], h['tags'], h['members'])
                    h.update(texts={}, tags=set(), members=[]); changed = True
        groups = [g for g in groups if g['members']]
    # 2. the best copy of every record, taking the copies in the order given
    merged, where = [], {}
    for g in sorted(groups, key=lambda g: min(x[0] for x in g['members'])):
        m = dict(recs={}, vkeys=set(), texts=g['texts'], tags=g['tags'], seen=[])
        for _, d, ii, it in sorted(g['members'], key=lambda x: x[0]):
            where[(d['tag'], ii)] = len(merged)
            m['seen'].append('%s@0x%08X' % (d['tag'], it[0]['counter']))
            m['vkeys'] |= {(r['seq'], r['text']) for r in it if r['ok']}
            for r in it:
                fin = [s for s, x in m['recs'].items() if x['ok'] and x['last']]
                if fin and r['seq'] > fin[0]:
                    continue                      # beyond this item's verified final record: foreign
                if r['ok'] and r['last']:
                    for s in [s for s in m['recs'] if s > r['seq'] and not m['recs'][s]['ok']]:
                        del m['recs'][s]
                cur = m['recs'].get(r['seq'])
                better = (cur is None or (r['ok'] and not cur['ok'])
                          or (not cur['ok'] and not r['ok'] and r['complete'] and not cur['complete']))
                if better:
                    m['recs'][r['seq']] = dict(r, src=d['tag'])
        merged.append(m)
    return merged, where


def load(fn):
    d = json.load(open(fn, encoding='utf-8'))
    d['items'] = fold_split_finals(d['items'])
    return d


def main():
    out, files = sys.argv[1], sys.argv[2:]
    merged, _ = merge([load(fn) for fn in files])
    nrec = sum(len(m['recs']) for m in merged)
    nok = sum(r['ok'] for m in merged for r in m['recs'].values())
    nch = sum(len(r['text']) for m in merged for r in m['recs'].values())
    nchok = sum(len(r['text']) for m in merged for r in m['recs'].values() if r['ok'])
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# KCBS-2 side channel, merged bulletin from %s\n' % ', '.join(files))
        f.write('# %d distinct items, %d records, %d verified (%.1f%%); %.1f%% of characters exact.\n'
                % (len(merged), nrec, nok, 100 * nok / max(nrec, 1), 100 * nchok / max(nch, 1)))
        f.write('# Plain text = CRC + structure verified (exact). ⟦...⟧ = verified in no copy: best guess.\n'
                '# A missing sequence number = record never received.\n')
        for k, m in enumerate(merged):
            seqs = sorted(m['recs'])
            nv = sum(m['recs'][s]['ok'] for s in seqs)
            last = m['recs'][seqs[-1]]['last']
            f.write('\n=== item %d: records %d, verified %d%s; seen %s ===\n'
                    % (k, len(seqs), nv, '' if last else ', final record not seen', ', '.join(m['seen'])))
            prev = None
            body = []
            for s in seqs:
                r = m['recs'][s]
                if prev is not None and s != prev + 1:
                    body.append(' [… record(s) missing …] ')
                body.append(r['text'] if r['ok'] else '⟦' + r['text'] + '⟧')
                prev = s
            f.write(''.join(body) + '\n')
    print('%s: %d items, %d records, %d verified (%.1f%%), %.1f%% of characters exact'
          % (out, len(merged), nrec, nok, 100 * nok / max(nrec, 1), 100 * nchok / max(nch, 1)))
    import collections
    print('   verified records taken from:', dict(collections.Counter(
        r['src'] for m in merged for r in m['recs'].values() if r['ok'])))


if __name__ == '__main__':
    main()
