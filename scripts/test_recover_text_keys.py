"""Positive and negative controls for additive text-key recovery."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from recover_text_keys import recover
from test_text_records import make_record


def data_units(tails_by_tick):
    """Minimal data_units.npz arrays: counters, phases and 716-byte units carrying tails."""
    ticks = sorted(tails_by_tick)
    du = np.zeros((len(ticks), 716), np.uint8)
    for i, t in enumerate(ticks):
        du[i, 698:714] = tails_by_tick[t]
    return (np.array([t // 3 for t in ticks], np.uint32),
            np.array([t % 3 for t in ticks], np.uint8), du)


class RecoverTextKeysTest(unittest.TestCase):
    GROUPS = (100, 101, 102, 103)
    PERIOD = 6  # blocks between the two carousel copies

    def setUp(self):
        self.rng = np.random.default_rng(7)
        # A six-block final record with pure-text interior blocks, so a phrase from
        # the corpus can propose a group key; repeated one carousel period apart.
        self.text = "".join(chr(int(c)) for c in self.rng.integers(0xac00, 0xd7a4, 40))
        self.record = make_record(self.text.encode("utf-16le"), sequence=3, final=True)
        self.assertEqual(len(self.record), 96)  # six 16-byte blocks

    def _build(self, corrupt=False, randomize=False):
        plain = np.frombuffer(self.record, np.uint8).reshape(6, 16)
        keys = {g: self.rng.integers(0, 256, 16, np.uint8) for g in self.GROUPS}
        tails = {}
        for copy in (0, 1):  # copy 1 at blocks 0..5, copy 2 at blocks PERIOD..PERIOD+5
            for b in range(6):
                block = copy * self.PERIOD + b
                group, phase = self.GROUPS[block // 3], block % 3
                tails[3 * group + phase] = plain[b] ^ keys[group]
        if corrupt:  # break the second copy's CRC so it never repeats
            tails[3 * self.GROUPS[3] + 2] = tails[3 * self.GROUPS[3] + 2].copy()
            tails[3 * self.GROUPS[3] + 2][5] ^= 1
        if randomize:
            tails = {t: self.rng.integers(0, 256, 16, np.uint8) for t in tails}
        return tails, keys

    def _run(self, tails, directory):
        du_c, du_p, du = data_units(tails)
        np.savez(directory / "units.npz", counters=du_c, phases=du_p, du=du)
        np.savez(directory / "supplied.npz", counters=np.zeros(0, np.uint32),
                 ks=np.zeros((0, 16), np.uint8), crc_supported=np.zeros((0, 16), bool))
        corpus = directory / "ref"
        corpus.mkdir()
        rows = [{"ok": True, "text": self.text[i:i + 8]} for i in range(len(self.text) - 7)]
        (corpus / "items_x.json").write_text(json.dumps({"items": [rows]}), encoding="utf-8")
        out = directory / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            recover(directory / "units.npz", directory / "supplied.npz", corpus, self.PERIOD, out)
        with np.load(out / "keys.npz") as k:
            return {int(c): row for c, row, m in zip(k["counters"], k["ks"], k["crc_supported"])
                    if m.all()}

    def test_carousel_verified_record_recovers_all_group_keys(self):
        tails, keys = self._build()
        with tempfile.TemporaryDirectory() as td:
            recovered = self._run(tails, Path(td))
        self.assertEqual(set(recovered), set(self.GROUPS))
        for g in self.GROUPS:
            np.testing.assert_array_equal(recovered[g], keys[g])

    def test_single_copy_without_carousel_repeat_is_not_recovered(self):
        tails, _ = self._build(corrupt=True)
        with tempfile.TemporaryDirectory() as td:
            recovered = self._run(tails, Path(td))
        # Only one clean copy remains: the CRC passes once but never repeats.
        self.assertEqual(recovered, {})

    def test_random_tails_recover_nothing(self):
        tails, _ = self._build(randomize=True)
        with tempfile.TemporaryDirectory() as td:
            recovered = self._run(tails, Path(td))
        self.assertEqual(recovered, {})

    def test_carousel_xor_link_keys_a_non_record_sibling(self):
        # Two clean copies verify groups 100-103 via carousel repeat. A third copy one
        # period later (groups 104-105) has a corrupted CRC, so it is not a valid record
        # and cannot be found by the scanner; its text blocks still match the verified
        # copies at the carousel distance, so the XOR-link keys it.
        plain = np.frombuffer(self.record, np.uint8).reshape(6, 16)
        groups = (100, 101, 102, 103, 104, 105)
        keys = {g: self.rng.integers(0, 256, 16, np.uint8) for g in groups}
        tails = {}
        for copy in range(3):
            raw = bytearray(self.record)
            if copy == 2:
                raw[90] ^= 1  # corrupt the CRC region of the third copy
            blocks = np.frombuffer(bytes(raw), np.uint8).reshape(6, 16)
            for b in range(6):
                block = copy * self.PERIOD + b
                group, phase = groups[block // 3], block % 3
                tails[3 * group + phase] = blocks[b] ^ keys[group]
        with tempfile.TemporaryDirectory() as td:
            recovered = self._run(tails, Path(td))
        self.assertEqual(set(recovered), set(groups))
        for g in (104, 105):
            np.testing.assert_array_equal(recovered[g], keys[g])

    def test_period_auto_detected_when_omitted(self):
        from recover_text_keys import detect_period
        tails, _ = self._build()
        tailmap = {t: v for t, v in tails.items()}
        first = min(t // 3 for t in tails)
        span = max(t // 3 for t in tails) - first + 1
        # supplied keys are empty here, so detect from the already-plain records instead:
        # build a keyed map by brute truth to confirm the dominant distance is PERIOD.
        plain = np.frombuffer(self.record, np.uint8).reshape(6, 16)
        keys = {}
        for copy in (0, 1):
            for b in range(6):
                block = copy * self.PERIOD + b
                g = self.GROUPS[block // 3]
                keys[g] = tailmap[3 * g + block % 3] ^ plain[b]
        self.assertEqual(detect_period(keys, tailmap, first, span), self.PERIOD)


if __name__ == "__main__":
    unittest.main()
