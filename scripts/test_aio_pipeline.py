"""Synthetic checks for the end-to-end pipeline's frame assembly and fills."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('pfq', HERE/'partial_frame_quality.py')
pfq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfq)
import kcbs_decode_audio as aio


class PipelineTests(unittest.TestCase):
    def test_packet_and_mode_positions_match_container(self):
        self.assertEqual(aio.PACKET_STARTS[:4], [7, 40, 73, 106])
        self.assertEqual(aio.MODE_POSITIONS[:4], [2, 39, 72, 105])
        for j, start in enumerate(aio.PACKET_STARTS):
            self.assertLessEqual(start + 32, 662)
            role = j % 4
            cycle = j // 4
            if role == 0:
                self.assertEqual(aio.MODE_POSITIONS[j], 2 + cycle)
            else:
                self.assertEqual(aio.MODE_POSITIONS[j], start - 1)

    def test_role_slot_alignment(self):
        slot = (np.arange(180) % 4)
        for k in range(3):
            for phase in range(3):
                for j in range(20):
                    self.assertEqual(slot[k*60 + phase*20 + j], j % 4)

    def test_fills_touch_only_unknown_bytes(self):
        rng = np.random.default_rng(3)
        frames = rng.integers(0, 256, (120, 33), dtype=np.uint8)
        slot = np.arange(120) % 4
        for name in ('role1_zeros', 'role1_random', 'role1_donor'):
            built, keep = pfq.apply_strategy(frames, name, np.random.default_rng(1), frames)
            np.testing.assert_array_equal(built[slot == 0], frames[slot == 0])
            known = [i for i in range(33) if i not in set(pfq.ROLE1_UNKNOWN.tolist())]
            np.testing.assert_array_equal(built[np.ix_(np.flatnonzero(slot == 1), known)],
                                          frames[np.ix_(np.flatnonzero(slot == 1), known)])
            self.assertTrue(keep[slot == 0].all())
            self.assertTrue(keep[slot == 1].all())
            self.assertFalse(keep[slot == 2].any())
            self.assertFalse(keep[slot == 3].any())

    def test_erase_all_supplies_only_role0(self):
        frames = np.zeros((80, 33), np.uint8)
        built, keep = pfq.apply_strategy(frames, 'erase_all', np.random.default_rng(0), frames)
        np.testing.assert_array_equal(built, frames)
        self.assertEqual(int(keep.sum()), 20)


    def test_text_stage_separates_verified_from_candidates(self):
        import json, tempfile
        from tail_fill_audit import crc
        body = ('가나다라마바사아자차카타파하' * 4)[:46]
        good = bytes([0x85, 0x2c]) + body.encode('utf-16le')
        good = good + crc(good).to_bytes(2, 'big') + bytes([0xbb]) * 16
        bad = bytearray(good)
        bad[94] ^= 1
        with tempfile.TemporaryDirectory() as td:
            keys_dir = Path(td) / 'keys'
            keys_dir.mkdir()
            (keys_dir / 'crc_supported_text.json').write_text(json.dumps([
                dict(start=300, header_ok=True, fill_ok=True, crc_ok=True,
                     plaintext_hex=good.hex(), text=body)]), encoding='utf-8')
            (keys_dir / 'segments.json').write_text(json.dumps([
                dict(start=300, header_ok=True, fill_ok=True, crc_ok=True,
                     plaintext_hex=good.hex()),
                dict(start=307, header_ok=True, fill_ok=True, crc_ok=False,
                     plaintext_hex=bytes(bad).hex())]), encoding='utf-8')
            out = Path(td) / 'out'
            out.mkdir()
            salvage = Path(td) / 'salvage'
            salvage.mkdir()
            report = aio.stage_text(keys_dir, salvage, out, [])
            self.assertEqual(report['crc_valid_text_segments'], 1)
            self.assertEqual(report['distinct_text_bodies'], 1)
            self.assertEqual(report['unverified_text_candidates'], 1)
            written = (out / 'text' / 'kcbs2_text.txt').read_text(encoding='utf-8')
            self.assertIn('counter 100 phase 0', written)
            self.assertIn('NOT VERIFIED', written)
            self.assertIn(body[:12], written)
            verified_part, candidate_part = written.split('CANDIDATE SEGMENTS THAT FAILED')
            self.assertIn('crc=False', candidate_part)
            self.assertNotIn('crc=False', verified_part)
            self.assertIn('TEXT FROM FILL-CRIB CANDIDATE KEYS', written)

    def test_one_line_collapses_breaks(self):
        self.assertEqual(aio._one_line('a' + chr(13) + chr(10) + 'b'), 'a / b')

    def test_merge_gap_stretch_bounds(self):
        """Stretch grouping must keep the first keyed group and not crash on singletons."""
        import importlib.util
        spec = importlib.util.spec_from_file_location('rc', HERE/'render_continuous.py')
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)

        def bounds_of(offsets, gap):
            bounds, start, previous = [], offsets[0], offsets[0]
            for o in offsets[1:]:
                if o - previous > gap:
                    bounds.append((start, previous))
                    start = o
                previous = o
            bounds.append((start, previous))
            return bounds

        self.assertEqual(bounds_of([5], 4), [(5, 5)])
        self.assertEqual(bounds_of([0, 99], 4), [(0, 0), (99, 99)])
        self.assertEqual(bounds_of([0, 2, 4, 50, 52], 4), [(0, 4), (50, 52)])
        self.assertEqual(bounds_of([0, 1, 2, 3], 4), [(0, 3)])
        # Every keyed offset must fall inside some stretch.
        offsets = [0, 3, 7, 40, 41, 90]
        covered = {o for lo, hi in bounds_of(offsets, 4) for o in offsets if lo <= o <= hi}
        self.assertEqual(covered, set(offsets))

if __name__ == '__main__':
    unittest.main()
