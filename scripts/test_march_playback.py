"""Exercise the March listening renderer without Dream or EVS."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kcbs_decode_fast import (
    IDLE_TAILS, assemble_frames, assemble_salvaged_idle, attenuate,
    group_runs, render_timeline, replay_nearby, salvage_idle_packets,
)
from march_listening_filter import reduce_flutter
from render_supported_nonidle import join_supported
from tail_fill_audit import crc


class MarchPlaybackTests(unittest.TestCase):
    def test_only_unknown_role1_bytes_are_replaced(self):
        du = np.random.default_rng(12).integers(0, 256, (1, 3, 716), dtype=np.uint8)
        key = np.arange(1, 17, dtype=np.uint8)
        frames, supplied = assemble_frames(du, key[None, :], [1], span=2)
        self.assertFalse(supplied[:60].any())
        self.assertFalse(frames[:60].any())
        np.testing.assert_array_equal(supplied[60:], np.arange(60) % 4 < 2)
        for phase in range(3):
            for cycle in range(5):
                shift = 131*cycle
                base = 60 + 20*phase + 4*cycle
                original = du[0, phase]
                role0 = original[shift+7:shift+39].copy()
                role1 = original[shift+40:shift+72].copy()
                role0[::3] ^= key[:11]
                role1[:15:3] ^= key[11:]
                role1[15::3] = role0[15::3]
                self.assertEqual(frames[base, 0], original[2+cycle])
                self.assertEqual(frames[base+1, 0], original[shift+39])
                np.testing.assert_array_equal(frames[base, 1:], role0)
                np.testing.assert_array_equal(frames[base+1, 1:], role1)

    def test_replay_keeps_sources_and_reception_holes_intact(self):
        frames = np.repeat(np.arange(16, dtype=np.uint8)[:, None], 33, axis=1)
        supplied = np.array([True, True, False, False]*4)
        supplied[8:12] = False
        out, keep = replay_nearby(frames, supplied)
        np.testing.assert_array_equal(out[:, 0], [0, 1, 1, 4, 4, 5, 5, 5, 8, 9, 10, 11, 12, 13, 13, 13])
        np.testing.assert_array_equal(out[supplied], frames[supplied])
        np.testing.assert_array_equal(keep, np.repeat([True, True, False, True], 4))

    def test_replay_keeps_a_partial_final_cycle(self):
        frames = np.repeat(np.arange(7, dtype=np.uint8)[:, None], 33, axis=1)
        supplied = np.array([True, True, False, False, True, True, False])
        out, keep = replay_nearby(frames, supplied)
        np.testing.assert_array_equal(out[:, 0], [0, 1, 1, 4, 4, 5, 5])
        self.assertTrue(keep.all())

    def test_attenuation_is_signed_and_does_not_change_length(self):
        source = np.array([-32768, -5000, 0, 5000, 32767], dtype='<i2')
        got = attenuate(source, 20)
        np.testing.assert_array_equal(got, [-3277, -500, 0, 500, 3277])
        self.assertEqual(got.dtype, source.dtype)

    def test_nonidle_requires_every_byte_to_have_crc_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            du = np.zeros((3, 3, 716), dtype=np.uint8)
            du[:, 0, 0] = [70, 71, 72]
            groups, keys = root/'groups.npz', root/'keys.npz'
            np.savez(groups, counters=[70, 71, 72], du=du)
            key_bytes = np.tile(np.arange(16, dtype=np.uint8), (4, 1))
            key_bytes[:, 0] = [71, 132, 70, 72]
            mask = np.ones((4, 16), dtype=bool)
            mask[0, 15] = False
            mask[3, :] = False
            np.savez(keys, counters=[71, 900, 70, 72], ks=key_bytes, crc_supported=mask,
                     candidate_mask=np.ones_like(mask))
            matched, matched_key, matched_du, _ = join_supported(groups, keys)
            np.testing.assert_array_equal(matched, [70])
            np.testing.assert_array_equal(matched_key, key_bytes[2:3])
            np.testing.assert_array_equal(matched_du, du[:1])

    def test_gap_limit_and_singleton_runs(self):
        offsets = [0, 5, 11, 12]
        self.assertEqual(group_runs(offsets, 4), [(0, 6), (11, 13)])
        self.assertEqual(group_runs(offsets, 0), [(0, 1), (5, 6), (11, 13)])
        self.assertEqual(group_runs([0], 4), [(0, 1)])

    def test_alignment_and_fades_do_not_change_gap_length(self):
        rate = 16000
        per_frame = rate//50
        per_group = per_frame*60
        frames = np.zeros((4*60, 33), dtype=np.uint8)
        supplied = np.array([True]*60 + [False]*120 + [True]*60)
        calls = []

        def decode(part, keep):
            calls.append(len(part))
            self.assertTrue(keep.all())
            return np.full(len(part)*per_frame, 1234, dtype=np.int16)

        pcm = render_timeline(frames, supplied, [(0, 1), (3, 4)], rate, decode)
        self.assertEqual(calls, [60, 60])
        self.assertEqual(len(pcm), 4*per_group)
        self.assertFalse(pcm[per_group:3*per_group].any())
        self.assertEqual(pcm[per_group-1], 0)
        self.assertEqual(pcm[3*per_group], 0)
        self.assertEqual(pcm[per_group-per_frame], 1234)
        self.assertEqual(pcm[3*per_group+per_frame-1], 1234)

    def test_listening_filter_is_attenuation_only(self):
        n = 16000//50
        source = np.tile(np.repeat([14000, -12000, 8000, -5000], n), 120).astype(np.int16)
        source[:3*4*n] = 0
        out, stats = reduce_flutter(source, 16000)
        self.assertEqual(out.shape, source.shape)
        self.assertFalse(out[source == 0].any())
        self.assertTrue((np.abs(out) <= np.abs(source)).all())
        self.assertLess(stats['held_out_slot_spread_db_after'], stats['held_out_slot_spread_db_before'])

    def test_salvaged_idle_recovers_partial_group_and_quarantines_conflicts(self):
        rng = np.random.default_rng(77)
        key = np.arange(10, 26, dtype=np.uint8)
        du = rng.integers(0, 256, (3, 716), dtype=np.uint8)
        counter = 500
        du[0, 664:668] = np.frombuffer(counter.to_bytes(4, 'big'), dtype=np.uint8)
        for p in range(3):
            du[p, 1] = p << 4
            du[p, 698:714] = np.frombuffer(IDLE_TAILS[p], dtype=np.uint8) ^ key
            du[p, 714:716] = np.frombuffer(crc(du[p, :714]).to_bytes(2, 'big'), dtype=np.uint8)

        def pack_record(unit, corrupt_packet=None):
            chunks = []
            for q in range(4):
                body = bytearray([q]) + bytearray(unit[q * 179 : (q + 1) * 179])
                pkt_crc = crc(body)
                if q == corrupt_packet:
                    pkt_crc ^= 0xFFFF
                chunks.append(bytes(body) + pkt_crc.to_bytes(2, 'big'))
            raw = b''.join(chunks)
            return len(raw).to_bytes(4, 'little') + raw

        with tempfile.TemporaryDirectory() as td:
            dump = Path(td) / 'partial.msc.bin'
            # Phase 2 loses packet 2 (bytes 358..537), while phases 0 and 1 are intact.
            dump.write_bytes(pack_record(du[0]) + pack_record(du[1]) + pack_record(du[2], corrupt_packet=2))
            payloads = salvage_idle_packets([dump])
            counters, frames, supplied, exact_r0 = assemble_salvaged_idle(payloads)
            np.testing.assert_array_equal(counters, [counter])
            self.assertGreater(exact_r0, 10)
            self.assertLess(exact_r0, 15)
            # Injecting a conflicting packet observation on phase 0 packet 3 drops that phase's key vote;
            # injecting a second conflict on phase 1 packet 3 leaves <2 agreeing phases and rejects the group.
            payloads[(counter, 0, 3)].add(b'\x00' * 179)
            payloads[(counter, 1, 3)].add(b'\x00' * 179)
            counters2, _, _, _ = assemble_salvaged_idle(payloads)
            self.assertEqual(len(counters2), 0)


if __name__ == '__main__':
    unittest.main()
