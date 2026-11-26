"""Positive/negative controls for repeated text-key recovery and packet salvage."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tail_fill_audit import crc, fill_candidates
from repeated_text_keys import recover, XorUF
from salvage_msc_packets import salvage
from test_key_recovery import group
from expand_verified_text import propose
from align_november_evs import score_offsets


class NovemberRecoveryTests(unittest.TestCase):
    def test_verified_crib_and_bad_crc(self):
        rng = np.random.default_rng(42)
        text = ('가나다라마바사아자차카타파하'*4)[:46].encode('utf-16le')
        head_body = b'\x85\x2c'+text
        message = head_body+crc(head_body).to_bytes(2,'big')+b'\xbb'*16
        keys = {g:rng.integers(0,256,16,dtype=np.uint8) for g in range(100,103)}
        tails = {300+j:np.frombuffer(message[j*16:(j+1)*16],np.uint8)^keys[(300+j)//3]
                 for j in range(7)}
        result = propose(300,text,tails,{})
        self.assertEqual(result['status'],'crc_supported')
        self.assertEqual(result['plaintext_hex'],message[:96].hex())
        bad = {k:v.copy() for k,v in tails.items()}
        bad[305][14] ^= 1
        result = propose(300,text,bad,{})
        self.assertIsNotNone(result)
        self.assertEqual(result['status'],'bad_crc')
        del bad[302]
        self.assertNotEqual((propose(300,text,bad,{}) or {}).get('status'),'crc_supported')

    def test_alignment_scoring_known_offset(self):
        rng = np.random.default_rng(8)
        reference = rng.normal(size=(300,16)).astype(np.float32)
        indices = np.arange(0,120,3)
        scores = score_offsets(reference[indices+37],reference,indices,100)
        self.assertEqual(int(scores.argmax()),37)
        self.assertAlmostEqual(float(scores[37]),1.0,places=5)
        zeros = score_offsets(np.zeros((len(indices),16)),reference,indices,100)
        self.assertTrue(np.isfinite(zeros).all())
        self.assertTrue((zeros == 0).all())

    def test_xor_constraints(self):
        uf = XorUF()
        self.assertTrue(uf.bind(10, -1, 0x34))
        self.assertTrue(uf.bind(20, 10, 0x12))
        self.assertEqual(uf.value(20), 0x26)
        trial = uf.copy()
        self.assertFalse(trial.bind(20, -1, 0x55))
        self.assertEqual(uf.value(20), 0x26)

    def test_repeated_segment_recovery_and_random_control(self):
        rng = np.random.default_rng(2711)
        text = ''.join(chr(int(c)) for c in rng.integers(0xac00, 0xd7a4, 46)).encode('utf-16le')
        segments = []
        for i in range(9):
            body = bytes([5 if i % 2 else 0x85, 0x2c]) + text
            segments.append(body + crc(body).to_bytes(2, 'big') + b'\xbb'*16)
        plain = np.frombuffer(b''.join(segments), np.uint8).reshape(-1, 3, 16)
        n = len(plain)
        keys = rng.integers(0, 256, (n, 16), dtype=np.uint8)
        cipher = plain ^ keys[:, None, :]
        counters = np.arange(10000, 10000+n, dtype=np.uint32)
        candidate_keys = np.zeros_like(keys)
        mask = np.zeros_like(keys, dtype=bool)
        offsets = np.full(n, -1, np.int16)
        for i in range(n):
            cs = fill_candidates(cipher[i])
            if len(cs) == 1:
                candidate_keys[i] = cs[0]['key']; mask[i] = True; offsets[i] = cs[0]['offset']
        self.assertGreater(int(mask.all(1).sum()), 0)
        du = np.zeros((n, 3, 716), np.uint8); du[:, :, 698:714] = cipher
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gp, up = root/'groups.npz', root/'units.npz'
            np.savez(gp, counters=counters, candidate_ks=candidate_keys,
                     candidate_mask=mask, fill_offset=offsets)
            np.savez(up, counters=np.repeat(counters, 3), phases=np.tile(np.arange(3), n),
                     du=du.reshape(-1, 716))
            with contextlib.redirect_stdout(io.StringIO()):
                recover(gp, up, root/'positive')
            report = json.loads((root/'positive/report.json').read_text())
            self.assertEqual(report['crc_supported_segments'], 9)
            with np.load(root/'positive/keys.npz') as result:
                self.assertTrue(result['candidate_mask'].all())
                np.testing.assert_array_equal(result['ks'], keys)
            # Same anchors and seed hypotheses, independent random ciphertext.
            du[:, :, 698:714] = rng.integers(0, 256, cipher.shape, dtype=np.uint8)
            np.savez(up, counters=np.repeat(counters, 3), phases=np.tile(np.arange(3), n),
                     du=du.reshape(-1, 716))
            with contextlib.redirect_stdout(io.StringIO()):
                recover(gp, up, root/'negative')
            negative = json.loads((root/'negative/report.json').read_text())
            self.assertEqual(negative['crc_supported_segments'], 0)
            self.assertEqual(len(negative['matched_pairs']), 0)

    def test_packet_salvage_positive_control(self):
        # Two observations each have a different bad phase-1 packet. No full
        # group survives separately, but bracketing phase-0 counters are intact.
        base = group(100) + group(101)
        a, b = bytearray(base), bytearray(base)
        # record length 732; phase 1, packet 1 versus packet 2. Neither is q0/q3.
        a[732 + 4 + 182 + 20] ^= 1
        b[732 + 4 + 364 + 20] ^= 1
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); pa, pb = root/'a.bin', root/'b.bin'
            pa.write_bytes(a); pb.write_bytes(b)
            with contextlib.redirect_stdout(io.StringIO()):
                salvage([pa, pb], root/'output')
            report = json.loads((root/'output/report.json').read_text())
            self.assertEqual(report['original_complete_groups'], 1)
            self.assertEqual(report['salvaged_complete_groups'], 2)
            self.assertEqual(report['extra_groups'], 1)
            self.assertEqual(report['reassembly_crc_failures'], 0)


if __name__ == '__main__':
    unittest.main()
