"""Verify that record review only keeps bytes already present in a supplied archive."""
import unittest

import numpy as np

from parse_text_records import crc
from render_supported_nonidle import with_loss_policy
from review_supplied_records import review
from test_text_records import make_record


class SuppliedRecordsTest(unittest.TestCase):
    def setUp(self):
        self.c = np.arange(100,103,dtype=np.uint32)
        self.count = np.repeat(self.c,3)
        self.phase = np.tile(np.arange(3,dtype=np.uint8),3)
        self.keys = np.arange(48,dtype=np.uint8).reshape(3,16)
        self.mask = np.ones_like(self.keys,dtype=bool)
        self.du = np.zeros((9,716),dtype=np.uint8)
        self.du[:,1] = self.phase*16
        stream = bytearray(144)
        stream[:32] = make_record('한'.encode('utf-16le'),sequence=2)
        stream[64:96] = make_record('한'.encode('utf-16le'),sequence=2,toggle=1)
        self.du[:,698:714] = np.frombuffer(stream,dtype=np.uint8).reshape(9,16) ^ np.repeat(self.keys,3,axis=0)
        for i, row in enumerate(self.du):
            if not self.phase[i]:
                row[664:668] = list(int(self.count[i]).to_bytes(4,'big'))
            row[714:] = list(crc(row[:714]).to_bytes(2,'big'))

    def checked(self, period):
        return review(self.c,self.keys,self.mask,self.count,self.phase,self.du,period)

    def test_repeat_includes_both_observed_copies_but_not_other_prefixes(self):
        mask, rows = self.checked(4)
        np.testing.assert_array_equal(mask.all(1), [True, True, False])
        self.assertEqual([r['matching_copies'] for r in rows], [2,2])
        mask, rows = self.checked(3)
        self.assertFalse(mask.any())
        self.assertEqual([r['matching_copies'] for r in rows], [1,1])

    def test_unknown_byte_is_never_added_even_with_repeat(self):
        self.mask[1,7] = False
        mask, rows = self.checked(4)
        self.assertFalse(mask.any())
        self.assertEqual(len(rows),1)

    def test_data_unit_crc_is_rechecked(self):
        self.du[0,698] ^= 1
        with self.assertRaisesRegex(ValueError,'data-unit CRC'):
            self.checked(0)

    def test_future_policy_leaves_role_two_and_unreceived_slots_as_loss(self):
        frames = np.repeat(np.arange(16,dtype=np.uint8)[:,None],33,axis=1)
        source = np.array([True, True, False, False]*4)
        source[8:12] = False
        changed, keep = with_loss_policy(frames,source,'future')
        np.testing.assert_array_equal(np.flatnonzero(keep & ~source), [3])
        np.testing.assert_array_equal(changed[3],frames[4])


if __name__ == '__main__':
    unittest.main()
