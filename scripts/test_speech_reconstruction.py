"""Verify that packet replay preserves time and never uses unknown source slots."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from november_speech_reconstruction import write_stream


def read_stream(path):
    words = np.fromfile(path, '<u2')
    rows = []
    i = 0
    while i < len(words):
        sync, length = map(int, words[i:i+2]); i += 2
        payload = words[i:i+length]; i += length
        rows.append((sync, np.packbits(payload == 0x81).tobytes()))
    return rows


class SpeechReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.random.default_rng(23).integers(0,256,(12,33),dtype=np.uint8)

    def render(self, method):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'frames.192'
            write_stream(p,self.frames,method)
            return read_stream(p)

    def test_loss_slots_preserve_frame_count(self):
        rows = self.render('native_plc')
        self.assertEqual(len(rows),12)
        for i,(sync,data) in enumerate(rows):
            if i%4:
                self.assertEqual((sync,data),(0x6b20,b''))
            else:
                self.assertEqual(sync,0x6b21)
                self.assertEqual(data,self.frames[i].tobytes())

    def test_nearest_only_uses_known_packets(self):
        rows = self.render('nearest_packet')
        expected = [0,0,4,4,4,4,8,8,8,8,8,8]
        self.assertEqual(len(rows),12)
        for (sync,data),source in zip(rows,expected):
            self.assertEqual(sync,0x6b21)
            self.assertEqual(data,self.frames[source].tobytes())

    def test_previous_only_uses_known_packets(self):
        rows = self.render('previous_packet')
        self.assertEqual(len(rows),12)
        for i,(sync,data) in enumerate(rows):
            self.assertEqual(sync,0x6b21)
            self.assertEqual(data,self.frames[(i//4)*4].tobytes())


if __name__=='__main__':
    unittest.main()
