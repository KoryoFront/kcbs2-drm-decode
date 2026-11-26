"""Check exact reported slot coordinates and rejection of gap-crossing mode edges."""
import unittest

import numpy as np

from audit_capture_layout import inspect
from render_continuous import MODE_POSITIONS
from rf_pulse_check import localized_pulses


class CaptureLayoutTest(unittest.TestCase):
    def test_mode_edge_does_not_bridge_a_missing_group(self):
        counters = [40, 42, 43]
        du = np.zeros((3, 3, 716), dtype=np.uint8)
        du[:, 0, 664:668] = np.array(counters, dtype='>u4').view(np.uint8).reshape(3, 4)
        du[:, 1, 1] = 0x10
        du[:, 2, 1] = 0x20
        du[0, :, MODE_POSITIONS] = 0xfe
        du[1, :, MODE_POSITIONS] = 0xf8
        du[2, :, MODE_POSITIONS] = 0xfe
        du[2, 1, MODE_POSITIONS[9:]] = 0xf8
        du[2, 2, MODE_POSITIONS] = 0xf8
        a = inspect(counters, du, 0xfe, 0xf8, 20)
        self.assertEqual(a['phase_layout_matching'], 3)
        self.assertEqual(a['counter_fields_matching'], 3)
        self.assertEqual(a['zero_layout_matching'], 3)
        self.assertEqual(a['distinct_trailers'], 1)
        self.assertEqual(a['edges'], [dict(counter=43, phase=1, slot=9,
                                           relative_seconds=4.18, lead_seconds=.58)])

    def test_local_pulse_requires_two_receivers_and_two_windows(self):
        t = np.arange(6000)/50
        rng = np.random.default_rng(4135)
        x, y = rng.standard_normal((2,len(t)))
        pulse = 5*(np.sin(2*np.pi*.8*t)>0)
        self.assertEqual(localized_pulses(x+pulse,y)['adjacent_window_matches'], [])
        self.assertEqual(localized_pulses(x+pulse*(t<40),y+pulse*(t<40))['adjacent_window_matches'], [])
        matches = localized_pulses(x+pulse,y+pulse)['adjacent_window_matches']
        self.assertIn(dict(start_seconds=0.,hz=.8),matches)


if __name__ == '__main__':
    unittest.main()
