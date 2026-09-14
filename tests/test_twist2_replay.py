"""Replay timing and command order, exercised in the simulator container."""

import unittest

try:
    import numpy as np
    from embodied_control.sim.twist2_replay import body_command, ticks_for_frame
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "requires TWIST2 runtime")
class ReplayTests(unittest.TestCase):
    def test_dataset_body_last_root_becomes_controller_root_first(self):
        command = body_command(np.arange(47))
        np.testing.assert_array_equal(command[:6], [29, 30, 31, 32, 33, 34])
        np.testing.assert_array_equal(command[6:], np.arange(29))

    def test_hands_do_not_affect_body(self):
        action = np.zeros(47)
        command = body_command(action)
        action[35:] = 100
        np.testing.assert_array_equal(command, body_command(action))

    def test_no_accumulated_clock_drift(self):
        self.assertEqual(sum(ticks_for_frame(i) for i in range(600)), 1000)
        self.assertEqual(sum(ticks_for_frame(i) for i in range(633)), 1055)

    def test_invalid_command_rejected(self):
        for action in (np.zeros(46), np.full(47, np.nan)):
            with self.assertRaises(ValueError):
                body_command(action)


if __name__ == "__main__":
    unittest.main()
