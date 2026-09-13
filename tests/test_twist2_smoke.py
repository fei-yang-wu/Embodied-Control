"""TWIST2 body bridge contracts; run inside the smoke container."""
import unittest

try:
    import numpy as np
    from embodied_control.sim.twist2_eval import ACTION_DIMS, DEFAULT, body_command
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "requires the TWIST2 smoke runtime")
class BodyBridgeTests(unittest.TestCase):
    def test_named_order_and_bounded_targets(self):
        action = np.zeros(53)
        start = 0
        for name, width in ACTION_DIMS.items():
            if name in ("waist", "left_arm", "right_arm"):
                action[start:start + width] = 10
            start += width
        command = body_command(action, DEFAULT, np.tile([-3, 3], (29, 1)))
        np.testing.assert_array_equal(command[:6], [0, 0, .793, 0, 0, 0])
        np.testing.assert_array_equal(command[6:18], DEFAULT[:12])
        np.testing.assert_allclose(command[18:], DEFAULT[12:] + .05)

    def test_invalid_actions_are_rejected(self):
        for action in ([0] * 52, [float("nan")] * 53):
            with self.assertRaises(ValueError):
                body_command(action, DEFAULT, np.tile([-3, 3], (29, 1)))

    def test_joint_limits_are_applied(self):
        command = body_command(np.ones(53), np.zeros(29), np.tile([-.01, .01], (29, 1)))
        self.assertTrue(np.all(np.abs(command[6:]) <= .01))


if __name__ == "__main__":
    unittest.main()
