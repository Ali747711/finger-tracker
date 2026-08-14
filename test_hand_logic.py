import math

import pytest

from hand_logic import MovementTracker, count_fingers, fingers_up


def make_landmarks(overrides=None):
    """21 landmarks for a neutral open hand facing the camera.

    Base pose: wrist low (y=0.9), fingers extended upward, thumb spread
    out to the side. Hand scale (wrist to middle MCP) is 0.2.
    """
    lm = list([(0.5, 0.9)] * 21)
    lm[0] = (0.50, 0.90)   # wrist
    lm[9] = (0.50, 0.70)   # middle MCP -> hand scale 0.2
    lm[5] = (0.45, 0.70)   # index MCP
    lm[17] = (0.55, 0.70)  # pinky MCP
    lm[3] = (0.40, 0.75)   # thumb IP
    lm[4] = (0.35, 0.70)   # thumb tip, spread away from the index MCP
    for tip in (8, 12, 16, 20):
        lm[tip - 2] = (0.5, 0.55)   # PIP joints
        lm[tip] = (0.5, 0.35)       # fingertips, well past the joints
    if overrides:
        for i, pos in overrides.items():
            lm[i] = pos
    return lm


def rotate(landmarks, degrees):
    """Rotate a hand about its wrist — what tilting your hand looks like."""
    angle = math.radians(degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ox, oy = landmarks[0]
    return [(ox + (x - ox) * cos_a - (y - oy) * sin_a,
             oy + (x - ox) * sin_a + (y - oy) * cos_a) for x, y in landmarks]


class TestFingersUp:
    def test_open_hand_all_up(self):
        assert count_fingers(make_landmarks()) == 5

    def test_curled_index_is_down(self):
        lm = make_landmarks({8: (0.5, 0.65)})   # tip pulled back to the palm
        up = fingers_up(lm)
        assert up['index'] is False
        assert count_fingers(lm) == 4

    def test_extension_survives_hand_rotation(self):
        # People tilt their hand constantly. Judging extension by tip
        # height instead of distance from the wrist starts misreading
        # around 45 degrees and inverts completely past 90.
        lm = make_landmarks()
        for degrees in (0, 30, 45, 60, 90, 135, 180, 225, 270, 315):
            assert count_fingers(rotate(lm, degrees)) == 5, degrees

    def test_curled_finger_stays_curled_when_rotated(self):
        lm = make_landmarks({8: (0.5, 0.65)})
        for degrees in (0, 45, 90, 180, 270):
            assert fingers_up(rotate(lm, degrees))['index'] is False, degrees

    def test_folded_thumb_reads_as_down(self):
        # tucked across the palm, close to the index knuckle
        lm = make_landmarks({4: (0.47, 0.68)})
        assert fingers_up(lm)['thumb'] is False

    def test_rejects_wrong_landmark_count(self):
        with pytest.raises(ValueError):
            fingers_up([(0.5, 0.5)] * 20)


class TestMovementTracker:
    def test_first_frame_is_still(self):
        assert MovementTracker().update(0.5, 0.5) == 'still'

    def test_small_jitter_is_still(self):
        t = MovementTracker()
        t.update(0.5, 0.5)
        assert t.update(0.51, 0.505) == 'still'

    def test_moving_right(self):
        t = MovementTracker()
        t.update(0.3, 0.5)
        assert t.update(0.4, 0.5) == 'right'

    def test_moving_up_means_smaller_y(self):
        t = MovementTracker()
        t.update(0.5, 0.6)
        assert t.update(0.5, 0.4) == 'up'

    def test_reset_clears_history(self):
        t = MovementTracker()
        t.update(0.1, 0.5)
        t.reset()
        # after reset this is a "first frame" again -> still
        assert t.update(0.9, 0.5) == 'still'

    def test_diagonal_prefers_dominant_axis(self):
        t = MovementTracker()
        t.update(0.5, 0.5)
        assert t.update(0.7, 0.55) == 'right'

    def test_slow_drift_stays_still(self):
        # 0.005/frame: within the 5-frame window the span is 0.02 < 0.04.
        # An unbounded history would accumulate the drift and report 'right'.
        t = MovementTracker()
        results = [t.update(0.5 + 0.005 * i, 0.5) for i in range(20)]
        assert all(r == 'still' for r in results)

    def test_window_drops_old_samples(self):
        t = MovementTracker(window=3)
        t.update(0.50, 0.5)
        t.update(0.53, 0.5)
        assert t.update(0.56, 0.5) == 'right'  # span 0.06 across the window
        assert t.update(0.59, 0.5) == 'right'  # 0.50 dropped; span still 0.06
        assert t.update(0.59, 0.5) == 'still'  # window now 0.56..0.59 = 0.03
