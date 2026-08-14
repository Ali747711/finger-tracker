from hand_logic import MovementTracker, count_fingers, fingers_up


def make_landmarks(overrides=None):
    """21 landmarks for a neutral open right hand facing the camera.

    Base pose: wrist low (y=0.9), all tips high (y small) -> all fingers up.
    """
    lm = [(0.5, 0.9)] * 21
    lm = list(lm)
    # thumb: IP joint (3) and tip (4); right hand thumb tip further left
    lm[3] = (0.40, 0.75)
    lm[4] = (0.35, 0.70)
    # each finger: PIP at tip-2 below the tip
    for tip in (8, 12, 16, 20):
        lm[tip - 2] = (0.5, 0.55)
        lm[tip] = (0.5, 0.35)
    if overrides:
        for i, pos in overrides.items():
            lm[i] = pos
    return lm


class TestFingersUp:
    def test_open_hand_all_up(self):
        assert count_fingers(make_landmarks()) == 5

    def test_curled_index_is_down(self):
        # index tip (8) below its PIP joint (6)
        lm = make_landmarks({8: (0.5, 0.65)})
        up = fingers_up(lm)
        assert up['index'] is False
        assert count_fingers(lm) == 4

    def test_left_hand_thumb_flips(self):
        # same geometry read as a left hand -> thumb reads as folded
        lm = make_landmarks()
        assert fingers_up(lm, 'Right')['thumb'] is True
        assert fingers_up(lm, 'Left')['thumb'] is False

    def test_rejects_wrong_landmark_count(self):
        import pytest

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
