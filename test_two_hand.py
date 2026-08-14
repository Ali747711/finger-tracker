import pytest

from two_hand import IndexTouchDetector


def hand(index_x, index_up=True, scale=0.2):
    """One hand whose index tip sits at index_x, with the given hand
    scale (wrist to middle MCP). Returns (points, fingers_up)."""
    points = [(0.5, 0.9)] * 21
    points = list(points)
    points[0] = (0.5, 0.9)
    points[9] = (0.5, 0.9 - scale)     # wrist -> middle MCP sets the scale
    points[5] = points[17] = (0.5, 0.9 - scale)
    points[8] = (index_x, 0.5)
    fingers = {'thumb': False, 'index': index_up, 'middle': False,
               'ring': False, 'pinky': False}
    return points, fingers


def touching(gap=0.02):
    """Two hands with index tips `gap` apart (scale 0.2, so a gap of
    0.02 is a ratio of 0.1 — comfortably a touch)."""
    return [hand(0.5), hand(0.5 + gap)]


def apart(gap=0.3):
    return [hand(0.5), hand(0.5 + gap)]


class TestIndexTouchDetector:
    def test_touch_fires_once(self):
        d = IndexTouchDetector()
        assert d.update(touching(), 0.0) == 'index_touch'
        assert d.is_touching is True
        # holding the pose must not launch anything again
        assert d.update(touching(), 0.1) is None
        assert d.update(touching(), 0.2) is None

    def test_holding_the_pose_never_refires(self):
        # resting the fingertips together must not relaunch the app once
        # per cooldown — only separating and touching again may fire
        d = IndexTouchDetector(cooldown=1.0)
        assert d.update(touching(), 0.0) == 'index_touch'
        for moment in (0.5, 1.1, 2.0, 5.0):
            assert d.update(touching(), moment) is None, moment

    def test_hands_apart_do_not_fire(self):
        d = IndexTouchDetector()
        assert d.update(apart(), 0.0) is None
        assert d.is_touching is False

    def test_separating_then_touching_fires_again(self):
        d = IndexTouchDetector()
        assert d.update(touching(), 0.0) == 'index_touch'
        assert d.update(apart(), 1.0) is None      # latch clears
        assert d.is_touching is False
        assert d.update(touching(), 2.0) == 'index_touch'

    def test_cooldown_blocks_a_quick_retouch(self):
        d = IndexTouchDetector(cooldown=1.5)
        assert d.update(touching(), 0.0) == 'index_touch'
        d.update(apart(), 0.2)                     # separated already
        assert d.update(touching(), 0.5) is None   # still inside cooldown

    def test_one_hand_cannot_fire(self):
        d = IndexTouchDetector()
        assert d.update([hand(0.5)], 0.0) is None
        assert d.update([], 0.0) is None

    def test_needs_both_index_fingers_extended(self):
        # a fist bump must not trigger it
        d = IndexTouchDetector()
        hands = [hand(0.5, index_up=False), hand(0.52, index_up=False)]
        assert d.update(hands, 0.0) is None

    def test_one_curled_finger_is_not_enough(self):
        d = IndexTouchDetector()
        hands = [hand(0.5), hand(0.52, index_up=False)]
        assert d.update(hands, 0.0) is None

    def test_dropped_frame_does_not_refire(self):
        # a frame where only one hand is seen must not re-arm the trigger,
        # or the same touch would fire twice
        d = IndexTouchDetector()
        assert d.update(touching(), 0.0) == 'index_touch'
        assert d.update([hand(0.5)], 0.1) is None
        assert d.update(touching(), 0.2) is None

    def test_losing_a_hand_reports_no_contact(self):
        # the camera loop holds off single-hand control while the tips are
        # touching, so a stale True here froze the cursor, clicks and
        # scrolling permanently once a touch had happened
        d = IndexTouchDetector()
        d.update(touching(), 0.0)
        assert d.is_touching is True
        d.update([hand(0.5)], 0.1)
        assert d.is_touching is False
        for moment in (1.0, 5.0, 60.0):
            d.update([hand(0.5)], moment)
            assert d.is_touching is False, moment

    def test_one_index_curled_reports_no_contact(self):
        d = IndexTouchDetector()
        d.update(touching(), 0.0)
        d.update([hand(0.5), hand(0.52, index_up=False)], 0.1)
        assert d.is_touching is False

    def test_hands_apart_report_no_contact(self):
        d = IndexTouchDetector()
        d.update(touching(), 0.0)
        d.update(apart(), 0.5)
        assert d.is_touching is False

    def test_works_at_any_distance_from_camera(self):
        # the same gesture, hands further from the camera (smaller scale)
        d = IndexTouchDetector()
        small = [hand(0.5, scale=0.08), hand(0.508, scale=0.08)]
        assert d.update(small, 0.0) == 'index_touch'

    def test_wide_gap_scaled_by_small_hands_is_not_a_touch(self):
        d = IndexTouchDetector()
        small = [hand(0.5, scale=0.08), hand(0.56, scale=0.08)]
        assert d.update(small, 0.0) is None

    def test_degenerate_scale_ignored(self):
        d = IndexTouchDetector()
        flat = [hand(0.5, scale=0.0), hand(0.5, scale=0.0)]
        assert d.update(flat, 0.0) is None

    def test_reset_clears_the_latch(self):
        d = IndexTouchDetector()
        d.update(touching(), 0.0)
        d.reset()
        assert d.is_touching is False

    def test_rejects_bad_thresholds(self):
        with pytest.raises(ValueError):
            IndexTouchDetector(on_ratio=0.8, off_ratio=0.3)
        with pytest.raises(ValueError):
            IndexTouchDetector(cooldown=-1)
