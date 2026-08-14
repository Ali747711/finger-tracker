import pytest

from hands import (HandRegistry, HandTracker, assign_labels, other_hand,
                   pick_primary)


def hand_landmarks(pinch_ratio=1.0, index_up=True, x=0.5):
    """21 landmarks for one hand.

    Hand scale is 0.2 (wrist to middle MCP), and the thumb tip is placed
    beside the index tip so the thumb-index gap is exactly
    pinch_ratio * scale — i.e. pinch_ratio is the ratio PinchDetector
    compares against its thresholds.
    """
    lm = [(x, 0.9)] * 21
    lm = list(lm)
    lm[0] = (x, 0.9)          # wrist
    lm[9] = (x, 0.7)          # middle MCP -> scale 0.2
    lm[5] = lm[17] = (x, 0.7)  # knuckle span collapsed, so 0-9 sets scale
    lm[3] = (x - 0.10, 0.75)  # thumb IP
    for tip in (8, 12, 16, 20):
        lm[tip - 2] = (x, 0.55)   # PIP joints
        lm[tip] = (x, 0.35)       # extended fingertips

    tip_y = 0.35 if index_up else 0.65   # 0.65 is below the PIP -> curled
    lm[8] = (x, tip_y)
    lm[4] = (x - pinch_ratio * 0.2, tip_y)  # thumb reaches to the index tip
    return lm


class TestLabelAssignment:
    def test_distinct_labels_pass_through(self):
        assert assign_labels([('Left', 0.9), ('Right', 0.8)]) == ['Left', 'Right']

    def test_duplicate_labels_are_split(self):
        # both hands classified 'Right': the confident one keeps it
        assert assign_labels([('Right', 0.6), ('Right', 0.95)]) == \
            ['Left', 'Right']

    def test_duplicate_left_is_split(self):
        assert assign_labels([('Left', 0.99), ('Left', 0.5)]) == \
            ['Left', 'Right']

    def test_three_way_collision_gets_unique_keys(self):
        out = assign_labels([('Right', 0.9), ('Right', 0.8), ('Right', 0.7)])
        assert len(set(out)) == 3
        assert out[0] == 'Right'

    def test_single_hand(self):
        assert assign_labels([('Left', 0.9)]) == ['Left']

    def test_no_hands(self):
        assert assign_labels([]) == []

    def test_other_hand(self):
        assert other_hand('Right') == 'Left'
        assert other_hand('Left') == 'Right'


class TestPrimarySelection:
    def test_prefers_configured_hand(self):
        assert pick_primary(['Left', 'Right']) == 'Right'

    def test_falls_back_to_the_only_hand(self):
        assert pick_primary(['Left']) == 'Left'

    def test_none_without_hands(self):
        assert pick_primary([]) is None

    def test_preference_is_configurable(self):
        assert pick_primary(['Left', 'Right'], preferred='Left') == 'Left'

    def test_unavailable_preference_falls_back(self):
        assert pick_primary(['Left'], preferred='Right') == 'Left'


class TestHandTracker:
    def test_reports_fingers_and_pinch(self):
        t = HandTracker()
        events = t.update(hand_landmarks(pinch_ratio=0.2), 'Right', 0.0)
        assert 'pinch_start' in events
        assert t.is_pinching is True
        assert t.fingers['index'] is True

    def test_open_hand_no_events(self):
        t = HandTracker()
        assert t.update(hand_landmarks(), 'Right', 0.0) == []

    def test_lost_closes_a_held_pinch(self):
        t = HandTracker()
        t.update(hand_landmarks(pinch_ratio=0.2), 'Right', 0.0)
        assert t.lost() == ['pinch_end']
        assert t.is_pinching is False

    def test_lost_without_pinch_owes_nothing(self):
        t = HandTracker()
        t.update(hand_landmarks(), 'Right', 0.0)
        assert t.lost() == []

    def test_lost_requires_open_hand_before_new_pinch(self):
        # the reacquire guard must survive going through the tracker
        t = HandTracker()
        t.update(hand_landmarks(pinch_ratio=0.2), 'Right', 0.0)
        t.lost()
        assert t.update(hand_landmarks(pinch_ratio=0.2), 'Right', 0.1) == []

    def test_lost_resets_the_palm_gate(self):
        t = HandTracker()
        open_hand = hand_landmarks()
        for _ in range(3):
            t.update(open_hand, 'Right', 0.0)
        armed = dict(t.fingers)
        assert t.palm.update(armed) is True   # palm held long enough
        t.lost()
        # a returning hand has to hold the palm again before swipes arm
        assert t.palm.update(armed) is False

    def test_hands_do_not_share_state(self):
        left, right = HandTracker(), HandTracker()
        right.update(hand_landmarks(pinch_ratio=0.2), 'Right', 0.0)
        assert right.is_pinching is True
        assert left.is_pinching is False
        assert left.update(hand_landmarks(), 'Left', 0.0) == []


def flick_landmarks(x, y, palm=True):
    """An open hand at (x, y). palm=False is the blurred frame where
    the fingers misread as curled, which happens mid-flick."""
    lm = list([(x, 0.9 + y)] * 21)
    lm[0] = (x, 0.9 + y)
    lm[9] = lm[5] = lm[17] = (x, 0.7 + y)
    lm[3] = (x - 0.10, 0.75 + y)
    lm[4] = (x - 0.20, 0.35 + y)
    for tip in (8, 12, 16, 20):
        lm[tip - 2] = (x, 0.55 + y)
        lm[tip] = (x, (0.35 if palm else 0.65) + y)
    return lm


class TestSwipeDetection:
    """A swipe is a fast flick, and motion blur makes MediaPipe misread
    the fingers exactly while it happens. These pin that a real flick
    still fires, in the right direction."""

    def flick(self, blur=(), dx=0.06, dy=0.0, frames=6):
        t = HandTracker()
        now, x, y = 0.0, 0.30, -0.30
        for _ in range(4):                    # hold the palm to arm
            t.update(flick_landmarks(x, y), 'Right', now)
            now += 1 / 25
        events = []
        for i in range(frames):
            x, y = x + dx, y + dy
            events += t.update(
                flick_landmarks(x, y, i not in blur), 'Right', now)
            now += 1 / 25
        return events

    def test_clean_flick_fires(self):
        assert self.flick() == ['swipe_right']

    def test_flick_survives_blurred_frames(self):
        # the palm gate must not disarm and wipe the swipe history here
        assert self.flick(blur=(1, 2)) == ['swipe_right']

    def test_blur_does_not_bend_the_direction(self):
        # tracking a fingertip would read the finger curl as a big
        # vertical move and report swipe_down for a sideways flick
        assert self.flick(blur=(1, 2, 3)) == ['swipe_right']

    def test_vertical_flick(self):
        assert self.flick(dx=0.0, dy=0.06) == ['swipe_down']

    def test_slow_drift_does_not_fire(self):
        assert self.flick(dx=0.008) == []


class TestHandRegistry:
    def test_creates_one_tracker_per_hand(self):
        r = HandRegistry()
        left, right = r.get('Left'), r.get('Right')
        assert left is not right
        assert r.get('Left') is left  # stable across frames

    def test_active_lists_seen_hands(self):
        r = HandRegistry()
        r.get('Left')
        r.get('Right')
        assert r.active() == ['Left', 'Right']

    def test_brief_dropout_keeps_hand_active(self):
        r = HandRegistry(lost_frames=3)
        r.get('Right')
        assert r.sweep([]) == []          # 1 missed
        assert r.active() == ['Right']
        assert r.sweep([]) == []          # 2 missed
        assert r.active() == ['Right']

    def test_sustained_absence_reports_loss_once(self):
        r = HandRegistry(lost_frames=3)
        tracker = r.get('Right')
        r.sweep([])
        r.sweep([])
        assert r.sweep([]) == [('Right', tracker)]  # crosses the threshold
        assert r.sweep([]) == []                    # not reported again
        assert r.active() == []

    def test_returning_hand_reuses_its_tracker(self):
        r = HandRegistry(lost_frames=2)
        tracker = r.get('Right')
        r.sweep([])
        r.sweep([])
        assert r.get('Right') is tracker
        assert r.active() == ['Right']

    def test_one_hand_leaving_does_not_disturb_the_other(self):
        r = HandRegistry(lost_frames=2)
        left = r.get('Left')
        right = r.get('Right')
        r.sweep(['Left'])                       # Right missed once
        assert r.sweep(['Left']) == [('Right', right)]
        assert r.active() == ['Left']
        assert r.get('Left') is left

    def test_production_default_tolerates_a_dropout(self):
        # main.py builds HandRegistry() with the module default; a
        # 1-frame threshold would reset gesture state on every flicker
        r = HandRegistry()
        r.get('Right')
        assert r.sweep([]) == []
        assert r.active() == ['Right']

    def test_rejects_bad_threshold(self):
        with pytest.raises(ValueError):
            HandRegistry(lost_frames=0)
