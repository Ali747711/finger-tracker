import pytest

from gestures import PinchDetector, SwipeDetector, hand_scale


def pinch_landmarks(ratio):
    """21 landmarks with hand scale 0.2 and thumb–index tip distance
    set to `ratio` * scale."""
    lm = [(0.5, 0.5)] * 21
    lm[0] = (0.5, 0.9)   # wrist
    lm[9] = (0.5, 0.7)   # middle MCP -> scale = 0.2
    lm[4] = (0.4, 0.5)   # thumb tip
    lm[8] = (0.4 + ratio * 0.2, 0.5)  # index tip at requested ratio
    return lm


class TestHandScale:
    def test_scale_is_wrist_to_middle_mcp(self):
        assert hand_scale(pinch_landmarks(1.0)) == pytest.approx(0.2)


class TestPinchDetector:
    def test_open_hand_no_event(self):
        p = PinchDetector()
        assert p.update(pinch_landmarks(1.0)) is None
        assert p.is_pinching is False

    def test_closing_fires_pinch_start_once(self):
        p = PinchDetector()
        p.update(pinch_landmarks(1.0))
        assert p.update(pinch_landmarks(0.2)) == 'pinch_start'
        assert p.update(pinch_landmarks(0.2)) is None  # no repeat
        assert p.is_pinching is True

    def test_hysteresis_band_keeps_state(self):
        p = PinchDetector()
        p.update(pinch_landmarks(0.2))          # start pinch
        assert p.update(pinch_landmarks(0.45)) is None  # between on and off
        assert p.is_pinching is True

    def test_opening_fires_pinch_end(self):
        p = PinchDetector()
        p.update(pinch_landmarks(0.2))
        # the release needs consecutive open frames
        assert p.update(pinch_landmarks(0.8)) is None
        assert p.update(pinch_landmarks(0.8)) is None
        assert p.update(pinch_landmarks(0.8)) == 'pinch_end'
        assert p.is_pinching is False

    def test_drag_survives_one_noisy_frame(self):
        # landmarks get noisy while the hand moves; a single bad frame
        # dropping the pinch is what makes a drag let go halfway
        p = PinchDetector()
        p.update(pinch_landmarks(0.2))
        assert p.update(pinch_landmarks(0.8)) is None   # noise
        assert p.update(pinch_landmarks(0.2)) is None   # still pinching
        assert p.is_pinching is True
        # and the open-frame count restarted, so a real release still
        # takes the full run of open frames
        assert p.update(pinch_landmarks(0.8)) is None
        assert p.update(pinch_landmarks(0.8)) is None
        assert p.update(pinch_landmarks(0.8)) == 'pinch_end'

    def test_rejects_bad_release_frames(self):
        with pytest.raises(ValueError):
            PinchDetector(release_frames=0)

    def test_degenerate_scale_ignored(self):
        lm = [(0.5, 0.5)] * 21  # wrist == middle MCP -> scale 0
        p = PinchDetector()
        assert p.update(lm) is None

    def test_rejects_wrong_landmark_count(self):
        with pytest.raises(ValueError):
            PinchDetector().update([(0.5, 0.5)] * 20)

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            PinchDetector(on_ratio=0.6, off_ratio=0.4)

    def test_near_pinch_does_not_start(self):
        # inside the hysteresis band (ON 0.35 < 0.45 < OFF 0.55) a hand that
        # is NOT yet pinching must stay un-pinched — pins the ON threshold
        # so it can't silently degrade to the OFF threshold
        p = PinchDetector()
        assert p.update(pinch_landmarks(0.45)) is None
        assert p.is_pinching is False

    def test_just_above_on_threshold_no_start(self):
        p = PinchDetector()
        assert p.update(pinch_landmarks(0.36)) is None
        assert p.is_pinching is False

    def test_foreshortened_palm_keeps_scale(self):
        # hand pitched at the camera: wrist->middle-MCP projection collapses,
        # but the knuckle span (5->17) stays wide, so a held pinch still
        # reads as a pinch
        lm = [(0.5, 0.5)] * 21
        lm[0] = (0.5, 0.9)
        lm[9] = (0.5, 0.87)    # foreshortened: only 0.03 long
        lm[5] = (0.42, 0.85)
        lm[17] = (0.58, 0.85)  # knuckle span 0.16 stays lateral
        lm[4] = (0.49, 0.5)
        lm[8] = (0.52, 0.5)    # tips 0.03 apart -> ratio 0.1875 vs span
        p = PinchDetector()
        assert p.update(lm) == 'pinch_start'

    def test_reset_clears_pinch(self):
        p = PinchDetector()
        p.update(pinch_landmarks(0.2))
        p.reset()
        assert p.is_pinching is False

    def test_reset_requires_open_hand_before_new_pinch(self):
        # hand lost mid-pinch, re-detected still pinched: must NOT auto-fire
        # pinch_start (would re-press the mouse at a jumped position)
        p = PinchDetector()
        p.update(pinch_landmarks(0.2))
        p.reset()
        assert p.update(pinch_landmarks(0.2)) is None
        assert p.is_pinching is False
        p.update(pinch_landmarks(0.8))  # user visibly opens the hand
        assert p.update(pinch_landmarks(0.2)) == 'pinch_start'


class TestSwipeDetector:
    def test_fast_right_flick_fires(self):
        s = SwipeDetector()
        s.update(0.3, 0.5, 0.0)
        s.update(0.45, 0.5, 0.1)
        assert s.update(0.6, 0.5, 0.2) == 'swipe_right'

    def test_up_means_smaller_y(self):
        s = SwipeDetector()
        s.update(0.5, 0.7, 0.0)
        assert s.update(0.5, 0.4, 0.15) == 'swipe_up'

    def test_slow_drift_does_not_fire(self):
        s = SwipeDetector()
        # 0.01/frame at 30fps -> only ~0.09 inside the 0.3s window
        t, x = 0.0, 0.2
        for _ in range(60):
            assert s.update(x, 0.5, t) is None
            t += 1 / 30
            x += 0.01

    def test_diagonal_is_ambiguous(self):
        s = SwipeDetector()
        s.update(0.3, 0.3, 0.0)
        assert s.update(0.55, 0.55, 0.2) is None

    def test_cooldown_blocks_second_swipe(self):
        s = SwipeDetector()
        s.update(0.3, 0.5, 0.0)
        assert s.update(0.6, 0.5, 0.2) == 'swipe_right'
        # a second full flick inside the cooldown is swallowed
        s.update(0.3, 0.5, 0.3)
        assert s.update(0.6, 0.5, 0.5) is None

    def test_fires_again_after_cooldown(self):
        s = SwipeDetector()
        s.update(0.3, 0.5, 0.0)
        assert s.update(0.6, 0.5, 0.2) == 'swipe_right'
        # cooldown (0.6s) has passed; new flick starting fresh
        s.update(0.6, 0.5, 1.0)
        assert s.update(0.5, 0.5, 1.1) is None  # not far enough yet
        assert s.update(0.28, 0.5, 1.2) == 'swipe_left'

    def test_old_samples_fall_out_of_window(self):
        s = SwipeDetector()
        s.update(0.2, 0.5, 0.0)
        s.update(0.4, 0.5, 0.1)   # moved, but not enough yet
        # long pause; earlier movement must not count anymore
        assert s.update(0.42, 0.5, 2.0) is None

    def test_return_stroke_after_cooldown_does_not_fire(self):
        s = SwipeDetector()
        s.update(0.3, 0.5, 0.0)
        assert s.update(0.6, 0.5, 0.2) == 'swipe_right'  # quiet until 0.8
        # natural return stroke while the detector is quiet
        s.update(0.6, 0.5, 0.55)
        s.update(0.45, 0.5, 0.65)
        s.update(0.3, 0.5, 0.75)
        # first frame past the cooldown with a near-still hand: the buffered
        # return stroke must NOT fire a phantom swipe_left
        assert s.update(0.28, 0.5, 0.82) is None

    def test_reset_clears_history(self):
        s = SwipeDetector()
        s.update(0.2, 0.5, 0.0)
        s.reset()
        assert s.update(0.6, 0.5, 0.1) is None  # fresh start, one sample

    def test_reset_preserves_cooldown(self):
        # hand lost right after a swipe: the cooldown must survive reset so
        # a re-detected hand can't double-fire the same flick
        s = SwipeDetector()
        s.update(0.3, 0.5, 0.0)
        assert s.update(0.6, 0.5, 0.2) == 'swipe_right'  # quiet until 0.8
        s.reset()
        s.update(0.3, 0.5, 0.4)
        assert s.update(0.6, 0.5, 0.6) is None  # full flick, still quiet
