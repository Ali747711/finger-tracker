import pytest

from control import (ActionRouter, ControlSession, CursorMapper, PalmGate,
                     ScrollTracker, is_open_palm, is_pointer_pose,
                     is_scroll_pose)


def fingers(**overrides):
    up = {'thumb': False, 'index': False, 'middle': False,
          'ring': False, 'pinky': False}
    up.update(overrides)
    return up

POINTER = fingers(index=True)
SCROLL = fingers(index=True, middle=True)
OPEN_PALM = fingers(thumb=True, index=True, middle=True, ring=True, pinky=True)
FIST = fingers()


class TestCursorMapper:
    def test_center_maps_to_screen_center(self):
        m = CursorMapper(1000, 800, min_alpha=1.0, max_alpha=1.0)
        assert m.map(0.5, 0.5) == (500, 400)

    def test_margin_edges_clamp_to_screen_edges(self):
        m = CursorMapper(1000, 800, margin=0.2, min_alpha=1.0, max_alpha=1.0)
        assert m.map(0.2, 0.2) == (0, 0)
        m.reset()
        # last addressable pixel, not screen_w — coordinate 1000 would land
        # on an adjacent display in multi-monitor layouts
        assert m.map(0.8, 0.8) == (999, 799)
        m.reset()
        assert m.map(0.05, 0.95) == (0, 799)  # outside margin still clamps

    def test_smoothing_moves_partway(self):
        m = CursorMapper(1000, 800, margin=0.0, min_alpha=0.5, max_alpha=0.5)
        m.map(0.0, 0.0)
        # target jumps to (1000, 800); EMA with 0.5 lands halfway
        assert m.map(1.0, 1.0) == (500, 400)

    def test_reset_makes_next_map_jump(self):
        m = CursorMapper(1000, 800, margin=0.0, min_alpha=0.5, max_alpha=0.5)
        m.map(0.0, 0.0)
        m.reset()
        assert m.map(1.0, 1.0) == (999, 799)  # no glide after reset

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError):
            CursorMapper(1000, 800, margin=0.5)
        with pytest.raises(ValueError):
            CursorMapper(1000, 800, min_alpha=0)

    def test_production_defaults_are_pinned(self):
        # main.py builds CursorMapper with the module defaults; pin the
        # shipped cursor feel so it can't silently regress
        m = CursorMapper(1000, 800)
        assert m.map(0.5, 0.5) == (500, 400)      # first frame jumps
        # a big move is far above the speed reference: damping released
        assert m.map(0.65, 0.65) == (749, 599)

    def test_fast_motion_has_no_lag(self):
        m = CursorMapper(1000, 800, margin=0.0)
        m.map(0.0, 0.5)
        assert m.map(1.0, 0.5) == (999, 400)  # lands exactly on target

    def test_slow_motion_is_damped(self):
        # 1% of the screen is well under the speed reference, so the
        # cursor moves only part way — this is what holds it steady
        # enough that a pinch clicks where the user is aiming
        m = CursorMapper(1000, 800, margin=0.0)
        m.map(0.5, 0.5)
        assert m.map(0.51, 0.5)[0] == 503     # target is 509

    def test_rejects_inverted_alphas(self):
        with pytest.raises(ValueError):
            CursorMapper(1000, 800, min_alpha=0.9, max_alpha=0.2)
        with pytest.raises(ValueError):
            CursorMapper(1000, 800, speed_ref=0)


class TestPoses:
    def test_pointer_pose_ignores_thumb(self):
        assert is_pointer_pose(fingers(index=True)) is True
        assert is_pointer_pose(fingers(index=True, thumb=True)) is True

    def test_pointer_pose_needs_others_curled(self):
        assert is_pointer_pose(fingers(index=True, middle=True)) is False
        assert is_pointer_pose(FIST) is False

    def test_open_palm_needs_four_fingers(self):
        assert is_open_palm(OPEN_PALM) is True
        assert is_open_palm(fingers(index=True, middle=True,
                                    ring=True, pinky=True)) is True
        assert is_open_palm(fingers(index=True, middle=True, ring=True)) is False

    def test_scroll_pose_is_index_plus_middle(self):
        assert is_scroll_pose(SCROLL) is True
        assert is_scroll_pose(fingers(index=True, middle=True,
                                      thumb=True)) is True  # thumb ignored
        assert is_scroll_pose(POINTER) is False
        assert is_scroll_pose(OPEN_PALM) is False
        assert is_scroll_pose(fingers(index=True, middle=True,
                                      ring=True)) is False

    def test_ok_sign_does_not_read_as_an_open_palm(self):
        # index curled into the ring plus thumb and three fingers up is
        # four extended fingers, which a plain count would arm swipes on
        ok = fingers(thumb=True, middle=True, ring=True, pinky=True)
        assert is_open_palm(ok) is False

    def test_poses_are_mutually_exclusive(self):
        for pose in (POINTER, SCROLL, OPEN_PALM, FIST):
            claims = [is_pointer_pose(pose), is_scroll_pose(pose),
                      is_open_palm(pose)]
            assert sum(claims) <= 1


class TestScrollTracker:
    def test_first_frame_no_scroll(self):
        assert ScrollTracker(sensitivity=10).update(0.5, 0.5) == (0, 0)

    def test_hand_down_scrolls_positive(self):
        t = ScrollTracker(sensitivity=10, natural=True)
        t.update(0.5, 0.5)
        # 0.25 travel * 10 = 2.5 steps -> emit 2, carry 0.5
        assert t.update(0.5, 0.75) == (0, 2)
        # 2.5 more + 0.5 carried -> 3
        assert t.update(0.5, 1.0) == (0, 3)

    def test_hand_up_scrolls_negative(self):
        t = ScrollTracker(sensitivity=10, natural=True)
        t.update(0.5, 0.5)
        assert t.update(0.5, 0.25) == (0, -2)  # -2.5 -> -2, carry -0.5

    def test_classic_mode_inverts(self):
        t = ScrollTracker(sensitivity=10, natural=False)
        t.update(0.5, 0.5)
        assert t.update(0.5, 0.75) == (0, -2)

    def test_slow_motion_accumulates(self):
        t = ScrollTracker(sensitivity=10)
        t.update(0.5, 0.5)
        assert t.update(0.5, 0.5625) == (0, 0)  # 0.625 steps: too small
        assert t.update(0.5, 0.625) == (0, 1)   # 1.25 accumulated -> 1

    def test_horizontal_axis(self):
        t = ScrollTracker(sensitivity=10)
        t.update(0.5, 0.5)
        assert t.update(0.75, 0.5) == (2, 0)

    def test_reset_forgets_position(self):
        t = ScrollTracker(sensitivity=10)
        t.update(0.5, 0.5)
        t.reset()
        assert t.update(0.9, 0.9) == (0, 0)  # re-primed, no catch-up jump

    def test_rejects_bad_sensitivity(self):
        with pytest.raises(ValueError):
            ScrollTracker(sensitivity=0)

    def test_production_default_sensitivity(self):
        # ActionRouter falls back to ScrollTracker() with module defaults,
        # so pin the shipped scroll feel. pynput posts each step as 10 px,
        # so a quarter-frame sweep here is ~370 px of scrolling.
        t = ScrollTracker()
        t.update(0.5, 0.5)
        assert t.update(0.5, 0.75) == (0, 75)  # 0.25 * 300 = 75 steps


class TestPalmGate:
    def test_requires_sustained_open_palm(self):
        g = PalmGate(frames=3)
        assert g.update(OPEN_PALM) is False
        assert g.update(OPEN_PALM) is False
        assert g.update(OPEN_PALM) is True

    def test_flicker_does_not_arm_swipes(self):
        # two-frame misread during a scroll stroke must not arm swipes
        g = PalmGate(frames=3)
        g.update(OPEN_PALM)
        g.update(OPEN_PALM)
        assert g.update(SCROLL) is False
        assert g.update(OPEN_PALM) is False  # counter restarted

    def test_armed_palm_rides_out_flick_blur(self):
        # a real swipe blurs the fingers mid-flick; disarming there would
        # clear the swipe history exactly when the swipe is happening
        g = PalmGate(frames=3, gap=4)
        for _ in range(3):
            g.update(OPEN_PALM)
        assert g.update(POINTER) is True   # misread during the flick
        assert g.update(FIST) is True
        assert g.update(OPEN_PALM) is True  # recovered, still armed

    def test_sustained_non_palm_disarms(self):
        g = PalmGate(frames=3, gap=4)
        for _ in range(3):
            g.update(OPEN_PALM)
        results = [g.update(FIST) for _ in range(4)]
        assert results == [True, True, True, False]

    def test_production_default_needs_more_than_one_frame(self):
        # main.py builds PalmGate() with the module default; a 1-frame
        # gate is exactly the misread-during-scroll hazard it exists for
        g = PalmGate()
        assert g.update(OPEN_PALM) is False


def make_router():
    return ActionRouter(CursorMapper(1000, 800, margin=0.0, min_alpha=1.0, max_alpha=1.0))


class TestActionRouter:
    def test_pointer_pose_moves_cursor(self):
        r = make_router()
        actions = r.route(POINTER, False, [], (0.5, 0.5))
        assert actions == [('move', 500, 400)]

    def test_fist_does_not_move_cursor(self):
        r = make_router()
        assert r.route(FIST, False, [], (0.5, 0.5)) == []

    def test_pinch_start_presses_at_hover_position(self):
        r = make_router()
        actions = r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        assert ('press',) in actions
        assert r.is_dragging is True
        # no move on the press frame: curling into the pinch drags the
        # fingertip, so the click must land where the user was hovering
        assert not any(a[0] == 'move' for a in actions)
        # subsequent drag frames track the finger again
        follow = r.route(FIST, True, [], (0.25, 0.25))
        assert follow == [('move', 250, 200)]

    def test_ok_sign_ring_does_not_click(self):
        # an OK sign closes the same thumb-index ring a click does, so the
        # press must be suppressed or every OK sign clicks whatever is
        # under the cursor
        r = make_router()
        ok = fingers(thumb=True, middle=True, ring=True, pinky=True)
        actions = r.route(ok, True, ['pinch_start'], (0.5, 0.5),
                          ok_showing=True)
        assert ('press',) not in actions
        assert r.is_dragging is False

    def test_holding_the_ok_sign_does_not_drag_the_cursor(self):
        # the press frame never moves the cursor anyway, so this has to
        # check a later frame of the same held sign
        r = make_router()
        ok = fingers(thumb=True, middle=True, ring=True, pinky=True)
        r.route(ok, True, ['pinch_start'], (0.5, 0.5), ok_showing=True)
        assert r.route(ok, True, [], (0.25, 0.25), ok_showing=True) == []

    def test_relaxed_pinch_still_clicks(self):
        # `fingers_up` calls a merely relaxed finger extended, so an
        # ordinary pinch often reports middle/ring/pinky up. Only the OK
        # detector decides it's an OK sign — reading the finger dict here
        # stopped normal clicks working.
        r = make_router()
        relaxed = fingers(thumb=True, middle=True, ring=True, pinky=True)
        actions = r.route(relaxed, True, ['pinch_start'], (0.5, 0.5))
        assert ('press',) in actions
        assert r.is_dragging is True

    def test_ok_reading_cannot_strand_a_held_drag(self):
        # a stray OK reading mid-drag must not freeze the cursor while the
        # button stays down
        r = make_router()
        r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        actions = r.route(FIST, True, [], (0.25, 0.25), ok_showing=True)
        assert actions == [('move', 250, 200)]

    def test_click_pinch_still_presses(self):
        # the same event with the three fingers curled is a real click
        r = make_router()
        actions = r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        assert ('press',) in actions
        assert r.is_dragging is True

    def test_second_pinch_start_cannot_double_press(self):
        r = make_router()
        r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        actions = r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        assert ('press',) not in actions  # exactly one press per drag

    def test_pinch_end_releases(self):
        r = make_router()
        r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        actions = r.route(FIST, False, ['pinch_end'], (0.5, 0.5))
        assert ('release',) in actions
        assert r.is_dragging is False

    def test_release_without_press_is_ignored(self):
        r = make_router()
        assert r.route(FIST, False, ['pinch_end'], (0.5, 0.5)) == []

    def test_open_palm_swipe_fires_hotkey(self):
        r = make_router()
        actions = r.route(OPEN_PALM, False, ['swipe_left'], (0.5, 0.5))
        assert ('hotkey', 'swipe_left') in actions

    def test_pointer_pose_swipe_is_suppressed(self):
        # flicking while mousing must not switch desktops
        r = make_router()
        actions = r.route(POINTER, False, ['swipe_right'], (0.5, 0.5))
        assert ('hotkey', 'swipe_right') not in actions

    def test_suspend_releases_held_drag(self):
        r = make_router()
        r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        assert r.suspend() == [('release',)]
        assert r.is_dragging is False

    def test_suspend_without_drag_is_empty(self):
        assert make_router().suspend() == []

    def test_drag_keeps_cursor_tracking_after_pose_lost(self):
        r = make_router()
        r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        # pinch detector may flicker is_pinching off a frame before the
        # event arrives; the held drag alone must keep the cursor moving
        actions = r.route(FIST, False, [], (0.25, 0.25))
        assert actions == [('move', 250, 200)]

    def test_brief_pose_gap_keeps_smoothing(self):
        # 1-2 dropout frames (index curling into a pinch) must NOT reset
        # the mapper — otherwise the eventual click jumps unsmoothed
        r = ActionRouter(CursorMapper(1000, 800, margin=0.0, min_alpha=0.5, max_alpha=0.5))
        r.route(POINTER, False, [], (0.25, 0.25))   # smoothed at (250, 200)
        r.route(FIST, False, [], (0.9, 0.9))        # 1-frame gap: no reset
        actions = r.route(POINTER, False, [], (0.75, 0.75))
        assert actions == [('move', 500, 400)]      # EMA continues, no jump

    def test_sustained_pose_loss_resets_smoothing(self):
        r = ActionRouter(CursorMapper(1000, 800, margin=0.0, min_alpha=0.5, max_alpha=0.5))
        r.route(POINTER, False, [], (0.25, 0.25))
        for _ in range(3):                          # >= POSE_GAP_FRAMES
            r.route(FIST, False, [], (0.9, 0.9))
        actions = r.route(POINTER, False, [], (0.75, 0.75))
        assert actions == [('move', 749, 599)]      # jump, no cross-screen glide

    def test_suspend_resets_smoothing(self):
        r = ActionRouter(CursorMapper(1000, 800, margin=0.0, min_alpha=0.5, max_alpha=0.5))
        r.route(POINTER, False, [], (0.25, 0.25))
        r.suspend()
        actions = r.route(POINTER, False, [], (0.75, 0.75))
        assert actions == [('move', 749, 599)]      # jump to the finger

    def scroll_router(self):
        return ActionRouter(CursorMapper(1000, 800, margin=0.0, min_alpha=1.0, max_alpha=1.0),
                            ScrollTracker(sensitivity=10))

    def engage_scroll(self, r):
        """Hold the scroll pose long enough to arm it, ending at y=0.5.

        Steps are exact binary fractions so the accumulator carry is
        predictable: it leaves 0.5 of a step carried.
        """
        for y in (0.25, 0.375, 0.5):
            r.route(SCROLL, False, [], (0.5, y))

    def test_scroll_pose_emits_scroll_not_move(self):
        r = self.scroll_router()
        assert r.route(SCROLL, False, [], (0.5, 0.5)) == []   # priming
        assert r.route(SCROLL, False, [], (0.5, 0.75)) == []  # still engaging
        assert r.route(SCROLL, False, [], (0.5, 1.0)) == [('scroll', 0, 3)]

    def test_transient_scroll_pose_does_not_scroll(self):
        # the V pose is passed through while opening the hand for a swipe
        r = self.scroll_router()
        r.route(SCROLL, False, [], (0.5, 0.5))
        r.route(SCROLL, False, [], (0.5, 0.9))
        actions = r.route(OPEN_PALM, False, [], (0.5, 1.0))
        assert not any(a[0] == 'scroll' for a in actions)

    def test_scroll_survives_one_frame_pose_flicker(self):
        r = self.scroll_router()
        self.engage_scroll(r)
        # middle finger misreads as curled for a single frame: the cursor
        # must not take over and teleport the mouse
        assert r.route(POINTER, False, [], (0.5, 0.6)) == []
        # session continues, and the motion spanning the flicker isn't
        # lost: 0.5 -> 0.75 is 2.5 steps plus the 0.5 carried
        assert r.route(SCROLL, False, [], (0.5, 0.75)) == [('scroll', 0, 3)]

    def test_sustained_pose_change_ends_scroll(self):
        # relax to a fist to reposition the hand, then scroll again: the
        # re-entry must prime, not emit a catch-up jump
        r = self.scroll_router()
        self.engage_scroll(r)
        for _ in range(3):
            r.route(FIST, False, [], (0.5, 0.9))
        assert r.route(SCROLL, False, [], (0.5, 0.1)) == []

    def test_suspend_resets_scroll(self):
        r = self.scroll_router()
        self.engage_scroll(r)
        r.suspend()
        assert r.route(SCROLL, False, [], (0.5, 0.1)) == []

    def test_drag_beats_scroll_pose(self):
        # scroll pose while a drag is held keeps moving the cursor
        r = self.scroll_router()
        r.route(FIST, True, ['pinch_start'], (0.5, 0.5))
        actions = r.route(SCROLL, False, [], (0.25, 0.25))
        assert actions == [('move', 250, 200)]


class TestControlSession:
    def make(self):
        return ControlSession(make_router())

    def test_starts_off_and_routes_nothing(self):
        s = self.make()
        assert s.control_on is False
        assert s.frame(POINTER, False, [], (0.5, 0.5)) == []
        assert s.frame(FIST, True, ['pinch_start'], (0.5, 0.5)) == []

    def test_toggle_on_enables_routing(self):
        s = self.make()
        assert s.toggle() == []
        assert s.control_on is True
        assert s.frame(POINTER, False, [], (0.5, 0.5)) == [('move', 500, 400)]

    def test_toggle_off_mid_drag_releases(self):
        s = self.make()
        s.toggle()
        s.frame(FIST, True, ['pinch_start'], (0.5, 0.5))
        assert s.toggle() == [('release',)]
        assert s.control_on is False

    def test_hand_lost_mid_drag_releases(self):
        s = self.make()
        s.toggle()
        s.frame(FIST, True, ['pinch_start'], (0.5, 0.5))
        assert s.hand_lost() == [('release',)]

    def test_hand_lost_while_off_is_safe(self):
        assert self.make().hand_lost() == []

    def test_frame_carries_the_ok_sign_flag_through(self):
        ok = fingers(thumb=True, middle=True, ring=True, pinky=True)
        held = self.make()
        held.toggle()
        assert held.frame(ok, True, ['pinch_start'], (0.5, 0.5),
                          ok_showing=True) == []
        # the identical frame without the flag is an ordinary click
        clicking = self.make()
        clicking.toggle()
        assert ('press',) in clicking.frame(ok, True, ['pinch_start'],
                                           (0.5, 0.5))

    def test_bound_action_blocked_while_off(self):
        # a gesture binding must not launch anything before the user
        # deliberately turns control on
        s = self.make()
        assert s.bound(('launch', 'Terminal')) == []

    def test_bound_action_runs_while_on(self):
        s = self.make()
        s.toggle()
        assert s.bound(('launch', 'Terminal')) == [('launch', 'Terminal')]

    def test_unbound_gesture_does_nothing(self):
        s = self.make()
        s.toggle()
        assert s.bound(None) == []
