"""Pure decision logic for gesture -> macOS control.

No pynput, no OpenCV: hand state comes in, abstract actions come out.
Actions are tuples executed by mac_actions.MacController:
    ('move', screen_x, screen_y)
    ('press',) / ('release',)          # left mouse button
    ('scroll', dx_steps, dy_steps)     # scroll steps, ints (1 step = 10 px)
    ('hotkey', 'swipe_left' | 'swipe_right' | 'swipe_up' | 'swipe_down')
"""

import math

# Fraction of the camera frame's edge that is dead zone: the inner
# region maps to the full screen so the fingertip never has to reach
# the actual frame border.
CURSOR_MARGIN = 0.2
# Cursor smoothing is speed-adaptive. A fixed coefficient forces a bad
# trade: enough damping to hold the cursor steady for a click also makes
# it lag ~250 ms behind fast motion. Instead, damp hard when the hand is
# nearly still and release the damping as it speeds up.
CURSOR_MIN_ALPHA = 0.25   # nearly still: heavy damping, steady enough to click
CURSOR_MAX_ALPHA = 1.0    # fast: track the finger with no lag at all
# Hand speed (fraction of the screen per frame) at which damping is fully
# released. ~0.05 is a brisk deliberate move at 25-30 fps.
CURSOR_SPEED_REF = 0.05
# An open-palm flick needs at least this many extended fingers.
OPEN_PALM_FINGERS = 4
# Frames the open palm must be held before swipe detection is armed.
OPEN_PALM_HOLD_FRAMES = 3
# Frames the pointer pose may drop out (e.g. while the index curls into a
# pinch) before cursor smoothing state is discarded.
POSE_GAP_FRAMES = 3
# Scroll steps per full frame-height of hand travel. pynput posts each
# step as a 10-pixel scroll event on macOS, so 150 ~= 1500 px per sweep.
SCROLL_SENSITIVITY = 150.0
# Frames the scroll pose must be held before it emits, so poses passed
# through in transit (e.g. opening the hand for a swipe) don't scroll.
SCROLL_ENGAGE_FRAMES = 3
# True = hand down scrolls content down. Direction of synthetic scroll
# events is independent of the macOS "Natural scrolling" preference, so
# this constant alone decides; flip it if the direction feels wrong.
SCROLL_NATURAL = True


class CursorMapper:
    """Maps normalized camera coords (mirrored frame) to screen pixels.

    Applies an edge margin, then speed-adaptive smoothing. reset() clears
    the smoothing state so a re-engaged pointer jumps to the finger
    instead of gliding across the screen.

    Pass min_alpha == max_alpha for a plain fixed-coefficient filter.
    """

    def __init__(self, screen_w, screen_h, margin=CURSOR_MARGIN,
                 min_alpha=CURSOR_MIN_ALPHA, max_alpha=CURSOR_MAX_ALPHA,
                 speed_ref=CURSOR_SPEED_REF):
        if not 0 <= margin < 0.5:
            raise ValueError('margin must be in [0, 0.5)')
        if not 0 < min_alpha <= 1 or not 0 < max_alpha <= 1:
            raise ValueError('alphas must be in (0, 1]')
        if min_alpha > max_alpha:
            raise ValueError('min_alpha must not exceed max_alpha')
        if speed_ref <= 0:
            raise ValueError('speed_ref must be positive')
        self._screen = (screen_w, screen_h)
        self._margin = margin
        self._min_alpha = min_alpha
        self._max_alpha = max_alpha
        self._speed_ref = speed_ref
        self._pos = None

    def map(self, x, y):
        """x, y in normalized camera coords (0..1). Returns (px, py) ints."""
        span = 1 - 2 * self._margin
        nx = min(max((x - self._margin) / span, 0.0), 1.0)
        ny = min(max((y - self._margin) / span, 0.0), 1.0)
        # scale to screen-1: coordinate screen_w is one past the last pixel
        # and lands on an adjacent display in multi-monitor layouts
        target = (nx * (self._screen[0] - 1), ny * (self._screen[1] - 1))

        if self._pos is None:
            self._pos = target
        else:
            alpha = self._alpha_for(target)
            self._pos = (
                self._pos[0] + alpha * (target[0] - self._pos[0]),
                self._pos[1] + alpha * (target[1] - self._pos[1]),
            )
        return int(round(self._pos[0])), int(round(self._pos[1]))

    def _alpha_for(self, target):
        """Damping coefficient for this frame, from how far the target
        moved as a fraction of the screen."""
        speed = math.hypot(
            (target[0] - self._pos[0]) / max(self._screen[0] - 1, 1),
            (target[1] - self._pos[1]) / max(self._screen[1] - 1, 1),
        )
        ramp = min(speed / self._speed_ref, 1.0)
        return self._min_alpha + (self._max_alpha - self._min_alpha) * ramp

    def reset(self):
        self._pos = None


def is_pointer_pose(up):
    """Index extended, middle/ring/pinky curled. Thumb is ignored —
    it drifts in and out during pinches."""
    return up['index'] and not up['middle'] and not up['ring'] and not up['pinky']


def is_scroll_pose(up):
    """Index + middle extended (a V), ring/pinky curled. Thumb ignored."""
    return (up['index'] and up['middle']
            and not up['ring'] and not up['pinky'])


class PalmGate:
    """Reports whether the open palm has been held long enough to arm
    swipe detection.

    A vigorous scroll stroke is one flickered finger away from reading as
    an open palm, and a 1-2 frame misread would fire a desktop switch, so
    the pose has to persist before swipes are fed.
    """

    def __init__(self, frames=OPEN_PALM_HOLD_FRAMES):
        self._frames = frames
        self._held = 0

    def update(self, up):
        self._held = self._held + 1 if is_open_palm(up) else 0
        return self._held >= self._frames


class ScrollTracker:
    """Converts fingertip motion into integer wheel steps.

    Fractional steps accumulate across frames, so slow hand motion still
    scrolls eventually and fast motion isn't truncated per frame.
    """

    def __init__(self, sensitivity=SCROLL_SENSITIVITY, natural=SCROLL_NATURAL):
        if sensitivity <= 0:
            raise ValueError('sensitivity must be positive')
        self._sens = sensitivity * (1 if natural else -1)
        self._last = None
        self._acc = [0.0, 0.0]

    def update(self, x, y):
        """Feed the tracked point; returns (dx_steps, dy_steps) ints."""
        if self._last is None:
            self._last = (x, y)
            return 0, 0
        self._acc[0] += (x - self._last[0]) * self._sens
        self._acc[1] += (y - self._last[1]) * self._sens
        self._last = (x, y)
        dx, dy = int(self._acc[0]), int(self._acc[1])
        self._acc[0] -= dx
        self._acc[1] -= dy
        return dx, dy

    def reset(self):
        """Call when leaving scroll pose so re-entering can't emit a
        catch-up jump from the stale position."""
        self._last = None
        self._acc = [0.0, 0.0]


def is_open_palm(up):
    return sum(up.values()) >= OPEN_PALM_FINGERS


class ActionRouter:
    """Turns one frame of hand state into a list of actions.

    Owns the drag state so a pinch that started a drag is always paired
    with a release, even if control is toggled off or the hand is lost.
    """

    def __init__(self, mapper, scroll_tracker=None):
        self._mapper = mapper
        self._scroll = scroll_tracker or ScrollTracker()
        self._pose_lost_frames = 0
        self._scroll_held = 0   # frames the scroll pose has been held
        self._scroll_gap = 0    # frames it has been missing mid-session
        self.is_dragging = False

    def route(self, up, is_pinching, events, raw_index):
        """up: fingers_up dict; events: gesture event strings this frame;
        raw_index: (x, y) index tip in normalized camera coords."""
        actions = []
        pinch_started = 'pinch_start' in events
        pinch_active = is_pinching or self.is_dragging or pinch_started
        scroll_pose = is_scroll_pose(up)

        steps = self._track_scroll(scroll_pose, pinch_active, raw_index)
        scrolling = self._scroll_held >= SCROLL_ENGAGE_FRAMES

        if self._scroll_held:
            # A scroll session owns the hand: never move the cursor. Once
            # engaged, drop cursor state so returning to pointer jumps to
            # the finger rather than gliding from where scrolling began.
            if scrolling:
                self._mapper.reset()
                if any(steps):
                    actions.append(('scroll', *steps))
        # Cursor follows the finger in pointer pose, and keeps following
        # during a pinch even though pinching curls the pose.
        elif is_pointer_pose(up) or pinch_active:
            self._pose_lost_frames = 0
            # No move on the press frame: curling into the pinch drags the
            # fingertip, so the click must land where the user was hovering.
            if not pinch_started:
                actions.append(('move', *self._mapper.map(*raw_index)))
        else:
            # The pose flickers off for a frame or two while the index curls
            # toward the thumb; only discard smoothing on a sustained gap.
            self._pose_lost_frames += 1
            if self._pose_lost_frames >= POSE_GAP_FRAMES:
                self._mapper.reset()

        for event in events:
            if event == 'pinch_start' and not self.is_dragging:
                self.is_dragging = True
                actions.append(('press',))
            elif event == 'pinch_end' and self.is_dragging:
                self.is_dragging = False
                actions.append(('release',))
            elif event.startswith('swipe_') and is_open_palm(up):
                actions.append(('hotkey', event))
        return actions

    def _track_scroll(self, scroll_pose, pinch_active, raw_index):
        """Advance the scroll session state; returns this frame's steps.

        The tracker is fed on every genuine scroll-pose frame (so the
        session is primed and ready the moment it engages) but the caller
        only emits once the pose has been held long enough to be
        deliberate.
        """
        if pinch_active:
            self._end_scroll()      # a pinch always outranks scrolling
        elif scroll_pose:
            self._scroll_gap = 0
            self._scroll_held += 1
        elif self._scroll_held:
            # Mid-scroll the middle finger can read as curled for a frame
            # or two. Without this tolerance that misread would hand the
            # hand back to the cursor and teleport the mouse.
            self._scroll_gap += 1
            if self._scroll_gap >= POSE_GAP_FRAMES:
                self._end_scroll()
        else:
            self._end_scroll()

        if scroll_pose and self._scroll_held:
            return self._scroll.update(*raw_index)
        return 0, 0

    def _end_scroll(self):
        self._scroll_held = 0
        self._scroll_gap = 0
        self._scroll.reset()

    def suspend(self):
        """Control turned off or hand lost: release a held drag and clear
        cursor smoothing. Returns the actions still owed to the OS."""
        self._mapper.reset()
        self._end_scroll()
        self._pose_lost_frames = 0
        if self.is_dragging:
            self.is_dragging = False
            return [('release',)]
        return []


class ControlSession:
    """Owns the ON/OFF safety gate so it is testable, not main-loop glue.

    Control always starts OFF; while OFF no frame produces actions.
    Every method returns the actions owed to the OS.
    """

    def __init__(self, router):
        self._router = router
        self.control_on = False

    def toggle(self):
        self.control_on = not self.control_on
        # turning off mid-drag must never leave the button held
        return [] if self.control_on else self._router.suspend()

    def frame(self, up, is_pinching, events, raw_index):
        if not self.control_on:
            return []
        return self._router.route(up, is_pinching, events, raw_index)

    def hand_lost(self):
        return self._router.suspend()
