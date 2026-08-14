"""Gesture detectors built on per-frame hand landmarks.

Pure logic: positions and timestamps come in, gesture events come out.
Time is always injected (never read from the clock) so detectors are
deterministic and unit-testable.
"""

import math
from collections import deque

# Pinch thresholds as a fraction of hand size (wrist -> middle-finger MCP).
# Two thresholds (on < off) give hysteresis so the state can't flicker
# when the distance hovers around a single cutoff.
PINCH_ON_RATIO = 0.35
PINCH_OFF_RATIO = 0.55
# Consecutive open frames needed to end a pinch. Landmarks get noisy
# while the hand moves, so a drag must survive a single bad frame.
PINCH_RELEASE_FRAMES = 3

# An OK sign closes thumb and index into a ring, the same shape a
# click-pinch makes, so it reuses the pinch thresholds.
OK_ON_RATIO = PINCH_ON_RATIO
OK_OFF_RATIO = PINCH_OFF_RATIO
OK_COOLDOWN_SEC = 1.5
OK_FINGER_TIPS = (12, 16, 20)   # middle, ring, pinky
# How far past its PIP joint a fingertip must reach, as a fraction of hand
# size, to count as deliberately extended. `fingers_up` only asks whether
# the tip is above the joint at all, which a merely relaxed finger clears
# easily — and treating a relaxed pinch as an OK sign would stop it
# clicking. A fully extended finger clears its joint by ~1.0 of hand size.
OK_EXTENSION_MARGIN = 0.40

SWIPE_WINDOW_SEC = 0.30     # look-back window for displacement
SWIPE_MIN_DISTANCE = 0.18   # normalized-screen units the point must travel
SWIPE_COOLDOWN_SEC = 0.60   # ignore further swipes right after one fires
SWIPE_DOMINANCE = 1.5       # dominant axis must beat the other by this factor


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_scale(landmarks):
    """Reference length for ratio thresholds.

    Max of two roughly perpendicular palm segments — wrist (0) to
    middle-finger MCP (9), and the knuckle span index MCP (5) to pinky
    MCP (17). Either alone collapses under 2D projection when the hand
    pitches or yaws toward the camera; the max stays usable in any
    orientation and is independent of distance from the camera.
    """
    return max(distance(landmarks[0], landmarks[9]),
               distance(landmarks[5], landmarks[17]))


class PinchDetector:
    """Thumb-tip (4) to index-tip (8) pinch with hysteresis."""

    def __init__(self, on_ratio=PINCH_ON_RATIO, off_ratio=PINCH_OFF_RATIO,
                 release_frames=PINCH_RELEASE_FRAMES):
        if on_ratio >= off_ratio:
            raise ValueError('on_ratio must be smaller than off_ratio')
        if release_frames < 1:
            raise ValueError('release_frames must be at least 1')
        self._on = on_ratio
        self._off = off_ratio
        self._release_frames = release_frames
        self._open_frames = 0
        self._need_open = False
        self.is_pinching = False

    def update(self, landmarks):
        """Feed one frame of 21 (x, y) landmarks.

        Returns 'pinch_start', 'pinch_end', or None.
        """
        if len(landmarks) != 21:
            raise ValueError(f'expected 21 landmarks, got {len(landmarks)}')

        scale = hand_scale(landmarks)
        if scale < 1e-6:  # degenerate frame — ignore rather than divide by ~0
            return None

        ratio = distance(landmarks[4], landmarks[8]) / scale

        # After a reset, wait for one clearly-open frame before allowing a
        # new pinch: a hand re-detected mid-pinch must not auto-fire
        # pinch_start (which would re-press the mouse at a jumped position).
        if self._need_open:
            if ratio > self._off:
                self._need_open = False
            return None

        if not self.is_pinching:
            if ratio < self._on:
                self.is_pinching = True
                self._open_frames = 0
                return 'pinch_start'
            return None

        # Releasing takes consecutive open frames. Landmarks get noisy
        # while the hand moves, and a single bad frame dropping the pinch
        # is what makes a drag let go halfway.
        if ratio > self._off:
            self._open_frames += 1
            if self._open_frames >= self._release_frames:
                self.is_pinching = False
                self._open_frames = 0
                return 'pinch_end'
        else:
            self._open_frames = 0
        return None

    def reset(self):
        """Call when the hand is lost."""
        self.is_pinching = False
        self._open_frames = 0
        self._need_open = True


def fingers_clearly_extended(landmarks, tips=OK_FINGER_TIPS,
                             margin_ratio=OK_EXTENSION_MARGIN):
    """True when every named fingertip reaches well past its PIP joint.

    Deliberately stricter than `fingers_up`: this has to tell a hand
    holding an OK sign apart from a hand pinching to click with its other
    fingers merely relaxed, and those look identical to a bare
    tip-above-joint test.
    """
    scale = hand_scale(landmarks)
    if scale < 1e-6:
        return False
    margin = margin_ratio * scale
    return all(landmarks[tip][1] < landmarks[tip - 2][1] - margin
               for tip in tips)


class OkSignDetector:
    """Fires once when the hand shows an OK sign: thumb and index tips
    closed into a ring while the other three fingers stay extended.

    The ring is the same shape as a click-pinch, so those three extended
    fingers are the only thing telling the two gestures apart — and they
    have to be *clearly* extended, or ordinary pinches stop clicking.

    Latches like a trigger: holding the sign fires once, and the ring has
    to open (or the fingers curl) before it can fire again.
    """

    def __init__(self, on_ratio=OK_ON_RATIO, off_ratio=OK_OFF_RATIO,
                 cooldown=OK_COOLDOWN_SEC):
        if on_ratio >= off_ratio:
            raise ValueError('on_ratio must be smaller than off_ratio')
        if cooldown < 0:
            raise ValueError('cooldown must not be negative')
        self._on = on_ratio
        self._off = off_ratio
        self._cooldown = cooldown
        self._quiet_until = None
        self.is_showing = False

    def update(self, landmarks, now):
        """Returns 'ok_sign' once per showing, otherwise None."""
        if len(landmarks) != 21:
            raise ValueError(f'expected 21 landmarks, got {len(landmarks)}')

        scale = hand_scale(landmarks)
        if scale < 1e-6:
            return None
        ratio = distance(landmarks[4], landmarks[8]) / scale
        extended = fingers_clearly_extended(landmarks)

        if self.is_showing:
            if ratio > self._off or not extended:
                self.is_showing = False
            return None

        if not extended or ratio >= self._on:
            return None
        if self._quiet_until is not None and now < self._quiet_until:
            return None

        self.is_showing = True
        self._quiet_until = now + self._cooldown
        return 'ok_sign'

    def reset(self):
        """Call when the hand is lost."""
        self.is_showing = False


class SwipeDetector:
    """Detects fast directional flicks of one tracked point.

    A swipe fires when the point travels `min_distance` within
    `window_sec` with one axis clearly dominant, then the detector goes
    quiet for `cooldown_sec` so a single flick can't fire twice.
    """

    def __init__(self, window_sec=SWIPE_WINDOW_SEC,
                 min_distance=SWIPE_MIN_DISTANCE,
                 cooldown_sec=SWIPE_COOLDOWN_SEC,
                 dominance=SWIPE_DOMINANCE):
        self._window = window_sec
        self._min_distance = min_distance
        self._cooldown = cooldown_sec
        self._dominance = dominance
        self._history = deque()  # (t, x, y), trimmed to the window
        self._quiet_until = None

    def update(self, x, y, now):
        """Feed the tracked point and current timestamp (seconds).

        Returns 'swipe_left' / 'swipe_right' / 'swipe_up' / 'swipe_down'
        or None.
        """
        if self._quiet_until is not None:
            if now < self._quiet_until:
                # Quiet period: don't record the hand's return stroke, or it
                # would fire a phantom reverse swipe the moment cooldown ends.
                return None
            self._quiet_until = None
            self._history.clear()

        self._history.append((now, x, y))
        while self._history and self._history[0][0] < now - self._window:
            self._history.popleft()

        if len(self._history) < 2:
            return None

        _, x0, y0 = self._history[0]
        dx, dy = x - x0, y - y0
        adx, ady = abs(dx), abs(dy)
        if max(adx, ady) < self._min_distance:
            return None

        if adx >= ady * self._dominance:
            direction = 'swipe_right' if dx > 0 else 'swipe_left'
        elif ady >= adx * self._dominance:
            # y grows downward in image coordinates
            direction = 'swipe_down' if dy > 0 else 'swipe_up'
        else:
            return None  # diagonal — ambiguous, ignore

        self._quiet_until = now + self._cooldown
        self._history.clear()
        return direction

    def reset(self):
        """Call when the hand is lost. Keeps an active cooldown so a
        re-detected hand can't double-fire the same flick."""
        self._history.clear()
