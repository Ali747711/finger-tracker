"""Gestures that need both hands at once.

Pure logic: landmarks in, gesture events out, time injected.
"""

from gestures import distance, hand_scale

INDEX_TIP = 8
# Gap between the two index tips, as a fraction of hand size, that counts
# as a touch. Measuring against hand size keeps it working at any
# distance from the camera.
TOUCH_ON_RATIO = 0.30
# The hands must separate past this before another touch can fire, so
# resting the fingers together does not repeat.
TOUCH_OFF_RATIO = 0.60
# Minimum gap between two firings, so a jittery contact fires once.
TOUCH_COOLDOWN_SEC = 1.5


class IndexTouchDetector:
    """Fires once when the two index fingertips meet.

    Latching plus a cooldown is what makes this usable as a trigger: the
    hands have to come apart before it can fire again, so holding the
    pose does not launch an app on every frame.
    """

    def __init__(self, on_ratio=TOUCH_ON_RATIO, off_ratio=TOUCH_OFF_RATIO,
                 cooldown=TOUCH_COOLDOWN_SEC):
        if on_ratio >= off_ratio:
            raise ValueError('on_ratio must be smaller than off_ratio')
        if cooldown < 0:
            raise ValueError('cooldown must not be negative')
        self._on = on_ratio
        self._off = off_ratio
        self._cooldown = cooldown
        self._quiet_until = None
        self.is_touching = False

    def update(self, hands, now):
        """hands: [(points, fingers_up), ...] for the hands seen now,
        where points are aspect-corrected landmarks.

        Returns 'index_touch' once per touch, otherwise None.
        """
        if len(hands) != 2:
            # Can't measure a gap with one hand. Leave the latch alone so
            # a dropped frame doesn't let the same touch fire twice.
            return None

        (a_points, a_up), (b_points, b_up) = hands
        if not (a_up.get('index') and b_up.get('index')):
            return None   # both index fingers must be extended

        scale = (hand_scale(a_points) + hand_scale(b_points)) / 2
        if scale < 1e-6:
            return None

        ratio = distance(a_points[INDEX_TIP], b_points[INDEX_TIP]) / scale
        if self.is_touching:
            if ratio > self._off:
                self.is_touching = False
            return None

        if ratio >= self._on:
            return None
        if self._quiet_until is not None and now < self._quiet_until:
            return None

        self.is_touching = True
        self._quiet_until = now + self._cooldown
        return 'index_touch'

    def reset(self):
        """Control turned off, or both hands lost."""
        self.is_touching = False
