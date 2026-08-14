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
        # Whether the fingertips are in contact right now. The caller uses
        # this to hold off single-hand control, so it must never report a
        # reading the camera can't currently see.
        self.is_touching = False
        # Whether this touch has already fired. Kept separately, because it
        # has to survive a frame where a hand isn't visible — otherwise one
        # touch fires twice.
        self._latched = False

    def update(self, hands, now):
        """hands: [(points, fingers_up), ...] for the hands seen now,
        where points are aspect-corrected landmarks.

        Returns 'index_touch' once per touch, otherwise None.
        """
        if not self._measurable(hands):
            self.is_touching = False
            return None

        (a_points, _), (b_points, _) = hands
        scale = (hand_scale(a_points) + hand_scale(b_points)) / 2
        if scale < 1e-6:
            self.is_touching = False
            return None

        ratio = distance(a_points[INDEX_TIP], b_points[INDEX_TIP]) / scale
        self.is_touching = ratio <= self._off

        if self._latched:
            if ratio > self._off:
                self._latched = False   # separated: armed again
            return None
        if ratio >= self._on:
            return None
        if self._quiet_until is not None and now < self._quiet_until:
            return None

        self._latched = True
        self._quiet_until = now + self._cooldown
        return 'index_touch'

    @staticmethod
    def _measurable(hands):
        """A gap needs two hands with both index fingers extended."""
        if len(hands) != 2:
            return False
        return all(up.get('index') for _points, up in hands)

    def reset(self):
        """Control turned off, or both hands lost."""
        self.is_touching = False
        self._latched = False
