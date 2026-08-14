"""Pure hand-analysis logic: no camera, no MediaPipe imports.

Works on plain (x, y) landmark tuples so it stays easy to test and reuse.
MediaPipe hand landmark indices used here:
  0 = wrist, 4 = thumb tip, 8 = index tip, 12 = middle tip,
  16 = ring tip, 20 = pinky tip. Each tip's PIP joint is tip - 2.
"""

from collections import deque

# hand_scale lives in gestures because the pinch needed it first; importing
# it here rather than copying keeps its foreshortening fix in one place.
from gestures import distance, hand_scale

FINGER_TIPS = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
WRIST = 0
INDEX_MCP = 5
# How far a thumb tip must sit from the index knuckle, as a fraction of
# hand size, to count as extended. A folded thumb tucks across the palm.
# No gesture depends on the thumb — the poses that matter all ignore it —
# so this only has to be good enough for the on-screen readout.
THUMB_SPREAD_RATIO = 0.45

# Normalized-coordinate distance the fingertip must travel between the
# oldest and newest smoothed positions before we call it a movement.
MOVEMENT_THRESHOLD = 0.04


def fingers_up(landmarks):
    """Return a dict of finger name -> bool (extended or not).

    landmarks: sequence of 21 (x, y) tuples in normalized image coords.

    Extension is judged by distance from the wrist: an extended fingertip
    reaches further from the wrist than its own middle joint does. That
    ratio survives the hand being rotated, which matters because people
    tilt their hand constantly. Comparing tip and joint *heights* instead
    only holds while the hand points upwards — it starts misreading around
    45 degrees of tilt and inverts completely past 90.

    Handedness is deliberately not a parameter: nothing here depends on
    which hand it is, so a mislabelled hand can no longer misread a pose.
    """
    if len(landmarks) != 21:
        raise ValueError(f'expected 21 landmarks, got {len(landmarks)}')

    wrist = landmarks[WRIST]
    up = {}
    for name in ('index', 'middle', 'ring', 'pinky'):
        tip = FINGER_TIPS[name]
        up[name] = (distance(wrist, landmarks[tip])
                    > distance(wrist, landmarks[tip - 2]))

    # The thumb folds sideways across the palm instead of curling towards
    # the wrist, so the wrist rule doesn't apply; how far it sits from the
    # index knuckle does separate the poses.
    scale = hand_scale(landmarks)
    up['thumb'] = bool(
        scale >= 1e-6
        and distance(landmarks[FINGER_TIPS['thumb']], landmarks[INDEX_MCP])
        > THUMB_SPREAD_RATIO * scale)
    return up


def count_fingers(landmarks):
    return sum(fingers_up(landmarks).values())


class MovementTracker:
    """Tracks a single point over time and reports its dominant direction."""

    def __init__(self, window=5, threshold=MOVEMENT_THRESHOLD):
        self._positions = deque(maxlen=window)
        self._threshold = threshold

    def update(self, x, y):
        """Add a new position and return 'left'/'right'/'up'/'down'/'still'."""
        self._positions.append((x, y))
        if len(self._positions) < 2:
            return 'still'

        old_x, old_y = self._positions[0]
        dx, dy = x - old_x, y - old_y
        return self._classify(dx, dy)

    def reset(self):
        """Call when the hand is lost so stale positions don't count."""
        self._positions.clear()

    def _classify(self, dx, dy):
        if max(abs(dx), abs(dy)) < self._threshold:
            return 'still'
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        # y grows downward in image coordinates
        return 'down' if dy > 0 else 'up'
