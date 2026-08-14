"""Per-hand gesture state and identity, for tracking two hands at once.

Pure logic: landmark lists and timestamps in, gesture events out. Each
hand gets its own detectors, so a pinch on one hand cannot disturb the
other's state.

Identity is keyed on MediaPipe's handedness label ('Left'/'Right'). The
label is what makes state stick to the same physical hand across frames,
so the two failure modes it has — brief dropouts and duplicate labels —
are both handled here rather than leaking into the camera loop.
"""

from control import PalmGate, is_open_palm
from gestures import PinchDetector, SwipeDetector
from hand_logic import MovementTracker, fingers_up

INDEX_TIP = 8
# Middle-finger MCP — the centre of the palm. A swipe is a whole-hand
# flick, and this point barely moves when the fingers curl or blur,
# unlike a fingertip, so it gives the swipe a stable direction.
PALM_CENTER = 9
# A hand missing for fewer than this many consecutive frames is treated as
# still present: MediaPipe drops a hand routinely, and reacting instantly
# would reset gesture state mid-gesture.
HAND_LOST_FRAMES = 3
# Which hand drives the cursor when both are visible.
PRIMARY_HAND = 'Right'


def assign_labels(candidates):
    """Give each detected hand a unique identity key.

    candidates: list of (handedness_label, confidence score), in
    detection order. Returns a list of unique labels in the same order.

    MediaPipe sometimes classifies both hands the same way. Sharing one
    key would make two physical hands share gesture state, so the more
    confident hand keeps the contested label and the other takes the
    free one.
    """
    labels = [label for label, _ in candidates]
    if len(set(labels)) == len(labels):
        return labels

    assigned = [None] * len(candidates)
    used = set()
    # most confident first, so it keeps the label it was given
    for i in sorted(range(len(candidates)), key=lambda i: -candidates[i][1]):
        label = candidates[i][0]
        if label in used:
            label = other_hand(label)
            suffix = 2
            while label in used:
                label = f'{candidates[i][0]}#{suffix}'
                suffix += 1
        assigned[i] = label
        used.add(label)
    return assigned


def other_hand(label):
    return 'Left' if label == 'Right' else 'Right'


def pick_primary(labels, preferred=PRIMARY_HAND):
    """Which hand drives the cursor: the preferred one when it is there,
    otherwise whichever single hand is. None when no hand is active."""
    if not labels:
        return None
    if preferred in labels:
        return preferred
    return sorted(labels)[0]


class HandTracker:
    """Every gesture detector for one hand, advanced together."""

    def __init__(self):
        self.movement = MovementTracker()
        self.pinch = PinchDetector()
        self.swipe = SwipeDetector()
        self.palm = PalmGate()
        self.fingers = {}
        self.direction = 'still'
        self.palm_armed = False
        self.points = []   # this frame's landmarks, for two-hand gestures

    @property
    def is_pinching(self):
        return self.pinch.is_pinching

    def update(self, points, label, now):
        """points: 21 aspect-corrected (x, y) landmarks for this hand.
        Returns the gesture events it produced this frame."""
        self.points = points
        self.fingers = fingers_up(points, label)
        self.direction = self.movement.update(*points[INDEX_TIP])

        events = []
        pinch_event = self.pinch.update(points)
        if pinch_event:
            events.append(pinch_event)

        # Swipes are an open-palm gesture, and the palm must be held: a
        # fast pointer or scroll move with a flickered finger would
        # otherwise fire a desktop switch.
        self.palm_armed = self.palm.update(self.fingers)
        if self.palm_armed:
            swipe_event = self.swipe.update(*points[PALM_CENTER], now)
            if swipe_event:
                events.append(swipe_event)
        else:
            self.swipe.reset()
        return events

    def lost(self):
        """The hand is gone. Resets the detectors and returns the events
        still owed, so a pinch held at the moment of loss is closed out
        instead of dangling."""
        events = ['pinch_end'] if self.pinch.is_pinching else []
        self.movement.reset()
        self.pinch.reset()
        self.swipe.reset()
        self.palm.reset()
        self.fingers = {}
        self.direction = 'still'
        self.palm_armed = False
        self.points = []
        return events

    def open_palm(self):
        return bool(self.fingers) and is_open_palm(self.fingers)


class HandRegistry:
    """Keeps one HandTracker per hand, and decides when a hand is gone.

    Trackers are reset rather than discarded when a hand is lost, so a
    hand that comes back keeps the guards that depend on history (a hand
    re-detected mid-pinch must open before it can pinch again).
    """

    def __init__(self, factory=HandTracker, lost_frames=HAND_LOST_FRAMES):
        if lost_frames < 1:
            raise ValueError('lost_frames must be at least 1')
        self._factory = factory
        self._lost_frames = lost_frames
        self._hands = {}
        self._missing = {}

    def get(self, label):
        """Fetch (creating if needed) the tracker for a hand seen now."""
        if label not in self._hands:
            self._hands[label] = self._factory()
        self._missing[label] = 0
        return self._hands[label]

    def sweep(self, present):
        """Age out hands not seen this frame.

        Returns [(label, tracker)] for hands that just crossed the lost
        threshold — exactly once each, on the frame they cross it.
        """
        lost = []
        for label, tracker in self._hands.items():
            if label in present:
                continue
            self._missing[label] += 1
            if self._missing[label] == self._lost_frames:
                lost.append((label, tracker))
        return lost

    def active(self):
        """Labels of hands considered present, including ones dropped for
        a frame or two."""
        return sorted(label for label, missed in self._missing.items()
                      if missed < self._lost_frames)
