"""Finger movement tracker v0.5 — two hands, optional macOS control.

Mirror view tracking up to two hands, each with independent gesture
state (fingers, pinch, scroll, swipe). Press `c` to toggle macOS
control, which the right hand drives when both are visible: index finger
moves the cursor, pinch clicks/drags, index+middle scrolls, open-palm
flicks switch desktop spaces. Starts with control OFF. Press `q` (or
close the window) to quit; Esc kills control from anywhere.
"""

import sys
import threading
import time

import cv2
import mediapipe as mp

from bindings import BINDINGS
from control import ActionRouter, ControlSession, CursorMapper
from hands import HandRegistry, assign_labels, pick_primary
from mac_actions import (MacController, binding_problems, is_trusted,
                         screen_size, start_kill_listener)
from two_hand import IndexTouchDetector

THUMB_TIP = 4
INDEX_TIP = 8
TEXT_COLOR = (80, 220, 80)
PINCH_COLOR = (80, 220, 80)
IDLE_COLOR = (180, 180, 180)
FLASH_COLOR = (60, 160, 255)
FLASH_SEC = 0.8
WINDOW_NAME = 'Finger Tracker'


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            'Could not open the camera. If macOS asked for camera permission, '
            'grant it to your terminal app and run again.'
        )
    return cap


def draw_status(frame, lines):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)


def draw_pinch_line(frame, hand, pinching):
    h, w = frame.shape[:2]
    thumb = hand.landmark[THUMB_TIP]
    index = hand.landmark[INDEX_TIP]
    p1 = (int(thumb.x * w), int(thumb.y * h))
    p2 = (int(index.x * w), int(index.y * h))
    cv2.line(frame, p1, p2, PINCH_COLOR if pinching else IDLE_COLOR, 2)


def draw_flash(frame, text):
    h, w = frame.shape[:2]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)[0]
    org = ((w - size[0]) // 2, (h + size[1]) // 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, FLASH_COLOR, 3)


def read_hands(results):
    """Pair up MediaPipe's landmark and handedness lists.

    Returns [(label, score, landmarks)] with a unique label per hand.
    The two lists are guarded against a length mismatch, which would
    otherwise attribute one hand's gestures to the other.
    """
    landmarks = results.multi_hand_landmarks or []
    handedness = results.multi_handedness or []
    paired = min(len(landmarks), len(handedness))
    candidates = [(handedness[i].classification[0].label,
                   handedness[i].classification[0].score)
                  for i in range(paired)]
    labels = assign_labels(candidates)
    return [(labels[i], candidates[i][1], landmarks[i])
            for i in range(paired)]


def process_hand(frame, hand, label, tracker, now, drawer):
    """Advance one hand's gesture state and draw it.

    Returns (status_line, events, raw_index_pos) where raw_index_pos is
    the index tip in unscaled 0..1 camera coords for cursor mapping.
    """
    drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    # MediaPipe normalizes x by frame width and y by height. Rescale x so
    # both axes share height-normalized units — otherwise distances and
    # axis comparisons skew ~1.78x on a 16:9 camera and pinch/swipe
    # behavior changes with hand orientation.
    h, w = frame.shape[:2]
    aspect = w / h
    points = [(lm.x * aspect, lm.y) for lm in hand.landmark]

    events = tracker.update(points, label, now)
    draw_pinch_line(frame, hand, tracker.is_pinching)

    up_names = [name for name, is_up in tracker.fingers.items() if is_up]
    status = (f'{label}: {", ".join(up_names) or "fist"}'
              f' | {tracker.direction}'
              f'{" | PINCH" if tracker.is_pinching else ""}'
              f'{" | PALM(swipe armed)" if tracker.palm_armed else ""}')
    raw_index = (hand.landmark[INDEX_TIP].x, hand.landmark[INDEX_TIP].y)
    return status, events, raw_index


def main():
    for problem in binding_problems(BINDINGS):
        print(f'bindings.py: {problem}')

    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        model_complexity=0,  # lightest model — fine for an Intel CPU
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    drawer = mp.solutions.drawing_utils
    registry = HandRegistry()
    two_hand = IndexTouchDetector()
    session = ControlSession(ActionRouter(CursorMapper(*screen_size())))
    controller = MacController()
    kill = threading.Event()
    start_kill_listener(kill.set)  # Esc works even without window focus

    cap = open_camera()
    last_time = time.time()
    flash = None       # (text, expires_at)
    controlling = None  # label of the hand currently driving the cursor

    print('Tracking… press q to quit, c to toggle control, Esc to kill control.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Camera frame not available, stopping.')
                break

            frame = cv2.flip(frame, 1)  # mirror view feels natural
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            if kill.is_set():
                kill.clear()
                if session.control_on:
                    controller.execute(session.toggle())
                    print('macOS control OFF (Esc)')

            status = [
                f'FPS: {fps:.0f}',
                f'control: {"ON  (Esc kills)" if session.control_on else "OFF"}'
                f' | c toggles | driving: {controlling or "-"}',
            ]

            events = []   # (label, event) for the log and the flash
            frames = {}   # label -> (tracker, events, raw_index)

            for label, _score, landmarks in read_hands(results):
                tracker = registry.get(label)
                line, hand_events, raw_index = process_hand(
                    frame, landmarks, label, tracker, now, drawer)
                frames[label] = (tracker, hand_events, raw_index)
                events.extend((label, e) for e in hand_events)
                status.append(line)

            # A hand missing for a frame or two is still 'active', so a
            # brief dropout can't hand control to the other hand.
            for label, tracker in registry.sweep(list(frames)):
                events.extend((label, e) for e in tracker.lost())

            active = registry.active()
            if not active:
                status.append('no hand detected')

            # Two-hand gestures, measured across both hands at once
            pair = [(frames[label][0].points, frames[label][0].fingers)
                    for label in sorted(frames)]
            touch = two_hand.update(pair, now)
            if touch:
                events.append(('both hands', touch))
                # a deliberate two-hand gesture must not also leave a drag
                # held by whichever hand was driving
                controller.execute(session.hand_lost())

            primary = pick_primary(active)
            if primary != controlling:
                # control changed hands (or ran out of hands): never leave
                # a drag held by the hand that just left
                controller.execute(session.hand_lost())
                controlling = primary
            if primary in frames and not two_hand.is_touching:
                tracker, hand_events, raw_index = frames[primary]
                # only the controlling hand's gestures reach macOS
                controller.execute(session.frame(
                    tracker.fingers, tracker.is_pinching,
                    hand_events, raw_index))

            for label, event in events:
                print(f'event: {label} {event}')
                flash = (f'{label} {event.replace("_", " ")}'.upper(),
                         now + FLASH_SEC)
                # any gesture named in bindings.py runs its action
                controller.execute(session.bound(BINDINGS.get(event)))

            if flash and now < flash[1]:
                draw_flash(frame, flash[0])

            draw_status(frame, status)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                if not session.control_on and not is_trusted(prompt=True):
                    print('Accessibility permission missing: System Settings '
                          '> Privacy & Security > Accessibility > enable your '
                          'terminal app, then restart it.')
                    flash = ('NEED ACCESSIBILITY PERMISSION', now + 2.5)
                else:
                    controller.execute(session.toggle())
                    print(f'macOS control '
                          f'{"ON" if session.control_on else "OFF"}')
            # closing the window with the red button must also stop the app
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        # runs on EVERY exit path — never leave the mouse button held down
        controller.execute(session.hand_lost())
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == '__main__':
    main()
