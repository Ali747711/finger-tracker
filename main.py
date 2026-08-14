"""Finger movement tracker v0.3 — gestures + optional macOS control.

Mirror view with hand landmarks, finger states, pinch, and swipe events.
Press `c` to toggle macOS control: index finger moves the cursor, pinch
clicks/drags, open-palm flicks switch desktop spaces. Starts with
control OFF. Press `q` (or close the window) to quit.
"""

import sys
import threading
import time

import cv2
import mediapipe as mp

from control import (ActionRouter, ControlSession, CursorMapper, PalmGate)
from gestures import PinchDetector, SwipeDetector
from hand_logic import MovementTracker, fingers_up
from mac_actions import (MacController, is_trusted, screen_size,
                         start_kill_listener)

THUMB_TIP = 4
INDEX_TIP = 8
TEXT_COLOR = (80, 220, 80)
PINCH_COLOR = (80, 220, 80)
IDLE_COLOR = (180, 180, 180)
FLASH_COLOR = (60, 160, 255)
FLASH_SEC = 0.8
WINDOW_NAME = 'Finger Tracker'
# MediaPipe tracking drops out for a frame or two routinely; only treat the
# hand as lost (and reset gesture state) after this many missed frames.
HAND_LOST_FRAMES = 3


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


def process_hand(frame, results, tracker, pinch, swipe, palm, now, drawer):
    """Analyze the detected hand.

    Returns (status_lines, events, fingers_up_dict, raw_index_pos) where
    raw_index_pos is the index tip in unscaled 0..1 camera coords for
    cursor mapping.
    """
    hand = results.multi_hand_landmarks[0]
    handedness = results.multi_handedness[0].classification[0].label
    drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    # MediaPipe normalizes x by frame width and y by height. Rescale x so
    # both axes share height-normalized units — otherwise distances and
    # axis comparisons skew ~1.78x on a 16:9 camera and pinch/swipe
    # behavior changes with hand orientation.
    h, w = frame.shape[:2]
    aspect = w / h
    points = [(lm.x * aspect, lm.y) for lm in hand.landmark]
    up = fingers_up(points, handedness)
    up_names = [name for name, is_up in up.items() if is_up] or ['fist']
    direction = tracker.update(*points[INDEX_TIP])

    events = []
    pinch_event = pinch.update(points)
    if pinch_event:
        events.append(pinch_event)
    # Swipes are an open-palm gesture: feeding the detector during pointer,
    # scroll or pinch poses would fire phantom desktop switches from fast
    # hand moves and burn the cooldown a genuine flick then runs into.
    if palm.update(up):
        swipe_event = swipe.update(*points[INDEX_TIP], now)
        if swipe_event:
            events.append(swipe_event)
    else:
        swipe.reset()  # clears history, keeps any active cooldown

    draw_pinch_line(frame, hand, pinch.is_pinching)
    status = [
        f'{handedness} hand | fingers: {", ".join(up_names)}',
        f'index finger: {direction}',
        f'pinch: {"yes" if pinch.is_pinching else "no"}',
    ]
    raw_index = (hand.landmark[INDEX_TIP].x, hand.landmark[INDEX_TIP].y)
    return status, events, up, raw_index


def main():
    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        model_complexity=0,  # lightest model — fine for an Intel CPU
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    drawer = mp.solutions.drawing_utils
    tracker = MovementTracker()
    pinch = PinchDetector()
    swipe = SwipeDetector()
    palm = PalmGate()
    session = ControlSession(ActionRouter(CursorMapper(*screen_size())))
    controller = MacController()
    kill = threading.Event()
    start_kill_listener(kill.set)  # Esc works even without window focus

    cap = open_camera()
    last_time = time.time()
    flash = None  # (text, expires_at)
    missed = 0    # consecutive frames without a hand detection

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
                ' | c toggles',
            ]

            events = []
            if results.multi_hand_landmarks:
                missed = 0
                hand_status, events, up, raw_index = process_hand(
                    frame, results, tracker, pinch, swipe, palm, now, drawer)
                status.extend(hand_status)
                controller.execute(
                    session.frame(up, pinch.is_pinching, events, raw_index))
            else:
                missed += 1
                if missed == HAND_LOST_FRAMES:
                    if pinch.is_pinching:
                        events.append('pinch_end')  # keep events paired
                    tracker.reset()
                    pinch.reset()
                    swipe.reset()
                    controller.execute(session.hand_lost())
                status.append('no hand detected')

            for event in events:
                print(f'event: {event}')
                flash = (event.replace('_', ' ').upper(), now + FLASH_SEC)

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
