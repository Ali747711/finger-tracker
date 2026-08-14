"""Find out which key-posting method macOS accepts as a system shortcut.

    .venv/bin/python diagnose_hotkey.py

BEFORE RUNNING: press Ctrl+Up physically. If Mission Control does not
open, the shortcut is switched off on this Mac and no method here can
work — turn it on in System Settings > Keyboard > Keyboard Shortcuts >
Mission Control.

The script fires Ctrl+Up several different ways, pausing between each so
you can watch. Note which numbered method opens Mission Control, and
close it before the next one fires.
"""

import subprocess
import sys
import time

from mac_actions import SWIPE_KEYCODES, is_trusted

# Physical modifier keys carry a device-dependent bit alongside the
# generic mask. Some macOS hotkey matching checks for it, so an event
# without it is delivered to apps but ignored as a shortcut.
NX_DEVICELCTLKEYMASK = 0x00000001


def quartz_post(keycode, tap_name, extra_flags=0, hold=0.02):
    import Quartz

    tap = getattr(Quartz, tap_name)
    flags = Quartz.kCGEventFlagMaskControl | extra_flags

    def post(code, down, event_flags):
        event = Quartz.CGEventCreateKeyboardEvent(None, code, down)
        Quartz.CGEventSetFlags(event, event_flags)
        Quartz.CGEventPost(tap, event)

    post(59, True, flags)          # Control down
    time.sleep(hold)
    post(keycode, True, flags)
    time.sleep(hold)
    post(keycode, False, flags)
    post(59, False, 0)


def method_hid_plain(keycode):
    quartz_post(keycode, 'kCGHIDEventTap')


def method_hid_device_bit(keycode):
    quartz_post(keycode, 'kCGHIDEventTap', NX_DEVICELCTLKEYMASK)


def method_session_tap(keycode):
    quartz_post(keycode, 'kCGSessionEventTap', NX_DEVICELCTLKEYMASK)


def method_applescript(keycode):
    subprocess.run(['osascript', '-e',
                    f'tell application "System Events" to '
                    f'key code {keycode} using control down'],
                   capture_output=True)


def method_open_mission_control(_keycode):
    subprocess.run(['open', '-a', 'Mission Control'], capture_output=True)


METHODS = [
    ('1. Quartz HID tap, plain Ctrl flag', method_hid_plain),
    ('2. Quartz HID tap + device-dependent bit', method_hid_device_bit),
    ('3. Quartz session tap + device bit', method_session_tap),
    ('4. AppleScript System Events', method_applescript),
    ('5. open -a "Mission Control" (up only)', method_open_mission_control),
]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'swipe_up'
    keycode = SWIPE_KEYCODES.get(name)
    if keycode is None:
        sys.exit(f'unknown gesture {name!r}; '
                 f'choose from {", ".join(SWIPE_KEYCODES)}')

    print(f'Accessibility granted: {is_trusted(prompt=False)}')
    if not is_trusted(prompt=False):
        sys.exit('Grant Accessibility to your terminal app first.')

    print(f'\nTesting {name} (key code {keycode}).')
    print('Watch the screen. Note which numbered method reacts, and close')
    print('Mission Control before the next one fires.\n')
    time.sleep(3)

    for label, method in METHODS:
        if method is method_open_mission_control and name != 'swipe_up':
            continue
        print(f'--- {label}', flush=True)
        time.sleep(2)
        try:
            method(keycode)
        except Exception as exc:            # a method may not apply here
            print(f'    failed: {exc}')
        time.sleep(4)

    print('\nDone. Which numbered method worked?')


if __name__ == '__main__':
    main()
