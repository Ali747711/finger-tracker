"""Test the macOS hotkey path on its own, with no camera or gestures.

Run it, then watch your screen:

    .venv/bin/python diagnose_hotkey.py

It counts down, fires Ctrl+Up (Mission Control), waits, then fires it
again to close. If Mission Control opens, the keyboard layer works and
any remaining swipe problem is in gesture detection. If nothing happens,
the problem is permissions or the shortcut itself.
"""

import sys
import time

from mac_actions import SWIPE_KEYCODES, is_trusted, post_system_hotkey


def countdown(seconds, message):
    for remaining in range(seconds, 0, -1):
        print(f'{message} in {remaining}...', flush=True)
        time.sleep(1)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'swipe_up'
    keycode = SWIPE_KEYCODES.get(name)
    if keycode is None:
        sys.exit(f'unknown gesture {name!r}; '
                 f'choose from {", ".join(SWIPE_KEYCODES)}')

    print(f'Accessibility permission granted: {is_trusted(prompt=False)}')
    if not is_trusted(prompt=False):
        print('\nWithout it macOS silently discards every synthetic key.')
        print('System Settings > Privacy & Security > Accessibility >')
        print('enable your terminal app, then restart the terminal.')
        return

    countdown(3, f'Firing {name} (Ctrl+key {keycode})')
    post_system_hotkey(keycode)
    print('sent — did the screen react?')

    if name == 'swipe_up':
        countdown(3, 'Firing it again to close Mission Control')
        post_system_hotkey(keycode)
        print('sent')


if __name__ == '__main__':
    main()
