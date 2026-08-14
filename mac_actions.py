"""Executes abstract control actions on macOS via pynput.

The only file that touches the OS. Requires the Accessibility
permission: System Settings -> Privacy & Security -> Accessibility ->
enable your terminal app.
"""

import time

from pynput.mouse import Button, Controller as MouseController

# macOS virtual key codes for the default Mission Control shortcuts.
SWIPE_KEYCODES = {
    'swipe_left': 123,   # Ctrl+Left  -> previous desktop space
    'swipe_right': 124,  # Ctrl+Right -> next desktop space
    'swipe_up': 126,     # Ctrl+Up    -> Mission Control
    'swipe_down': 125,   # Ctrl+Down  -> App Expose
}
KEY_CONTROL = 59
# The system tap that owns Mission Control shortcuts can miss a modifier
# pressed in the same instant as the key, so hold it briefly.
HOTKEY_HOLD_SEC = 0.02


def post_system_hotkey(keycode, hold=HOTKEY_HOLD_SEC):
    """Send Ctrl+<keycode> as real key events.

    Mission Control and Spaces shortcuts are claimed by an event tap that
    reads the modifier flags carried on the key event itself, not just
    the fact that a modifier key is down. Posting the modifier key AND
    stamping the flag on the arrow event is what makes them respond;
    pressing the modifier alone is silently ignored.
    """
    import Quartz

    flags = Quartz.kCGEventFlagMaskControl

    def post(code, down, event_flags):
        event = Quartz.CGEventCreateKeyboardEvent(None, code, down)
        Quartz.CGEventSetFlags(event, event_flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    post(KEY_CONTROL, True, flags)
    time.sleep(hold)
    post(keycode, True, flags)
    time.sleep(hold)
    post(keycode, False, flags)
    post(KEY_CONTROL, False, 0)


def screen_size():
    """Main-screen size in points (what pynput's coordinates use)."""
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    size = (root.winfo_screenwidth(), root.winfo_screenheight())
    root.destroy()
    return size


def is_trusted(prompt=False):
    """True if this process has the Accessibility permission.

    Without it, macOS silently discards every synthetic event pynput
    posts — control would 'work' as a no-op with no error anywhere.
    With prompt=True, macOS shows the grant dialog on first ask.
    """
    from ApplicationServices import (AXIsProcessTrustedWithOptions,
                                     kAXTrustedCheckOptionPrompt)

    return bool(AXIsProcessTrustedWithOptions(
        {kAXTrustedCheckOptionPrompt: prompt}))


def start_kill_listener(callback):
    """Global Esc listener so control can be cut even when a pinch-click
    moved keyboard focus to another app and cv2's keys are unreachable.
    Runs `callback` from a background thread on every Esc press."""
    from pynput import keyboard

    def on_press(key):
        if key == keyboard.Key.esc:
            callback()

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


class MacController:
    def __init__(self):
        self._mouse = MouseController()
        self._hotkey = post_system_hotkey

    def execute(self, actions):
        for action in actions:
            kind = action[0]
            if kind == 'move':
                self._mouse.position = (action[1], action[2])
            elif kind == 'press':
                self._mouse.press(Button.left)
            elif kind == 'release':
                self._mouse.release(Button.left)
            elif kind == 'scroll':
                # ('scroll', dx, dy) -> scroll(dx, dy): horizontal first
                self._mouse.scroll(action[1], action[2])
            elif kind == 'hotkey':
                keycode = SWIPE_KEYCODES.get(action[1])
                if keycode is not None:
                    self._hotkey(keycode)
