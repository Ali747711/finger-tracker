"""Executes abstract control actions on macOS via pynput.

The only file that touches the OS. Requires the Accessibility
permission: System Settings -> Privacy & Security -> Accessibility ->
enable your terminal app.
"""

from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button, Controller as MouseController

# macOS default Mission Control shortcuts (Ctrl+Arrow).
SWIPE_KEYS = {
    'swipe_left': Key.left,    # previous desktop space
    'swipe_right': Key.right,  # next desktop space
    'swipe_up': Key.up,        # Mission Control
    'swipe_down': Key.down,    # App Expose
}


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
        self._keyboard = KeyboardController()

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
                self._mouse.scroll(action[2], action[1])
            elif kind == 'hotkey':
                key = SWIPE_KEYS.get(action[1])
                if key is not None:
                    with self._keyboard.pressed(Key.ctrl):
                        self._keyboard.press(key)
                        self._keyboard.release(key)
