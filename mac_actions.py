"""Executes abstract control actions on macOS.

The only file that touches the OS. Mouse actions go through pynput;
system shortcuts go through AppleScript, because macOS ignores
synthetic key events for those (see post_system_hotkey).

Needs the Accessibility permission (System Settings -> Privacy &
Security -> Accessibility -> enable your terminal app), and macOS will
also ask to allow controlling System Events the first time a swipe
fires.
"""

import subprocess
import threading

from pynput.mouse import Button, Controller as MouseController

# macOS virtual key codes for the default Mission Control shortcuts.
SWIPE_KEYCODES = {
    'swipe_left': 123,   # Ctrl+Left  -> previous desktop space
    'swipe_right': 124,  # Ctrl+Right -> next desktop space
    'swipe_up': 126,     # Ctrl+Up    -> Mission Control
    'swipe_down': 125,   # Ctrl+Down  -> App Expose
}
def hotkey_script(keycode):
    """AppleScript that presses Ctrl+<keycode>."""
    return ('tell application "System Events" to '
            f'key code {keycode} using control down')


def post_system_hotkey(keycode, background=True):
    """Fire Ctrl+<keycode> as a genuine system shortcut.

    Spawning osascript is a strange-looking way to press a key, and
    posting the event with Quartz is far cheaper — but macOS hands those
    events to the focused application as a plain Ctrl+key without ever
    matching them against Mission Control and Spaces shortcuts, so they
    silently do nothing. System Events is what actually triggers the
    shortcut. diagnose_hotkey.py compares all the alternatives if this
    ever needs revisiting.

    osascript costs ~100 ms, so it runs on a throwaway thread instead of
    stalling the camera loop; the swipe cooldown stops these overlapping.
    """
    script = hotkey_script(keycode)
    if not background:
        return _run_script(script)
    threading.Thread(target=_run_script, args=(script,), daemon=True).start()
    return None


def _run_script(script):
    # capture_output keeps AppleScript errors off the app's own output,
    # and run() reaps the child so a long session can't leak zombies
    return subprocess.run(['osascript', '-e', script], capture_output=True)


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
