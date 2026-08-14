"""Dispatch tests for the macOS executor.

Real pynput controllers are never constructed — the fakes below record
calls instead, so nothing is posted to the OS.
"""

from pynput.keyboard import Key
from pynput.mouse import Button

from mac_actions import MacController


class FakeMouse:
    def __init__(self):
        self.calls = []

    @property
    def position(self):
        return None

    @position.setter
    def position(self, value):
        self.calls.append(('position', value))

    def press(self, button):
        self.calls.append(('press', button))

    def release(self, button):
        self.calls.append(('release', button))

    def scroll(self, dx, dy):
        self.calls.append(('scroll', dx, dy))


class FakeKeyboard:
    def __init__(self):
        self.calls = []

    def pressed(self, key):
        calls = self.calls

        class Ctx:
            def __enter__(self):
                calls.append(('hold', key))
                return self

            def __exit__(self, *exc):
                calls.append(('unhold', key))
                return False

        return Ctx()

    def press(self, key):
        self.calls.append(('press', key))

    def release(self, key):
        self.calls.append(('release', key))


def make_controller():
    # __new__ skips __init__, so no real pynput controllers are created
    controller = MacController.__new__(MacController)
    mouse, keyboard = FakeMouse(), FakeKeyboard()
    controller._mouse = mouse
    controller._keyboard = keyboard
    return controller, mouse, keyboard


class TestDispatch:
    def test_move_press_release(self):
        controller, mouse, _ = make_controller()
        controller.execute([('move', 12, 34), ('press',), ('release',)])
        assert mouse.calls == [
            ('position', (12, 34)),
            ('press', Button.left),
            ('release', Button.left),
        ]

    def test_scroll_axis_order(self):
        # pynput's scroll(dx, dy) takes horizontal first, vertical second
        controller, mouse, _ = make_controller()
        controller.execute([('scroll', 2, -3)])
        assert mouse.calls == [('scroll', -3, 2)]

    def test_hotkey_is_ctrl_plus_arrow(self):
        controller, _, keyboard = make_controller()
        controller.execute([('hotkey', 'swipe_right')])
        assert keyboard.calls == [
            ('hold', Key.ctrl),
            ('press', Key.right),
            ('release', Key.right),
            ('unhold', Key.ctrl),
        ]

    def test_unknown_hotkey_is_ignored(self):
        controller, _, keyboard = make_controller()
        controller.execute([('hotkey', 'swipe_diagonal')])
        assert keyboard.calls == []

    def test_empty_action_list_is_noop(self):
        controller, mouse, keyboard = make_controller()
        controller.execute([])
        assert mouse.calls == [] and keyboard.calls == []
