"""Dispatch tests for the macOS executor.

Real pynput controllers are never constructed — the fakes below record
calls instead, so nothing is posted to the OS.
"""

from pynput.mouse import Button

from mac_actions import SWIPE_KEYCODES, MacController, hotkey_script


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


class FakeHotkey:
    """Stands in for post_system_hotkey — records key codes instead of
    posting real events to the OS."""

    def __init__(self):
        self.calls = []

    def __call__(self, keycode):
        self.calls.append(keycode)


def make_controller():
    # __new__ skips __init__, so no real OS controllers are created
    controller = MacController.__new__(MacController)
    mouse, hotkey = FakeMouse(), FakeHotkey()
    controller._mouse = mouse
    controller._hotkey = hotkey
    return controller, mouse, hotkey


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
        # ('scroll', dx, dy) must reach pynput's scroll(dx, dy) in the
        # same order: horizontal first, vertical second. Swapping these
        # sends vertical hand motion to the horizontal axis, which most
        # apps ignore — so scrolling looks like it barely works.
        controller, mouse, _ = make_controller()
        controller.execute([('scroll', 2, -3)])
        assert mouse.calls == [('scroll', 2, -3)]

    def test_each_swipe_maps_to_its_arrow_key(self):
        for name, keycode in SWIPE_KEYCODES.items():
            controller, _, hotkey = make_controller()
            controller.execute([('hotkey', name)])
            assert hotkey.calls == [keycode], name

    def test_arrow_key_codes_are_the_macos_ones(self):
        # left/right/down/up virtual key codes; a wrong code here would
        # silently fire some other shortcut
        assert SWIPE_KEYCODES == {'swipe_left': 123, 'swipe_right': 124,
                                  'swipe_down': 125, 'swipe_up': 126}

    def test_hotkey_script_presses_ctrl_and_the_key(self):
        # Quartz-posted events reach apps but are never matched against
        # Mission Control shortcuts; System Events is what works
        script = hotkey_script(126)
        assert script == ('tell application "System Events" to '
                          'key code 126 using control down')

    def test_unknown_hotkey_is_ignored(self):
        controller, _, hotkey = make_controller()
        controller.execute([('hotkey', 'swipe_diagonal')])
        assert hotkey.calls == []

    def test_empty_action_list_is_noop(self):
        controller, mouse, hotkey = make_controller()
        controller.execute([])
        assert mouse.calls == [] and hotkey.calls == []
