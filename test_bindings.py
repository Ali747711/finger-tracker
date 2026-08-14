"""bindings.py is meant to be edited, so these check its shape rather
than its contents — a customised binding must not fail the suite, but a
typo that would silently do nothing should be caught."""

from bindings import BINDINGS
from mac_actions import binding_problems


def test_shipped_bindings_are_valid():
    assert binding_problems(BINDINGS) == []


class TestBindingProblems:
    def test_unknown_action_kind_is_reported(self):
        problems = binding_problems({'ok_sign': ('lanch', 'Telegram')})
        assert len(problems) == 1
        assert 'lanch' in problems[0]

    def test_unknown_hotkey_is_reported(self):
        problems = binding_problems({'ok_sign': ('hotkey', 'swipe_sideways')})
        assert len(problems) == 1
        assert 'swipe_sideways' in problems[0]

    def test_known_hotkey_is_accepted(self):
        assert binding_problems({'ok_sign': ('hotkey', 'swipe_up')}) == []

    def test_wrong_shape_is_reported(self):
        assert len(binding_problems({'ok_sign': 'Telegram'})) == 1
        assert len(binding_problems({'ok_sign': ('launch',)})) == 1

    def test_empty_name_is_reported(self):
        assert len(binding_problems({'ok_sign': ('launch', '')})) == 1

    def test_shell_and_launch_accept_any_string(self):
        assert binding_problems({'a': ('launch', 'Telegram'),
                                 'b': ('shell', 'say hi')}) == []

    def test_no_bindings_is_fine(self):
        assert binding_problems({}) == []
