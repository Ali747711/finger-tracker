"""Custom gesture -> action bindings. Edit this file to change what a
gesture does; nothing else needs touching.

Each value is an action tuple that mac_actions.MacController executes:

    ('launch', 'Terminal')     open an application by name
    ('hotkey', 'swipe_up')     fire a Mission Control shortcut
    ('shell', 'say hello')     run a shell command

Gesture names currently available:

    'index_touch'   touch both index fingertips together

Bindings are gated by the same `c` control toggle as everything else, so
nothing here can fire while control is off.
"""

BINDINGS = {
    'index_touch': ('launch', 'Terminal'),
}
