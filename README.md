# Finger Tracker

Real-time finger movement detection from the Mac webcam, using MediaPipe Hands + OpenCV.

## v0.6 — custom gesture bindings

Touch **both index fingertips together** to fire a bound action. Out of the box that opens Terminal. Edit [`bindings.py`](bindings.py) to change it — nothing else needs touching:

```python
BINDINGS = {
    'index_touch': ('launch', 'Terminal'),
}
```

Available actions: `('launch', 'AppName')` opens an app, `('hotkey', 'swipe_up')` fires a Mission Control shortcut, `('shell', 'command')` runs a shell command.

The touch **latches**: it fires once when your fingertips meet, and won't fire again until they separate — so resting your fingers together doesn't relaunch anything. A 1.5s cooldown covers jittery contact. The gap is measured against your hand size, so it works at any distance from the camera, and both index fingers must be extended, so a fist bump won't trigger it. While the touch is held, the driving hand stops moving the cursor — a deliberate two-hand gesture shouldn't also click something.

Bindings are gated by the same `c` toggle as everything else: nothing fires while control is off.

## v0.5 — two hands

Both hands are tracked at once, each with completely independent gesture state — a pinch on one hand can't disturb the other's detectors. Hand identity is keyed on MediaPipe's handedness label, with the two ways that goes wrong handled explicitly:

- **Brief dropouts**: a hand missing for 1–2 frames is still considered present, so a flicker can't reset gesture state mid-gesture or hand control to the other hand.
- **Duplicate labels**: MediaPipe sometimes classifies both hands the same way. The more confident hand keeps the contested label and the other takes the free one, so two physical hands never share state.

**The right hand drives macOS control** when both are visible (`PRIMARY_HAND` in `hands.py`); with one hand up, that hand drives regardless of which it is. If control changes hands, any held drag is released first. The overlay shows a line per hand plus which one is `driving`.

The off hand is fully tracked — its fingers, pinch, scroll and swipe events are all detected and logged — but only drives macOS through two-hand gestures (see v0.6), never on its own.

## v0.3 — macOS control

Press `c` in the video window to toggle macOS control (starts **OFF**):

| Gesture | Action |
|---|---|
| Index finger up (others curled) | Move the mouse cursor |
| Pinch (thumb + index) | Mouse press → release = click; hold to drag |
| Index + middle up (V sign) *(v0.4)* | Scroll — move hand up/down (or sideways) |
| Open-palm flick left/right | Switch desktop space (Ctrl+←/→) |
| Open-palm flick up | Mission Control (Ctrl+↑) |
| Open-palm flick down | App Exposé (Ctrl+↓) |

Poses are debounced: scroll and swipe both need their pose held ~3 frames before firing, so a hand passing through a pose in transit doesn't trigger them. A held scroll survives a 1-2 frame finger misread without handing the cursor back.

Tuning knobs in `control.py`:

| Constant | Effect |
|---|---|
| `SCROLL_SENSITIVITY` | Scroll steps per frame-height of hand travel. pynput posts each step as **10 px** on macOS, so 300 ≈ 3000 px per full sweep. |
| `SCROLL_NATURAL` | Flip if scroll direction feels wrong. Synthetic scroll direction is independent of the macOS "Natural scrolling" setting. |
| `CURSOR_MIN_ALPHA` | Damping when the hand is nearly still. Lower = steadier for clicking, slower to settle. |
| `CURSOR_MAX_ALPHA` | Damping when moving fast. 1.0 = zero lag. |
| `CURSOR_SPEED_REF` | Hand speed (fraction of screen per frame) at which damping fully releases. Lower = snappier overall. |
| `CURSOR_MARGIN` | Camera edge dead zone; the inner region maps to the whole screen. |

Swipe shortcuts go through AppleScript (System Events) rather than synthetic key events: macOS delivers Quartz-posted key events to the focused app as a plain Ctrl+key but never matches them against Mission Control or Spaces shortcuts, so they silently do nothing. Run `diagnose_hotkey.py` to see that comparison for yourself. Left/right also need **more than one desktop Space** to have any visible effect.

First use: grant the **Accessibility** permission — System Settings → Privacy & Security → Accessibility → enable your terminal app (the app checks and prompts when you first press `c`; without the permission macOS silently discards synthetic input). The Esc kill switch additionally uses **Input Monitoring** — macOS will prompt for it on first run.

Safety:
- Control always starts **OFF**
- **Esc is a global kill switch** — it works even when a pinch-click moved focus to another app and the tracker window can't hear `q`/`c` anymore; pulling your hand out of frame also pauses all actions
- Toggling off, losing the hand, quitting, or **Ctrl+C mid-drag** all force-release a held mouse button (cleanup runs in `finally`)
- A hand re-detected mid-pinch after a dropout won't auto-re-press — it must open first
- Swipes require an open palm both while moving and at fire time, so fast mousing can't switch desktops
- Clicks land at the hovered position, not where the fingertip drifted while curling into the pinch

## v0.2 — gesture detection

- Opens the built-in camera (mirror view)
- Tracks one hand and draws its 21 landmarks
- Detects which fingers are up (thumb/index/middle/ring/pinky, or fist)
- Detects index-fingertip movement direction: left / right / up / down / still
- **Pinch** (thumb + index tip together) → `pinch_start` / `pinch_end` events, with hysteresis so the state can't flicker
- **Swipes** (fast index-finger flicks) → `swipe_left` / `swipe_right` / `swipe_up` / `swipe_down`, with a cooldown so one flick fires one event; big on-screen flash when one fires

Robustness (added after an adversarial review pass):
- Coordinates are aspect-corrected, so gestures behave the same regardless of hand orientation and swipe axis
- Pinch survives tilting the hand toward the camera (foreshortening-resistant hand scale)
- The return stroke after a swipe can't fire a phantom opposite swipe
- Brief 1–2 frame tracking dropouts no longer reset gesture state (3-frame debounce; a real hand loss mid-pinch emits a paired `pinch_end`)
- Quit with `q` or by closing the window

## Run

```bash
source .venv/bin/activate
python main.py        # press q in the video window to quit
```

First run: macOS will ask for camera permission for your terminal app — grant it and run again.

## Test

```bash
python -m pytest
```

## Layout

| File | Purpose |
|---|---|
| `main.py` | Camera loop, MediaPipe wiring, on-screen overlay, control toggle |
| `hands.py` | Per-hand gesture state, hand identity/registry, primary-hand selection — unit-tested |
| `two_hand.py` | Gestures needing both hands (index-fingertip touch) — unit-tested |
| `bindings.py` | **Edit this** — maps gesture names to actions |
| `diagnose_hotkey.py` | Compares key-posting methods when a shortcut won't fire |
| `hand_logic.py` | Pure logic (fingers up, movement direction) — no camera deps, unit-tested |
| `gestures.py` | Pure gesture detectors (pinch with hysteresis, swipe with cooldown) — time injected, unit-tested |
| `control.py` | Pure control logic (cursor mapping with margin+smoothing, action routing, drag state) — unit-tested |
| `mac_actions.py` | The only file that touches macOS: mouse via pynput, shortcuts and launches via AppleScript/`open` |
| `test_*.py` | Test suites for all pure modules |

## Upgrade ideas (roadmap)

- [x] Pinch detection (thumb–index distance) → click gesture *(v0.2)*
- [x] Swipe gestures (fast directional moves) with cooldown *(v0.2)*
- [x] Map gestures to macOS actions (cursor, click/drag, spaces) via pynput *(v0.3)*
- [x] Scroll gesture (index + middle up, move hand) *(v0.4)*
- [x] Two-hand tracking with independent per-hand state *(v0.5)*
- [x] Custom gesture bindings + two-hand index touch *(v0.6)*
- [ ] More two-hand gestures (modifier hand, zoom/rotate from hand distance)
- [ ] Expose events over WebSocket (FastAPI) so other apps can subscribe
- [ ] Config file for gesture→action mappings
