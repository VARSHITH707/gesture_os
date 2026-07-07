# GestureOS — Morning Physical Validation Plan

All 128 automated tests pass. The remaining validation requires a physical hand
in front of a webcam. Work through each section in order; failures at any step
include a diagnosis checklist.

---

## Pre-flight (2 minutes)

```
python gesture_os/main.py
```

Expected log lines (first ~5 seconds):
```
GestureOS started — camera XX.X FPS  screen 1920x1080
```
System tray icon should appear (green circle).

**If camera FPS < 20**: check lighting, close other apps using the camera.  
**If no tray icon appears**: `pip install pystray pillow`.  
**If ModuleNotFoundError**: run from `I:\gesture_os` with the full Python path.

---

## 1. Cursor Control

**Action**: Hold your right hand in front of the webcam with all fingers extended
(open relaxed hand). Move hand left/right/up/down slowly.

**Expected**: Mouse cursor follows with ~1 second of warmup lag then smooth
trackpad-style relative motion. Moving 10 cm should move the cursor
roughly half the screen.

**Pass criteria**: Cursor moves without jumping. No clicks fire.

**Fail diagnosis**:
- Cursor jitters → lower `filter_beta` in `config.py` (`FilterConfig.beta = 0.005`)
- Cursor doesn't move → check `cursor.sensitivity` in `CursorConfig`
- Cursor jumps on re-detection → `on_hand_lost()` / reacquisition path broken

---

## 2. Index-Thumb Tap (Left Click)

**Action**: While tracking cursor, briefly touch index fingertip to thumb tip
and release within 300 ms.

**Expected**: Single left-click fires once. Log: `Left click`.  
No click fires during the pinch — only after release.

**Pass criteria**: A UI element under the cursor activates (e.g. click the
desktop to deselect, or tap a button).

**Fail diagnosis**:
- Click fires immediately on pinch → debounce broken (`debounce_ms`)
- No click fires → `pinch_enter` threshold too tight; measure `thumb_index_dist`
  from debug overlay (`show_debug_window: true` in `config.py`)
- Double-click fires instead of single → `double_tap_ms` window too long

---

## 3. Double Tap (Right Click)

**Action**: Two rapid index-thumb pinch-release cycles within 300 ms.

**Expected**: Context menu appears (right-click). Log shows right-click. No
left-click fires.

**Pass criteria**: Right-click context menu visible on screen.

**Fail diagnosis**:
- First tap fires left-click before second tap → `double_tap_ms` too short
  (increase to 400 ms)
- Both taps produce left-clicks → second pinch not detected within window

---

## 4. Long-Press Drag

**Action**: Touch index-thumb, hold for 600+ ms, then move hand while held.

**Expected**: Log: `mouse_press`. Dragging a window: window moves with hand.
On release (open fingers): `mouse_release`.

**Pass criteria**: Can drag a window from one position to another.

**Fail diagnosis**:
- Press fires immediately → `long_press_ms` too short
- Window doesn't follow → cursor not moving during drag (check `cursor_active`
  in drag state)
- Mouse stays pressed after release → `mouse_release` not emitting; check
  `GestureMode.DRAG` exit condition

---

## 5. Fist Short Drag

**Action**: Make a fist, hold for 200–400 ms, move hand while fist, then open.

**Expected**: Mouse button held during fist drag, released when hand opens.

**Pass criteria**: Can move a selected item while fist is held.

**Fail diagnosis**:
- Goes straight to pause → `fist_pause_ms` not 2000 ms
- No drag starts → short fist not entering FIST_DRAG

---

## 6. Two-Second Fist Pause

**Action**: Make a tight fist and hold for 2+ seconds without moving.

**Expected**: Log: `GestureOS paused`. Tray icon turns grey.  
Cursor stops responding. All mouse buttons released.

**Pass criteria**: Icon is grey, cursor ignores hand movement.

**Resuming**: After pause, show fist again for 2 seconds → Log: `GestureOS resumed`.
Tray icon turns green.

**Fail diagnosis**:
- Pause fires after 0.5 s → `fist_pause_ms` is in seconds not ms (check config)
- No pause fires → finger extension detection: curl all 4 fingers firmly

---

## 7. Two-Finger Scroll

**Action**: Extend only index and middle fingers (like peace sign but pointed
forward, not V-shape), curl ring and pinky. Move hand up and down.

**Expected**: Page scrolls smoothly. Moving down → scrolls page down.
Cursor invisible during scroll.

**Pass criteria**: Browser/text editor scrolls 3–4 lines per second of movement.

**Fail diagnosis**:
- No scroll → scroll pose not detected; try curling ring/pinky more firmly
- Scroll too fast → reduce `scroll_speed_scale` (default 100.0)
- Cursor still moves during scroll → `cursor_active` not False in SCROLL state

---

## 8. Swipe Right (Ctrl+Tab — next tab)

**Action**: Open palm, move right quickly (>18 cm equivalent at arm distance,
within 0.6 s).

**Expected**: Log: `Swipe RIGHT → next tab`. Sends Ctrl+Tab, which switches to
the next tab *within the focused application* (browser, editor) — not the
OS-level window switcher.

**Pass criteria**: Tab focus changes within the active app.

**Fail diagnosis**:
- No swipe → increase `swipe_min_displacement` or reduce `swipe_min_velocity`
- Multiple swipes fire → `swipe_cooldown_s` too short (default 0.6 s)
- Wrong direction classified → check palm angle; hand must be flat

---

## 9. Swipe Left (Ctrl+Shift+Tab — previous tab)

Same as above but move hand left.

**Expected**: Log: `Swipe LEFT → prev tab`. Sends Ctrl+Shift+Tab (previous
tab within the focused app, not the OS window switcher).

---

## 10. Swipe Down (Win+M, minimize all)

**Action**: Open palm, flick downward quickly.

**Expected**: All windows minimized. Log: `Swipe DOWN → Win+M`.

---

## 11. Swipe Up (Win+Shift+M, restore all)

**Action**: Open palm, flick upward quickly.

**Expected**: All minimized windows restored.

---

## 12. Middle-Thumb Wispr Toggle

**Action**: Touch thumb tip to middle fingertip briefly (not index — the finger
next to it). Release.

**Expected**: Wispr Flow dictation starts (overlay appears, microphone activates).
Log: `Wispr shortcut — local toggle=True`.

Second tap: Wispr Flow stops (overlay closes, log: `local toggle=False`).

**Pass criteria**: Wispr Flow dictation toggles on and off.

**Fail diagnosis**:
- Index pinch triggers instead → finger geometry ambiguous; separate index
  finger more from the thumb-middle contact
- Nothing happens → Wispr Flow not running in background; start it first
- Three-finger gesture fires instead → middle finger not isolated; curl index
  and ring slightly

---

## 13. Three-Finger Double Tap (Left Click alternative)

**Action**: Touch thumb to both index AND middle fingertips simultaneously
(three fingers meeting). Release. Repeat within 400 ms.

**Expected**: Left click fires on second tap. Log shows left_click.

**Pass criteria**: UI element activated.

**Fail diagnosis**:
- First tap fires immediately → THREE_SETTLE debounce issue
- Only wispr fires → index not close enough to thumb; bring all three together

---

## 14. Tray Menu Controls

Right-click the tray icon.

**Expected menu items**: Pause / Resume (toggles), Exit.

**Pause via tray**: Click "Pause" → icon goes grey, gestures stop.  
**Resume via tray**: Click "Resume" → icon goes green, gestures work.  
**Exit via tray**: Click "Exit" → process terminates cleanly, no hung keys.

---

## 15. Autostart Verification

```
python -c "from gesture_os.startup import verify_autostart; import pprint; pprint.pprint(verify_autostart())"
```

**Expected output**:
```python
{'registered': True, 'match': True, 'command': '"C:\\...python.exe" "I:\\gesture_os\\gesture_os\\main.py"', ...}
```

**Fail diagnosis**: `registered=False` → `register_autostart()` failed silently;
run with admin or check registry manually at
`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`.

---

## 16. Clean Shutdown Stress Test

With the app running:
1. Start a drag gesture (hold pinch)
2. Immediately close the terminal (Ctrl+C)

**Expected**: Log shows safe release sequence. Mouse button is NOT left held
down after process exits.

**Pass criteria**: Can click normally with real mouse after kill.

---

## Performance Targets

Check the stats (add a print to `get_stats()` or log it every 30 s):

| Metric | Target | Acceptable |
|--------|--------|------------|
| cam_fps | ≥ 25 | ≥ 20 |
| pipeline_fps | ≥ 25 | ≥ 20 |
| avg_latency_ms | ≤ 50 | ≤ 80 |
| p95_latency_ms | ≤ 100 | ≤ 150 |

If pipeline_fps < cam_fps, the bottleneck is MediaPipe inference, not camera.
Consider reducing resolution to 320×240 in `CameraConfig`.

---

## Quick Config Tuning Reference

| Symptom | Parameter | Direction |
|---------|-----------|-----------|
| Click fires too easily | `pinch_enter` | increase (0.06 → 0.07) |
| Pinch not detected | `pinch_enter` | decrease (0.06 → 0.05) |
| Cursor jitter | `FilterConfig.beta` | decrease |
| Cursor lag | `FilterConfig.beta` | increase |
| Scroll too sensitive | `scroll_speed_scale` | decrease |
| Swipe too hard to trigger | `swipe_min_displacement` | decrease |
| Accidental swipes | `swipe_min_velocity` | increase |
| Drag starts too late | `long_press_ms` | decrease |
| Accidental drag | `long_press_ms` | increase |
| Pause fires too easily | `fist_pause_ms` | increase |

All parameters are in `gesture_os/config.py`.

---

## Summary Checklist

- [ ] Cursor moves smoothly
- [ ] Left click (index-thumb tap)
- [ ] Right click (double tap)
- [ ] Long-press drag
- [ ] Fist short drag
- [ ] 2-second fist pause/resume
- [ ] Two-finger scroll (up + down)
- [ ] Swipe right (Ctrl+Tab, next tab)
- [ ] Swipe left (Ctrl+Shift+Tab, prev tab)
- [ ] Swipe down (Win+M)
- [ ] Swipe up (Win+Shift+M)
- [ ] Wispr Flow toggle
- [ ] Three-finger double tap
- [ ] Tray pause/resume/exit
- [ ] Autostart registered
- [ ] Clean shutdown (no stuck keys)
