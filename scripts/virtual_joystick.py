"""Virtual joystick GUI for remote control without a physical gamepad.

Drop-in replacement for JoystickReader. Uses tkinter (no extra deps).

Controls:
  Left pad  / WASD        : lin_vel_x, lin_vel_y
  Right pad / Arrow keys   : ang_vel_z, height (ry)
  D-pad up/down / Q (up) E (down)    : shoulder pitch (hold)
  D-pad left/right / Z (left) C (right) : waist yaw (hold; teleop tasks only)
  Keys 1,2,3,4            : toggle buttons 0–3 (Cross, Circle, Triangle, Square)
  Space                   : zero all axes (emergency stop)
"""
from __future__ import annotations

import threading
import tkinter as tk
from typing import Tuple

# ── Geometry ──────────────────────────────────────────────────────────
PAD_RADIUS = 100
KNOB_RADIUS = 16
MARGIN = 30

PAD_CX_L = MARGIN + PAD_RADIUS                       # left stick center
DPAD_CX = PAD_CX_L + PAD_RADIUS + 45                 # D-pad center (between pads)
PAD_CX_R = DPAD_CX + 45 + PAD_RADIUS                 # right stick center
PAD_CY = PAD_RADIUS + 50                              # shared vertical center

# PS4-style button diamond to the right of the right pad.
BTN_R = 18          # button circle radius
BTN_SP = 36         # spacing from diamond center to each button
BTN_CX = PAD_CX_R + PAD_RADIUS + 70
BTN_CY = PAD_CY

# D-pad arrow buttons.
DPAD_BTN_W = 36
DPAD_BTN_H = 30
DPAD_GAP = 6        # gap between up and down arrows

WIN_W = BTN_CX + BTN_SP + BTN_R + MARGIN
WIN_H = 2 * PAD_RADIUS + 130

# ── Keyboard-axis config ─────────────────────────────────────────────
KB_STEP = 1.0


class VirtualJoystickReader:
  """Tkinter GUI joystick with the same interface as JoystickReader."""

  def __init__(self, poll_hz: float = 50.0, deadzone: float = 0.0):
    self._deadzone = deadzone
    self._lx = 0.0  # forward  / back
    self._ly = 0.0  # left     / right
    self._rz = 0.0  # rotation
    self._ry = 0.0  # height
    self._dpad_x = 0.0  # D-pad X: left = -1, right = +1 (waist yaw)
    self._dpad_y = 0.0  # D-pad Y: up = -1, down = +1 (shoulder pitch)
    self._button_toggles: dict[int, bool] = {}
    self._lock = threading.Lock()

    # Keyboard state tracking.
    self._keys_down: set[str] = set()
    # Mouse-held D-pad direction: None, "up", "down", "left", or "right".
    self._dpad_mouse: str | None = None

    # Start GUI in its own thread.
    self._thread = threading.Thread(target=self._run_gui, daemon=True)
    self._thread.start()

  # ── public API (same as JoystickReader) ───────────────────────────
  def get_values(self) -> Tuple[float, float, float, float]:
    with self._lock:
      return (self._lx, self._ly, self._rz, self._ry)

  def get_dpad_y(self) -> float:
    """Returns D-pad Y axis: -1 = up, +1 = down, 0 = neutral."""
    with self._lock:
      return self._dpad_y

  def get_dpad_x(self) -> float:
    """Returns D-pad X axis: -1 = left, +1 = right, 0 = neutral."""
    with self._lock:
      return self._dpad_x

  def get_button_toggle(self, button: int) -> bool:
    with self._lock:
      return self._button_toggles.get(button, False)

  # ── GUI ───────────────────────────────────────────────────────────
  def _run_gui(self):
    self._root = root = tk.Tk()
    root.title("Virtual Joystick")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=WIN_W, height=WIN_H, bg="#2b2b2b")
    canvas.pack()
    self._canvas = canvas

    # Draw stick pads.
    for cx, label in [(PAD_CX_L, "Move (WASD)"), (PAD_CX_R, "Rot/Height (\u2190\u2191\u2192\u2193)")]:
      canvas.create_oval(
        cx - PAD_RADIUS, PAD_CY - PAD_RADIUS,
        cx + PAD_RADIUS, PAD_CY + PAD_RADIUS,
        outline="#555", width=2,
      )
      canvas.create_line(cx - PAD_RADIUS, PAD_CY, cx + PAD_RADIUS, PAD_CY, fill="#333")
      canvas.create_line(cx, PAD_CY - PAD_RADIUS, cx, PAD_CY + PAD_RADIUS, fill="#333")
      canvas.create_text(cx, PAD_CY + PAD_RADIUS + 16, text=label, fill="#aaa", font=("sans", 10))

    # Knobs (draggable).
    self._knob_l = self._make_knob(canvas, PAD_CX_L, PAD_CY)
    self._knob_r = self._make_knob(canvas, PAD_CX_R, PAD_CY)

    # ── D-pad (up/down arrows between the pads) ─────────────────────
    dpad_top = PAD_CY - DPAD_BTN_H - DPAD_GAP // 2
    dpad_bot = PAD_CY + DPAD_GAP // 2
    self._dpad_up_rect = canvas.create_rectangle(
      DPAD_CX - DPAD_BTN_W // 2, dpad_top,
      DPAD_CX + DPAD_BTN_W // 2, dpad_top + DPAD_BTN_H,
      fill="#334", outline="#555", width=2,
    )
    canvas.create_text(DPAD_CX, dpad_top + DPAD_BTN_H // 2,
                       text="\u25b2", fill="#ccc", font=("sans", 14))
    self._dpad_down_rect = canvas.create_rectangle(
      DPAD_CX - DPAD_BTN_W // 2, dpad_bot,
      DPAD_CX + DPAD_BTN_W // 2, dpad_bot + DPAD_BTN_H,
      fill="#334", outline="#555", width=2,
    )
    canvas.create_text(DPAD_CX, dpad_bot + DPAD_BTN_H // 2,
                       text="\u25bc", fill="#ccc", font=("sans", 14))
    # Left / Right arrows for waist yaw (placed alongside up arrow).
    dpad_left_x = DPAD_CX - DPAD_BTN_W - DPAD_GAP // 2 - DPAD_BTN_W // 2
    dpad_right_x = DPAD_CX + DPAD_GAP // 2 + DPAD_BTN_W // 2
    self._dpad_left_rect = canvas.create_rectangle(
      dpad_left_x, dpad_top,
      dpad_left_x + DPAD_BTN_W, dpad_top + DPAD_BTN_H,
      fill="#334", outline="#555", width=2,
    )
    canvas.create_text(dpad_left_x + DPAD_BTN_W // 2, dpad_top + DPAD_BTN_H // 2,
                       text="\u25c0", fill="#ccc", font=("sans", 14))
    self._dpad_right_rect = canvas.create_rectangle(
      dpad_right_x, dpad_top,
      dpad_right_x + DPAD_BTN_W, dpad_top + DPAD_BTN_H,
      fill="#334", outline="#555", width=2,
    )
    canvas.create_text(dpad_right_x + DPAD_BTN_W // 2, dpad_top + DPAD_BTN_H // 2,
                       text="\u25b6", fill="#ccc", font=("sans", 14))
    # Store rects for hit-test.
    self._dpad_up_bbox = (
      DPAD_CX - DPAD_BTN_W // 2, dpad_top,
      DPAD_CX + DPAD_BTN_W // 2, dpad_top + DPAD_BTN_H,
    )
    self._dpad_down_bbox = (
      DPAD_CX - DPAD_BTN_W // 2, dpad_bot,
      DPAD_CX + DPAD_BTN_W // 2, dpad_bot + DPAD_BTN_H,
    )
    self._dpad_left_bbox = (
      dpad_left_x, dpad_top,
      dpad_left_x + DPAD_BTN_W, dpad_top + DPAD_BTN_H,
    )
    self._dpad_right_bbox = (
      dpad_right_x, dpad_top,
      dpad_right_x + DPAD_BTN_W, dpad_top + DPAD_BTN_H,
    )
    canvas.create_text(DPAD_CX, dpad_bot + DPAD_BTN_H + 14,
                       text="Shoulder Q/E  |  Waist Z/C", fill="#aaa", font=("sans", 9))

    # ── PS4 button diamond (right side) ──────────────────────────────
    # pygame mapping: 0=Cross, 1=Circle, 2=Triangle, 3=Square
    #        Triangle [3]
    #   Square [4]   Circle [2]
    #        Cross [1]
    self._btn_ids: dict[int, int] = {}
    self._btn_label_ids: dict[int, int] = {}
    # (button_index, dx, dy, symbol, off_color, on_color, key_label)
    ps4_buttons = [
      (0, 0, +BTN_SP, "\u2715", "#334", "#5577dd", "1"),    # Cross    – bottom
      (1, +BTN_SP, 0, "\u25cb", "#334", "#dd5555", "2"),    # Circle   – right
      (2, 0, -BTN_SP, "\u25b3", "#334", "#55bb77", "3"),    # Triangle – top
      (3, -BTN_SP, 0, "\u25a1", "#334", "#cc55aa", "4"),    # Square   – left
    ]
    for idx, dx, dy, sym, off_col, on_col, kl in ps4_buttons:
      bx = BTN_CX + dx
      by = BTN_CY + dy
      oid = canvas.create_oval(
        bx - BTN_R, by - BTN_R, bx + BTN_R, by + BTN_R,
        fill=off_col, outline="#666", width=2,
      )
      canvas.create_text(bx, by - 1, text=sym, fill="#ccc", font=("sans", 14))
      lid = canvas.create_text(bx, by + BTN_R + 10, text=kl, fill="#666", font=("sans", 8))
      self._btn_ids[idx] = oid
      self._btn_label_ids[idx] = lid
    self._btn_off = {i: c[3] for i, *c in ps4_buttons}
    self._btn_on = {i: c[4] for i, *c in ps4_buttons}
    canvas.create_text(BTN_CX, BTN_CY + BTN_SP + BTN_R + 26,
                       text="Buttons", fill="#aaa", font=("sans", 10))

    # Info line.
    self._info = canvas.create_text(
      WIN_W // 2, 16, text="Space = stop | click or 1/2/3/4 = toggle buttons",
      fill="#777", font=("sans", 9),
    )

    # Value readout.
    self._readout = canvas.create_text(
      WIN_W // 2, 34, text="", fill="#8f8", font=("mono", 9),
    )

    # ── Mouse drag logic ────────────────────────────────────────────
    self._dragging: str | None = None  # "L" or "R"

    canvas.bind("<ButtonPress-1>", self._on_press)
    canvas.bind("<B1-Motion>", self._on_drag)
    canvas.bind("<ButtonRelease-1>", self._on_release)

    # ── Keyboard ────────────────────────────────────────────────────
    root.bind("<KeyPress>", self._on_key_press)
    root.bind("<KeyRelease>", self._on_key_release)

    # Periodic update.
    self._update_from_keys()

    root.protocol("WM_DELETE_WINDOW", self._on_close)
    print("[VirtualJoystick] GUI ready")
    root.mainloop()

  def _make_knob(self, canvas: tk.Canvas, cx: float, cy: float) -> int:
    return canvas.create_oval(
      cx - KNOB_RADIUS, cy - KNOB_RADIUS,
      cx + KNOB_RADIUS, cy + KNOB_RADIUS,
      fill="#4a90d9", outline="#6ab0ff", width=2,
    )

  @staticmethod
  def _in_rect(x: int, y: int, bbox: tuple) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

  # ── Mouse handlers ────────────────────────────────────────────────
  def _on_press(self, event):
    # D-pad arrows (hold-to-activate).
    if self._in_rect(event.x, event.y, self._dpad_up_bbox):
      self._dpad_mouse = "up"
      self._canvas.itemconfig(self._dpad_up_rect, fill="#4a90d9")
      with self._lock:
        self._dpad_y = -1.0
      return
    if self._in_rect(event.x, event.y, self._dpad_down_bbox):
      self._dpad_mouse = "down"
      self._canvas.itemconfig(self._dpad_down_rect, fill="#4a90d9")
      with self._lock:
        self._dpad_y = 1.0
      return
    if self._in_rect(event.x, event.y, self._dpad_left_bbox):
      self._dpad_mouse = "left"
      self._canvas.itemconfig(self._dpad_left_rect, fill="#4a90d9")
      with self._lock:
        self._dpad_x = -1.0
      return
    if self._in_rect(event.x, event.y, self._dpad_right_bbox):
      self._dpad_mouse = "right"
      self._canvas.itemconfig(self._dpad_right_rect, fill="#4a90d9")
      with self._lock:
        self._dpad_x = 1.0
      return

    # PS4 buttons.
    btn_offsets = {
      0: (0, +BTN_SP),   # Cross    – bottom
      1: (+BTN_SP, 0),   # Circle   – right
      2: (0, -BTN_SP),   # Triangle – top
      3: (-BTN_SP, 0),   # Square   – left
    }
    for idx, (dx, dy) in btn_offsets.items():
      bx = BTN_CX + dx
      by = BTN_CY + dy
      if ((event.x - bx) ** 2 + (event.y - by) ** 2) ** 0.5 <= BTN_R:
        self._toggle_button(idx)
        return

    # Stick pads.
    dl = ((event.x - PAD_CX_L) ** 2 + (event.y - PAD_CY) ** 2) ** 0.5
    dr = ((event.x - PAD_CX_R) ** 2 + (event.y - PAD_CY) ** 2) ** 0.5
    if dl <= PAD_RADIUS:
      self._dragging = "L"
      self._move_knob("L", event.x, event.y)
    elif dr <= PAD_RADIUS:
      self._dragging = "R"
      self._move_knob("R", event.x, event.y)

  def _on_drag(self, event):
    if self._dragging:
      self._move_knob(self._dragging, event.x, event.y)

  def _on_release(self, _event):
    # Release D-pad.
    if self._dpad_mouse == "up":
      self._canvas.itemconfig(self._dpad_up_rect, fill="#334")
      with self._lock:
        self._dpad_y = 0.0
      self._dpad_mouse = None
      return
    if self._dpad_mouse == "down":
      self._canvas.itemconfig(self._dpad_down_rect, fill="#334")
      with self._lock:
        self._dpad_y = 0.0
      self._dpad_mouse = None
      return
    if self._dpad_mouse == "left":
      self._canvas.itemconfig(self._dpad_left_rect, fill="#334")
      with self._lock:
        self._dpad_x = 0.0
      self._dpad_mouse = None
      return
    if self._dpad_mouse == "right":
      self._canvas.itemconfig(self._dpad_right_rect, fill="#334")
      with self._lock:
        self._dpad_x = 0.0
      self._dpad_mouse = None
      return

    # Release stick.
    if self._dragging == "L":
      self._set_knob_pos("L", 0, 0)
      self._move_knob_visual(self._knob_l, PAD_CX_L, PAD_CY)
      with self._lock:
        self._lx = 0.0
        self._ly = 0.0
    elif self._dragging == "R":
      self._set_knob_pos("R", 0, 0)
      self._move_knob_visual(self._knob_r, PAD_CX_R, PAD_CY)
      with self._lock:
        self._rz = 0.0
        self._ry = 0.0
    self._dragging = None

  def _move_knob(self, side: str, mx: int, my: int):
    cx = PAD_CX_L if side == "L" else PAD_CX_R
    dx = (mx - cx) / PAD_RADIUS
    dy = (my - PAD_CY) / PAD_RADIUS
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > 1.0:
      dx /= dist
      dy /= dist
    knob = self._knob_l if side == "L" else self._knob_r
    self._move_knob_visual(knob, cx + dx * PAD_RADIUS, PAD_CY + dy * PAD_RADIUS)
    self._set_knob_pos(side, dx, dy)

  def _move_knob_visual(self, knob_id: int, x: float, y: float):
    self._canvas.coords(
      knob_id,
      x - KNOB_RADIUS, y - KNOB_RADIUS,
      x + KNOB_RADIUS, y + KNOB_RADIUS,
    )

  def _set_knob_pos(self, side: str, dx: float, dy: float):
    """dx: right+, dy: down+.  Map to robot axes."""
    with self._lock:
      if side == "L":
        self._lx = -dy  # stick up = forward
        self._ly = -dx  # stick left = positive ly
      else:
        self._rz = -dx  # stick left = positive rz
        self._ry = -dy  # stick up = positive ry

  def _toggle_button(self, idx: int):
    """Toggle a PS4 button and update its visual."""
    with self._lock:
      self._button_toggles[idx] = not self._button_toggles.get(idx, False)
      state = self._button_toggles[idx]
    color = self._btn_on[idx] if state else self._btn_off[idx]
    self._canvas.itemconfig(self._btn_ids[idx], fill=color)

  # ── Keyboard handlers ─────────────────────────────────────────────
  def _on_key_press(self, event):
    key = event.keysym.lower()
    self._keys_down.add(key)

    # Toggle buttons on key-down.
    btn_map = {"1": 0, "2": 1, "3": 2, "4": 3}
    if key in btn_map:
      self._toggle_button(btn_map[key])

    # Emergency stop.
    if key == "space":
      with self._lock:
        self._lx = self._ly = self._rz = self._ry = 0.0
      self._move_knob_visual(self._knob_l, PAD_CX_L, PAD_CY)
      self._move_knob_visual(self._knob_r, PAD_CX_R, PAD_CY)

  def _on_key_release(self, event):
    self._keys_down.discard(event.keysym.lower())

  def _update_from_keys(self):
    """Called every 20ms to update axes from held keys."""
    keys = self._keys_down

    if self._dragging != "L":
      lx = ly = 0.0
      if "w" in keys:
        lx += KB_STEP
      if "s" in keys:
        lx -= KB_STEP
      if "a" in keys:
        ly += KB_STEP
      if "d" in keys:
        ly -= KB_STEP
      with self._lock:
        self._lx = max(-1.0, min(1.0, lx))
        self._ly = max(-1.0, min(1.0, ly))
      self._move_knob_visual(
        self._knob_l,
        PAD_CX_L - self._ly * PAD_RADIUS,
        PAD_CY - self._lx * PAD_RADIUS,
      )

    if self._dragging != "R":
      rz = ry = 0.0
      if "up" in keys:
        ry += KB_STEP
      if "down" in keys:
        ry -= KB_STEP
      if "left" in keys:
        rz += KB_STEP
      if "right" in keys:
        rz -= KB_STEP
      with self._lock:
        self._rz = max(-1.0, min(1.0, rz))
        self._ry = max(-1.0, min(1.0, ry))
      self._move_knob_visual(
        self._knob_r,
        PAD_CX_R - self._rz * PAD_RADIUS,
        PAD_CY - self._ry * PAD_RADIUS,
      )

    # D-pad Y via Q/E and D-pad X via Z/C (only when not mouse-holding).
    if self._dpad_mouse is None:
      dpad_y = 0.0
      if "q" in keys:
        dpad_y -= 1.0  # up
      if "e" in keys:
        dpad_y += 1.0  # down
      dpad_x = 0.0
      if "z" in keys:
        dpad_x -= 1.0  # left
      if "c" in keys:
        dpad_x += 1.0  # right
      with self._lock:
        self._dpad_y = max(-1.0, min(1.0, dpad_y))
        self._dpad_x = max(-1.0, min(1.0, dpad_x))
      # Visual feedback on D-pad arrows for keyboard.
      self._canvas.itemconfig(
        self._dpad_up_rect, fill="#4a90d9" if dpad_y < 0 else "#334")
      self._canvas.itemconfig(
        self._dpad_down_rect, fill="#4a90d9" if dpad_y > 0 else "#334")
      self._canvas.itemconfig(
        self._dpad_left_rect, fill="#4a90d9" if dpad_x < 0 else "#334")
      self._canvas.itemconfig(
        self._dpad_right_rect, fill="#4a90d9" if dpad_x > 0 else "#334")

    # Update readout text.
    with self._lock:
      lx, ly, rz, ry = self._lx, self._ly, self._rz, self._ry
      dpx, dpy = self._dpad_x, self._dpad_y
    self._canvas.itemconfig(
      self._readout,
      text=f"lx={lx:+.2f}  ly={ly:+.2f}  rz={rz:+.2f}  ry={ry:+.2f}  dpad=({dpx:+.0f},{dpy:+.0f})",
    )

    self._root.after(20, self._update_from_keys)

  def _on_close(self):
    with self._lock:
      self._lx = self._ly = self._rz = self._ry = 0.0
    self._root.destroy()
