"""Simple joystick reader used by play.py.

Uses pygame if available. Produces (lin_x, lin_y, ang_z, ry) in range [-1, 1].
"""
from __future__ import annotations

import threading
import time
from typing import Tuple


class JoystickReader:
  def __init__(self, poll_hz: float = 50.0, deadzone: float = 0.3):
    self._poll_hz = poll_hz
    self._deadzone = deadzone
    self._lx = 0.0
    self._ly = 0.0
    self._rz = 0.0
    self._ry = 0.0
    self._buttons: dict[int, bool] = {}
    self._button_toggles: dict[int, bool] = {}
    self._prev_buttons: dict[int, bool] = {}
    self._lock = threading.Lock()

    # Try pygame backend.
    try:
      import pygame

      pygame.init()
      pygame.joystick.init()
      if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick detected via pygame")
      self._joy = pygame.joystick.Joystick(0)
      self._joy.init()
      self._backend = "pygame"
      print(f"[Joystick] {self._joy.get_name()}, {self._joy.get_numaxes()} axes")
    except Exception as e:
      raise RuntimeError(f"Joystick initialization failed: {e}")

    self._thread = threading.Thread(target=self._poll_loop, daemon=True)
    self._thread.start()

  def _poll_loop(self):
    import pygame

    clock = pygame.time.Clock()
    while True:
      try:
        pygame.event.pump()
        # PS4/PS5 controller (6 axes):
        #   0: left stick X, 1: left stick Y
        #   2: L2 trigger,   3: right stick X
        #   4: right stick Y, 5: R2 trigger
        # Other controllers (4 axes): axis 2 = right stick X
        ax_lstick_x = self._safe_axis(0)
        ax_lstick_y = self._safe_axis(1)
        if self._joy.get_numaxes() >= 6:
          ax_rstick_x = self._safe_axis(3)
          ax_rstick_y = self._safe_axis(4)
        else:
          ax_rstick_x = self._safe_axis(2)
          ax_rstick_y = self._safe_axis(3)

        # Map axes: stick Y (forward/back) → lin_vel_x, stick X (left/right) → lin_vel_y.
        # Invert stick Y so pushing forward => positive lin_vel_x.
        lx = self._apply_deadzone(float(-(ax_lstick_y or 0.0)))
        ly = self._apply_deadzone(float(-(ax_lstick_x or 0.0)))
        rz = self._apply_deadzone(float(-(ax_rstick_x or 0.0)))
        ry = self._apply_deadzone(float(-(ax_rstick_y or 0.0)))

        # Read buttons and detect toggle (press edge).
        buttons = {}
        for i in range(self._joy.get_numbuttons()):
          buttons[i] = bool(self._joy.get_button(i))

        with self._lock:
          self._lx = lx
          self._ly = ly
          self._rz = rz
          self._ry = ry
          for i, pressed in buttons.items():
            was_pressed = self._prev_buttons.get(i, False)
            if pressed and not was_pressed:
              self._button_toggles[i] = not self._button_toggles.get(i, False)
            self._buttons[i] = pressed
          self._prev_buttons = buttons
      except Exception:
        pass
      clock.tick(self._poll_hz)

  def _apply_deadzone(self, val: float) -> float:
    if abs(val) < self._deadzone:
      return 0.0
    return val

  def _safe_axis(self, idx: int):
    try:
      if idx < self._joy.get_numaxes():
        return self._joy.get_axis(idx)
    except Exception:
      return None

  def get_values(self) -> Tuple[float, float, float, float]:
    with self._lock:
      return (self._lx, self._ly, self._rz, self._ry)

  def get_button_toggle(self, button: int) -> bool:
    """Returns toggle state for a button (flips on each press)."""
    with self._lock:
      return self._button_toggles.get(button, False)
