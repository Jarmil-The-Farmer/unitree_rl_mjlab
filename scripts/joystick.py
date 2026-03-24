"""Simple joystick reader used by play.py.

Uses pygame if available. Produces (lin_x, lin_y, ang_z) in range [-1, 1].
"""
from __future__ import annotations

import threading
import time
from typing import Tuple


class JoystickReader:
  def __init__(self, poll_hz: float = 50.0):
    self._poll_hz = poll_hz
    self._lx = 0.0
    self._ly = 0.0
    self._rz = 0.0
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
        # Default mapping guesses (may vary by controller):
        # axis 0: left stick X, axis 1: left stick Y
        # axis 2 or 3: right stick X (try 2 then 3)
        ax0 = self._safe_axis(0)
        ax1 = self._safe_axis(1)
        ax2 = self._safe_axis(2)
        if ax2 is None:
          ax2 = self._safe_axis(3) or 0.0

        # Map axes to commands. Invert Y axis so pushing forward => positive.
        lx = float(ax0 or 0.0)
        ly = float(-(ax1 or 0.0))
        rz = float(ax2 or 0.0)

        with self._lock:
          self._lx = lx
          self._ly = ly
          self._rz = rz
      except Exception:
        pass
      clock.tick(self._poll_hz)

  def _safe_axis(self, idx: int):
    try:
      if idx < self._joy.get_numaxes():
        return self._joy.get_axis(idx)
    except Exception:
      return None

  def get_values(self) -> Tuple[float, float, float]:
    with self._lock:
      return (self._lx, self._ly, self._rz)
