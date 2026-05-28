"""Log G1 motor temperatures + torque from the real robot to a CSV.

Standalone DDS subscriber — runs alongside the deploy controller without
interfering (DDS allows multiple subscribers to ``rt/lowstate``). Used to
collect calibration data for the simulated motor thermal model.

The robot publishes two per-motor temperatures:
  temperature[0] = casing  (Unitree protection limit ~85 C)
  temperature[1] = winding (Unitree protection limit ~120 C)

Usage (on the robot, after the deploy is running and holding a pose):
  python scripts/log_motor_thermal.py --iface eth0 --pose arms_extended

Stop with Ctrl-C. Writes logs/motor_thermal/<pose>_<timestamp>.csv.
"""

from __future__ import annotations

import argparse
import csv
import signal
import time
from datetime import datetime
from pathlib import Path

from unitree_sdk2py.core.channel import (
  ChannelFactoryInitialize,
  ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Deploy/hardware motor order (29-DoF G1), index -> joint name.
G1_JOINT_NAMES = [
  "left_hip_pitch_joint",       # 0
  "left_hip_roll_joint",        # 1
  "left_hip_yaw_joint",         # 2
  "left_knee_joint",            # 3
  "left_ankle_pitch_joint",     # 4
  "left_ankle_roll_joint",      # 5
  "right_hip_pitch_joint",      # 6
  "right_hip_roll_joint",       # 7
  "right_hip_yaw_joint",        # 8
  "right_knee_joint",           # 9
  "right_ankle_pitch_joint",    # 10
  "right_ankle_roll_joint",     # 11
  "waist_yaw_joint",            # 12
  "waist_roll_joint",           # 13
  "waist_pitch_joint",          # 14
  "left_shoulder_pitch_joint",  # 15
  "left_shoulder_roll_joint",   # 16
  "left_shoulder_yaw_joint",    # 17
  "left_elbow_joint",           # 18
  "left_wrist_roll_joint",      # 19
  "left_wrist_pitch_joint",     # 20
  "left_wrist_yaw_joint",       # 21
  "right_shoulder_pitch_joint", # 22
  "right_shoulder_roll_joint",  # 23
  "right_shoulder_yaw_joint",   # 24
  "right_elbow_joint",          # 25
  "right_wrist_roll_joint",     # 26
  "right_wrist_pitch_joint",    # 27
  "right_wrist_yaw_joint",      # 28
]
NUM_MOTORS = len(G1_JOINT_NAMES)

# Motors most relevant for the leg+waist thermal calibration. Used only for
# the live console summary, not for what gets logged (all 29 are logged).
SUMMARY_MOTORS = [
  "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
  "left_hip_pitch_joint", "right_hip_pitch_joint",
  "left_hip_roll_joint", "right_hip_roll_joint",
  "left_knee_joint", "right_knee_joint",
]


class _Latest:
  """Holds the most recent LowState message (written by the DDS callback)."""

  def __init__(self) -> None:
    self.msg: LowState_ | None = None

  def handler(self, msg: LowState_) -> None:
    self.msg = msg


def _short(name: str) -> str:
  return name.removesuffix("_joint").replace("left_", "L_").replace("right_", "R_")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--iface", required=True,
                  help="Network interface connected to the robot (e.g. eth0).")
  ap.add_argument("--pose", default="pose",
                  help="Short label for this run (e.g. arms_extended, lean_left).")
  ap.add_argument("--rate", type=float, default=10.0,
                  help="Logging rate in Hz (default 10).")
  ap.add_argument("--out-dir", default="logs/motor_thermal",
                  help="Output directory for the CSV.")
  ap.add_argument("--summary-period", type=float, default=5.0,
                  help="Seconds between live console summaries (default 5).")
  args = ap.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out_path = out_dir / f"{args.pose}_{stamp}.csv"

  ChannelFactoryInitialize(0, args.iface)
  latest = _Latest()
  sub = ChannelSubscriber("rt/lowstate", LowState_)
  sub.Init(latest.handler, 10)

  print(f"[log] Subscribing to rt/lowstate on '{args.iface}' ...")
  print("[log] Waiting for first message (is the deploy running?) ...")
  t_wait = time.time()
  while latest.msg is None:
    time.sleep(0.05)
    if time.time() - t_wait > 10.0:
      print("[log] ERROR: no LowState received in 10 s. Check --iface and that "
            "the robot/deploy is up.")
      return
  print(f"[log] Connected. Logging to {out_path}")
  print(f"[log] Pose='{args.pose}'  rate={args.rate} Hz. Press Ctrl-C to stop.\n")

  # CSV header: time + per-motor tau / Tcase / Twind / q / dq.
  header = ["t_s", "wall_clock", "pose"]
  for name in G1_JOINT_NAMES:
    s = _short(name)
    header += [f"tau_{s}", f"Tcase_{s}", f"Twind_{s}", f"q_{s}", f"dq_{s}"]

  summary_ids = [G1_JOINT_NAMES.index(n) for n in SUMMARY_MOTORS]

  stop = {"flag": False}
  signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

  dt = 1.0 / args.rate
  t0 = time.monotonic()
  next_summary = 0.0
  n_rows = 0

  with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    while not stop["flag"]:
      loop_start = time.monotonic()
      t_s = loop_start - t0
      msg = latest.msg
      if msg is not None:
        motors = msg.motor_state
        row = [f"{t_s:.3f}", datetime.now().isoformat(timespec="milliseconds"),
               args.pose]
        for i in range(NUM_MOTORS):
          m = motors[i]
          temp = m.temperature
          row += [f"{m.tau_est:.4f}", int(temp[0]), int(temp[1]),
                  f"{m.q:.5f}", f"{m.dq:.5f}"]
        writer.writerow(row)
        n_rows += 1

        if t_s >= next_summary:
          next_summary = t_s + args.summary_period
          f.flush()
          parts = []
          hot_case = hot_wind = 0
          for i in summary_ids:
            m = motors[i]
            c, w = int(m.temperature[0]), int(m.temperature[1])
            hot_case = max(hot_case, c)
            hot_wind = max(hot_wind, w)
            parts.append(f"{_short(G1_JOINT_NAMES[i])}:{w}/{c}C tau={m.tau_est:+.1f}")
          flag = ""
          if hot_wind >= 120 or hot_case >= 85:
            flag = "  <<< AT LIMIT — STOP & SET PASSIVE"
          elif hot_wind >= 100 or hot_case >= 75:
            flag = "  <<< getting hot"
          print(f"[t={t_s:6.1f}s] (winding/casing) " + "  ".join(parts) + flag)

      sleep = dt - (time.monotonic() - loop_start)
      if sleep > 0:
        time.sleep(sleep)

  print(f"\n[log] Stopped. Wrote {n_rows} rows ({t_s:.1f} s) to {out_path}")


if __name__ == "__main__":
  main()
