"""Send arm positions to the deploy controller via LCM.

Modes (cycle with 'm'):
  random  — nudge_arms_position style random targets
  zero    — all arm joints at 0
  forward — arms forward (-1.6, 0, 0, 1.57, 0, 0, 0)

In zero/forward modes, select a joint then adjust with +/-:
  s — select shoulder_pitch
  r — select shoulder_roll (abduction)
  e — select elbow
  +/= — increase selected joint by step
  -   — decrease selected joint by step
  0   — reset selected joint to mode default

Channel: "arm_action"  (arm_action_lcmt, 14 doubles)
Joint order: [left 7] + [right 7], each: shoulder_pitch, shoulder_roll,
  shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw.

Usage:
    python scripts/send_random_arms.py [--speed 0.5] [--offset 0.5] [--hz 50] [--step 0.1]
"""
from __future__ import annotations

import argparse
import random
import select
import struct
import sys
import termios
import time
import tty

import lcm

# ── LCM message encoding ────────────────────────────────────────────────
_BASE_HASH = 0x692F2BE95563F8FE
_FINGERPRINT = ((_BASE_HASH << 1) + (_BASE_HASH >> 63)) & 0xFFFFFFFFFFFFFFFF
_PACKED_FINGERPRINT = struct.pack(">Q", _FINGERPRINT)


def encode_arm_action(positions: list[float]) -> bytes:
    assert len(positions) == 14
    return _PACKED_FINGERPRINT + struct.pack(">14d", *positions)


# ── Joint limits from g1.xml ────────────────────────────────────────────
JOINT_LIMITS_LEFT = [
    (-3.0892, 2.6704),   # shoulder_pitch
    (-1.5882, 2.2515),   # shoulder_roll
    (-2.618, 2.618),     # shoulder_yaw
    (-1.0472, 2.0944),   # elbow
    (-1.9722, 1.9722),   # wrist_roll
    (-1.6144, 1.6144),   # wrist_pitch
    (-1.6144, 1.6144),   # wrist_yaw
]
JOINT_LIMITS_RIGHT = [
    (-3.0892, 2.6704),   # shoulder_pitch
    (-2.2515, 1.5882),   # shoulder_roll  (mirrored)
    (-2.618, 2.618),     # shoulder_yaw
    (-1.0472, 2.0944),   # elbow
    (-1.9722, 1.9722),   # wrist_roll
    (-1.6144, 1.6144),   # wrist_pitch
    (-1.6144, 1.6144),   # wrist_yaw
]
JOINT_LIMITS = JOINT_LIMITS_LEFT + JOINT_LIMITS_RIGHT

# ── Selectable joints ───────────────────────────────────────────────────
# "mirror": True means the right arm gets the opposite sign adjustment
# (shoulder_roll limits are mirrored: left +roll = abduction, right -roll = abduction)
JOINTS = {
    "s": {"name": "shoulder_pitch", "idx": 0, "mirror": False},
    "r": {"name": "shoulder_roll",  "idx": 1, "mirror": True},
    "e": {"name": "elbow",          "idx": 3, "mirror": False},
}

# ── Mode presets ─────────────────────────────────────────────────────────
ZERO_POS = [0.0] * 14
FORWARD_POS = [
    -1.6, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0,
    -1.6, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0,
]
MODES = ["random", "zero", "forward"]


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def sample_goal(current: list[float], offset_range: float) -> list[float]:
    goal = []
    for i in range(14):
        lo, hi = JOINT_LIMITS[i]
        raw = current[i] + random.uniform(-offset_range, offset_range)
        goal.append(clamp(raw, lo, hi))
    return goal


def get_key() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def base_for_mode(mode: str) -> list[float]:
    return ZERO_POS if mode == "zero" else FORWARD_POS


def print_status(mode: str, selected: str | None, goal: list[float],
                 positions: list[float]) -> None:
    sel_str = JOINTS[selected]["name"] if selected else "none"
    # Show goal values for selected joint (left + right).
    if selected and mode != "random":
        idx = JOINTS[selected]["idx"]
        sel_str += f" (goal L={goal[idx]:+.2f} R={goal[7 + idx]:+.2f})"
    print(
        f"\r[mode={mode}] [joint: {sel_str}] "
        f"L:[{', '.join(f'{p:+.2f}' for p in positions[:7])}] "
        f"R:[{', '.join(f'{p:+.2f}' for p in positions[7:])}]"
        "          ",
        end="", flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send arm positions via LCM")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Smooth movement speed in rad/s (default: 0.5)")
    parser.add_argument("--offset", type=float, default=0.5,
                        help="Random offset range in rad (default: 0.5)")
    parser.add_argument("--hz", type=float, default=50.0,
                        help="Publish rate in Hz (default: 50)")
    parser.add_argument("--step", type=float, default=0.1,
                        help="Joint step per +/- keypress in rad (default: 0.1)")
    args = parser.parse_args()

    lc = lcm.LCM()
    dt = 1.0 / args.hz

    mode_idx = 1  # start in "zero" mode
    mode = MODES[mode_idx]
    positions = list(ZERO_POS)
    goal = list(ZERO_POS)
    random_goal = list(ZERO_POS)
    selected: str | None = None

    print(f"Arm LCM controller @ {args.hz} Hz")
    print(f"  m     = cycle mode ({', '.join(MODES)})")
    print(f"  s/r/e = select shoulder_pitch / shoulder_roll / elbow")
    print(f"  +/-   = adjust selected joint by {args.step:.2f} rad")
    print(f"  0     = reset selected joint to mode default")
    print(f"  Ctrl+C = quit\n")

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())

        step = 0
        while True:
            # ── Handle keyboard ──────────────────────────────────
            key = get_key()
            if key == "\x03":  # Ctrl+C
                break
            elif key == "m":
                mode_idx = (mode_idx + 1) % len(MODES)
                mode = MODES[mode_idx]
                selected = None
                if mode == "random":
                    random_goal = sample_goal(positions, args.offset)
                else:
                    goal = list(base_for_mode(mode))
                print(f"\r\n>> Mode: {mode}          \r\n", end="", flush=True)
            elif key in JOINTS and mode != "random":
                selected = key
                jname = JOINTS[key]["name"]
                idx = JOINTS[key]["idx"]
                print(
                    f"\r\n>> Selected: {jname} "
                    f"(L={goal[idx]:+.2f} R={goal[7 + idx]:+.2f})"
                    f"          \r\n",
                    end="", flush=True,
                )
            elif key in ("+", "=") and selected and mode != "random":
                jcfg = JOINTS[selected]
                idx = jcfg["idx"]
                sign_l, sign_r = 1, (-1 if jcfg["mirror"] else 1)
                lo_l, hi_l = JOINT_LIMITS[idx]
                lo_r, hi_r = JOINT_LIMITS[7 + idx]
                goal[idx] = clamp(goal[idx] + sign_l * args.step, lo_l, hi_l)
                goal[7 + idx] = clamp(goal[7 + idx] + sign_r * args.step, lo_r, hi_r)
                print(
                    f"\r\n>> {jcfg['name']} +step "
                    f"-> L={goal[idx]:+.2f} R={goal[7 + idx]:+.2f}"
                    f"          \r\n",
                    end="", flush=True,
                )
            elif key == "-" and selected and mode != "random":
                jcfg = JOINTS[selected]
                idx = jcfg["idx"]
                sign_l, sign_r = 1, (-1 if jcfg["mirror"] else 1)
                lo_l, hi_l = JOINT_LIMITS[idx]
                lo_r, hi_r = JOINT_LIMITS[7 + idx]
                goal[idx] = clamp(goal[idx] - sign_l * args.step, lo_l, hi_l)
                goal[7 + idx] = clamp(goal[7 + idx] - sign_r * args.step, lo_r, hi_r)
                print(
                    f"\r\n>> {jcfg['name']} -step "
                    f"-> L={goal[idx]:+.2f} R={goal[7 + idx]:+.2f}"
                    f"          \r\n",
                    end="", flush=True,
                )
            elif key == "0" and selected and mode != "random":
                idx = JOINTS[selected]["idx"]
                base = base_for_mode(mode)
                goal[idx] = base[idx]
                goal[7 + idx] = base[7 + idx]
                jname = JOINTS[selected]["name"]
                print(
                    f"\r\n>> {jname} reset "
                    f"-> L={goal[idx]:+.2f} R={goal[7 + idx]:+.2f}"
                    f"          \r\n",
                    end="", flush=True,
                )

            # ── Compute target positions ─────────────────────────
            max_step = args.speed * dt

            if mode == "random":
                reached = True
                for i in range(14):
                    diff = random_goal[i] - positions[i]
                    if abs(diff) > 0.01:
                        reached = False
                    positions[i] += clamp(diff, -max_step, max_step)
                if reached:
                    random_goal = sample_goal(positions, args.offset)
            else:
                # Smoothly move toward goal.
                for i in range(14):
                    diff = goal[i] - positions[i]
                    positions[i] += clamp(diff, -max_step, max_step)

            # ── Publish ──────────────────────────────────────────
            lc.publish("arm_action", encode_arm_action(positions))

            step += 1
            if step % int(args.hz) == 0:
                print_status(mode, selected, goal, positions)

            time.sleep(dt)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nStopped.")


if __name__ == "__main__":
    main()
