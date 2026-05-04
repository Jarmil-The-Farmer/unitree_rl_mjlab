#!/usr/bin/env python3
"""Send Inspire hand finger-angle commands to the deploy controller via LCM.

Deploy receives channel "inspire_hand_action" and republishes to DDS topics:
  rt/inspire_hand/ctrl/l
  rt/inspire_hand/ctrl/r

LCM payload: inspire_hand_action_lcmt
Finger order per hand: pinky, ring, middle, index, thumb_bend, thumb_rot.
Hand order in arrays: left[0:6], right[6:12].
Values use Inspire's 0..1000 angle-register convention, where 1000 is open.
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


BASE_HASH = 0x8432D1E1A710B9BD
FINGERPRINT = ((BASE_HASH << 1) + (BASE_HASH >> 63)) & 0xFFFFFFFFFFFFFFFF
PACKED_FINGERPRINT = struct.pack(">Q", FINGERPRINT)

CHANNEL = "inspire_hand_action"
NUM_HAND_FINGERS = 6
NUM_HAND_VALUES = 12

HAND_MASK = {"left": 1, "right": 2, "both": 3}
HAND_LABEL = {"left": "LEFT", "right": "RIGHT", "both": "BOTH"}

FINGER_NAMES = [
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rot",
]

ANGLE_MIN = 0
ANGLE_MAX = 1000


def encode_inspire_hand_action(*, hand_mask: int, finger_angle: list[int]) -> bytes:
    assert len(finger_angle) == NUM_HAND_VALUES
    timestamp_us = time.time_ns() // 1000
    return PACKED_FINGERPRINT + struct.pack(
        ">qb12h",
        timestamp_us,
        hand_mask,
        *finger_angle,
    )


def clamp(value: int | float, lo: int = ANGLE_MIN, hi: int = ANGLE_MAX) -> int:
    return max(lo, min(hi, int(value)))


def get_key() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def hand_offsets(active_hand: str) -> list[int]:
    if active_hand == "left":
        return [0]
    if active_hand == "right":
        return [NUM_HAND_FINGERS]
    return [0, NUM_HAND_FINGERS]


def finger_indices(active_finger: int | None) -> list[int]:
    if active_finger is None:
        return list(range(NUM_HAND_FINGERS))
    return [active_finger]


def selected_abs_indices(active_hand: str, active_finger: int | None) -> list[int]:
    return [
        offset + idx
        for offset in hand_offsets(active_hand)
        for idx in finger_indices(active_finger)
    ]


def adjust(finger_angle: list[int], active_hand: str, active_finger: int | None, delta: int) -> None:
    for idx in selected_abs_indices(active_hand, active_finger):
        finger_angle[idx] = clamp(finger_angle[idx] + delta)


def set_selected(
    finger_angle: list[int],
    active_hand: str,
    active_finger: int | None,
    value: int,
) -> None:
    for idx in selected_abs_indices(active_hand, active_finger):
        finger_angle[idx] = clamp(value)


def render(
    *,
    active_hand: str,
    active_finger: int | None,
    step: int,
    finger_angle: list[int],
    random_mode: bool,
    last_msg: str,
) -> None:
    finger_label = FINGER_NAMES[active_finger] if active_finger is not None else "ALL"

    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write("=== Inspire hand LCM control ===\r\n")
    sys.stdout.write(
        f"Hand: {HAND_LABEL[active_hand]:5s}  Finger: {finger_label:10s}  "
        f"Step: {step:3d}  Random: {'ON' if random_mode else 'OFF'}\r\n\r\n"
    )
    sys.stdout.write("              " + " ".join(f"{n:>10s}" for n in FINGER_NAMES) + "\r\n")
    sys.stdout.write("finger_angle L " + " ".join(f"{v:>10d}" for v in finger_angle[:6]) + "\r\n")
    sys.stdout.write("finger_angle R " + " ".join(f"{v:>10d}" for v in finger_angle[6:]) + "\r\n\r\n")
    sys.stdout.write(
        "l/r/b hand  1..6 finger  a all  x random\r\n"
        "+/- adjust  [/] step  o/c selected open/close  O/C whole active hand  space resend\r\n"
        "h/? refresh help  q or Ctrl+C quit\r\n"
    )
    if last_msg:
        sys.stdout.write(f"\r\n> {last_msg}\r\n")
    if active_finger is not None:
        sys.stdout.write(
            f"\r\nSelected finger_angle: "
            f"L={finger_angle[active_finger]} "
            f"R={finger_angle[NUM_HAND_FINGERS + active_finger]}\r\n"
        )
    sys.stdout.flush()


def publish(lc: lcm.LCM, active_hand: str, finger_angle: list[int]) -> None:
    payload = encode_inspire_hand_action(
        hand_mask=HAND_MASK[active_hand],
        finger_angle=finger_angle,
    )
    lc.publish(CHANNEL, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send Inspire hand finger angles via LCM")
    parser.add_argument("--hz", type=float, default=50.0, help="publish rate in Hz")
    parser.add_argument("--step", type=int, default=50, help="initial step size")
    parser.add_argument(
        "--hand",
        choices=["left", "right", "both"],
        default="both",
        help="initial active hand",
    )
    parser.add_argument(
        "--random-speed",
        type=float,
        default=500.0,
        help="random angle smoothing speed in Inspire units/s",
    )
    args = parser.parse_args()

    lc = lcm.LCM()
    dt = 1.0 / args.hz

    active_hand = args.hand
    active_finger: int | None = None
    step = clamp(args.step, 1, 500)

    finger_angle = [ANGLE_MAX] * NUM_HAND_VALUES
    random_mode = False
    random_goal = list(finger_angle)
    last_msg = "ready"

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        render(
            active_hand=active_hand,
            active_finger=active_finger,
            step=step,
            finger_angle=finger_angle,
            random_mode=random_mode,
            last_msg=last_msg,
        )

        while True:
            key = get_key()

            if key is not None:
                if key in ("q", "\x03"):
                    break
                if key == "l":
                    active_hand = "left"; last_msg = "selected LEFT hand"
                elif key == "r":
                    active_hand = "right"; last_msg = "selected RIGHT hand"
                elif key == "b":
                    active_hand = "both"; last_msg = "selected BOTH hands"
                elif key in "123456":
                    active_finger = int(key) - 1
                    last_msg = f"selected {FINGER_NAMES[active_finger]}"
                elif key == "a":
                    active_finger = None; last_msg = "selected ALL fingers"
                elif key == "x":
                    random_mode = not random_mode
                    if random_mode:
                        random_goal = [random.randint(ANGLE_MIN, ANGLE_MAX) for _ in range(NUM_HAND_VALUES)]
                    last_msg = f"random = {'ON' if random_mode else 'OFF'}"
                elif key in ("+", "="):
                    adjust(finger_angle, active_hand, active_finger, +step)
                    last_msg = f"finger_angle +{step}"
                elif key in ("-", "_"):
                    adjust(finger_angle, active_hand, active_finger, -step)
                    last_msg = f"finger_angle -{step}"
                elif key == "]":
                    step = min(500, step + 10); last_msg = f"step = {step}"
                elif key == "[":
                    step = max(1, step - 10); last_msg = f"step = {step}"
                elif key == "o":
                    set_selected(finger_angle, active_hand, active_finger, ANGLE_MAX)
                    last_msg = "open selected"
                elif key == "c":
                    set_selected(finger_angle, active_hand, active_finger, ANGLE_MIN)
                    last_msg = "close selected"
                elif key == "O":
                    set_selected(finger_angle, active_hand, None, ANGLE_MAX)
                    last_msg = "open whole active hand"
                elif key == "C":
                    set_selected(finger_angle, active_hand, None, ANGLE_MIN)
                    last_msg = "close whole active hand"
                elif key in ("h", "?"):
                    last_msg = "help shown"
                elif key == " ":
                    last_msg = "resent"
                else:
                    last_msg = f"unknown key: {key!r}"

            if random_mode:
                max_delta = max(1.0, args.random_speed * dt)
                reached = True
                for idx in selected_abs_indices(active_hand, None):
                    diff = random_goal[idx] - finger_angle[idx]
                    if abs(diff) > 5:
                        reached = False
                    finger_angle[idx] = clamp(
                        finger_angle[idx] + max(-max_delta, min(max_delta, diff))
                    )
                if reached:
                    for idx in selected_abs_indices(active_hand, None):
                        random_goal[idx] = random.randint(ANGLE_MIN, ANGLE_MAX)

            publish(lc, active_hand, finger_angle)

            if key is not None:
                render(
                    active_hand=active_hand,
                    active_finger=active_finger,
                    step=step,
                    finger_angle=finger_angle,
                    random_mode=random_mode,
                    last_msg=last_msg,
                )

            time.sleep(dt)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nstopped")


if __name__ == "__main__":
    main()
