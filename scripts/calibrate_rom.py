#!/usr/bin/env python3
"""Range-of-motion kalibrace reálného Unitree G1 vůči MJCF joint_range.

BEZPEČNOST — PŘEČTI PŘED SPUŠTĚNÍM:
    Skript přepne VŠECHNY klouby do damping módu (kp=0, nízké kd). Robot se
    nebude držet vlastní silou. Před spuštěním MUSÍŠ robota zavěsit na stojan
    nebo ho bezpečně posadit. Nohy mají vyšší kd (default 5), takže se
    nepropadnou okamžitě, ale sám nestojí.

Postup:
    1. Zavěsit / posadit robota.
    2. `python scripts/calibrate_rom.py --iface <eth_iface>`
    3. Potvrdit 'y'. Robot přejde do damping módu.
    4. Rukama pomalu projet každý kloub od jednoho dorazu k druhému.
    5. Ctrl+C → skript vypíše porovnávací tabulku sim_range vs real_range.

Výstup:
    Per-kloub tabulka s MJCF rozsahem, reálným dosaženým rozsahem, a flagy:
        SHORT     — nedošel jsi do krajní polohy (chybí >0.3 rad)
        OVERSHOOT — reálný kloub šel za MJCF limit (MJCF je špatně nebo
                    encoder offset)
        NO_DATA   — motor nehlásí data

Příklad:
    python scripts/calibrate_rom.py --iface enp4s0
    python scripts/calibrate_rom.py --iface eth0 --kd-arms 0.3 --yes
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)


G1_NUM_MOTOR = 29

# Motor index → MJCF joint name. Pořadí z G1JointIndex
# (unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py:34-69).
MOTOR_NAMES = [
    "left_hip_pitch_joint",        # 0
    "left_hip_roll_joint",         # 1
    "left_hip_yaw_joint",          # 2
    "left_knee_joint",             # 3
    "left_ankle_pitch_joint",      # 4
    "left_ankle_roll_joint",       # 5
    "right_hip_pitch_joint",       # 6
    "right_hip_roll_joint",        # 7
    "right_hip_yaw_joint",         # 8
    "right_knee_joint",            # 9
    "right_ankle_pitch_joint",     # 10
    "right_ankle_roll_joint",      # 11
    "waist_yaw_joint",             # 12
    "waist_roll_joint",            # 13
    "waist_pitch_joint",           # 14
    "left_shoulder_pitch_joint",   # 15
    "left_shoulder_roll_joint",    # 16
    "left_shoulder_yaw_joint",     # 17
    "left_elbow_joint",            # 18
    "left_wrist_roll_joint",       # 19
    "left_wrist_pitch_joint",      # 20
    "left_wrist_yaw_joint",        # 21
    "right_shoulder_pitch_joint",  # 22
    "right_shoulder_roll_joint",   # 23
    "right_shoulder_yaw_joint",    # 24
    "right_elbow_joint",           # 25
    "right_wrist_roll_joint",      # 26
    "right_wrist_pitch_joint",     # 27
    "right_wrist_yaw_joint",       # 28
]

DEFAULT_MJCF = (
    Path(__file__).resolve().parents[1]
    / "src/assets/robots/unitree_g1/xmls/g1.xml"
)


def load_joint_ranges(xml_path: Path) -> dict[str, tuple[float, float]]:
    """Parsuj <joint name=... range="lo hi"/> z MJCF bez mujoco dependency."""
    tree = ET.parse(xml_path)
    ranges: dict[str, tuple[float, float]] = {}
    for j in tree.iter("joint"):
        name = j.attrib.get("name")
        rng = j.attrib.get("range")
        if name and rng:
            parts = rng.split()
            if len(parts) == 2:
                ranges[name] = (float(parts[0]), float(parts[1]))
    return ranges


@dataclass
class Tracker:
    q_min: list[float] = field(
        default_factory=lambda: [float("+inf")] * G1_NUM_MOTOR
    )
    q_max: list[float] = field(
        default_factory=lambda: [float("-inf")] * G1_NUM_MOTOR
    )
    q_last: list[float] = field(default_factory=lambda: [0.0] * G1_NUM_MOTOR)
    samples: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, positions: list[float]) -> None:
        with self.lock:
            for i in range(G1_NUM_MOTOR):
                q = positions[i]
                if q < self.q_min[i]:
                    self.q_min[i] = q
                if q > self.q_max[i]:
                    self.q_max[i] = q
                self.q_last[i] = q
            self.samples += 1


class G1Calibrator:
    def __init__(
        self, iface: str, kd_legs: float, kd_waist: float, kd_arms: float
    ):
        self.iface = iface
        self.kds = [kd_legs] * 12 + [kd_waist] * 3 + [kd_arms] * 14
        self.tracker = Tracker()
        self.low_state: LowState_ | None = None
        self.mode_machine: int = 0
        self.mode_machine_set = False
        self.running = False
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.cmd_thread: threading.Thread | None = None
        self.pub: ChannelPublisher | None = None
        self.sub: ChannelSubscriber | None = None

    def init_dds(self) -> None:
        ChannelFactoryInitialize(0, self.iface)

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        retries = 0
        while result.get("name") and retries < 10:
            print(f"[DDS] uvolňuji aktivní mode='{result['name']}'...")
            msc.ReleaseMode()
            time.sleep(0.5)
            status, result = msc.CheckMode()
            retries += 1
        if result.get("name"):
            raise RuntimeError(f"Nelze uvolnit motion switcher: {result}")
        print(f"[DDS] motion switcher uvolněn (iface={self.iface})")

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_low_state, 10)

        t0 = time.time()
        while not self.mode_machine_set:
            if time.time() - t0 > 5.0:
                raise RuntimeError(
                    "Žádný LowState během 5 s — zkontroluj iface / propojení."
                )
            time.sleep(0.05)
        print(
            f"[DDS] mode_machine={self.mode_machine}, první LowState přijat"
        )

    def _on_low_state(self, msg: LowState_) -> None:
        self.low_state = msg
        if not self.mode_machine_set:
            self.mode_machine = msg.mode_machine
            self.mode_machine_set = True
        positions = [msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]
        self.tracker.update(positions)

    def start_damping(self) -> None:
        self.running = True
        self.cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self.cmd_thread.start()
        print("[DDS] damping mode aktivní (500 Hz) — pohybuj klouby")

    def _cmd_loop(self) -> None:
        period = 0.002  # 500 Hz
        next_t = time.perf_counter() + period
        while self.running:
            self.low_cmd.mode_pr = 0  # PR mode
            self.low_cmd.mode_machine = self.mode_machine
            for i in range(G1_NUM_MOTOR):
                mc = self.low_cmd.motor_cmd[i]
                mc.mode = 1
                mc.kp = 0.0
                mc.kd = self.kds[i]
                mc.q = 0.0
                mc.dq = 0.0
                mc.tau = 0.0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            if self.pub is not None:
                self.pub.Write(self.low_cmd)
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            next_t += period

    def stop(self) -> None:
        self.running = False
        if self.cmd_thread is not None:
            self.cmd_thread.join(timeout=1.0)

    def live_line(self) -> str:
        q = self.tracker.q_last
        return (
            f"samples={self.tracker.samples:>6}  "
            f"LKnee={q[3]:+.2f} RKnee={q[9]:+.2f}  "
            f"Waist={q[12]:+.2f}  "
            f"LShdP={q[15]:+.2f} LElb={q[18]:+.2f}  "
            f"RShdP={q[22]:+.2f} RElb={q[25]:+.2f}"
        )


def print_report(
    tracker: Tracker, ranges: dict[str, tuple[float, float]]
) -> None:
    print()
    print("=" * 104)
    print(
        f"{'idx':>3}  {'joint':<28}  {'sim_range':>18}  "
        f"{'real_range':>18}  {'miss_lo':>7}  {'miss_hi':>7}  {'flag':<12}"
    )
    print("-" * 104)
    any_flag = False
    for i, name in enumerate(MOTOR_NAMES):
        sim = ranges.get(name)
        real_lo = tracker.q_min[i]
        real_hi = tracker.q_max[i]
        has_real = real_lo != float("+inf") and real_hi != float("-inf")

        flags: list[str] = []
        miss_lo = miss_hi = 0.0
        if not has_real:
            flags.append("NO_DATA")
        elif sim is not None:
            sim_lo, sim_hi = sim
            # Kolik z MJCF rozsahu jsi nepokryl (na reálu jsi nedošel až k limitu).
            miss_lo = max(0.0, real_lo - sim_lo)
            miss_hi = max(0.0, sim_hi - real_hi)
            # Přešel jsi za MJCF limit?
            over_lo = max(0.0, sim_lo - real_lo)
            over_hi = max(0.0, real_hi - sim_hi)
            if over_lo > 0.05 or over_hi > 0.05:
                flags.append("OVERSHOOT")
            if miss_lo > 0.3 or miss_hi > 0.3:
                flags.append("SHORT")

        flag = ",".join(flags) if flags else "ok"
        if flags:
            any_flag = True

        sim_str = f"[{sim[0]:+.2f},{sim[1]:+.2f}]" if sim else "—"
        real_str = f"[{real_lo:+.2f},{real_hi:+.2f}]" if has_real else "—"
        print(
            f"{i:>3}  {name:<28}  {sim_str:>18}  {real_str:>18}  "
            f"{miss_lo:>7.2f}  {miss_hi:>7.2f}  {flag:<12}"
        )

    print("-" * 104)
    print(
        "miss_lo / miss_hi [rad] — kolik ses nedostal k dolnímu / hornímu dorazu (vůči MJCF)"
    )
    print(
        "OVERSHOOT — reálný kloub šel za MJCF limit (>0.05 rad) → MJCF joint_range nesedí hw"
    )
    print(
        "SHORT     — neprojel jsi až k dorazu (>0.3 rad chybí) → opakuj ten kloub"
    )
    if not any_flag:
        print("\n✓ Všechny kloubové rozsahy sedí se simulací.")
    print(f"\nCelkem vzorků: {tracker.samples}")
    print()


def confirm(msg: str) -> bool:
    try:
        return input(f"{msg} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--iface", required=True, help="síťové rozhraní k robotovi (např. enp4s0)"
    )
    ap.add_argument(
        "--mjcf",
        default=str(DEFAULT_MJCF),
        help=f"cesta k g1.xml (default: {DEFAULT_MJCF})",
    )
    ap.add_argument("--kd-legs", type=float, default=5.0, help="damping nohy (def 5)")
    ap.add_argument("--kd-waist", type=float, default=3.0, help="damping pás (def 3)")
    ap.add_argument("--kd-arms", type=float, default=0.5, help="damping paže (def 0.5)")
    ap.add_argument("--yes", action="store_true", help="přeskočit potvrzení")
    args = ap.parse_args()

    mjcf_path = Path(args.mjcf)
    if not mjcf_path.is_file():
        print(f"ERROR: MJCF nenalezen: {mjcf_path}")
        return 1
    ranges = load_joint_ranges(mjcf_path)
    missing = [n for n in MOTOR_NAMES if n not in ranges]
    if missing:
        print(f"WARN: tyto klouby chybí v MJCF (reference bude prázdná): {missing}")

    print("=" * 70)
    print("BEZPEČNOSTNÍ UPOZORNĚNÍ")
    print("=" * 70)
    print("Skript přepne VŠECHNY klouby do damping módu (kp=0, nízké kd).")
    print("→ Robot se nebude držet vlastní silou.")
    print("→ Musí být ZAVĚŠENÝ na stojanu nebo BEZPEČNĚ POSAZENÝ.")
    print(
        f"Gainy: kd_legs={args.kd_legs}, kd_waist={args.kd_waist}, "
        f"kd_arms={args.kd_arms}"
    )
    print()
    if not args.yes and not confirm("Je robot bezpečně zavěšený / posazený?"):
        print("Zrušeno.")
        return 0

    cal = G1Calibrator(args.iface, args.kd_legs, args.kd_waist, args.kd_arms)

    stop_evt = threading.Event()

    def on_sigint(_sig, _frame):
        stop_evt.set()

    signal.signal(signal.SIGINT, on_sigint)

    try:
        cal.init_dds()
        cal.start_damping()
        print()
        print("Pomalu projeď každý kloub rukama od dorazu k dorazu.")
        print("Ctrl+C → ukončit a vypsat tabulku.")
        print()
        while not stop_evt.is_set():
            time.sleep(0.3)
            sys.stdout.write("\r" + cal.live_line() + "    ")
            sys.stdout.flush()
    finally:
        cal.stop()
        print()
        print_report(cal.tracker, ranges)
    return 0


if __name__ == "__main__":
    sys.exit(main())
