#!/usr/bin/env python3
"""Range-of-motion kalibrace reálného Unitree G1 vůči MJCF joint_range.

BEZPEČNOST — PŘEČTI PŘED SPUŠTĚNÍM:
    Skript přepne VŠECHNY klouby do damping módu (kp=0, kd=1 na všech
    motorech). Robot se nebude držet vlastní silou. Před spuštěním MUSÍŠ
    robota zavěsit na stojan nebo ho bezpečně posadit.

Průchod po kloubech (interactive):
    Pro každý kloub skript vypíše očekávaný MJCF rozsah a začne live ukazovat
    aktuální pozici + dosažený min/max. Pohybuj kloubem tam a zpět, potom:
        Enter   → další kloub
        s+Enter → přeskočit aktuální kloub (nedostupný / neoznačit)
        r+Enter → smazat naměřená data aktuálního kloubu a měřit znovu
        q+Enter → ukončit a vypsat souhrnnou tabulku

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
class CalibState:
    """Sdílený stav mezi DDS callbackem, display threadem a main loopem."""
    current_joint: int = 0
    q_min: list[float] = field(
        default_factory=lambda: [float("+inf")] * G1_NUM_MOTOR
    )
    q_max: list[float] = field(
        default_factory=lambda: [float("-inf")] * G1_NUM_MOTOR
    )
    q_live: list[float] = field(default_factory=lambda: [0.0] * G1_NUM_MOTOR)
    skipped: list[bool] = field(default_factory=lambda: [False] * G1_NUM_MOTOR)
    samples: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, positions: list[float]) -> None:
        with self.lock:
            for i in range(G1_NUM_MOTOR):
                q = positions[i]
                self.q_live[i] = q
                if q < self.q_min[i]:
                    self.q_min[i] = q
                if q > self.q_max[i]:
                    self.q_max[i] = q
            self.samples += 1

    def reset_joint(self, j: int) -> None:
        with self.lock:
            self.q_min[j] = float("+inf")
            self.q_max[j] = float("-inf")
            self.skipped[j] = False

    def mark_skipped(self, j: int) -> None:
        with self.lock:
            self.skipped[j] = True


class G1Calibrator:
    def __init__(
        self,
        iface: str,
        kd_legs: float,
        kd_waist: float,
        kd_arms: float,
        state: CalibState,
    ):
        self.iface = iface
        self.kds = [kd_legs] * 12 + [kd_waist] * 3 + [kd_arms] * 14
        self.state = state
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
        self.state.update(positions)

    def start_damping(self) -> None:
        self.running = True
        self.cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self.cmd_thread.start()
        print("[DDS] damping mode aktivní (500 Hz)")

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


def _fmt_rng(lo: float, hi: float) -> str:
    lo_s = f"{lo:+.2f}" if lo != float("+inf") else "  —  "
    hi_s = f"{hi:+.2f}" if hi != float("-inf") else "  —  "
    return f"[{lo_s}, {hi_s}]"


def _display_loop(
    state: CalibState,
    ranges: dict[str, tuple[float, float]],
    stop: threading.Event,
) -> None:
    """Background thread: 10 Hz rewrite jediné status řádky pro aktuální kloub."""
    while not stop.is_set():
        with state.lock:
            j = state.current_joint
            q_now = state.q_live[j]
            q_min = state.q_min[j]
            q_max = state.q_max[j]
        name = MOTOR_NAMES[j]
        sim = ranges.get(name)

        # Vyhodnocení coverage vůči MJCF limitu.
        lo_mark = hi_mark = "  "
        miss_str = ""
        if sim is not None and q_min != float("+inf"):
            sim_lo, sim_hi = sim
            reached_lo = q_min - sim_lo   # < 0 = overshoot, > 0 = miss
            reached_hi = sim_hi - q_max   # < 0 = overshoot, > 0 = miss
            lo_mark = "✓ " if reached_lo <= 0.1 else "✗ "
            hi_mark = "✓ " if reached_hi <= 0.1 else "✗ "
            parts = []
            if reached_lo > 0.1:
                parts.append(f"miss_lo={reached_lo:+.2f}")
            if reached_hi > 0.1:
                parts.append(f"miss_hi={reached_hi:+.2f}")
            if reached_lo < -0.05:
                parts.append(f"over_lo={-reached_lo:.2f}")
            if reached_hi < -0.05:
                parts.append(f"over_hi={-reached_hi:.2f}")
            miss_str = "  " + " ".join(parts) if parts else "  ✓ROZSAH POKRYT"

        line = (
            f"\r   q={q_now:+.3f}  "
            f"real={_fmt_rng(q_min, q_max)}  "
            f"{lo_mark}lo {hi_mark}hi{miss_str}          "
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(0.1)


def print_report(
    state: CalibState, ranges: dict[str, tuple[float, float]]
) -> None:
    print()
    print("=" * 108)
    print(
        f"{'idx':>3}  {'joint':<28}  {'sim_range':>18}  "
        f"{'real_range':>18}  {'miss_lo':>7}  {'miss_hi':>7}  {'flag':<14}"
    )
    print("-" * 108)
    any_flag = False
    for i, name in enumerate(MOTOR_NAMES):
        sim = ranges.get(name)
        real_lo = state.q_min[i]
        real_hi = state.q_max[i]
        has_real = real_lo != float("+inf") and real_hi != float("-inf")

        flags: list[str] = []
        miss_lo = miss_hi = 0.0
        if state.skipped[i]:
            flags.append("SKIPPED")
        elif not has_real:
            flags.append("NO_DATA")
        elif sim is not None:
            sim_lo, sim_hi = sim
            miss_lo = max(0.0, real_lo - sim_lo)
            miss_hi = max(0.0, sim_hi - real_hi)
            over_lo = max(0.0, sim_lo - real_lo)
            over_hi = max(0.0, real_hi - sim_hi)
            if over_lo > 0.05 or over_hi > 0.05:
                flags.append("OVERSHOOT")
            if miss_lo > 0.3 or miss_hi > 0.3:
                flags.append("SHORT")

        flag = ",".join(flags) if flags else "ok"
        if flags and flags != ["SKIPPED"]:
            any_flag = True

        sim_str = f"[{sim[0]:+.2f},{sim[1]:+.2f}]" if sim else "—"
        real_str = f"[{real_lo:+.2f},{real_hi:+.2f}]" if has_real else "—"
        print(
            f"{i:>3}  {name:<28}  {sim_str:>18}  {real_str:>18}  "
            f"{miss_lo:>7.2f}  {miss_hi:>7.2f}  {flag:<14}"
        )

    print("-" * 108)
    print(
        "miss_lo / miss_hi [rad] — kolik ses nedostal k dolnímu / hornímu dorazu (vůči MJCF)"
    )
    print(
        "OVERSHOOT — reálný kloub šel za MJCF limit (>0.05 rad) → MJCF joint_range nesedí hw"
    )
    print(
        "SHORT     — nedošel jsi až k dorazu (>0.3 rad chybí)"
    )
    print(
        "SKIPPED   — vynechal jsi tento kloub klávesou 's'"
    )
    if not any_flag:
        print("\n✓ Všechny označené klouby sedí se simulací.")
    print(f"\nCelkem vzorků: {state.samples}")
    print()


def confirm(msg: str) -> bool:
    try:
        return input(f"{msg} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _joint_header(idx: int, total: int, name: str,
                  sim: tuple[float, float] | None) -> None:
    sim_str = f"[{sim[0]:+.3f}, {sim[1]:+.3f}]" if sim else "—"
    print()
    print(f"━━ [{idx + 1:>2}/{total}] {name}")
    print(f"   sim_range: {sim_str}")
    print(f"   Pohybuj kloubem tam a zpět až k oběma dorazům.")
    print(f"   Enter=další  |  r+Enter=reset  |  s+Enter=skip  |  q+Enter=konec")


def interactive_walk(
    state: CalibState, ranges: dict[str, tuple[float, float]]
) -> None:
    """Sekvenční průchod kloubů s live displayem a Enter/r/s/q ovládáním."""
    total = len(MOTOR_NAMES)
    i = 0
    while i < total:
        with state.lock:
            state.current_joint = i
        # Reset je explicitní volba — čerstvý průchod začíná s prázdným
        # min/max pro tento kloub, aby dřívější náhodné dotyky neovlivnily
        # měření.
        state.reset_joint(i)

        name = MOTOR_NAMES[i]
        sim = ranges.get(name)
        _joint_header(i, total, name, sim)

        display_stop = threading.Event()
        display_thread = threading.Thread(
            target=_display_loop, args=(state, ranges, display_stop), daemon=True
        )
        display_thread.start()

        try:
            cmd = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = "q"
        finally:
            display_stop.set()
            display_thread.join(timeout=0.5)
            # Přidat newline pod smazanou live řádku.
            sys.stdout.write("\n")
            sys.stdout.flush()

        if cmd == "" or cmd == "n":
            i += 1
        elif cmd == "r":
            # Smažu data a opakuju stejný kloub.
            continue
        elif cmd == "s":
            state.mark_skipped(i)
            i += 1
        elif cmd == "q":
            return
        elif cmd == "b" and i > 0:
            i -= 1
        else:
            # Neznámý vstup → nic nedělej, opakuj stejný kloub.
            continue


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
    ap.add_argument("--kd-legs", type=float, default=1.0, help="damping nohy (def 1)")
    ap.add_argument("--kd-waist", type=float, default=1.0, help="damping pás (def 1)")
    ap.add_argument("--kd-arms", type=float, default=1.0, help="damping paže (def 1)")
    ap.add_argument("--yes", action="store_true", help="přeskočit bezpečnostní potvrzení")
    args = ap.parse_args()

    mjcf_path = Path(args.mjcf)
    if not mjcf_path.is_file():
        print(f"ERROR: MJCF nenalezen: {mjcf_path}")
        return 1
    ranges = load_joint_ranges(mjcf_path)
    missing = [n for n in MOTOR_NAMES if n not in ranges]
    if missing:
        print(f"WARN: tyto klouby chybí v MJCF (bez reference): {missing}")

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

    state = CalibState()
    cal = G1Calibrator(
        args.iface, args.kd_legs, args.kd_waist, args.kd_arms, state
    )

    # Ctrl+C → ukončí input() přes KeyboardInterrupt, který interactive_walk
    # ošetří jako 'q'. Další Ctrl+C v shutdown sekvenci se ignoruje.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        cal.init_dds()
        cal.start_damping()
        print()
        print("Začínáme průchod. Live status se zobrazuje na jedné řádce;")
        print("tvé klávesy se tam krátce objeví než je přepíše další update.")
        interactive_walk(state, ranges)
    finally:
        cal.stop()
        print_report(state, ranges)
    return 0


if __name__ == "__main__":
    sys.exit(main())
