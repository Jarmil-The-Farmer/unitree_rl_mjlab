#!/usr/bin/env python3
"""Step response test — per kloub měří dead-time a časovou konstantu.

Posílá malý skok v position targetu pro jeden kloub, zaznamenává q(t), dq(t),
tau_est(t). Fituje prahový odhad dead-time (td) a 63 % časové konstanty (tau).

BEZPEČNOST — přečti před spuštěním:
    - Robot MUSÍ být ZAVĚŠENÝ v bezpečném stojanu (ne na zemi!).
    - Test používá normální PD gainy — kloub se pohne svižně (~0.5 s
      dojede do cíle, ne pomalu).
    - Amplituda je default 0.1 rad (~5.7°). Pro klouby s malým rozsahem
      (kotník ±0.26, pás ±0.5) to zmenši `--amplitude 0.05`.
    - Kolem testovaného kloubu musí být VOLNO (nic do čeho kloub narazí).
    - Ctrl+C kdykoli → okamžitě ramp down na damping mode.

Postup:
    1. Zavěsit robota.
    2. `python scripts/step_response.py --iface <eth>`
    3. Interaktivně: vyber index kloubu (0-28), potvrď, skript:
         a) fade-in PD gainy na všech (robot drží aktuální polohu)
         b) do target kloubu pošle skok +amplitude
         c) nahrává data 2 s (hold)
         d) slow return do q0
         e) fade-out PD zpět na damping
    4. Vypíše `td` a `τ` + uloží CSV (data/time series).
    5. Loop další kloub nebo 'q' = konec.

Příklad:
    python scripts/step_response.py --iface enp4s0
    python scripts/step_response.py --iface eth0 --amplitude 0.05 --hold-time 3
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
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

# Sim-matched PD gains (z deploy/robots/g1/config/policy/velocity/
# balance_height_v6/params/deploy.yaml). Musí odpovídat simu, jinak testované
# τ neodpovídá tomu co policy "zažívá".
KP_DEFAULT = [
    40.2, 99.1, 40.2, 99.1, 28.5, 28.5,       # 0-5   left leg
    40.2, 99.1, 40.2, 99.1, 28.5, 28.5,       # 6-11  right leg
    40.2, 28.5, 28.5,                         # 12-14 waist
    40, 40, 40, 40, 40, 40, 40,               # 15-21 left arm
    40, 40, 40, 40, 40, 40, 40,               # 22-28 right arm
]
KD_DEFAULT = [
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8,             # 0-5   left leg
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8,             # 6-11  right leg
    2.6, 1.8, 1.8,                            # 12-14 waist
    5, 5, 5, 5, 5, 5, 5,                      # 15-21 left arm
    5, 5, 5, 5, 5, 5, 5,                      # 22-28 right arm
]

# Damping mode gains (test start/end — robot lehce drží, dá se hýbat).
KD_DAMPING = 1.0

CONTROL_HZ = 500  # cmd publish rate
CONTROL_DT = 1.0 / CONTROL_HZ


@dataclass
class Record:
    """Single step response recording."""
    joint_idx: int
    joint_name: str
    q0: float
    amplitude: float
    t: list[float] = field(default_factory=list)
    q: list[float] = field(default_factory=list)
    dq: list[float] = field(default_factory=list)
    tau_est: list[float] = field(default_factory=list)
    q_target: list[float] = field(default_factory=list)


class StepTester:
    def __init__(self, iface: str):
        self.iface = iface
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.pub: ChannelPublisher | None = None
        self.sub: ChannelSubscriber | None = None
        self.crc = CRC()
        self.mode_machine: int = 0
        self.mode_machine_set = False

        self.state_lock = threading.Lock()
        self.q_real = [0.0] * G1_NUM_MOTOR
        self.dq_real = [0.0] * G1_NUM_MOTOR
        self.tau_est = [0.0] * G1_NUM_MOTOR

        # Per-motor command state — atomic float assignments OK under GIL.
        self.kp_scale = [0.0] * G1_NUM_MOTOR  # 0 = damping, 1 = full PD
        self.q_target = [0.0] * G1_NUM_MOTOR

        self.running = False
        self.cmd_thread: threading.Thread | None = None

        # Recording state.
        self.record_lock = threading.Lock()
        self.recording = False
        self.record: Record | None = None
        self.record_start_time = 0.0

    def init_dds(self) -> None:
        ChannelFactoryInitialize(0, self.iface)

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        retries = 0
        while result.get("name") and retries < 10:
            print(f"[DDS] uvolňuji mode='{result['name']}'...")
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
        self.sub.Init(self._on_state, 10)

        t0 = time.time()
        while not self.mode_machine_set:
            if time.time() - t0 > 5.0:
                raise RuntimeError("Žádný LowState — zkontroluj iface.")
            time.sleep(0.05)

        # Init targets = current q (damping at start, no kick).
        with self.state_lock:
            for i in range(G1_NUM_MOTOR):
                self.q_target[i] = self.q_real[i]
        print("[DDS] inicializováno, damping mode aktivní")

    def _on_state(self, msg: LowState_) -> None:
        if not self.mode_machine_set:
            self.mode_machine = msg.mode_machine
            self.mode_machine_set = True
        with self.state_lock:
            for i in range(G1_NUM_MOTOR):
                self.q_real[i] = msg.motor_state[i].q
                self.dq_real[i] = msg.motor_state[i].dq
                self.tau_est[i] = msg.motor_state[i].tau_est

        # Recording hook.
        if self.recording:
            with self.record_lock:
                if self.recording and self.record is not None:
                    j = self.record.joint_idx
                    t = time.perf_counter() - self.record_start_time
                    self.record.t.append(t)
                    self.record.q.append(self.q_real[j])
                    self.record.dq.append(self.dq_real[j])
                    self.record.tau_est.append(self.tau_est[j])
                    self.record.q_target.append(self.q_target[j])

    def start_cmd_thread(self) -> None:
        self.running = True
        self.cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self.cmd_thread.start()

    def _cmd_loop(self) -> None:
        # Static fields: set once, never rewrite.
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(G1_NUM_MOTOR):
            mc = self.low_cmd.motor_cmd[i]
            mc.mode = 1
            mc.dq = 0.0
            mc.tau = 0.0

        # Cache refs once; avoid repeated attribute lookups in hot loop.
        motor_cmds = [self.low_cmd.motor_cmd[i] for i in range(G1_NUM_MOTOR)]
        kp_def = KP_DEFAULT
        kd_def = KD_DEFAULT
        pub = self.pub
        crc = self.crc
        low_cmd = self.low_cmd
        kp_scale = self.kp_scale
        q_target = self.q_target

        # Rate diagnostic.
        self._cmd_iters = 0
        self._cmd_last_report = time.perf_counter()
        self._cmd_rate_hz = 0.0

        next_t = time.perf_counter() + CONTROL_DT
        while self.running:
            for i in range(G1_NUM_MOTOR):
                mc = motor_cmds[i]
                s = kp_scale[i]
                mc.kp = s * kp_def[i]
                mc.kd = s * kd_def[i] + (1.0 - s) * KD_DAMPING
                mc.q = q_target[i]
            low_cmd.crc = crc.Crc(low_cmd)
            if pub is not None:
                pub.Write(low_cmd)

            self._cmd_iters += 1
            now = time.perf_counter()
            if now - self._cmd_last_report >= 1.0:
                self._cmd_rate_hz = self._cmd_iters / (now - self._cmd_last_report)
                self._cmd_iters = 0
                self._cmd_last_report = now

            rem = next_t - now
            if rem > 0:
                time.sleep(rem)
            next_t += CONTROL_DT

    def hold_current_positions(self) -> None:
        """Capture current q for all joints and set as target."""
        with self.state_lock:
            for i in range(G1_NUM_MOTOR):
                self.q_target[i] = self.q_real[i]

    def ramp_all_pd(self, from_s: float, to_s: float, duration: float) -> None:
        """Linearly interpolate kp_scale from `from_s` to `to_s` on all joints."""
        if duration <= 0:
            for i in range(G1_NUM_MOTOR):
                self.kp_scale[i] = to_s
            return
        n = max(1, int(duration * CONTROL_HZ))
        for k in range(n + 1):
            frac = k / n
            s = from_s + (to_s - from_s) * frac
            for i in range(G1_NUM_MOTOR):
                self.kp_scale[i] = s
            time.sleep(duration / n)

    def step_test(
        self, joint_idx: int, amplitude: float, hold_time: float
    ) -> Record:
        """Pošle skok v targetu a nahraje odpověď."""
        name = MOTOR_NAMES[joint_idx]

        with self.state_lock:
            q0 = self.q_real[joint_idx]
            # Ensure targets reflect the current held pose.
            for i in range(G1_NUM_MOTOR):
                self.q_target[i] = self.q_real[i]

        rec = Record(joint_idx=joint_idx, joint_name=name, q0=q0, amplitude=amplitude)

        with self.record_lock:
            self.record = rec
        self.record_start_time = time.perf_counter()
        self.recording = True

        # Step.
        self.q_target[joint_idx] = q0 + amplitude

        # Hold.
        time.sleep(hold_time)

        self.recording = False
        return rec

    def slow_return(
        self, joint_idx: int, from_q: float, to_q: float, duration: float
    ) -> None:
        """Smoothly move target back from from_q to to_q."""
        if duration <= 0:
            self.q_target[joint_idx] = to_q
            return
        n = max(1, int(duration * CONTROL_HZ))
        for k in range(n + 1):
            frac = k / n
            self.q_target[joint_idx] = from_q + (to_q - from_q) * frac
            time.sleep(duration / n)

    def emergency_damping(self) -> None:
        """Instantly drop to damping mode (no ramp — use only if needed)."""
        for i in range(G1_NUM_MOTOR):
            self.kp_scale[i] = 0.0

    def shutdown(self) -> None:
        self.running = False
        if self.cmd_thread is not None:
            self.cmd_thread.join(timeout=1.0)


def _crossing_time(
    rec: Record, threshold: float, increasing: bool
) -> float | None:
    """První čas, kdy q překročí threshold (rostoucí nebo klesající)."""
    for i, q in enumerate(rec.q):
        if increasing and q >= threshold:
            return rec.t[i]
        if not increasing and q <= threshold:
            return rec.t[i]
    return None


def analyze_response(rec: Record) -> dict:
    """Bohatší analýza step response.

    Vrací dict s klíči:
      q_final  — průměr posledních 10 % vzorků
      undershoot — q_final - (q0 + amplitude) (v rad; 0 = trefa)
      t10,t50,t63,t90 — absolute time od record startu do překročení n%
      td        — dead-time odhad (= t10 - 0.105·τ pro 1. řád)
      tau       — τ odhad z (t63 - t10) / 0.895 (konzistence 1. řádu)
      tau_alt   — alternativní τ z (t90 - t50) / 1.61 (pro cross-check)
      first_order_quality — |tau - tau_alt| / tau (0 = perfektní 1. řád)
    """
    out: dict = {
        "q_final": rec.q0,
        "undershoot": 0.0,
        "t10": None, "t50": None, "t63": None, "t90": None,
        "td": None, "tau": None, "tau_alt": None,
        "first_order_quality": None,
    }
    if not rec.t:
        return out

    n = len(rec.q)
    tail_n = max(1, n // 10)
    q_final = sum(rec.q[-tail_n:]) / tail_n
    out["q_final"] = q_final
    out["undershoot"] = q_final - (rec.q0 + rec.amplitude)

    total_response = q_final - rec.q0
    if abs(total_response) < 5e-4:
        return out

    inc = total_response > 0
    for frac, key in [(0.10, "t10"), (0.50, "t50"),
                      (0.632, "t63"), (0.90, "t90")]:
        thr = rec.q0 + frac * total_response
        out[key] = _crossing_time(rec, thr, inc)

    t10, t63 = out["t10"], out["t63"]
    t50, t90 = out["t50"], out["t90"]

    # 1. řád s dead-time: q(t) = q_inf - (q_inf-q0)·exp(-(t-td)/τ)
    # Frakce f v čase t - td:  1 - exp(-(t-td)/τ) = f → t-td = -ln(1-f)·τ
    # -ln(0.9) = 0.1054;  -ln(0.368) = 1.000;  -ln(0.1) = 2.303;  -ln(0.5) = 0.693
    # t63 - t10 = (1.000 - 0.105)·τ = 0.895·τ  → τ = (t63-t10)/0.895
    # td = t10 - 0.105·τ
    if t10 is not None and t63 is not None:
        tau = (t63 - t10) / 0.895
        if tau > 0:
            out["tau"] = tau
            out["td"] = t10 - 0.105 * tau
    if t50 is not None and t90 is not None:
        tau_alt = (t90 - t50) / 1.610
        if tau_alt > 0:
            out["tau_alt"] = tau_alt

    if out["tau"] is not None and out["tau_alt"] is not None:
        out["first_order_quality"] = (
            abs(out["tau"] - out["tau_alt"]) / max(out["tau"], out["tau_alt"])
        )
    return out


def save_csv(path: Path, rec: Record) -> None:
    with open(path, "w") as f:
        f.write("t,q,dq,tau_est,q_target\n")
        for i in range(len(rec.t)):
            f.write(
                f"{rec.t[i]:.6f},{rec.q[i]:.6f},{rec.dq[i]:.6f},"
                f"{rec.tau_est[i]:.6f},{rec.q_target[i]:.6f}\n"
            )


def list_joints() -> None:
    print()
    print("Indexy kloubů:")
    for i, name in enumerate(MOTOR_NAMES):
        print(f"  {i:>2}  {name}")


def confirm(msg: str) -> bool:
    try:
        return input(f"{msg} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--iface", required=True, help="síťové rozhraní k robotovi")
    ap.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="amplituda skoku [rad] (def 0.1 ≈ 5.7°)",
    )
    ap.add_argument(
        "--hold-time",
        type=float,
        default=2.0,
        help="doba nahrávání po skoku [s] (def 2)",
    )
    ap.add_argument(
        "--ramp-time",
        type=float,
        default=1.0,
        help="doba fade-in/out PD gainů [s] (def 1)",
    )
    ap.add_argument(
        "--return-time",
        type=float,
        default=2.0,
        help="doba pomalého návratu do q0 [s] (def 2)",
    )
    ap.add_argument(
        "--outdir",
        default="step_response_logs",
        help="adresář pro CSV (def step_response_logs)",
    )
    ap.add_argument(
        "--amplitudes",
        default=None,
        help="čárkou oddělené amplitudy pro sweep (např. '0.02,0.05,0.1,0.15'). "
             "Pokud je nastaveno, --amplitude se ignoruje.",
    )
    ap.add_argument(
        "--pre-wait",
        type=float,
        default=0.5,
        help="čas po ramp-up před step [s] (def 0.5) — testuje firmware "
             "startup transient. Zvyš na 2-3 pokud td klesá s delším čekáním.",
    )
    ap.add_argument("--yes", action="store_true", help="přeskočit potvrzení")
    args = ap.parse_args()

    if args.amplitudes:
        amplitudes = [float(x) for x in args.amplitudes.split(",")]
    else:
        amplitudes = [args.amplitude]

    print("=" * 70)
    print("BEZPEČNOSTNÍ UPOZORNĚNÍ — STEP RESPONSE TEST")
    print("=" * 70)
    print("• Robot MUSÍ být ZAVĚŠENÝ.")
    print(
        f"• Amplituda {args.amplitude:.3f} rad ≈ {math.degrees(args.amplitude):.1f}°."
    )
    print(
        "  Ankle (±15°) a waist roll/pitch (±30°) — zvaž --amplitude 0.05."
    )
    print("• Kolem testovaného kloubu MUSÍ být volno (nic v dráze).")
    print("• Ctrl+C kdykoli → ramp down na damping.")
    print()
    if not args.yes and not confirm("Je robot zavěšený a dráha kloubů volná?"):
        print("Zrušeno.")
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tester = StepTester(args.iface)

    # Ctrl+C v input() způsobí KeyboardInterrupt, pustí se cleanup.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        tester.init_dds()
        tester.start_cmd_thread()

        list_joints()

        while True:
            print()
            try:
                choice = input(
                    "Index kloubu pro test (0-28), 'l' = list, 'q' = konec: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if choice in ("q", "quit", "exit"):
                break
            if choice in ("l", "list", ""):
                list_joints()
                continue

            try:
                idx = int(choice)
                if not (0 <= idx < G1_NUM_MOTOR):
                    raise ValueError
            except ValueError:
                print("Neplatný index.")
                continue

            name = MOTOR_NAMES[idx]
            print(f"\nKloub: {name}  (index {idx})")
            print(
                f"  Kp={KP_DEFAULT[idx]}, Kd={KD_DEFAULT[idx]}  "
                f"amplitudes={amplitudes} rad  pre_wait={args.pre_wait:.2f}s"
            )
            if not confirm("Spustit test?"):
                continue

            try:
                # A) Fade in PD na všech kloubech (hold current pose).
                tester.hold_current_positions()
                print("  [ramp] ramp up PD...")
                tester.ramp_all_pd(0.0, 1.0, args.ramp_time)

                # A2) Pre-wait: dej firmware čas aby se ustálil po změně kp.
                if args.pre_wait > 0:
                    print(f"  [wait] settling {args.pre_wait:.2f}s...")
                    time.sleep(args.pre_wait)

                sweep_results = []
                for k, amp in enumerate(amplitudes):
                    print(f"\n  ── sweep {k+1}/{len(amplitudes)}: amplitude = "
                          f"{amp:+.3f} rad ({math.degrees(amp):+.1f}°) ──")

                    # Step + record.
                    print(f"    step + hold {args.hold_time:.1f}s...")
                    rec = tester.step_test(idx, amp, args.hold_time)

                    # Slow return to q0.
                    with tester.state_lock:
                        q_after = tester.q_real[idx]
                    print(f"    slow return ({args.return_time:.1f}s)...")
                    tester.slow_return(idx, q_after, rec.q0, args.return_time)

                    # Mezi kroky malá pauza aby se motor ustálil.
                    if k < len(amplitudes) - 1:
                        time.sleep(args.pre_wait)

                    # Analysis.
                    a = analyze_response(rec)
                    sweep_results.append((amp, rec, a))

                    # Save CSV per amplitude.
                    csv_name = f"{idx:02d}_{name}_amp{amp:+.3f}.csv"
                    csv_path = outdir / csv_name
                    save_csv(csv_path, rec)

                # D) Fade out PD.
                print("\n  [ramp] ramp down PD...")
                tester.ramp_all_pd(1.0, 0.0, args.ramp_time)

            except KeyboardInterrupt:
                print("\n  [!] Ctrl+C — emergency damping")
                tester.emergency_damping()
                break

            # Comparison table.
            print()
            print(f"  ══ Sweep summary: {name} ══")
            print(f"    cmd loop {tester._cmd_rate_hz:.0f} Hz / "
                  f"DDS {len(sweep_results[0][1].t)/args.hold_time:.0f} Hz")
            print(f"    Kp={KP_DEFAULT[idx]}, Kd={KD_DEFAULT[idx]}")
            print()
            print("    " + "─" * 100)
            print(
                f"    {'amp':>8}  {'q_final':>8}  {'under':>7}  "
                f"{'t10':>6}  {'t50':>6}  {'t63':>6}  {'t90':>6}  "
                f"{'td':>6}  {'τ':>6}  {'τ_alt':>6}  {'1st_ord':>7}"
            )
            print(
                f"    {'(rad)':>8}  {'(rad)':>8}  {'(rad)':>7}  "
                f"{'(ms)':>6}  {'(ms)':>6}  {'(ms)':>6}  {'(ms)':>6}  "
                f"{'(ms)':>6}  {'(ms)':>6}  {'(ms)':>6}  {'err %':>7}"
            )
            print("    " + "─" * 100)
            for amp, rec, a in sweep_results:
                def _ms(v): return f"{v*1000:>6.1f}" if v is not None else "   —  "
                foq = (f"{a['first_order_quality']*100:>6.1f}%"
                       if a["first_order_quality"] is not None else "   — ")
                print(
                    f"    {amp:>+8.3f}  {a['q_final']:>+8.4f}  "
                    f"{a['undershoot']:>+7.4f}  "
                    f"{_ms(a['t10'])}  {_ms(a['t50'])}  {_ms(a['t63'])}  "
                    f"{_ms(a['t90'])}  {_ms(a['td'])}  {_ms(a['tau'])}  "
                    f"{_ms(a['tau_alt'])}  {foq:>7}"
                )
            print("    " + "─" * 100)
            print(
                "    Čtení: td = dead-time, τ = časová konstanta (z t10↔t63),"
                " τ_alt (z t50↔t90)"
            )
            print(
                "    1st_ord err = |τ - τ_alt| / max; <10 % ~ 1. řád, >30 % ~"
                " jiná dynamika"
            )
            print(
                "    Pokud td ROSTE s amplitudou → rate limiter. Pokud KONSTANT"
                "Í → low-pass filter."
            )

            # Prvních 15 vzorků pro POSLEDNÍ amplitudu — zahrnout dq ať vidíme
            # jestli motor se hýbe i když q ještě nedala znamení.
            _, last_rec, _ = sweep_results[-1]
            print()
            print(f"    Prvních 15 vzorků (amp={sweep_results[-1][0]:+.3f}):")
            print(f"    {'t[ms]':>6}  {'q_rel':>8}  {'dq':>8}  {'q_tgt':>8}")
            for i in range(min(15, len(last_rec.t))):
                q_rel = last_rec.q[i] - last_rec.q0
                qt_rel = last_rec.q_target[i] - last_rec.q0
                print(f"    {last_rec.t[i]*1000:>6.1f}  "
                      f"{q_rel:>+8.4f}  {last_rec.dq[i]:>+8.4f}  "
                      f"{qt_rel:>+8.4f}")

    finally:
        print("\n[shutdown] ramp down PD → damping...")
        try:
            s_now = tester.kp_scale[0] if tester.kp_scale else 0.0
            tester.ramp_all_pd(s_now, 0.0, 1.0)
        except Exception:
            tester.emergency_damping()
        tester.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
