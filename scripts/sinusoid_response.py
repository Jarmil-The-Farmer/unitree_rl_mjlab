#!/usr/bin/env python3
"""Sinusoid response test — měří operační zpoždění při plynulém tracking.

Pošle target q(t) = q0 + A·sin(2π·f·t), zaznamená q(t), fituje sinusoidu
a z fázového posunu spočítá čas delay. Na rozdíl od step testu je to
reprezentativní pro normální provoz policy (plynule se měnící target).

BEZPEČNOST:
    - Robot MUSÍ být ZAVĚŠENÝ.
    - Kolem kloubu musí být prostor ±A od aktuální pozice.
    - Ctrl+C → ramp down na damping mode.

Výstup per frekvence:
    A_ratio  — poměr amplitud (q / q_target). 1.0 = dokonalé sledování.
               Pokles pod 0.7 (−3 dB) = překročen bandwidth.
    phase    — fázové zpoždění q za q_target [rad].
    delay    — time delay [ms] = phase / (2π·f). Tohle je OPERAČNÍ delay.
    R²       — jak dobře sedí fit (1.0 = perfektní sinus; nízké = šum/nelinearita).

Příklad:
    python scripts/sinusoid_response.py --iface enp4s0 --joint 0
    python scripts/sinusoid_response.py --iface eth0 --joint 4 \\
        --frequencies 0.3,0.5,1,2,3,5 --amplitude 0.03
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Reuse infrastructure.
from step_response import (
    CONTROL_DT,
    CONTROL_HZ,
    G1_NUM_MOTOR,
    KD_DEFAULT,
    KP_DEFAULT,
    MOTOR_NAMES,
    Record,
    StepTester,
    confirm,
    list_joints,
    save_csv,
)


def sine_test(
    tester: StepTester,
    joint_idx: int,
    amplitude: float,
    frequency: float,
    duration: float,
    ramp_cycles: float = 2.0,
) -> Record:
    """Spustí sinusoidní target pro jeden kloub, nahraje odpověď.

    Amplituda lineárně naramuje do plné hodnoty v průběhu `ramp_cycles`
    period (předejde skokový start). Pak drží plnou amplitudu zbytek duration.
    PD gains se nepočítají — předpokládáme že jsou už v plném stavu.
    """
    name = MOTOR_NAMES[joint_idx]

    with tester.state_lock:
        q0 = tester.q_real[joint_idx]
        for i in range(G1_NUM_MOTOR):
            tester.q_target[i] = tester.q_real[i]

    rec = Record(joint_idx=joint_idx, joint_name=name, q0=q0, amplitude=amplitude)

    with tester.record_lock:
        tester.record = rec
    tester.record_start_time = time.perf_counter()
    tester.recording = True

    ramp_duration = ramp_cycles / frequency if frequency > 0 else 0.0
    omega = 2.0 * math.pi * frequency

    t_start = time.perf_counter()
    next_t = t_start + CONTROL_DT
    while True:
        t = time.perf_counter() - t_start
        if t >= duration:
            break
        amp_now = amplitude if t >= ramp_duration else amplitude * (t / ramp_duration)
        tester.q_target[joint_idx] = q0 + amp_now * math.sin(omega * t)
        rem = next_t - time.perf_counter()
        if rem > 0:
            time.sleep(rem)
        next_t += CONTROL_DT

    tester.recording = False
    tester.q_target[joint_idx] = q0
    return rec


def analyze_sine(
    rec: Record, frequency: float, amplitude_commanded: float, skip_ratio: float = 0.35
) -> dict:
    """Fit q(t) ≈ bias + b·sin(ωt) + c·cos(ωt) a extrahuj amplitudu + fázi.

    Skip prvních `skip_ratio` dat jako přechod (ramp-in + settling).
    """
    out = {
        "n_samples": len(rec.t),
        "n_fit": 0,
        "A_q": 0.0,
        "A_ratio": 0.0,
        "phase_lag_rad": 0.0,
        "delay_ms": 0.0,
        "bias": 0.0,
        "r_squared": 0.0,
    }
    if len(rec.t) < 20 or frequency <= 0:
        return out

    skip = max(1, int(skip_ratio * len(rec.t)))
    t = np.array(rec.t[skip:], dtype=np.float64)
    q = np.array(rec.q[skip:], dtype=np.float64)
    out["n_fit"] = len(t)

    omega = 2.0 * np.pi * frequency
    A = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    a, b, c = coef

    A_q = float(np.hypot(b, c))
    # q(t) = A_q · sin(ωt - φ) = b·sin(ωt) + c·cos(ωt)  kde  b=A_q·cos(φ), c=-A_q·sin(φ)
    # → positive phase lag pokud c < 0
    phase_lag = float(np.arctan2(-c, b))
    delay_s = phase_lag / omega

    q_pred = A @ coef
    ss_res = float(np.sum((q - q_pred) ** 2))
    ss_tot = float(np.sum((q - q.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    out.update({
        "A_q": A_q,
        "A_ratio": A_q / amplitude_commanded if amplitude_commanded > 0 else 0.0,
        "phase_lag_rad": phase_lag,
        "delay_ms": delay_s * 1000.0,
        "bias": float(a),
        "r_squared": r_sq,
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--iface", required=True)
    ap.add_argument("--joint", type=int, required=True, help="index (0-28)")
    ap.add_argument(
        "--frequencies",
        default="0.3,0.5,1.0,2.0,3.0,5.0",
        help="čárkou oddělené frekvence [Hz] (def 0.3,0.5,1,2,3,5)",
    )
    ap.add_argument("--amplitude", type=float, default=0.05, help="[rad] (def 0.05 ≈ 2.9°)")
    ap.add_argument(
        "--cycles",
        type=float,
        default=8.0,
        help="minimum cycles per frekvence (def 8)",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="horní strop doby testu na frekvenci [s] (def 20)",
    )
    ap.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="spodní minimum doby testu na frekvenci [s] (def 4)",
    )
    ap.add_argument("--ramp-time", type=float, default=1.0, help="fade-in/out PD [s]")
    ap.add_argument("--pre-wait", type=float, default=0.5, help="pauza po ramp PD [s]")
    ap.add_argument(
        "--outdir",
        default="sine_response_logs",
        help="adresář pro CSV (def sine_response_logs)",
    )
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if not (0 <= args.joint < G1_NUM_MOTOR):
        print(f"Neplatný index kloubu: {args.joint}")
        return 1
    frequencies = [float(x) for x in args.frequencies.split(",")]
    if not frequencies:
        print("--frequencies prázdné")
        return 1

    joint_name = MOTOR_NAMES[args.joint]
    print("=" * 70)
    print("BEZPEČNOSTNÍ UPOZORNĚNÍ — SINUSOID RESPONSE TEST")
    print("=" * 70)
    print(f"• Kloub: {joint_name} (index {args.joint})")
    print(f"• Amplituda: ±{args.amplitude:.3f} rad ≈ ±{math.degrees(args.amplitude):.1f}°")
    print(f"• Frekvence [Hz]: {frequencies}")
    print("• Robot MUSÍ být ZAVĚŠENÝ; kolem kloubu volný prostor.")
    print("• Ctrl+C → ramp down na damping.")
    print()
    if not args.yes and not confirm("Pokračovat?"):
        print("Zrušeno.")
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tester = StepTester(args.iface)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    results = []  # (freq, record, analysis)

    try:
        tester.init_dds()
        tester.start_cmd_thread()

        # A) Ramp PD up.
        tester.hold_current_positions()
        print("[ramp] ramp up PD...")
        tester.ramp_all_pd(0.0, 1.0, args.ramp_time)
        if args.pre_wait > 0:
            time.sleep(args.pre_wait)

        for k, f in enumerate(frequencies):
            dur = max(args.min_duration, min(args.max_duration, args.cycles / f))
            print(
                f"\n── [{k+1}/{len(frequencies)}] f={f:.2f} Hz  "
                f"cycles≈{dur * f:.1f}  duration={dur:.1f}s"
            )
            rec = sine_test(
                tester,
                args.joint,
                args.amplitude,
                f,
                dur,
                ramp_cycles=min(2.0, dur * f / 4),
            )
            ana = analyze_sine(rec, f, args.amplitude)
            results.append((f, rec, ana))
            print(
                f"    A_ratio={ana['A_ratio']:.3f}  phase={math.degrees(ana['phase_lag_rad']):+.1f}°  "
                f"delay={ana['delay_ms']:+.1f} ms  R²={ana['r_squared']:.3f}"
            )

            csv_path = outdir / f"{args.joint:02d}_{joint_name}_f{f:.2f}.csv"
            save_csv(csv_path, rec)

            # Mezi frekvencemi chvíli počkat.
            tester.q_target[args.joint] = rec.q0
            time.sleep(0.5)

        # B) Ramp PD down.
        print("\n[ramp] ramp down PD...")
        tester.ramp_all_pd(1.0, 0.0, args.ramp_time)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C — emergency damping")
        tester.emergency_damping()
    finally:
        try:
            s_now = tester.kp_scale[0] if tester.kp_scale else 0.0
            if s_now > 0:
                tester.ramp_all_pd(s_now, 0.0, 1.0)
        except Exception:
            tester.emergency_damping()
        tester.shutdown()

    # ── Souhrnná tabulka ──
    if results:
        print()
        print("═" * 88)
        print(f" Sinusoid frequency response — {joint_name}")
        print(f" Kp={KP_DEFAULT[args.joint]}, Kd={KD_DEFAULT[args.joint]}, "
              f"amplitude=±{args.amplitude} rad")
        print(f" cmd loop ≈ {tester._cmd_rate_hz:.0f} Hz")
        print("═" * 88)
        print(f" {'f (Hz)':>8} {'A_ratio':>8} {'|H| [dB]':>10} {'phase (°)':>10} "
              f"{'delay (ms)':>12} {'R²':>6} {'n_fit':>7}")
        print("─" * 88)
        for f, rec, a in results:
            a_ratio = a["A_ratio"]
            dB = 20.0 * math.log10(max(a_ratio, 1e-6))
            print(
                f" {f:>8.2f} {a_ratio:>8.3f} {dB:>10.2f} "
                f"{math.degrees(a['phase_lag_rad']):>+10.1f} "
                f"{a['delay_ms']:>+12.1f} {a['r_squared']:>6.3f} {a['n_fit']:>7}"
            )
        print("─" * 88)
        # Identifikovat bandwidth (A_ratio = 0.707, -3 dB).
        bw = None
        for i in range(len(results) - 1):
            f1, _, a1 = results[i]
            f2, _, a2 = results[i + 1]
            if a1["A_ratio"] >= 0.707 >= a2["A_ratio"]:
                # Log interpolation.
                frac = (math.log(a1["A_ratio"]) - math.log(0.707)) / (
                    math.log(a1["A_ratio"]) - math.log(max(a2["A_ratio"], 1e-6))
                )
                bw = f1 + frac * (f2 - f1)
                break
        if bw is not None:
            print(f" Bandwidth (−3 dB) ≈ {bw:.2f} Hz")
        avg_delay = sum(a["delay_ms"] for _, _, a in results) / len(results)
        low_f_delays = [a["delay_ms"] for f, _, a in results if f <= 2.0]
        if low_f_delays:
            print(
                f" Avg delay (≤2 Hz, operační pásmo) ≈ "
                f"{sum(low_f_delays) / len(low_f_delays):+.1f} ms"
            )
        print(" Delay ve zlomku CONTROL_DT (20ms @ 50Hz): "
              f"{avg_delay / 20.0:.1f} lag_steps")
        print("═" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
