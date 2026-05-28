"""Fit 1-node motor thermal parameters from real-robot logs.

Reads one or more CSVs produced by ``scripts/log_motor_thermal.py`` and fits,
per motor type, the parameters of the lumped thermal model used in
``src/tasks/velocity/mdp/thermal.py``:

    dT/dt = (G * tau**2 - (T - T_amb)) / tau_th

Only the product ``G = k * R_th`` (steady-state gain, [K/(N*m)^2]) and the
time constant ``tau_th`` [s] are identifiable from temperature alone, so the
fit reports ``G`` and ``tau_th`` (plus an estimated ``T_amb``). The emitted
sim params set ``R_th = 1.0`` and ``k = G``.

Channel: by default fits the WINDING temperature (temperature[1], the fast
copper-loss-driven channel, protection limit ~120 C). Use --channel casing
to fit the casing temperature instead.

Usage:
  python scripts/fit_motor_thermal.py logs/motor_thermal/*.csv
  python scripts/fit_motor_thermal.py run1.csv run2.csv --plot
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

# Joint (short name) -> motor type. Mirrors g1_constants.py actuator groups.
# Short names match scripts/log_motor_thermal.py::_short (L_/R_ prefixes).
def _motor_type(short_name: str) -> str | None:
  n = short_name
  if "hip_pitch" in n or "hip_yaw" in n or "waist_yaw" in n:
    return "7520_14"
  if "hip_roll" in n or "knee" in n:
    return "7520_22"
  if "ankle" in n or "waist_pitch" in n or "waist_roll" in n:
    return "5020"  # paired 2x5020 on the linkage; folded into gain G
  if "shoulder" in n or "elbow" in n or "wrist_roll" in n:
    return "5020"
  if "wrist_pitch" in n or "wrist_yaw" in n:
    return "4010"
  return None


def _load_csv(path: str) -> dict[str, np.ndarray]:
  with open(path) as f:
    reader = csv.reader(f)
    header = next(reader)
    cols: list[list[float]] = [[] for _ in header]
    for row in reader:
      for i, v in enumerate(row):
        try:
          cols[i].append(float(v))
        except ValueError:
          cols[i].append(np.nan)
  return {h: np.asarray(c, dtype=float) for h, c in zip(header, cols)}


def _joint_short_names(header: list[str]) -> list[str]:
  return [h[len("tau_"):] for h in header if h.startswith("tau_")]


def _simulate(tau: np.ndarray, t: np.ndarray, G: float, tau_th: float,
              T_amb: float, T0: float) -> np.ndarray:
  """Forward-Euler integrate the 1-node model over the logged tau sequence."""
  T = np.empty_like(tau)
  T[0] = T0
  for n in range(len(tau) - 1):
    dt = t[n + 1] - t[n]
    dt = dt if dt > 0 else 0.0
    dT = dt * (G * tau[n] ** 2 - (T[n] - T_amb)) / tau_th
    T[n + 1] = T[n] + dT
  return T


def _fit_joint(tau: np.ndarray, temp: np.ndarray, t: np.ndarray,
               t_amb_fixed: float | None):
  """Fit (G, tau_th, T_amb) for one joint. Returns dict or None if bad data."""
  mask = np.isfinite(tau) & np.isfinite(temp) & np.isfinite(t)
  tau, temp, t = tau[mask], temp[mask], t[mask]
  if len(tau) < 20 or (temp.max() - temp.min()) < 1.0:
    return None  # not enough signal to fit
  T0 = float(temp[0])

  def resid(p):
    G, log_tau_th, T_amb = p
    tau_th = np.exp(log_tau_th)
    if t_amb_fixed is not None:
      T_amb = t_amb_fixed
    sim = _simulate(tau, t, G, tau_th, T_amb, T0)
    return sim - temp

  # Initial guess: gain from steady-state-ish, tau_th ~ 90 s.
  tau_rms2 = float(np.mean(tau ** 2)) or 1.0
  G0 = max((temp.max() - temp.min()) / tau_rms2, 1e-4)
  p0 = [G0, np.log(90.0), T0 - 5.0 if t_amb_fixed is None else t_amb_fixed]
  lb = [1e-5, np.log(5.0), 10.0 if t_amb_fixed is None else t_amb_fixed - 1e-6]
  ub = [10.0, np.log(2000.0), 50.0 if t_amb_fixed is None else t_amb_fixed + 1e-6]
  try:
    res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=4000)
  except Exception:
    return None
  G, log_tau_th, T_amb = res.x
  tau_th = float(np.exp(log_tau_th))
  if t_amb_fixed is not None:
    T_amb = t_amb_fixed
  sim = _simulate(tau, t, G, tau_th, T_amb, T0)
  rmse = float(np.sqrt(np.mean((sim - temp) ** 2)))
  return {
    "G": float(G), "tau_th": tau_th, "T_amb": float(T_amb),
    "rmse": rmse, "T_max": float(temp.max()), "tau_rms": float(np.sqrt(tau_rms2)),
    "n": int(len(tau)), "dur_s": float(t[-1] - t[0]),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("csv", nargs="+", help="CSV log file(s) or glob(s).")
  ap.add_argument("--channel", choices=["winding", "casing"], default="winding")
  ap.add_argument("--t-amb", type=float, default=None,
                  help="Fix ambient temperature [C] instead of fitting it.")
  ap.add_argument("--plot", action="store_true", help="Save fit plots (PNG).")
  args = ap.parse_args()

  paths: list[str] = []
  for pat in args.csv:
    paths += sorted(glob.glob(pat)) or [pat]

  prefix = "Twind_" if args.channel == "winding" else "Tcase_"
  by_type: dict[str, list[dict]] = defaultdict(list)
  per_joint_rows: list[tuple] = []
  plot_data: list[tuple] = []

  for path in paths:
    data = _load_csv(path)
    header = list(data.keys())
    shorts = _joint_short_names(header)
    for s in shorts:
      tau = data.get(f"tau_{s}")
      temp = data.get(f"{prefix}{s}")
      t = data.get("t_s")
      if tau is None or temp is None or t is None:
        continue
      fit = _fit_joint(tau, temp, t, args.t_amb)
      if fit is None:
        continue
      mtype = _motor_type(s)
      per_joint_rows.append((path.split("/")[-1], s, mtype, fit))
      if mtype is not None:
        by_type[mtype].append(fit)
      if args.plot:
        plot_data.append((path, s, mtype, tau, temp, t, fit))

  if not per_joint_rows:
    print("No fittable joints found. Need logs where temperature actually "
          "rises (hold a loaded pose for a few minutes).")
    return

  # Per-joint table.
  print(f"\n=== Per-joint fits (channel={args.channel}) ===")
  print(f"{'file':<28}{'joint':<16}{'type':<9}"
        f"{'G':>10}{'tau_th[s]':>11}{'T_amb':>8}{'T_max':>8}"
        f"{'tau_rms':>9}{'rmse':>7}{'dur[s]':>8}")
  for fname, s, mtype, fit in per_joint_rows:
    print(f"{fname:<28}{s:<16}{str(mtype):<9}"
          f"{fit['G']:>10.4f}{fit['tau_th']:>11.1f}{fit['T_amb']:>8.1f}"
          f"{fit['T_max']:>8.0f}{fit['tau_rms']:>9.2f}{fit['rmse']:>7.2f}"
          f"{fit['dur_s']:>8.0f}")

  # Aggregate per motor type (median, weighted by signal would be nicer but
  # median is robust to one bad joint).
  print(f"\n=== Suggested DEFAULT_PARAMS (channel={args.channel}) ===")
  print("# Paste into src/tasks/velocity/mdp/thermal.py (R_th fixed to 1.0,")
  print("# k = fitted gain G). Re-check T_warn/T_crit/T_max against T_max above.")
  print("DEFAULT_PARAMS = {")
  for mtype in ("5020", "7520_14", "7520_22", "4010"):
    fits = by_type.get(mtype)
    if not fits:
      continue
    G = float(np.median([f["G"] for f in fits]))
    tau_th = float(np.median([f["tau_th"] for f in fits]))
    T_amb = float(np.median([f["T_amb"] for f in fits]))
    print(f'  "{mtype}": MotorThermalParams(k={G:.5f}, R_th=1.0, '
          f'tau_th={tau_th:.1f}),  # n={len(fits)}, T_amb~{T_amb:.1f}')
  print("}")

  if args.plot and plot_data:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for path, s, mtype, tau, temp, t, fit in plot_data:
      m = np.isfinite(tau) & np.isfinite(temp) & np.isfinite(t)
      tau, temp, t = tau[m], temp[m], t[m]
      sim = _simulate(tau, t, fit["G"], fit["tau_th"], fit["T_amb"],
                      float(temp[0]))
      fig, ax1 = plt.subplots(figsize=(9, 4))
      ax1.plot(t, temp, "k.", ms=2, label=f"{args.channel} (obs)")
      ax1.plot(t, sim, "r-", lw=1.5, label="fit")
      ax1.set_xlabel("t [s]"); ax1.set_ylabel("T [C]")
      ax2 = ax1.twinx()
      ax2.plot(t, tau, "b-", alpha=0.3, lw=0.8)
      ax2.set_ylabel("tau_est [N·m]", color="b")
      ax1.legend(loc="upper left")
      ax1.set_title(f"{s} ({mtype})  G={fit['G']:.4f} tau_th={fit['tau_th']:.0f}s "
                    f"rmse={fit['rmse']:.2f}")
      out = f"{path.rsplit('.', 1)[0]}__{s}.png"
      fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
      print(f"[plot] {out}")


if __name__ == "__main__":
  main()
