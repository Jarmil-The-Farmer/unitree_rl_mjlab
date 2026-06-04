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


def _function(short_name: str) -> str:
  """Joint function key (left/right share params). Matches thermal.py keys.

  Short names use the L_/R_ prefixes from log_motor_thermal.py::_short.
  """
  return short_name.replace("L_", "").replace("R_", "")


# Coupled groups: each motor's winding is heated by the COMBINED squared
# torque of the whole group (shared physical motors via 4-bar linkage).
# Keyed by short name -> tuple of short names whose tau drives this one.
def _torque_sources(short_name: str) -> tuple[str, ...]:
  fn = _function(short_name)
  if fn in ("waist_pitch", "waist_roll"):
    return ("waist_pitch", "waist_roll")
  if fn in ("ankle_pitch", "ankle_roll"):
    side = "L_" if short_name.startswith("L_") else "R_"
    return (f"{side}ankle_pitch", f"{side}ankle_roll")
  return (short_name,)


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

  # Initial guess: gain from steady-state-ish, tau_th ~ 40 s.
  tau_rms2 = float(np.mean(tau ** 2)) or 1.0
  G0 = max((temp.max() - temp.min()) / tau_rms2, 1e-4)
  if t_amb_fixed is None:
    amb_lb, amb_ub = 10.0, 50.0
    # The motor may start already hot, so anchor the ambient guess to the
    # coldest observed temperature, clamped into the bounds.
    amb_guess = float(np.clip(min(T0 - 5.0, float(temp.min())), amb_lb, amb_ub))
  else:
    amb_lb, amb_ub = t_amb_fixed - 1e-6, t_amb_fixed + 1e-6
    amb_guess = t_amb_fixed
  p0 = [G0, np.log(40.0), amb_guess]
  lb = [1e-5, np.log(2.0), amb_lb]
  ub = [10.0, np.log(2000.0), amb_ub]
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
  by_function: dict[str, list[dict]] = defaultdict(list)
  per_joint_rows: list[tuple] = []
  plot_data: list[tuple] = []

  for path in paths:
    data = _load_csv(path)
    header = list(data.keys())
    shorts = _joint_short_names(header)
    t = data.get("t_s")
    for s in shorts:
      temp = data.get(f"{prefix}{s}")
      if temp is None or t is None:
        continue
      # Driving torque: combined over the coupled group (shared physical
      # motors). tau_drive**2 == sum of source tau**2, matching the sim model.
      srcs = _torque_sources(s)
      sq = None
      missing = False
      for src in srcs:
        col = data.get(f"tau_{src}")
        if col is None:
          missing = True
          break
        sq = col ** 2 if sq is None else sq + col ** 2
      if missing or sq is None:
        continue
      tau_drive = np.sqrt(sq)
      fit = _fit_joint(tau_drive, temp, t, args.t_amb)
      if fit is None:
        continue
      fn = _function(s)
      fit["coupled"] = len(srcs) > 1
      per_joint_rows.append((path.split("/")[-1], s, fn, fit))
      by_function[fn].append(fit)
      if args.plot:
        plot_data.append((path, s, fn, tau_drive, temp, t, fit))

  if not per_joint_rows:
    print("No fittable joints found. Need logs where temperature actually "
          "rises (hold a loaded pose for a few minutes).")
    return

  # Per-joint table.
  print(f"\n=== Per-joint fits (channel={args.channel}) ===")
  print(f"{'file':<22}{'joint':<16}{'function':<13}{'cpl':<4}"
        f"{'G':>9}{'tau_th[s]':>10}{'T_amb':>7}{'T_max':>7}"
        f"{'tau_rms':>8}{'rmse':>6}{'dur[s]':>7}")
  for fname, s, fn, fit in per_joint_rows:
    print(f"{fname:<22}{s:<16}{fn:<13}{'Y' if fit['coupled'] else '.':<4}"
          f"{fit['G']:>9.4f}{fit['tau_th']:>10.1f}{fit['T_amb']:>7.1f}"
          f"{fit['T_max']:>7.0f}{fit['tau_rms']:>8.2f}{fit['rmse']:>6.2f}"
          f"{fit['dur_s']:>7.0f}")

  # Aggregate per joint function (median across left/right and logs).
  print(f"\n=== Suggested params per function (channel={args.channel}) ===")
  print("# Paste/merge into DEFAULT_PARAMS in src/tasks/velocity/mdp/thermal.py")
  print("# (R_th fixed to 1.0, k = fitted gain G). Coupled joints (waist,")
  print("# ankle) were fit against combined group torque.")
  for fn in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch",
             "ankle_roll", "waist_yaw", "waist_pitch", "waist_roll"):
    fits = by_function.get(fn)
    if not fits:
      continue
    G = float(np.median([f["G"] for f in fits]))
    tau_th = float(np.median([f["tau_th"] for f in fits]))
    T_amb = float(np.median([f["T_amb"] for f in fits]))
    Tmax = max(f["T_max"] for f in fits)
    print(f'  "{fn}": MotorThermalParams(k={G:.4f}, R_th=1.0, '
          f'tau_th={tau_th:.1f}),  # n={len(fits)} Tmax~{Tmax:.0f} T_amb~{T_amb:.1f}')

  if args.plot and plot_data:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for path, s, fn, tau, temp, t, fit in plot_data:
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
      ax2.set_ylabel("tau_drive [N·m]", color="b")
      ax1.legend(loc="upper left")
      cpl = " (coupled)" if fit["coupled"] else ""
      ax1.set_title(f"{s} [{fn}{cpl}]  G={fit['G']:.4f} tau_th={fit['tau_th']:.0f}s "
                    f"rmse={fit['rmse']:.2f}")
      out = f"{path.rsplit('.', 1)[0]}__{s}.png"
      fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
      print(f"[plot] {out}")


if __name__ == "__main__":
  main()
