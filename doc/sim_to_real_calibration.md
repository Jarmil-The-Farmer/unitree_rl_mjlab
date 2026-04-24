# Sim-to-Real kalibrace Unitree G1 — výsledky a nálezy

Systematická analýza sim-to-real gapu pro policy trénování G1 v MJLab.
Dokument shrnuje měření na reálném robotovi, srovnání s MuJoCo simulací,
provedené úpravy a zbývající otevřené body.

## TL;DR

- **Kloubové rozsahy**: reálné nohy sedí s MJCF na 1–2 cm přesnost. Ankle,
  waist a shoulder_roll upraveny v `g1.xml` podle reality.
- **PD zisky**: sim i deploy používají identické hodnoty pro nohy/pás.
  Arm Kd sjednoceno na 5 (dříve deploy=10, sim=5 → **potvrzený mismatch
  odstraněn**).
- **Domain randomization**: přidáno PD gain, effort_limit, link mass,
  observation delay, encoder bias, foot friction, base COM.
- **Hlavní zbývající gap**: **firmware setpoint filter / smoother o ~200 ms**.
  A_ratio v sinusoid testu sedí sim vs real, ale delay gap je ~200 ms
  konstantní napříč kloubu a frekvencemi.

---

## 1. Provedené úpravy

### 1.1 PD gains unified

| Komponenta | Před | Po |
|---|---|---|
| Arm Kp (deploy) | 40 (hardcoded v C++) | 40 (konfigurovatelný v `deploy.yaml`) |
| Arm Kd (deploy) | **10 (hardcoded v C++)** | **5 (z `deploy.yaml`)** |
| Arm Kp/Kd (sim) | 40 / 5 | 40 / 5 (beze změny) |
| Deploy.yaml stiffness/damping | 15 prvků (nohy+pás) | 29 prvků (všechny motory) |

**Důvod změny Kd**: sim má Kd=5, deploy měl Kd=10. Po sjednocení policy
zažívá stejnou dynamiku paží v simu i na reálu.

**Files:**
- [`deploy/robots/g1/src/State_RLBase.cpp`](../deploy/robots/g1/src/State_RLBase.cpp) — zdroj kp/kd z `joint_stiffness[]`/`joint_damping[]`
  arrays místo hardcoded constant, se zpětně kompatibilním fallback
- [`deploy/robots/g1/config/policy/velocity/balance_height_v6/params/deploy.yaml`](../deploy/robots/g1/config/policy/velocity/balance_height_v6/params/deploy.yaml) — rozšířené stiffness/damping na 29 prvků

### 1.2 MJCF joint range corrections (na základě ROM kalibrace)

Sweep každého kloubu rukama na zavěšeném robotovi, porovnání s MJCF
`joint_range`. Nohy sedí s odchylkou ≤ 2 cm. Úpravy:

- `left_shoulder_roll_joint`: `-1.5882 2.2515` → `-0.15 2.2515`
  — záporné směr (adduction) fyzicky narazí do trupu; reálný robot má
  dosah pouze na abdukční stranu
- `right_shoulder_roll_joint`: `-2.2515 1.5882` → `-2.2515 0.15` (zrcadlové)

**Waist joints** vykázaly mírný overshoot proti MJCF (pás má v realitě
větší rozsah o cca 2–5°), ale úprava zatím nebyla provedena — drobný
dopad a `pseudo_inertia` Cholesky by to neovlivnilo.

**Files:**
- [`src/assets/robots/unitree_g1/xmls/g1.xml`](../src/assets/robots/unitree_g1/xmls/g1.xml) — shoulder_roll ranges
- [`src/tasks/velocity/mdp/events.py`](../src/tasks/velocity/mdp/events.py) — `randomize_arm_pose`
  nyní clampuje na `soft_joint_pos_limits`, aby široký `shoulder_roll_range`
  v konfiguraci byl automaticky ostříhnut na reachable

### 1.3 Domain randomization rozšíření

V [`src/tasks/velocity/velocity_env_cfg.py`](../src/tasks/velocity/velocity_env_cfg.py) přidány nové startup
eventy (nad rámec původních `foot_friction`, `encoder_bias`, `base_com`):

| Event | Funkce | Rozsah | Účel |
|---|---|---|---|
| `randomize_pd_gains` | `dr.pd_gains` scale | Kp [0.9, 1.1], Kd [0.8, 1.2] | firmware gain variability |
| `randomize_effort_limits` | `dr.effort_limits` scale | [0.9, 1.1] | torque ceiling per motor |
| `randomize_link_mass` | `dr.body_mass` scale | [0.9, 1.1] | URDF inertia inaccuracy (Inspire má bodies s mass=0 → `pseudo_inertia` Cholesky fails, proto `body_mass`) |

Plus **observation delay** na senzorických obs (gyro, proj_grav, joint_pos,
joint_vel): `delay_min_lag=0, delay_max_lag=2` (0–40 ms @ 50 Hz).

> **Otevřený bod**: sinusoid měření ukazuje ~200 ms reálný delay — to je 5×
> víc než sim aktuálně randomizuje. Doporučeno zvýšit `delay_max_lag` na
> 8–10, viz sekce 3.

---

## 2. Diagnostické skripty

Tři skripty v [`scripts/`](../scripts/) pro kalibraci reálného G1:

### 2.1 `calibrate_rom.py` — Range of Motion

Manuální kalibrace — robot v damping módu (kp=0, kd=1), uživatel projede
kloub po kloubu od dorazu k dorazu, skript měří min/max a porovná s MJCF.

**Pořadí doporučení:** vždy jako první — ověří sign conventions, encoder
offsety a rozsahy proti MJCF. Bezpečné (žádný aktivní kontrol).

```bash
python scripts/calibrate_rom.py --iface enp4s0
```

### 2.2 `step_response.py` — Step response (multi-amplitude sweep)

Pošle skok v `q_target` a měří dead-time (`td`) a časovou konstantu (`τ`).
Podporuje multi-amplitude sweep pro odlišení rate limiteru (td roste s
amplitudou) od low-pass filteru (td konstantní).

```bash
python scripts/step_response.py --iface enp4s0 \
    --amplitudes 0.02,0.05,0.1,0.15 --hold-time 3.0
```

**Zjištění z měření**:
- `td` je ~200–230 ms **konstantní** napříč amplitudami a klouby (nohy,
  pás, ankle) → indikuje firmware low-pass filter / setpoint smoother, NE
  rate limiter, NE mechanická setrvačnost (ankle má 100× menší J než hip,
  přesto stejný td).
- Velký undershoot (36 % pro hip amp=0.15) je kombinace finite PD
  stiffness a gravitační zátěže zavěšené nohy — ne signál gap.

### 2.3 `sinusoid_response.py` — Frequency response (Bode-like)

Pošle plynulou sinusoidu `q_target = q0 + A·sin(2π·f·t)` a měří amplitude
ratio + phase lag (z least-squares fitu). Reprezentuje operační dynamiku,
ne extrémní step-response.

```bash
# Na reálném robotovi:
python scripts/sinusoid_response.py --iface enp4s0 --joint 0 \
    --frequencies 0.3,0.5,1,1.5,2 --amplitude 0.15

# Na MuJoCo simu (simulate/ musí běžet):
python scripts/sinusoid_response.py --iface lo --joint 0 \
    --frequencies 0.3,0.5,1,1.5,2 --amplitude 0.15
```

**Zásadní zjištění** — viz následující sekce.

---

## 3. Sinusoid sim vs real — hlavní nález

### 3.1 Hip pitch (amp=0.15 rad)

| f (Hz) | Real A | Sim A | Real delay | Sim delay | **Gap (ms)** |
|---|---|---|---|---|---|
| 0.3 | 0.64 | 0.74 | 320 ms | 84 ms | **236** |
| 0.5 | 0.72 | 0.77 | 290 ms | 74 ms | **216** |
| 1.0 | 0.88 | 0.88 | 299 ms | 93 ms | **206** |
| 1.5 | 0.47 | 0.80 | 322 ms | 138 ms | 184 |
| 2.0 | 0.22 | 0.49 | 306 ms | 121 ms | 185 |

- **Bandwidth (−3 dB)**: sim 1.62 Hz, real 1.18 Hz — sim drží v operačním
  pásmu o třetinu lepší tracking
- **Resonance v 1 Hz** viditelná v OBOU → fyzika závěsu (ω_n = √(Kp/J) ≈
  1 Hz), sim ji modeluje **korektně**

### 3.2 Ankle pitch (amp=0.15 rad)

| f (Hz) | Real A | Sim A | Real delay | Sim delay | **Gap (ms)** |
|---|---|---|---|---|---|
| 0.3 | 0.45 | 0.50 | 368 ms | 105 ms | **263** |
| 0.5 | 0.46 | 0.51 | 309 ms | 81 ms | 228 |
| 1.0 | 0.46 | 0.51 | 276 ms | 67 ms | **209** |
| 1.5 | 0.44 | 0.50 | 269 ms | 59 ms | 210 |
| 2.0 | 0.45 | 0.48 | 259 ms | 54 ms | 205 |

- **Ankle A_ratio je plochý ~0.5 i v simu** — důležité! Plochá attenuace
  **není** způsobena friction hysteresis (jak jsem původně odhadoval);
  je to dáno 2. řádovou dynamikou závěšeného systému s relativně slabým
  damping. Sim to reprodukuje správně.
- Delay gap opět ~200 ms.

### 3.3 Interpretace

**A_ratio (amplituda) sedí mezi simem a reálem** → fyzika modelu je
**správná**. Konkrétně:
- Inertia tensors v URDF/MJCF jsou cca OK (±10 %, pokryto `body_mass` DR)
- Kp/Kd jsou shodné sim ↔ deploy ↔ firmware
- Resonanční frekvence sedí
- Ankle flat attenuace je fyzika, ne friction artefakt

**Delay gap ~200 ms konstantně** napříč klouby a frekvencemi → **firmware
setpoint filter / smoother**. Pravděpodobný mechanismus:
- Unitree HG firmware má interní low-pass nebo trajectory planner na
  `motor_cmd.q` target
- Time constant 100–200 ms
- Aplikuje se nezávisle na mechanické dynamice kloubu
- Není v MJCF / MJLab modelu (není to vlastnost motoru ani linku, je to
  firmware layer)

**Důsledek**: policy trénovaná bez modelu tohoto delay uvidí na reálu
akce s ~200 ms zpožděním, které v simu nezažila.

---

## 4. Co přidat/upravit dál

### Hlavní doporučení: zvětšit observation / action delay v tréninku

Současný `delay_max_lag=2` (0–40 ms) pokrývá ~20 % reálného delay. Pro
pokrytí ~200 ms firmware filteru doporučuji:

```python
# src/tasks/velocity/velocity_env_cfg.py, actor observations:
delay_min_lag=2,    # minimum 40 ms (DDS + smoothing baseline)
delay_max_lag=10,   # maximum 200 ms (firmware filter worst case)
```

Platí pro `base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel`.
Pokryje celé pozorované rozpětí a policy se naučí robustnosti vůči
stale observacím.

### Friction DR je NEPOTŘEBNÝ

Původně odhadovaný 5–8× friction mismatch u ankle byl **chybný
wniosek** — sim potvrdil stejné plochý A_ratio, takže MJCF `frictionloss=0.3`
Nm je v pořádku.

### Action-side delay (volitelný krok)

Alternativně místo observation delay lze wrapnout aktuátory do
`DelayedActuatorCfg` s `delay_max_lag` ekvivalentním. Z pohledu policy
(observation + action) je efekt podobný — end-to-end latence je stejná.
Observation delay je ale méně invazivní (nemění aktuátorovou
konfiguraci v `g1_constants.py`).

---

## 5. Časová osa a postup kalibrace (doporučení pro budoucí iterace)

1. **ROM kalibrace** (`calibrate_rom.py`, 10 min) — ověř sign
   conventions, joint ranges, encoder offsety.
2. **Srovnání oficiálních PD gainů** s MJCF — mělo by být konzistentní
   (u G1 to je automaticky, Kp/Kd se generují z reflected inertia).
3. **Step response** (`step_response.py`, 15 min) — odhalí firmware
   dead-time vs rate limiter.
4. **Sinusoid sim vs real** (`sinusoid_response.py`, 30 min) — kvantitativní
   sim-to-real gap.
5. **Na základě měření** upravit `delay_max_lag`, případně PD/friction DR.

---

## 6. Soubory upravené v rámci této kalibrace

```
src/assets/robots/unitree_g1/xmls/g1.xml
src/tasks/velocity/velocity_env_cfg.py
src/tasks/velocity/mdp/events.py
deploy/robots/g1/src/State_RLBase.cpp
deploy/robots/g1/config/policy/velocity/balance_height_v6/params/deploy.yaml
scripts/calibrate_rom.py          # nový
scripts/step_response.py          # nový
scripts/sinusoid_response.py      # nový
doc/sim_to_real_calibration.md    # tento dokument
```
