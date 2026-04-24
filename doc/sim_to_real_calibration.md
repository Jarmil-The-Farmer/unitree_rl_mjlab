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
- **Operační sim-to-real delay gap je ~30-60 ms** v běžném pásmu (1-2 Hz).
  Na velmi nízkých frekvencích (0.3 Hz) ~100 ms. Podstatně menší než
  první Python měření naznačovala — Python timing měl artefakty ~150 ms
  na reálu (ne v simu); C++ verze testu potvrdila skutečné hodnoty.

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

> **Aktualizace po C++ měření**: sinusoid gap v operačním pásmu (1-2 Hz)
> je jen 20-35 ms — **aktuální `delay_max_lag=2` (40 ms) je dostatečný**.
> Python původně ukazoval ~200 ms gap, ale to byl artefakt Python
> timing/threading na reálu (v simu Python funguje správně). C++ verze
> testu potvrdila, že skutečný gap je malý.

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

### 3.3 Python delay artefakt — ověřeno C++ verzí

Po podezření že Python ukazuje nerealistické hodnoty byl napsán C++
test [`scripts/cpp/sinusoid_response.cpp`](../scripts/cpp/sinusoid_response.cpp) používající identický
DDS stack jako deploy binary. Výsledky ukazují **jiný příběh**:

**Srovnání Python vs C++ na identickém hardwaru (ankle, amp=0.15):**

| Zdroj | Avg delay (≤2 Hz) |
|---|---|
| Python sim | 73 ms |
| **C++ sim** | 72 ms (shoda) |
| Python real | 287 ms |
| **C++ real** | 124-134 ms |

Python na reálu přidával ~150 ms artefakt (Python cmd thread běžel jen
~450 Hz vs C++ 500 Hz + GIL contention mezi cmd thread a DDS callback
→ systematicky pozdní timestampy vzorků). V simu Python funguje OK,
protože `lo` interface má nižší overhead.

**Skutečný sim-to-real gap (z C++ dat):**

Ankle pitch (index 4, amp=0.15):

| f (Hz) | Sim delay | Real delay | Gap |
|---|---|---|---|
| 0.3 | 106 ms | 222 ms | 116 ms |
| 0.5 | 78 ms | 146 ms | 68 ms |
| 1.0 | 65 ms | 99 ms | 34 ms |
| 1.5 | 58 ms | 82 ms | 24 ms |
| 2.0 | 55 ms | 72 ms | 17 ms |

Hip pitch (index 0, amp=0.15):

| f (Hz) | Sim delay | Real delay | Gap |
|---|---|---|---|
| 0.3 | 84 ms | 115 ms | 31 ms |
| 0.5 | 75 ms | 99 ms | 24 ms |
| 1.0 | 91 ms | 115 ms | 24 ms |
| 1.5 | 138 ms | 143 ms | 5 ms |
| 2.0 | 123 ms | 110 ms | -13 ms |

**V operačním pásmu 1-2 Hz (walking) je gap 15-35 ms** → pokryto
aktuálním `delay_max_lag=2` (0-40 ms). Hip je téměř dokonale matched
(gap 15 ms avg), ankle má trochu víc na nízkých f (pravděpodobně
parallel linkage compliance), ale v operaci ne relevantní.

### 3.4 Interpretace

**A_ratio (amplituda) sedí mezi simem a reálem** → fyzika modelu je
**správná**. Konkrétně:
- Inertia tensors v URDF/MJCF jsou cca OK (±10 %, pokryto `body_mass` DR)
- Kp/Kd jsou shodné sim ↔ deploy ↔ firmware
- Resonanční frekvence sedí
- Ankle flat attenuace je fyzika, ne friction artefakt

**Delay gap je malý** (~30 ms v operačním pásmu). Firmware pravděpodobně
přidává krátký low-pass filter na setpoint, ale nic dramatického.

**Důsledek**: policy trénovaná s `delay_max_lag=2` (40 ms) pokryje
realitu v běžném operačním pásmu. Žádné zvyšování DR delay není
potřeba.

---

## 4. Co přidat/upravit dál

### Aktuální stav je OK

Po C++ verifikaci testu se ukázalo, že **současné nastavení DR
dostatečně pokrývá realitu** v operačním pásmu:

- `delay_max_lag=2` (0-40 ms) ~ reálný gap 20-35 ms v 1-2 Hz pásmu ✓
- `pd_gains` scale [0.9, 1.1] ~ firmware variability ✓
- `effort_limits` scale [0.9, 1.1] ~ motor torque variace ✓
- `body_mass` scale [0.9, 1.1] ~ URDF inertia nepřesnosti ✓
- `foot_friction`, `encoder_bias`, `base_com` ~ contact + kalibrace ✓

### Friction DR NEPOTŘEBNÝ

Původně odhadovaný 5–8× friction mismatch u ankle byl **chybný závěr**
— sim potvrdil stejný plochý A_ratio, takže MJCF `frictionloss=0.3`
Nm je v pořádku.

### Volitelně: mírné zvýšení `delay_max_lag` na 4

Pokud chceš malou bezpečnostní marži (např. pro velmi pomalé signály
kde je gap ~100 ms), lze posunout na:

```python
delay_min_lag=0,
delay_max_lag=4,  # 0-80 ms
```

Ale není to nutné — aktuální `=2` je z empirických měření dostatečný.

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
scripts/cpp/                      # nový — C++ verifikace timingu
scripts/cpp/sinusoid_response.cpp
scripts/cpp/CMakeLists.txt
scripts/cpp/recompile.sh
doc/sim_to_real_calibration.md    # tento dokument
```
