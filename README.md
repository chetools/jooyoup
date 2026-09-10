# Heat Conduction in Spherical Coordinates

An interactive Streamlit app for teaching the spherical-geometry heat transfer
problems that a chemical engineering student actually meets: **insulated storage
vessels**, **catalyst pellets with reaction heat**, and **transient quenching or
sterilisation**.

Every result comes from a closed-form analytical solution — no finite differences —
so the sliders respond instantly and the numbers can be checked against a textbook.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## What's inside

| Tab | Problem | Key result |
|---|---|---|
| 📖 Theory | Derivation from the shell energy balance | The governing PDE, all three cases, Bi and Fo |
| 1️⃣ Insulated sphere | Steady state, composite wall, convection | $q = \Delta T / \sum R$, critical radius $r_c = 2k/h$ |
| 2️⃣ Heat generation | Catalyst pellet / fuel sphere | $T_c - T_s = \dot q R^2/6k$ — the hot spot |
| 3️⃣ Transient | Quenching a steel ball, cooking a potato | Full eigenvalue series vs. the lumped model |
| 4️⃣ Bi–Fo explorer | The Heisler charts, live | Where the one-term approximation is valid |

### Teaching points the app is built around

- **Temperature is linear in $1/r$, not $r$.** The curvature in the profile plot is
  the geometry showing itself.
- **Critical radius of insulation.** For a small enough sphere, adding insulation
  *increases* heat loss. Tab 1 flags when your design is on the wrong side of
  $r_c = 2k_{\text{ins}}/h$.
- **Hot spots scale with $R^2$.** Halving a catalyst pellet cuts the internal
  $\Delta T$ by four but the film $\Delta T$ only by two — which is why pellet size
  is the strongest design lever against thermal runaway.
- **Where the lumped model breaks.** Tab 3 plots the lumped prediction against the
  true centre temperature and reports the peak error, making the $\mathrm{Bi} < 0.1$
  rule concrete rather than memorised.
- **Fo > 0.2 and the Heisler charts.** Tab 4 shows the higher-order terms decaying,
  so the one-term approximation stops being a rule and becomes an observation.

## The physics

The transient solution is the standard separation-of-variables series

$$\theta(r^*,\mathrm{Fo}) = \sum_{n=1}^{\infty} C_n \exp(-\lambda_n^2 \mathrm{Fo})
\frac{\sin(\lambda_n r^*)}{\lambda_n r^*}$$

with eigenvalues from $1 - \lambda_n \cot \lambda_n = \mathrm{Bi}$. The app sums
**20 terms**, so unlike the one-term charts it stays accurate at short times
($\mathrm{Fo} \lesssim 0.2$) too. Roots are found with a bracketed solver on
$((n-1)\pi, n\pi)$, where the function is guaranteed to change sign.

### Assumptions

Constant properties, uniform initial temperature, constant film coefficient, and
spherical symmetry — the same assumptions that sit behind the Heisler/Gröber charts.

## Tests

```bash
pytest tests/ -q
```

30 tests covering limiting behaviour and textbook values:

- Eigenvalues $\lambda_1$ and coefficients $C_1$ against Incropera Table 5.1
  (Bi = 0.01 … 100)
- Thin spherical shell → flat-wall resistance
- Series solution → lumped model as $\mathrm{Bi} \to 0$, and its failure at large Bi
- One-term approximation within 1 % of the full series for $\mathrm{Fo} > 0.2$,
  and demonstrably wrong below it
- Energy balance closure on the composite wall and the generating sphere
- $Q/Q_{max}$ spanning 0 → 1 monotonically

## Layout

```
app.py                     Streamlit UI (5 tabs)
heat_sphere.py             Analytical solutions — no Streamlit dependency
tests/test_heat_sphere.py  Verification against limits and textbook tables
```

`heat_sphere.py` is deliberately importable on its own, so the same functions can be
used from a notebook or a problem set.
