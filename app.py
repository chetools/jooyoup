"""Heat Conduction in Spherical Coordinates - an interactive teaching app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import heat_sphere as hs

st.set_page_config(
    page_title="Spherical Heat Conduction",
    page_icon="🔴",
    layout="wide",
)

# --------------------------------------------------------------------------
# Material library - typical values a ChemE student meets in practice
# --------------------------------------------------------------------------
MATERIALS = {
    #  name              k [W/m-K]  rho [kg/m3]  cp [J/kg-K]
    "Stainless steel 304": (14.9, 7900.0, 477.0),
    "Carbon steel": (60.5, 7854.0, 434.0),
    "Copper": (401.0, 8933.0, 385.0),
    "Alumina catalyst pellet": (1.5, 1300.0, 1000.0),
    "Glass (borosilicate)": (1.14, 2225.0, 835.0),
    "Polymer (HDPE)": (0.44, 960.0, 1900.0),
    "Fireclay brick": (1.0, 2050.0, 960.0),
    "Calcium silicate insulation": (0.062, 190.0, 835.0),
    "Potato / food solid (~80% water)": (0.50, 1055.0, 3640.0),
    "Water (stagnant)": (0.613, 997.0, 4179.0),
}

FILM_COEFFICIENTS = {
    "Free convection, air": 10.0,
    "Forced convection, air": 80.0,
    "Free convection, water": 500.0,
    "Forced convection, water": 3000.0,
    "Boiling / quenching": 8000.0,
}

PLOT_LAYOUT = dict(
    template="simple_white",
    height=430,
    margin=dict(l=60, r=20, t=50, b=50),
    hovermode="x unified",
)


def material_picker(label: str, default: str, key: str):
    """Sidebar-style material selector that allows manual override."""
    name = st.selectbox(label, list(MATERIALS), index=list(MATERIALS).index(default), key=key)
    k0, rho0, cp0 = MATERIALS[name]
    c1, c2, c3 = st.columns(3)
    k = c1.number_input("k [W/m·K]", 0.01, 500.0, float(k0), key=f"{key}_k")
    rho = c2.number_input("ρ [kg/m³]", 100.0, 20000.0, float(rho0), key=f"{key}_rho")
    cp = c3.number_input("c_p [J/kg·K]", 100.0, 5000.0, float(cp0), key=f"{key}_cp")
    return k, rho, cp


# ==========================================================================
# Header
# ==========================================================================
st.title("🔴 Heat Conduction in Spherical Coordinates")
st.markdown(
    "An interactive companion for the spherical-geometry problems that show up in a "
    "chemical engineering heat-transfer course: **insulated storage vessels**, "
    "**catalyst pellets with reaction heat**, and **transient quenching or sterilisation**. "
    "Move the sliders and watch the physics respond."
)

tab_theory, tab_shell, tab_gen, tab_transient, tab_dimensionless = st.tabs(
    [
        "📖 Theory",
        "1️⃣ Steady state: insulated sphere",
        "2️⃣ Steady state: heat generation",
        "3️⃣ Transient: quenching & cooking",
        "4️⃣ Bi–Fo explorer",
    ]
)

# ==========================================================================
# THEORY
# ==========================================================================
with tab_theory:
    st.header("Where the equations come from")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("The governing equation")
        st.markdown(
            "Start from the energy balance on a differential shell of thickness $dr$. "
            "With spherical symmetry, temperature depends only on the radius $r$ and time $t$, "
            "so the general conduction equation collapses to"
        )
        st.latex(
            r"\rho c_p \frac{\partial T}{\partial t}"
            r"= \frac{1}{r^{2}}\frac{\partial}{\partial r}"
            r"\!\left(k\,r^{2}\frac{\partial T}{\partial r}\right) + \dot q"
        )
        st.markdown(
            "The $1/r^{2}$ factor is the whole story of spherical geometry: as heat flows "
            "outward the area $A = 4\\pi r^{2}$ grows, so the **flux** $q''$ falls even though "
            "the **rate** $q$ is constant. For constant $k$ this becomes"
        )
        st.latex(
            r"\frac{1}{\alpha}\frac{\partial T}{\partial t}"
            r"= \frac{1}{r^{2}}\frac{\partial}{\partial r}"
            r"\!\left(r^{2}\frac{\partial T}{\partial r}\right) + \frac{\dot q}{k},"
            r"\qquad \alpha = \frac{k}{\rho c_p}"
        )

        st.subheader("Case 1 — Steady state, no generation")
        st.markdown(
            "Set $\\partial T/\\partial t = 0$ and $\\dot q = 0$. Integrating twice gives a "
            "profile that is **linear in $1/r$**, not in $r$:"
        )
        st.latex(
            r"\frac{T(r)-T_1}{T_2-T_1} = \frac{1/r_1 - 1/r}{1/r_1 - 1/r_2},"
            r"\qquad q = \frac{4\pi k (T_1-T_2)}{1/r_1 - 1/r_2}"
        )
        st.markdown(
            "Written as a thermal resistance (the form you use for composite walls):"
        )
        st.latex(
            r"R_{\mathrm{cond,sph}} = \frac{1/r_1 - 1/r_2}{4\pi k},"
            r"\qquad R_{\mathrm{conv}} = \frac{1}{4\pi r^{2} h}"
        )
        st.info(
            "**Critical radius of insulation.** Because the outer *area* also grows with "
            "radius, adding insulation to a small sphere can make it lose *more* heat. "
            "Setting $dq/dr_{o} = 0$ gives $r_c = 2k_{\\mathrm{ins}}/h$ "
            "(compare $r_c = k/h$ for a cylinder). Explore this in Tab 1."
        )

        st.subheader("Case 2 — Steady state with uniform generation")
        st.markdown(
            "A catalyst pellet running an exothermic reaction, or a fuel sphere, generates "
            "$\\dot q$ [W/m³] internally. With the symmetry condition $dT/dr = 0$ at the "
            "centre and convection at the surface:"
        )
        st.latex(
            r"T(r) = T_s + \frac{\dot q\,(R^{2}-r^{2})}{6k},"
            r"\qquad T_s = T_\infty + \frac{\dot q R}{3h}"
        )
        st.markdown(
            "The centre–surface difference $\\Delta T = \\dot q R^{2}/6k$ scales with $R^2$: "
            "**halving the pellet diameter cuts the internal hot spot by a factor of four.** "
            "That is why industrial catalyst pellets are millimetres, not centimetres."
        )

        st.subheader("Case 3 — Transient cooling with convection")
        st.markdown(
            "A sphere initially at $T_i$ is dropped into a fluid at $T_\\infty$. "
            "Separation of variables gives an infinite series in the dimensionless "
            "temperature $\\theta = (T-T_\\infty)/(T_i-T_\\infty)$:"
        )
        st.latex(
            r"\theta(r^{*},\mathrm{Fo}) = \sum_{n=1}^{\infty} C_n"
            r"\exp\!\left(-\lambda_n^{2}\mathrm{Fo}\right)"
            r"\frac{\sin(\lambda_n r^{*})}{\lambda_n r^{*}}"
        )
        st.markdown("where the eigenvalues $\\lambda_n$ come from the surface energy balance,")
        st.latex(
            r"1 - \lambda_n \cot \lambda_n = \mathrm{Bi},"
            r"\qquad C_n = \frac{4\,[\sin\lambda_n - \lambda_n\cos\lambda_n]}"
            r"{2\lambda_n - \sin(2\lambda_n)}"
        )
        st.markdown(
            "For $\\mathrm{Fo} > 0.2$ every term but the first has died out — that single "
            "surviving term *is* the Heisler chart. This app sums 20 terms, so it stays "
            "correct at short times too."
        )

    with right:
        st.subheader("The two numbers that matter")
        st.latex(r"\mathrm{Bi} = \frac{hR}{k} = \frac{\text{internal resistance}}{\text{external resistance}}")
        st.latex(r"\mathrm{Fo} = \frac{\alpha t}{R^{2}} = \text{dimensionless time}")
        st.markdown(
            """
| Bi | Physical picture | Model to use |
|---|---|---|
| $< 0.1$ | sphere is nearly isothermal; the film controls | **Lumped capacitance** |
| $0.1 - 100$ | both resistances matter | **Full series solution** |
| $> 100$ | surface pinned at $T_\\infty$; conduction controls | Fixed-surface-temperature limit |
"""
        )
        st.caption(
            "Note the Biot number for the *lumped* criterion is conventionally built on "
            "$L_c = V/A = R/3$, so $\\mathrm{Bi}_{L_c} = \\mathrm{Bi}/3$. This app reports "
            "$\\mathrm{Bi} = hR/k$ throughout and uses "
            "$\\theta_{\\text{lumped}} = e^{-3\\,\\mathrm{Bi}\\,\\mathrm{Fo}}$, which is the "
            "same thing."
        )

        st.subheader("Boundary conditions")
        st.markdown(
            """
- **Symmetry at the centre:** $\\left.\\dfrac{\\partial T}{\\partial r}\\right|_{r=0}=0$
  (no heat source at a point of zero area).
- **Convection at the surface:** $-k\\left.\\dfrac{\\partial T}{\\partial r}\\right|_{r=R}
  = h\\,[T(R)-T_\\infty]$.
- **Initial condition:** $T(r,0)=T_i$.
"""
        )

        st.subheader("The mass-transfer twin")
        st.markdown(
            "The identical equation, with $k \\to D_{AB}$ and $\\dot q \\to -k_r C_A$, "
            "gives the **Thiele modulus** and **effectiveness factor** of a porous catalyst "
            "pellet. If you can solve the sphere here, you have already solved diffusion "
            "with reaction:"
        )
        st.latex(r"\eta = \frac{3}{\phi}\left[\frac{1}{\tanh \phi} - \frac{1}{\phi}\right]")

# ==========================================================================
# TAB 1 - COMPOSITE SHELL
# ==========================================================================
with tab_shell:
    st.header("Steady state: an insulated spherical vessel")
    st.markdown(
        "A spherical tank holds a hot (or cryogenic) fluid. How thick should the insulation "
        "be, and where does the temperature actually drop?"
    )

    controls, results = st.columns([1, 2])

    with controls:
        st.subheader("Geometry & materials")
        r_in = st.slider("Inner radius r₁ [m]", 0.05, 3.0, 1.0, 0.05)
        t_wall = st.slider("Wall thickness [mm]", 1.0, 100.0, 10.0, 1.0) / 1000.0
        t_ins = st.slider("Insulation thickness [mm]", 0.0, 500.0, 100.0, 5.0) / 1000.0

        k_wall = st.number_input("k, vessel wall [W/m·K]", 0.1, 500.0, 14.9)
        k_ins = st.number_input("k, insulation [W/m·K]", 0.01, 5.0, 0.062, format="%.3f")

        st.subheader("Driving force")
        T_in = st.slider("Inner fluid T₁ [°C]", -200.0, 400.0, 150.0, 5.0)
        T_inf = st.slider("Ambient T∞ [°C]", -50.0, 60.0, 20.0, 1.0)

        film = st.select_slider(
            "Outside film", options=list(FILM_COEFFICIENTS), value="Free convection, air"
        )
        h_out = st.number_input("h_out [W/m²·K]", 1.0, 20000.0, FILM_COEFFICIENTS[film])

    layers = [hs.Layer("Vessel wall", r_in + t_wall, k_wall)]
    if t_ins > 0:
        layers.append(hs.Layer("Insulation", r_in + t_wall + t_ins, k_ins))

    res = hs.composite_shell(r_in, layers, T_in, T_inf, h_out)
    r_outer = layers[-1].r_out

    # Bare (uninsulated) reference case for the savings metric.
    bare = hs.composite_shell(r_in, [hs.Layer("Vessel wall", r_in + t_wall, k_wall)],
                              T_in, T_inf, h_out)

    with results:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Heat loss q", f"{res.q:,.0f} W")
        m2.metric("vs. bare vessel", f"{res.q - bare.q:+,.0f} W",
                  delta=f"{100*(res.q-bare.q)/bare.q:+.1f} %", delta_color="inverse")
        T_surf_out = T_inf + res.q * hs.convective_resistance(r_outer, h_out)
        m3.metric("Outer surface T", f"{T_surf_out:.1f} °C")
        m4.metric("U (outer area)", f"{res.U_out:.2f} W/m²·K")

        # ---- temperature profile ----
        fig = go.Figure()
        colors = ["#1f77b4", "#d62728", "#2ca02c"]
        r_start = r_in
        for i, layer in enumerate(layers):
            rr = np.linspace(r_start, layer.r_out, 200)
            TT = hs.shell_profile(rr, r_start, layer.r_out, res.T_nodes[i], res.T_nodes[i + 1])
            fig.add_trace(
                go.Scatter(x=rr * 1000, y=TT, mode="lines", name=layer.name,
                           line=dict(width=3, color=colors[i % len(colors)]))
            )
            r_start = layer.r_out

        T_surf = T_surf_out
        fig.add_trace(
            go.Scatter(x=[r_outer * 1000, r_outer * 1150], y=[T_surf, T_inf],
                       mode="lines", name="Film (convection)",
                       line=dict(width=3, dash="dot", color="#7f7f7f"))
        )
        fig.add_hline(y=T_inf, line_dash="dash", line_color="lightgrey",
                      annotation_text="T∞")
        fig.update_layout(
            title="Temperature profile — note the curvature: T is linear in 1/r, not r",
            xaxis_title="radius [mm]", yaxis_title="Temperature [°C]", **PLOT_LAYOUT
        )
        st.plotly_chart(fig, width='stretch')

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Resistance network** — the biggest resistance controls.")
            df = pd.DataFrame(
                {"Resistance [K/W]": [R for _, R in res.resistances],
                 "Share of ΔT": [R / sum(x for _, x in res.resistances) for _, R in res.resistances]},
                index=[n for n, _ in res.resistances],
            )
            st.dataframe(
                df.style.format({"Resistance [K/W]": "{:.4f}", "Share of ΔT": "{:.1%}"}),
                width='stretch',
            )

        with c_right:
            r_crit = hs.critical_radius_sphere(k_ins, h_out)
            st.markdown("**Critical radius check**")
            st.latex(r"r_c = \frac{2k_{ins}}{h} = " + f"{r_crit*1000:.1f}" + r"\ \mathrm{mm}")
            if r_outer < r_crit:
                st.warning(
                    f"Outer radius ({r_outer*1000:.0f} mm) is **below** r_c "
                    f"({r_crit*1000:.0f} mm): adding insulation here still *increases* "
                    "heat loss. Relevant for small spheres with poor films — e.g. a "
                    "thermocouple bead, not a storage tank."
                )
            else:
                st.success(
                    f"Outer radius ({r_outer*1000:.0f} mm) is above r_c "
                    f"({r_crit*1000:.0f} mm): every extra millimetre of insulation reduces "
                    "the heat loss."
                )

    # ---- insulation sweep ----
    st.subheader("How does the loss respond to insulation thickness?")
    t_sweep = np.linspace(0.001, max(0.5, t_ins * 2), 200)
    q_sweep = [
        hs.composite_shell(
            r_in,
            [hs.Layer("w", r_in + t_wall, k_wall), hs.Layer("i", r_in + t_wall + t, k_ins)],
            T_in, T_inf, h_out,
        ).q
        for t in t_sweep
    ]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t_sweep * 1000, y=q_sweep, mode="lines",
                              line=dict(width=3, color="#d62728"), name="q"))
    fig2.add_vline(x=t_ins * 1000, line_dash="dash", line_color="#1f77b4",
                   annotation_text="current design")
    fig2.update_layout(title="Diminishing returns: heat loss vs. insulation thickness",
                       xaxis_title="insulation thickness [mm]",
                       yaxis_title="heat loss q [W]", **PLOT_LAYOUT)
    st.plotly_chart(fig2, width='stretch')
    st.caption(
        "The curve flattens once the insulation resistance dominates the network. Past that "
        "knee you are buying material, not performance — the real design question becomes "
        "capital cost vs. energy cost."
    )

# ==========================================================================
# TAB 2 - HEAT GENERATION
# ==========================================================================
with tab_gen:
    st.header("Steady state with internal generation: the catalyst pellet hot spot")
    st.markdown(
        "An exothermic reaction inside a porous pellet releases heat that must conduct out "
        "through the solid and then cross the gas film. The centre runs hotter than the bulk "
        "gas — sometimes hot enough to sinter the catalyst or run away."
    )

    controls, results = st.columns([1, 2])

    with controls:
        st.subheader("Pellet")
        d_p = st.slider("Pellet diameter [mm]", 0.5, 50.0, 10.0, 0.5)
        R = d_p / 2000.0
        k_p = st.slider("k, pellet [W/m·K]", 0.05, 20.0, 0.5, 0.05)

        st.subheader("Reaction & environment")
        qgen_M = st.slider("Volumetric generation q̇ [MW/m³]", 0.01, 20.0, 8.0, 0.01)
        qgen = qgen_M * 1e6
        T_inf_g = st.slider("Bulk gas T∞ [°C]", 0.0, 800.0, 350.0, 5.0)
        h_g = st.slider("Film coefficient h [W/m²·K]", 5.0, 2000.0, 300.0, 5.0)
        T_limit = st.slider("Max allowable pellet T [°C]", 100.0, 1200.0, 500.0, 10.0)

    T_s = hs.generation_surface_temperature(R, qgen, h_g, T_inf_g)
    T_c = hs.generation_center_temperature(R, k_p, qgen, h_g, T_inf_g)
    q_total = qgen * (4.0 / 3.0) * np.pi * R**3

    with results:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Centre T", f"{T_c:.0f} °C",
                  delta=f"{T_c - T_limit:+.0f} °C vs limit",
                  delta_color="inverse")
        m2.metric("Surface T", f"{T_s:.0f} °C")
        m3.metric("Internal ΔT (q̇R²/6k)", f"{T_c - T_s:.1f} °C")
        m4.metric("Film ΔT (q̇R/3h)", f"{T_s - T_inf_g:.1f} °C")

        if T_c > T_limit:
            st.error(
                f"**Centre exceeds the limit by {T_c-T_limit:.0f} °C.** Options: shrink the "
                "pellet (ΔT_internal ∝ R²), dilute the catalyst to cut q̇, raise h, or lower "
                "the bulk temperature."
            )
        else:
            st.success(f"Centre temperature is {T_limit - T_c:.0f} °C below the limit.")

        rr = np.linspace(0, R, 300)
        TT = hs.generation_profile(rr, R, k_p, qgen, h_g, T_inf_g)
        fig = go.Figure()
        # Mirror the profile so the parabola reads as a diameter cut through the pellet.
        fig.add_trace(go.Scatter(
            x=np.concatenate([-rr[::-1], rr]) * 1000,
            y=np.concatenate([TT[::-1], TT]),
            mode="lines", name="T(r) in solid", line=dict(width=3, color="#d62728"),
            fill="tozeroy", fillcolor="rgba(214,39,40,0.06)"))
        fig.add_hline(y=T_inf_g, line_dash="dash", line_color="#1f77b4",
                      annotation_text="T∞ (bulk gas)")
        span = max(T_c - T_inf_g, 1.0)
        y_lo, y_hi = T_inf_g - 0.12 * span, T_c + 0.25 * span
        if y_lo <= T_limit <= y_hi:
            fig.add_hline(y=T_limit, line_dash="dot", line_color="#ff7f0e",
                          annotation_text="max allowable")
        fig.add_annotation(x=0, y=T_c, text=f"hot spot {T_c:.0f} °C",
                           showarrow=True, arrowhead=2, ay=-35)
        fig.update_layout(
            title="Parabolic profile across the pellet diameter",
            xaxis_title="position across the pellet [mm]",
            yaxis_title="Temperature [°C]",
            yaxis_range=[y_lo, y_hi],
            **PLOT_LAYOUT)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Design sweep: pellet size is the strongest lever")
    d_sweep = np.linspace(0.5, 50.0, 200)
    R_sweep = d_sweep / 2000.0
    Tc_sweep = [hs.generation_center_temperature(Rs, k_p, qgen, h_g, T_inf_g) for Rs in R_sweep]
    Ts_sweep = [hs.generation_surface_temperature(Rs, qgen, h_g, T_inf_g) for Rs in R_sweep]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d_sweep, y=Tc_sweep, mode="lines", name="centre T",
                              line=dict(width=3, color="#d62728")))
    fig2.add_trace(go.Scatter(x=d_sweep, y=Ts_sweep, mode="lines", name="surface T",
                              line=dict(width=3, color="#1f77b4")))
    fig2.add_hline(y=T_limit, line_dash="dot", line_color="#ff7f0e",
                   annotation_text="max allowable")
    fig2.add_vline(x=d_p, line_dash="dash", line_color="grey",
                   annotation_text="current pellet")
    fig2.update_layout(title="Centre and surface temperature vs. pellet diameter",
                       xaxis_title="pellet diameter [mm]",
                       yaxis_title="Temperature [°C]", **PLOT_LAYOUT)
    st.plotly_chart(fig2, width='stretch')
    st.caption(
        f"Total heat released per pellet: **{q_total*1000:.2f} mW**. The centre curve is a "
        "parabola in diameter (ΔT ∝ R²) while the surface curve is only linear (ΔT ∝ R) — "
        "which is why crushing the pellet fixes an internal hot spot but barely touches a "
        "film-limited one."
    )

# ==========================================================================
# TAB 3 - TRANSIENT
# ==========================================================================
with tab_transient:
    st.header("Transient conduction: quenching, chilling, sterilising")
    st.markdown(
        "A sphere at a uniform $T_i$ is plunged into a fluid at $T_\\infty$. "
        "How long until the **centre** — always the last point to respond — reaches target?"
    )

    controls, results = st.columns([1, 2])

    with controls:
        preset = st.radio(
            "Scenario",
            ["Quenching a steel ball", "Cooking / sterilising a food sphere", "Custom"],
            index=0,
        )
        if preset == "Quenching a steel ball":
            d0, mat0, Ti0, Tinf0, h0 = 50.0, "Stainless steel 304", 800.0, 25.0, 3000.0
        elif preset == "Cooking / sterilising a food sphere":
            d0, mat0, Ti0, Tinf0, h0 = 60.0, "Potato / food solid (~80% water)", 20.0, 100.0, 500.0
        else:
            d0, mat0, Ti0, Tinf0, h0 = 40.0, "Glass (borosilicate)", 200.0, 20.0, 100.0

        st.subheader("Sphere")
        diam = st.slider("Diameter [mm]", 1.0, 300.0, d0, 1.0)
        R = diam / 2000.0
        k, rho, cp = material_picker("Material", mat0, key=f"trans_{preset}")
        alpha = hs.thermal_diffusivity(k, rho, cp)

        st.subheader("Quench bath")
        Ti = st.slider("Initial temperature Tᵢ [°C]", -50.0, 1000.0, Ti0, 5.0)
        T_inf_t = st.slider("Fluid temperature T∞ [°C]", -50.0, 300.0, Tinf0, 1.0)
        h_t = st.slider("h [W/m²·K]", 1.0, 20000.0, h0, 1.0)
        T_target = st.slider("Target centre temperature [°C]", -50.0, 1000.0,
                             float(T_inf_t + 0.25 * (Ti0 - Tinf0)), 5.0)

    Bi = h_t * R / k
    n_terms = 20

    theta_target = (T_target - T_inf_t) / (Ti - T_inf_t) if Ti != T_inf_t else float("nan")
    Fo_target = hs.time_to_reach(theta_target, Bi) if 0 < theta_target < 1 else float("nan")
    t_target = Fo_target * R**2 / alpha if np.isfinite(Fo_target) else float("nan")

    with results:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Biot number  hR/k", f"{Bi:.2f}")
        m2.metric("α = k/ρc_p", f"{alpha*1e6:.3f} mm²/s")
        m3.metric("Time to target (centre)",
                  f"{t_target:,.0f} s" if np.isfinite(t_target) else "—",
                  delta=f"{t_target/60:.1f} min" if np.isfinite(t_target) else None,
                  delta_color="off")
        m4.metric("Fo at target", f"{Fo_target:.3f}" if np.isfinite(Fo_target) else "—")

        if Bi < 0.1:
            st.success(
                f"**Bi = {Bi:.3f} < 0.1** — the sphere is essentially isothermal. The film "
                "controls, and the one-line lumped model is all you need."
            )
        elif Bi > 100:
            st.warning(
                f"**Bi = {Bi:.0f} > 100** — the surface is effectively pinned at T∞. "
                "Internal conduction is the whole resistance; the centre lags badly."
            )
        else:
            st.info(
                f"**Bi = {Bi:.2f}** — internal and external resistances are comparable. "
                "Neither shortcut applies; you need the series solution."
            )

        # ---- time axis ----
        Fo_end = max(1.2, (Fo_target * 1.8 if np.isfinite(Fo_target) else 1.2))
        Fo_grid = np.linspace(1e-4, Fo_end, 400)
        t_grid = Fo_grid * R**2 / alpha

        theta_c = hs.transient_center_theta(Fo_grid, Bi, n_terms)
        theta_s = np.array([hs.transient_theta(np.array([1.0]), F, Bi, n_terms)[0]
                            for F in Fo_grid])
        theta_l = hs.lumped_theta(Fo_grid, Bi)
        to_T = lambda th: T_inf_t + th * (Ti - T_inf_t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_grid, y=to_T(theta_c), mode="lines", name="centre (r=0)",
                                 line=dict(width=3, color="#d62728")))
        fig.add_trace(go.Scatter(x=t_grid, y=to_T(theta_s), mode="lines", name="surface (r=R)",
                                 line=dict(width=3, color="#1f77b4")))
        fig.add_trace(go.Scatter(x=t_grid, y=to_T(theta_l), mode="lines",
                                 name="lumped model (approx.)",
                                 line=dict(width=2, dash="dash", color="#7f7f7f")))
        fig.add_hline(y=T_inf_t, line_dash="dot", line_color="lightgrey",
                      annotation_text="T∞")
        if np.isfinite(t_target):
            fig.add_vline(x=t_target, line_dash="dash", line_color="#2ca02c",
                          annotation_text=f"target reached: {t_target:,.0f} s")
        fig.update_layout(title="Temperature history — the centre is always the laggard",
                          xaxis_title="time [s]", yaxis_title="Temperature [°C]", **PLOT_LAYOUT)
        st.plotly_chart(fig, width='stretch')

        err = 100.0 * abs(to_T(theta_l) - to_T(theta_c)) / max(abs(Ti - T_inf_t), 1e-9)
        st.caption(
            f"Peak disagreement between the lumped model and the true centre temperature: "
            f"**{err.max():.1f} %** of the initial driving force. That gap is exactly what "
            "the Bi < 0.1 criterion is protecting you from."
        )

    # ---- profile snapshots ----
    st.subheader("Snapshots of the internal profile")
    slider_col, plot_col = st.columns([1, 2])
    with slider_col:
        t_snap = st.slider(
            "Time [s]", 0.0, float(t_grid[-1]),
            float(t_target if np.isfinite(t_target) else t_grid[-1] / 3), step=float(t_grid[-1]/200)
        )
        Fo_snap = alpha * t_snap / R**2
        st.metric("Fourier number", f"{Fo_snap:.3f}")
        if Fo_snap < 0.2:
            st.caption("⚠️ Fo < 0.2 — the one-term/Heisler approximation is not yet valid here; "
                       "this plot sums 20 terms, so it stays accurate.")
        Q_frac = float(hs.energy_fraction(np.array([max(Fo_snap, 1e-6)]), Bi, n_terms)[0])
        mass = rho * (4.0 / 3.0) * np.pi * R**3
        Q_max = mass * cp * abs(Ti - T_inf_t)
        st.metric("Energy exchanged Q/Q_max", f"{Q_frac:.1%}")
        st.metric("Q so far", f"{Q_frac*Q_max/1000:,.2f} kJ")

    with plot_col:
        r_star = np.linspace(0, 1, 200)
        fig3 = go.Figure()
        for frac, alpha_line in [(0.25, 0.35), (0.5, 0.55), (1.0, 1.0)]:
            F = max(Fo_snap * frac, 1e-6)
            th = hs.transient_theta(r_star, F, Bi, n_terms)
            fig3.add_trace(go.Scatter(
                x=np.concatenate([-r_star[::-1], r_star]) * R * 1000,
                y=to_T(np.concatenate([th[::-1], th])),
                mode="lines", name=f"t = {t_snap*frac:,.0f} s  (Fo={F:.3f})",
                line=dict(width=3, color=f"rgba(214,39,40,{alpha_line})")))
        fig3.add_hline(y=Ti, line_dash="dot", line_color="lightgrey", annotation_text="Tᵢ")
        fig3.add_hline(y=T_inf_t, line_dash="dot", line_color="#1f77b4", annotation_text="T∞")
        fig3.update_layout(
            title="Cooling front eating inward from the surface",
            xaxis_title="position across the sphere [mm]",
            yaxis_title="Temperature [°C]", **PLOT_LAYOUT)
        st.plotly_chart(fig3, width='stretch')

# ==========================================================================
# TAB 4 - DIMENSIONLESS EXPLORER
# ==========================================================================
with tab_dimensionless:
    st.header("The Bi–Fo picture behind the Heisler charts")
    st.markdown(
        "Strip away the units and every sphere in the world collapses onto one family of "
        "curves parameterised by the Biot number. This is the chart in the back of your "
        "textbook — except you can drag it."
    )

    c1, c2, c3 = st.columns(3)
    Bi_focus = c1.slider("Biot number (highlighted curve)", 0.01, 100.0, 1.0, 0.01)
    Fo_max = c2.slider("Fourier number range", 0.2, 5.0, 1.5, 0.1)
    show_one_term = c3.checkbox("Overlay one-term (Heisler) approximation", value=True)

    Fo_grid = np.linspace(1e-3, Fo_max, 400)

    fig = go.Figure()
    for Bi_c in [0.05, 0.2, 1.0, 5.0, 50.0]:
        fig.add_trace(go.Scatter(
            x=Fo_grid, y=hs.transient_center_theta(Fo_grid, Bi_c),
            mode="lines", name=f"Bi = {Bi_c:g}",
            line=dict(width=1.6, color="rgba(120,120,120,0.55)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=Fo_grid, y=hs.transient_center_theta(Fo_grid, Bi_focus),
        mode="lines", name=f"Bi = {Bi_focus:g} (selected)",
        line=dict(width=4, color="#d62728")))
    if show_one_term:
        fig.add_trace(go.Scatter(
            x=Fo_grid, y=hs.one_term_center_theta(Fo_grid, Bi_focus),
            mode="lines", name="one-term approximation",
            line=dict(width=2, dash="dash", color="#1f77b4")))
    fig.add_vrect(x0=0, x1=0.2, fillcolor="orange", opacity=0.10, line_width=0,
                  annotation_text="Fo < 0.2:<br>one term is<br>not enough",
                  annotation_position="bottom right",
                  annotation_font_size=11, annotation_font_color="#8a5a00")
    fig.update_layout(title="Centre temperature θ₀ vs. Fourier number",
                      xaxis_title="Fo = αt/R²", yaxis_title="θ₀ = (T₀−T∞)/(Tᵢ−T∞)",
                      yaxis_type="log", yaxis_range=[-2, 0.05], **PLOT_LAYOUT)
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns(2)
    with left:
        lam = hs.biot_eigenvalues(Bi_focus, 6)
        C = hs.series_coefficients(lam)
        st.markdown(f"**Eigenvalues for Bi = {Bi_focus:g}**")
        st.dataframe(
            pd.DataFrame({"λₙ": lam, "Cₙ": C, "decay exp(−λₙ²·0.2)": np.exp(-lam**2 * 0.2)},
                         index=[f"n = {i+1}" for i in range(len(lam))])
            .style.format("{:.4f}"),
            width='stretch')
        st.caption(
            "By Fo = 0.2 the second term has already decayed to a fraction of a percent — "
            "that is the whole justification for the one-term Heisler chart."
        )

    with right:
        st.markdown("**Energy released, Q/Q_max**")
        figE = go.Figure()
        for Bi_c in [0.05, 0.2, 1.0, 5.0, 50.0]:
            figE.add_trace(go.Scatter(
                x=Fo_grid, y=hs.energy_fraction(Fo_grid, Bi_c), mode="lines",
                name=f"Bi = {Bi_c:g}", line=dict(width=1.6, color="rgba(120,120,120,0.55)"),
                hoverinfo="skip"))
        figE.add_trace(go.Scatter(
            x=Fo_grid, y=hs.energy_fraction(Fo_grid, Bi_focus), mode="lines",
            name=f"Bi = {Bi_focus:g}", line=dict(width=4, color="#2ca02c")))
        figE.update_layout(xaxis_title="Fo = αt/R²", yaxis_title="Q/Q_max",
                           yaxis_range=[0, 1.02], **PLOT_LAYOUT)
        st.plotly_chart(figE, width='stretch')
        st.caption(
            "Useful for sizing a quench tank or a chiller duty: it tells you what fraction of "
            "the sphere's total enthalpy change has actually happened."
        )

st.divider()
st.caption(
    "Series solution truncated at 20 eigenvalues (accurate for Fo ≳ 10⁻³). "
    "Constant properties, uniform initial temperature, and a constant film coefficient are "
    "assumed throughout — the standard assumptions behind the Heisler/Gröber charts."
)
