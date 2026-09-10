"""Checks of the analytical solutions against limits and textbook values."""

import numpy as np
import pytest

import heat_sphere as hs


# ---------------- steady state, no generation ----------------


def test_shell_resistance_matches_flat_wall_when_thin():
    """A thin shell should behave like a flat wall of area 4*pi*r^2."""
    r1, t, k = 1.0, 1e-4, 10.0
    R_sph = hs.shell_resistance(r1, r1 + t, k)
    R_flat = t / (k * 4 * np.pi * r1**2)
    assert R_sph == pytest.approx(R_flat, rel=1e-3)


def test_shell_profile_hits_both_boundaries():
    r = np.linspace(0.1, 0.3, 50)
    T = hs.shell_profile(r, 0.1, 0.3, 300.0, 100.0)
    assert T[0] == pytest.approx(300.0)
    assert T[-1] == pytest.approx(100.0)
    assert np.all(np.diff(T) < 0)


def test_composite_shell_energy_balance():
    """The same q must cross every resistance in the series."""
    layers = [hs.Layer("wall", 1.01, 15.0), hs.Layer("ins", 1.11, 0.06)]
    res = hs.composite_shell(1.0, layers, T_inner=150.0, T_inf=20.0, h_out=10.0)
    drops = np.diff(np.concatenate([res.T_nodes, [20.0]]))
    R_list = [R for _, R in res.resistances]
    for dT, R in zip(-drops, R_list):
        assert dT / R == pytest.approx(res.q, rel=1e-9)


def test_insulation_below_critical_radius_increases_loss():
    """Classic counter-intuitive result: r_c = 2k/h for a sphere."""
    k_ins, h = 0.1, 5.0
    r_c = hs.critical_radius_sphere(k_ins, h)
    assert r_c == pytest.approx(0.04)

    r_in = 0.005  # a small bead, well inside r_c
    def q_at(t):
        return hs.composite_shell(r_in, [hs.Layer("ins", r_in + t, k_ins)],
                                  100.0, 20.0, h).q
    assert q_at(0.005) > q_at(1e-6)          # below r_c: more insulation, more loss
    assert q_at(0.2) < q_at(r_c - r_in)      # well beyond r_c: loss falls again


# ---------------- steady state with generation ----------------


def test_generation_surface_from_overall_balance():
    R, qgen, h, T_inf = 0.01, 1e6, 200.0, 300.0
    T_s = hs.generation_surface_temperature(R, qgen, h, T_inf)
    q_total = qgen * (4 / 3) * np.pi * R**3
    assert h * 4 * np.pi * R**2 * (T_s - T_inf) == pytest.approx(q_total)


def test_generation_profile_is_flat_at_the_centre():
    R = 0.01
    r = np.linspace(0, R, 200)
    T = hs.generation_profile(r, R, 1.5, 2e6, 150.0, 350.0)
    # dT/dr = -qgen*r/(3k) vanishes at r=0; check the interior slope, since
    # np.gradient uses a one-sided formula at the first node.
    dTdr = np.gradient(T, r)[1:-1]
    assert abs(dTdr[0]) < 0.02 * abs(dTdr).max()
    assert np.all(dTdr <= 1e-9)
    assert T[0] == pytest.approx(hs.generation_center_temperature(R, 1.5, 2e6, 150.0, 350.0))


def test_internal_delta_t_scales_with_radius_squared():
    args = dict(k=1.5, qgen=2e6, h=150.0, T_inf=350.0)
    def internal(R):
        return (hs.generation_center_temperature(R=R, **args)
                - hs.generation_surface_temperature(R, args["qgen"], args["h"], args["T_inf"]))
    assert internal(0.01) / internal(0.005) == pytest.approx(4.0)


# ---------------- transient ----------------


@pytest.mark.parametrize(
    "Bi, lam1_expected",
    [   # Incropera Table 5.1, one-term coefficients for a sphere
        (0.01, 0.1730), (0.1, 0.5423), (1.0, 1.5708),
        (5.0, 2.5704), (10.0, 2.8363), (100.0, 3.1102),
    ],
)
def test_eigenvalues_match_textbook_table(Bi, lam1_expected):
    assert hs.biot_eigenvalues(Bi, 1)[0] == pytest.approx(lam1_expected, abs=5e-4)


@pytest.mark.parametrize(
    "Bi, C1_expected",
    [(0.1, 1.0298), (1.0, 1.2732), (10.0, 1.9249), (50.0, 1.9962), (100.0, 1.9990)],
)
def test_series_coefficients_match_textbook_table(Bi, C1_expected):
    lam1 = hs.biot_eigenvalues(Bi, 1)
    assert hs.series_coefficients(lam1)[0] == pytest.approx(C1_expected, abs=5e-4)


def test_eigenvalues_are_bracketed_and_increasing():
    lam = hs.biot_eigenvalues(3.0, 12)
    assert np.all(np.diff(lam) > 0)
    for n, l in enumerate(lam):
        assert n * np.pi < l < (n + 1) * np.pi


def test_initial_condition_recovered_at_small_time():
    theta = hs.transient_theta(np.linspace(0, 0.9, 20), 1e-6, 2.0, n_terms=200)
    assert np.allclose(theta, 1.0, atol=0.02)


def test_temperature_decays_monotonically_and_centre_lags_surface():
    Fo = np.linspace(0.01, 2.0, 60)
    centre = hs.transient_center_theta(Fo, 5.0)
    surface = np.array([hs.transient_theta(np.array([1.0]), F, 5.0)[0] for F in Fo])
    assert np.all(np.diff(centre) < 0)
    assert np.all(centre > surface)   # the centre is always the hottest point when cooling
    assert centre[-1] < 0.01


def test_lumped_agrees_with_series_at_small_biot():
    """Below Bi = 0.1 the lumped model should be within a couple of percent."""
    Fo = np.linspace(0.5, 20.0, 40)
    for Bi in (0.01, 0.05):
        rel = np.abs(hs.lumped_theta(Fo, Bi) - hs.transient_center_theta(Fo, Bi))
        assert rel.max() < 0.02


def test_lumped_fails_at_large_biot():
    Fo = np.linspace(0.2, 1.0, 30)
    err = np.abs(hs.lumped_theta(Fo, 20.0) - hs.transient_center_theta(Fo, 20.0))
    assert err.max() > 0.1


def test_one_term_matches_full_series_beyond_fo_02():
    """The textbook claim: past Fo = 0.2 one term is good to about 1 %."""
    Fo = np.linspace(0.2, 3.0, 40)
    for Bi in (0.5, 5.0, 50.0):
        exact = hs.transient_center_theta(Fo, Bi)
        rel = np.abs(hs.one_term_center_theta(Fo, Bi) - exact) / exact
        assert rel.max() < 0.01


def test_one_term_is_wrong_at_very_short_times():
    diff = abs(hs.one_term_center_theta(np.array([1e-3]), 50.0)[0]
               - hs.transient_center_theta(np.array([1e-3]), 50.0)[0])
    assert diff > 0.1


def test_energy_fraction_spans_zero_to_one():
    assert hs.energy_fraction(np.array([1e-8]), 1.0)[0] == pytest.approx(0.0, abs=2e-3)
    assert hs.energy_fraction(np.array([50.0]), 1.0)[0] == pytest.approx(1.0, abs=1e-6)
    Q = hs.energy_fraction(np.linspace(0.01, 3.0, 50), 2.0)
    assert np.all(np.diff(Q) > 0)


def test_time_to_reach_is_consistent_with_the_profile():
    Bi = 2.0
    Fo = hs.time_to_reach(0.3, Bi)
    assert hs.transient_theta(np.array([0.0]), Fo, Bi)[0] == pytest.approx(0.3, abs=1e-6)


def test_time_to_reach_returns_nan_for_impossible_targets():
    assert np.isnan(hs.time_to_reach(1.5, 1.0))
    assert np.isnan(hs.time_to_reach(0.0, 1.0))


def test_thermal_diffusivity():
    assert hs.thermal_diffusivity(14.9, 7900.0, 477.0) == pytest.approx(3.95e-6, rel=1e-2)


def test_biot_eigenvalues_rejects_zero_biot():
    with pytest.raises(ValueError):
        hs.biot_eigenvalues(0.0)
