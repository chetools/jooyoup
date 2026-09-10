"""Analytical solutions for heat conduction in spherical geometry.

The functions here are deliberately free of any Streamlit dependency so that
they can be unit-tested and reused from a notebook.

Symbols
-------
k     thermal conductivity                     [W/m-K]
rho   density                                  [kg/m^3]
cp    specific heat capacity                   [J/kg-K]
alpha thermal diffusivity  k/(rho*cp)          [m^2/s]
h     convective heat transfer coefficient     [W/m^2-K]
qgen  volumetric heat generation rate          [W/m^3]
Bi    Biot number      h*R/k                   [-]
Fo    Fourier number   alpha*t/R^2             [-]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# 1. Steady state through a spherical shell (hollow sphere / insulated vessel)
# --------------------------------------------------------------------------


def shell_resistance(r_in: float, r_out: float, k: float) -> float:
    """Conductive resistance of a spherical shell, R = (1/r1 - 1/r2)/(4*pi*k) [K/W]."""
    if r_out <= r_in:
        raise ValueError("r_out must be larger than r_in")
    return (1.0 / r_in - 1.0 / r_out) / (4.0 * np.pi * k)


def convective_resistance(r: float, h: float) -> float:
    """Film resistance on a spherical surface, R = 1/(h*4*pi*r^2) [K/W]."""
    return 1.0 / (h * 4.0 * np.pi * r**2)


@dataclass
class Layer:
    """One material layer of a composite spherical wall."""

    name: str
    r_out: float  # outer radius [m]
    k: float  # conductivity [W/m-K]


@dataclass
class CompositeShellResult:
    q: float  # heat rate through the wall [W]
    resistances: list  # [(label, R [K/W]), ...]
    r_nodes: np.ndarray  # radii at each material interface [m]
    T_nodes: np.ndarray  # temperature at each interface [K or degC]
    U_out: float  # overall coefficient on the outer area [W/m^2-K]


def composite_shell(
    r_inner: float,
    layers: list,
    T_inner: float,
    T_inf: float,
    h_out: float,
    h_in: float | None = None,
) -> CompositeShellResult:
    """Series-resistance solution for a multi-layer spherical wall.

    ``T_inner`` is the inner fluid temperature when ``h_in`` is given, otherwise
    it is imposed directly on the inner wall surface.
    """
    resistances = []
    if h_in is not None:
        resistances.append(("Inner film", convective_resistance(r_inner, h_in)))

    r_prev = r_inner
    for layer in layers:
        resistances.append((layer.name, shell_resistance(r_prev, layer.r_out, layer.k)))
        r_prev = layer.r_out

    r_outer = r_prev
    resistances.append(("Outer film", convective_resistance(r_outer, h_out)))

    R_total = sum(R for _, R in resistances)
    q = (T_inner - T_inf) / R_total

    # March the temperature drop through the series of resistances.
    r_nodes = [r_inner] + [layer.r_out for layer in layers]
    T_nodes = []
    T = T_inner
    start = 1 if h_in is not None else 0
    if h_in is not None:
        T = T_inner - q * resistances[0][1]  # inner wall surface
    T_nodes.append(T)
    for _, R in resistances[start:-1]:
        T = T - q * R
        T_nodes.append(T)

    A_out = 4.0 * np.pi * r_outer**2
    return CompositeShellResult(
        q=q,
        resistances=resistances,
        r_nodes=np.asarray(r_nodes),
        T_nodes=np.asarray(T_nodes),
        U_out=1.0 / (R_total * A_out),
    )


def shell_profile(r: np.ndarray, r1: float, r2: float, T1: float, T2: float) -> np.ndarray:
    """Temperature profile inside a single shell: T varies linearly in 1/r."""
    return T1 + (T1 - T2) * (1.0 / r - 1.0 / r1) / (1.0 / r1 - 1.0 / r2)


def critical_radius_sphere(k_ins: float, h: float) -> float:
    """Critical radius of insulation for a sphere, r_c = 2k/h [m].

    Below r_c, adding insulation *increases* the heat loss because the growth of
    the outer area beats the added conduction resistance.
    """
    return 2.0 * k_ins / h


# --------------------------------------------------------------------------
# 2. Steady state with uniform generation (catalyst pellet, fuel sphere)
# --------------------------------------------------------------------------


def generation_profile(
    r: np.ndarray, R: float, k: float, qgen: float, h: float, T_inf: float
) -> np.ndarray:
    """T(r) in a solid sphere with uniform generation cooled by convection.

    T(r) = T_s + qgen*(R^2 - r^2)/(6k),   T_s = T_inf + qgen*R/(3h)
    """
    T_s = generation_surface_temperature(R, qgen, h, T_inf)
    return T_s + qgen * (R**2 - np.asarray(r) ** 2) / (6.0 * k)


def generation_surface_temperature(R: float, qgen: float, h: float, T_inf: float) -> float:
    """Surface temperature from an energy balance on the whole sphere."""
    return T_inf + qgen * R / (3.0 * h)


def generation_center_temperature(
    R: float, k: float, qgen: float, h: float, T_inf: float
) -> float:
    """Hot-spot (centre) temperature of a generating sphere."""
    return generation_surface_temperature(R, qgen, h, T_inf) + qgen * R**2 / (6.0 * k)


# --------------------------------------------------------------------------
# 3. Transient conduction in a sphere with surface convection
# --------------------------------------------------------------------------


def biot_eigenvalues(Bi: float, n_terms: int = 20) -> np.ndarray:
    """Roots of the sphere characteristic equation ``1 - lam*cot(lam) = Bi``.

    The n-th root lies in ((n-1)*pi, n*pi); the function is monotonic there and
    changes sign across the interval, so a bracketed solver is safe.
    """
    if Bi <= 0:
        raise ValueError("Bi must be positive (use the lumped solution for Bi -> 0)")

    def f(lam: float) -> float:
        return 1.0 - lam / np.tan(lam) - Bi

    roots = []
    eps = 1e-9
    for n in range(n_terms):
        lo = n * np.pi + eps
        hi = (n + 1) * np.pi - eps
        # Guard against the finite-eps bracket accidentally losing the sign change.
        while f(lo) > 0 and lo < hi:
            lo += eps
            eps *= 10
        roots.append(brentq(f, lo, hi, xtol=1e-12, rtol=1e-14))
    return np.asarray(roots)


def series_coefficients(lam: np.ndarray) -> np.ndarray:
    """C_n = 4*(sin(lam) - lam*cos(lam)) / (2*lam - sin(2*lam))."""
    return 4.0 * (np.sin(lam) - lam * np.cos(lam)) / (2.0 * lam - np.sin(2.0 * lam))


def transient_theta(
    r_star: np.ndarray, Fo: float, Bi: float, n_terms: int = 20
) -> np.ndarray:
    """Dimensionless temperature theta = (T - T_inf)/(T_i - T_inf).

    theta(r*, Fo) = sum C_n * exp(-lam_n^2 * Fo) * sin(lam_n r*)/(lam_n r*)
    """
    lam = biot_eigenvalues(Bi, n_terms)
    C = series_coefficients(lam)
    r_star = np.atleast_1d(np.asarray(r_star, dtype=float))

    theta = np.zeros_like(r_star)
    for lam_n, C_n in zip(lam, C):
        # sin(x)/x -> 1 at the centre; np.sinc(x/pi) is the numerically safe form.
        shape = np.sinc(lam_n * r_star / np.pi)
        theta += C_n * np.exp(-(lam_n**2) * Fo) * shape
    return theta


def transient_center_theta(Fo: np.ndarray, Bi: float, n_terms: int = 20) -> np.ndarray:
    """Centre temperature history theta_0(Fo)."""
    lam = biot_eigenvalues(Bi, n_terms)
    C = series_coefficients(lam)
    Fo = np.atleast_1d(np.asarray(Fo, dtype=float))
    return np.sum(
        C[None, :] * np.exp(-(lam[None, :] ** 2) * Fo[:, None]), axis=1
    )


def one_term_center_theta(Fo: np.ndarray, Bi: float) -> np.ndarray:
    """One-term (Heisler chart) approximation, valid for Fo > 0.2."""
    lam1 = biot_eigenvalues(Bi, 1)[0]
    C1 = series_coefficients(np.array([lam1]))[0]
    return C1 * np.exp(-(lam1**2) * np.asarray(Fo, dtype=float))


def lumped_theta(Fo: np.ndarray, Bi: float) -> np.ndarray:
    """Lumped-capacitance model: theta = exp(-3*Bi*Fo) (since V/A = R/3)."""
    return np.exp(-3.0 * Bi * np.asarray(Fo, dtype=float))


def energy_fraction(Fo: np.ndarray, Bi: float, n_terms: int = 20) -> np.ndarray:
    """Q/Q_max, the fraction of the maximum extractable energy already exchanged.

    Q/Q_max = 1 - 3*sum[ C_n exp(-lam_n^2 Fo) (sin lam_n - lam_n cos lam_n)/lam_n^3 ]
    """
    lam = biot_eigenvalues(Bi, n_terms)
    C = series_coefficients(lam)
    Fo = np.atleast_1d(np.asarray(Fo, dtype=float))
    weight = (np.sin(lam) - lam * np.cos(lam)) / lam**3
    stored = np.sum(
        3.0 * C[None, :] * weight[None, :] * np.exp(-(lam[None, :] ** 2) * Fo[:, None]),
        axis=1,
    )
    return 1.0 - stored


def time_to_reach(
    theta_target: float, Bi: float, r_star: float = 0.0, n_terms: int = 20
) -> float:
    """Fourier number at which ``theta`` at ``r_star`` first falls to the target.

    Returns ``nan`` if the target is never reached (theta -> 0 monotonically, so
    this only happens for theta_target <= 0).
    """
    if not 0.0 < theta_target < 1.0:
        return float("nan")

    def g(Fo: float) -> float:
        return float(transient_theta(np.array([r_star]), Fo, Bi, n_terms)[0]) - theta_target

    lo, hi = 1e-6, 1.0
    for _ in range(60):
        if g(hi) < 0:
            break
        hi *= 2.0
    else:
        return float("nan")
    return brentq(g, lo, hi, xtol=1e-10)


def thermal_diffusivity(k: float, rho: float, cp: float) -> float:
    """alpha = k/(rho*cp) [m^2/s]."""
    return k / (rho * cp)
