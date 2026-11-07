"""
Fitting and conversion functions used by analysis.py (and tested in tests/).

Contents
- Ordinary least-squares fits of v = H0*r (free intercept and through the origin),
  jackknife and bootstrap.
- Inverse-variance weighted fits with an iteratively reweighted error model
  sigma_v^2 = (H0*f*r)^2 + sigma_pec^2, and the peculiar-velocity floor
  sigma_pec estimated from the data (chi2/dof = 1, or maximum likelihood).
- Errors-in-variables fit through the origin (orthogonal distance regression,
  written out as the equivalent effective-variance estimator).
- Low-redshift cosmology: conversion of luminosity and angular-diameter
  distances to comoving distance, and of the measured cz to H0*D_C.
- Hubble's (1929) model with the solar motion, and the reflex of the Sun's
  Galactocentric motion along a line of sight.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize, minimize_scalar

C_KMS = 299792.458   # speed of light, km/s
OMEGA_M = 0.315      # matter density (Planck 2020), flat LambdaCDM, as in Pesce et al. (2020)


# ---------- ordinary least squares ----------

def fit_free(r, v):
    """Free-intercept OLS fit v = H0*r + b. Returns H0, se(H0), intercept, R2."""
    n = len(r)
    A = np.vstack([r, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    H0, b = coef
    resid = v - A @ coef
    sigma2 = np.sum(resid**2) / (n - 2)
    cov = sigma2 * np.linalg.inv(A.T @ A)
    r2 = 1 - np.sum(resid**2) / np.sum((v - v.mean())**2)
    return H0, np.sqrt(cov[0, 0]), b, r2


def fit_origin(r, v):
    """Origin-forced OLS fit v = H0*r. Returns H0, se(H0), R2 (about the mean)."""
    H0 = np.sum(r * v) / np.sum(r * r)
    resid = v - H0 * r
    sigma2 = np.sum(resid**2) / (len(r) - 1)
    r2 = 1 - np.sum(resid**2) / np.sum((v - v.mean())**2)
    return H0, np.sqrt(sigma2 / np.sum(r * r)), r2


def jackknife_origin(r, v):
    """Leave-one-out estimates of the origin-forced H0."""
    idx = np.arange(len(r))
    return np.array([fit_origin(r[idx != i], v[idx != i])[0] for i in idx])


def jackknife_se(jk):
    n = len(jk)
    return np.sqrt((n - 1) / n * np.sum((jk - jk.mean())**2))


def bootstrap_origin(r, v, rng, n_boot=20000):
    n = len(r)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        out[b] = fit_origin(r[s], v[s])[0]
    return out


def effective_sample_size(r):
    """(sum r^2)^2 / sum r^4: the number of equally weighted points that would
    give the origin-forced OLS fit the same precision, given its r^2 weights."""
    w = r**2
    return w.sum()**2 / np.sum(w**2)


# ---------- weighted fits ----------

def fit_origin_weighted(r, v, w):
    """Weighted fit v = H0*r with weights w = 1/sigma_v^2.
    se is scaled by sqrt(chi2/dof). Returns H0, se, R2, chi2/dof."""
    H0 = np.sum(w * r * v) / np.sum(w * r * r)
    resid = v - H0 * r
    chi2dof = np.sum(w * resid**2) / (len(r) - 1)
    se = np.sqrt(chi2dof / np.sum(w * r * r))
    r2 = 1 - np.sum(resid**2) / np.sum((v - v.mean())**2)
    return H0, se, r2, chi2dof


def fit_free_weighted(r, v, w):
    """Weighted fit v = H0*r + b. Returns H0, se, b, R2, chi2/dof."""
    n = len(r)
    A = np.vstack([r, np.ones(n)]).T
    AtW = A.T * w
    cov = np.linalg.inv(AtW @ A)
    H0, b = cov @ AtW @ v
    resid = v - (H0 * r + b)
    chi2dof = np.sum(w * resid**2) / (n - 2)
    r2 = 1 - np.sum(resid**2) / np.sum((v - v.mean())**2)
    return H0, np.sqrt(cov[0, 0] * chi2dof), b, r2, chi2dof


def irls_origin(r, v, frac_err, sigma_pec=0.0, n_iter=50):
    """Iteratively reweighted origin-forced fit with
    sigma_v^2 = (H0*f*r)^2 + sigma_pec^2. Returns H0, se, R2, chi2/dof."""
    H0 = fit_origin(r, v)[0]
    for _ in range(n_iter):
        w = 1.0 / ((H0 * frac_err * r)**2 + sigma_pec**2)
        H0_new, se, r2, chi2dof = fit_origin_weighted(r, v, w)
        converged = abs(H0_new - H0) < 1e-8
        H0 = H0_new
        if converged:
            break
    return H0, se, r2, chi2dof


def irls_free(r, v, frac_err, sigma_pec=0.0, n_iter=50):
    """Free-intercept version of irls_origin. Returns H0, se, b, R2, chi2/dof."""
    H0 = fit_free(r, v)[0]
    for _ in range(n_iter):
        w = 1.0 / ((H0 * frac_err * r)**2 + sigma_pec**2)
        H0_new, se, b, r2, chi2dof = fit_free_weighted(r, v, w)
        converged = abs(H0_new - H0) < 1e-8
        H0 = H0_new
        if converged:
            break
    return H0, se, b, r2, chi2dof


def sigma_pec_chi2(r, v, frac_err, lo=1.0, hi=5000.0):
    """Peculiar-velocity floor for which the weighted origin-forced fit has
    chi2/dof = 1 (the scatter not explained by the distance errors).
    Returns 0 if the distance errors alone already give chi2/dof <= 1."""
    def g(s):
        return irls_origin(r, v, frac_err, sigma_pec=s)[3] - 1.0
    if g(lo) <= 0:
        return 0.0
    return brentq(g, lo, hi, xtol=1e-3)


def ml_origin(r, v, frac_err, start=(70.0, 300.0)):
    """Maximum-likelihood fit of H0 and sigma_pec together, Gaussian errors
    sigma_v^2 = (H0*f*r)^2 + sigma_pec^2. Returns H0, sigma_pec."""
    def nll(p):
        H0, log_s = p
        s2 = (H0 * frac_err * r)**2 + np.exp(log_s)**2
        return 0.5 * np.sum((v - H0 * r)**2 / s2 + np.log(s2))
    res = minimize(nll, [start[0], np.log(start[1])], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
    return res.x[0], np.exp(res.x[1])


# ---------- errors in both variables ----------

def odr_origin(r, v, sigma_r, sigma_v):
    """Orthogonal distance regression through the origin, v = H0*r, with
    errors sigma_r on r and sigma_v on v. For a straight line, ODR is
    equivalent to minimizing S(H0) = sum (v - H0 r)^2 / (sigma_v^2 + H0^2 sigma_r^2)
    (effective variance); the standard error is the Gauss-Newton one, scaled
    by S/(n-1), as in scipy.odr. Returns H0, se(H0)."""
    def S(H):
        return np.sum((v - H * r)**2 / (sigma_v**2 + H**2 * sigma_r**2))
    H0_ols = fit_origin(r, v)[0]
    H0 = minimize_scalar(S, bounds=(0.2 * H0_ols, 5 * H0_ols), method="bounded",
                         options=dict(xatol=1e-10)).x
    s = np.sqrt(sigma_v**2 + H0**2 * sigma_r**2)
    J = -r / s - (v - H0 * r) * H0 * sigma_r**2 / s**3
    se = np.sqrt(S(H0) / (len(r) - 1) / np.sum(J**2))
    return H0, se


# ---------- low-redshift cosmology ----------

def comoving_velocity(v, omega_m=OMEGA_M):
    """Convert a recession velocity cz (km/s) into H0*D_C, where D_C is the
    comoving distance in flat LambdaCDM: c * integral_0^z dz'/E(z').
    v_eff = H0 * D_C holds exactly, for every z."""
    def one(vi):
        z = vi / C_KMS
        return C_KMS * quad(lambda x: 1.0 / np.sqrt(omega_m * (1 + x)**3 + 1 - omega_m), 0, z)[0]
    return np.array([one(vi) for vi in np.atleast_1d(v)], dtype=float)


def to_comoving_distance(r, z, angular):
    """Luminosity distance D_L -> D_C = D_L/(1+z); angular-diameter distance
    D_A -> D_C = D_A*(1+z). `angular` is a boolean array (True for D_A)."""
    return np.where(angular, r * (1 + z), r / (1 + z))


# ---------- Hubble (1929): K term plus the solar motion ----------

def unit_vectors(ra_deg, dec_deg):
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(a) * np.cos(d), np.sin(a) * np.cos(d), np.sin(d)])


def fit_hubble_solar_motion(r, v, ra_deg, dec_deg):
    """Hubble's (1929) equations of condition,
    v = K r + X cos(a)cos(d) + Y sin(a)cos(d) + Z sin(d),
    solved by least squares. Returns dict with K, X, Y, Z, their standard
    errors, the solar speed V0, the apex (A, D) in degrees and the mean
    absolute residual."""
    A = np.column_stack([r, unit_vectors(ra_deg, dec_deg)])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    resid = v - A @ coef
    cov = (resid @ resid / (len(v) - 4)) * np.linalg.inv(A.T @ A)
    K, X, Y, Z = coef
    V0 = np.sqrt(X**2 + Y**2 + Z**2)
    # (X, Y, Z) is minus the solar velocity: the apex is in the direction -(X, Y, Z)
    apex_A = np.degrees(np.arctan2(-Y, -X)) % 360
    apex_D = np.degrees(np.arcsin(-Z / V0))
    return dict(K=K, X=X, Y=Y, Z=Z, se=np.sqrt(np.diag(cov)), V0=V0,
                apex_A=apex_A, apex_D=apex_D, mean_abs_resid=np.mean(np.abs(resid)))


# Rotation from equatorial (ICRS/J2000) to Galactic cartesian coordinates
_EQ_TO_GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                       [+0.4941094279, -0.4448296300, +0.7469822445],
                       [-0.8676661490, -0.1980763734, +0.4559837762]])
# Velocity of the Sun with respect to the Galactic centre, km/s, (U, V, W) =
# toward the Galactic centre, in the direction of rotation, toward the north
# Galactic pole: solar peculiar motion of Schoenrich, Binney & Dehnen (2010)
# plus a circular speed of 239 km/s (McMillan 2011), as in van der Marel et al. (2012).
V_SUN_GAL = np.array([11.1, 239.0 + 12.24, 7.25])


def solar_reflex(ra_deg, dec_deg):
    """Component of the Sun's Galactocentric velocity along the line of sight.
    Galactocentric radial velocity = heliocentric velocity + solar_reflex."""
    n_gal = unit_vectors(ra_deg, dec_deg) @ _EQ_TO_GAL.T
    return n_gal @ V_SUN_GAL
