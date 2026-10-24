"""
Analysis script for the Hubble constant project.
- Reproduces every fit quoted in the report, with standard errors and R^2
  (Hubble 1929 sample and modern NED sample, free-intercept and
  origin-forced fits, heliocentric and CMB-frame velocities).
- Repeats the modern fits on the nearby subsample alone (r <= 25 Mpc) to
  show how the heliocentric/CMB-frame discrepancy depends on distance.
- Runs jackknife and bootstrap sensitivity checks on the modern sample,
  focused on the 6 galaxies with geometric maser distances (NGC 4258 plus
  five Megamaser Cosmology Project galaxies), to quantify their leverage
  on the origin-forced fit, and reports the jackknife and bootstrap
  standard errors next to the analytic one.
- Runs an inverse-variance weighted fit (method-level distance errors plus
  a peculiar-velocity floor) and an orthogonal distance regression check.
- Compares the spread between the ten H0 estimators with their own
  sampling noise (paired bootstrap: every estimator recomputed on the same
  resamples).
- Validates the weighted-fit method against the published Megamaser
  Cosmology Project result (Pesce et al. 2020, ApJL 891, L1), using their
  six distances and velocities (mcp_pesce2020.csv).
- Regenerates the four figures used in the report.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

rng = np.random.default_rng(42)

# ---------- helpers ----------

def fit_free(r, v):
    """Free-intercept OLS fit v = H0*r + b. Returns H0, se(H0), intercept, R2."""
    n = len(r)
    A = np.vstack([r, np.ones(n)]).T
    coef, res, *_ = np.linalg.lstsq(A, v, rcond=None)
    H0, b = coef
    v_pred = A @ coef
    resid = v - v_pred
    dof = n - 2
    sigma2 = np.sum(resid**2) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se_H0 = np.sqrt(cov[0, 0])
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((v - v.mean())**2)
    r2 = 1 - ss_res / ss_tot
    return H0, se_H0, b, r2


def fit_origin(r, v):
    """Origin-forced OLS fit v = H0*r. Returns H0, se(H0), R2 (about the fit)."""
    n = len(r)
    H0 = np.sum(r * v) / np.sum(r * r)
    v_pred = H0 * r
    resid = v - v_pred
    dof = n - 1
    sigma2 = np.sum(resid**2) / dof
    se_H0 = np.sqrt(sigma2 / np.sum(r * r))
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((v - v.mean())**2)
    r2 = 1 - ss_res / ss_tot
    return H0, se_H0, r2


def jackknife_origin(r, v):
    """Leave-one-out jackknife for the origin-forced H0. Returns array of H0_i."""
    n = len(r)
    out = np.zeros(n)
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        out[i] = fit_origin(r[mask], v[mask])[0]
    return out


def bootstrap_origin(r, v, n_boot=20000):
    n = len(r)
    ests = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.integers(0, n, n)
        ests[b] = fit_origin(r[sample], v[sample])[0]
    return ests


# Representative FRACTIONAL distance uncertainties by method, adopted from the
# typical ranges summarized in Freedman & Madore (2010, ARA&A 48, 673). These
# are method-level, representative values (not individual NED-quoted errors
# for each galaxy), used here only to set relative weights between methods.
# The single TRGB distance (NGC 4594, McQuinn et al. 2016) is given 5%,
# slightly larger than its own quoted uncertainty (about 3.5%).
METHOD_FRAC_ERROR = {
    "Cepheids": 0.07,
    "SBF": 0.10,
    "SNIa": 0.08,
    "SNII (optical)": 0.15,
    "TRGB": 0.05,
    "Maser": 0.05,
}

# Peculiar-velocity floor adopted for the weighted fit and the ODR check, and
# a representative measurement error on the redshift-based velocities
# themselves (NED quotes a few km/s for these galaxies). Only these two terms
# belong in the velocity error of the ODR: the distance error enters the ODR
# separately, through sigma_r.
SIGMA_PEC = 200.0
SIGMA_V_MEAS = 5.0


def fit_origin_weighted(r, v, w):
    """Weighted origin-forced fit v = H0*r, weights w = 1/sigma_v^2."""
    H0 = np.sum(w * r * v) / np.sum(w * r * r)
    v_pred = H0 * r
    resid = v - v_pred
    # weighted residual variance, scaled so chi2/dof ~ 1 by construction of se
    chi2 = np.sum(w * resid**2)
    dof = len(r) - 1
    se_H0 = np.sqrt(1.0 / np.sum(w * r * r)) * np.sqrt(chi2 / dof)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((v - v.mean())**2)
    r2 = 1 - ss_res / ss_tot
    return H0, se_H0, r2, chi2 / dof


def fit_free_weighted(r, v, w):
    """Weighted free-intercept fit v = H0*r + b, weights w = 1/sigma_v^2."""
    n = len(r)
    A = np.vstack([r, np.ones(n)]).T
    W = np.diag(w)
    AtW = A.T @ W
    cov_beta = np.linalg.inv(AtW @ A)
    coef = cov_beta @ AtW @ v
    H0, b = coef
    v_pred = A @ coef
    resid = v - v_pred
    chi2 = np.sum(w * resid**2)
    dof = n - 2
    se_H0 = np.sqrt(cov_beta[0, 0]) * np.sqrt(chi2 / dof)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((v - v.mean())**2)
    r2 = 1 - ss_res / ss_tot
    return H0, se_H0, b, r2, chi2 / dof


def irls_origin(r, v, frac_err, n_iter=15, sigma_pec=0.0):
    """Iteratively reweighted origin-forced fit.
    sigma_v_i^2 = (H0_current * frac_err_i * r_i)^2 + sigma_pec^2
    The sigma_pec term is a distance-independent 'floor', representing
    peculiar-velocity scatter, which does not shrink at small r the way a
    purely fractional distance error would. Starts from the unweighted fit
    as the initial H0 guess, then iterates the weights to convergence."""
    H0, _, _ = fit_origin(r, v)
    for _ in range(n_iter):
        sigma_v = np.sqrt((H0 * frac_err * r) ** 2 + sigma_pec ** 2)
        w = 1.0 / sigma_v**2
        H0_new, se_H0, r2, chi2dof = fit_origin_weighted(r, v, w)
        if abs(H0_new - H0) < 1e-6:
            H0 = H0_new
            break
        H0 = H0_new
    return H0, se_H0, r2, chi2dof


def odr_origin(r, v, sigma_r, sigma_v):
    """Errors-in-variables (orthogonal distance regression) fit forced through
    the origin: v = H0 * r, with errors on BOTH r and v. Unlike ordinary least
    squares, ODR does not assume r is known exactly -- it minimizes the
    perpendicular (weighted) distance from each point to the line, accounting
    for both sigma_r and sigma_v. Returns H0 and its standard error."""
    def f(B, x):
        return B[0] * x
    model = Model(f)
    data = RealData(r, v, sx=sigma_r, sy=sigma_v)
    # start from the OLS forced-origin estimate for a good initial guess
    H0_guess, _, _ = fit_origin(r, v)
    odr = ODR(data, model, beta0=[H0_guess])
    out = odr.run()
    return out.beta[0], out.sd_beta[0]


def irls_free(r, v, frac_err, n_iter=15, sigma_pec=0.0):
    H0, _, _, _ = fit_free(r, v)
    for _ in range(n_iter):
        sigma_v = np.sqrt((H0 * frac_err * r) ** 2 + sigma_pec ** 2)
        w = 1.0 / sigma_v**2
        H0_new, se_H0, b, r2, chi2dof = fit_free_weighted(r, v, w)
        if abs(H0_new - H0) < 1e-6:
            H0 = H0_new
            break
        H0 = H0_new
    return H0, se_H0, b, r2, chi2dof


# ---------- Part 1: Hubble 1929 ----------
print("=" * 70)
print("PART 1: Hubble (1929) data")
print("=" * 70)

df29 = pd.read_csv("hubble1929.csv")
r29 = df29.r_Mpc.values
v29 = df29.v_kms.values
lg = df29.local_group.values.astype(bool)

for label, mask in [("Full sample (24)", np.ones(len(df29), dtype=bool)),
                     ("Excluding Local Group (18)", ~lg)]:
    H0f, se_f, b_f, r2_f = fit_free(r29[mask], v29[mask])
    H0o, se_o, r2_o = fit_origin(r29[mask], v29[mask])
    print(f"\n{label}, n={mask.sum()}")
    print(f"  Free intercept:      H0 = {H0f:6.1f} +/- {se_f:4.1f} km/s/Mpc   (b={b_f:+.1f}, R2={r2_f:.3f})")
    print(f"  Forced thru origin:  H0 = {H0o:6.1f} +/- {se_o:4.1f} km/s/Mpc   (R2={r2_o:.3f})")

# ---------- Part 2: modern NED sample ----------
print()
dfm = pd.read_csv("ned_modern.csv")
N_MODERN = len(dfm)
print("=" * 70)
print(f"PART 2: Modern NED sample (n={N_MODERN})")
print("=" * 70)

r_m = dfm.r_Mpc.values
v_helio = dfm.v_helio_kms.values
v_cmb = dfm.v_cmb_kms.values
is_maser = (dfm.method == "Maser").values
print(f"\nMaser galaxies (n={is_maser.sum()}):")
print(dfm.loc[is_maser, ["object", "r_Mpc", "v_helio_kms"]].to_string(index=False))

results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0f, se_f, b_f, r2_f = fit_free(r_m, v)
    H0o, se_o, r2_o = fit_origin(r_m, v)
    results[label] = dict(H0f=H0f, se_f=se_f, b_f=b_f, r2_f=r2_f, H0o=H0o, se_o=se_o, r2_o=r2_o)
    print(f"\n{label}")
    print(f"  Free intercept:      H0 = {H0f:5.2f} +/- {se_f:4.2f} km/s/Mpc  (b={b_f:+.1f}, R2={r2_f:.4f})")
    print(f"  Forced thru origin:  H0 = {H0o:5.2f} +/- {se_o:4.2f} km/s/Mpc  (R2={r2_o:.4f})")

# ---------- Nearby subsample only (r <= 25 Mpc) ----------
# Same fits, restricted to the galaxies within 25 Mpc (no galaxy lies between
# 24.1 and 29.7 Mpc, so any cut in that range selects the same objects).
# This shows how strongly the heliocentric -> CMB frame change moves H0 when
# the sample does not reach into the Hubble flow, compared with the full sample.
R_CUT = 25.0
near = r_m <= R_CUT
print()
print("=" * 70)
print(f"NEARBY SUBSAMPLE: r <= {R_CUT:.0f} Mpc (n={near.sum()}) vs. full sample (n={N_MODERN})")
print("=" * 70)
near_results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0f, se_f, b_f, r2_f = fit_free(r_m[near], v[near])
    H0o, se_o, r2_o = fit_origin(r_m[near], v[near])
    near_results[label] = dict(H0f=H0f, se_f=se_f, H0o=H0o, se_o=se_o)
    print(f"\n{label}, r <= {R_CUT:.0f} Mpc")
    print(f"  Free intercept:      H0 = {H0f:5.2f} +/- {se_f:4.2f} km/s/Mpc  (b={b_f:+.1f}, R2={r2_f:.4f})")
    print(f"  Forced thru origin:  H0 = {H0o:5.2f} +/- {se_o:4.2f} km/s/Mpc  (R2={r2_o:.4f})")

print("\nFrame shift, H0(v_CMB) - H0(v_helio):")
for name, res in [(f"r <= {R_CUT:.0f} Mpc (n={near.sum()})", near_results),
                  (f"Full sample   (n={N_MODERN})", results)]:
    d_free = res["v_CMB"]["H0f"] - res["v_helio"]["H0f"]
    d_orig = res["v_CMB"]["H0o"] - res["v_helio"]["H0o"]
    print(f"  {name}: free intercept {d_free:+6.2f}, forced thru origin {d_orig:+6.2f} km/s/Mpc")

# ---------- Sensitivity to the 6 maser-distance galaxies ----------
# NGC 4258 (the geometric anchor of the distance ladder, Reid et al. 2019)
# plus the five Megamaser Cosmology Project galaxies, all with the distances
# adopted in Pesce et al. (2020).
print()
print("=" * 70)
print(f"SENSITIVITY CHECK: influence of the {is_maser.sum()} maser-distance galaxies")
print("=" * 70)

for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0_all, se_all, r2_all = fit_origin(r_m, v)
    H0_nomaser, se_nomaser, r2_nomaser = fit_origin(r_m[~is_maser], v[~is_maser])
    shift = H0_all - H0_nomaser
    print(f"\n{label} (origin-forced fit):")
    print(f"  Full sample (n={N_MODERN}):        H0 = {H0_all:5.2f} +/- {se_all:4.2f} km/s/Mpc")
    print(f"  Excluding {is_maser.sum()} masers (n={(~is_maser).sum()}): H0 = {H0_nomaser:5.2f} +/- {se_nomaser:4.2f} km/s/Mpc")
    print(f"  Shift from removing masers: {shift:+.2f} km/s/Mpc")

    jk = jackknife_origin(r_m, v)
    # Which single galaxy shifts H0 the most when removed?
    shifts_1 = H0_all - jk
    order = np.argsort(-np.abs(shifts_1))[:5]
    print("  Top-5 single-galaxy leverage (H0_all - H0_excl_i):")
    for i in order:
        print(f"    {dfm.object.iloc[i]:15s} ({dfm.method.iloc[i]:>8s}, r={dfm.r_Mpc.iloc[i]:6.1f} Mpc): {shifts_1[i]:+.2f} km/s/Mpc")

    n_jk = len(jk)
    se_jk = np.sqrt((n_jk - 1) / n_jk * np.sum((jk - jk.mean()) ** 2))

    boot = bootstrap_origin(r_m, v)
    lo, hi = np.percentile(boot, [16, 84])
    print(f"  Bootstrap 68% CI (percentile): [{lo:.2f}, {hi:.2f}] km/s/Mpc (median {np.median(boot):.2f})")
    print(f"  Standard error on H0: analytic {se_all:.2f}, jackknife {se_jk:.2f}, "
          f"bootstrap SD {boot.std(ddof=1):.2f} km/s/Mpc")

# ---------- Weighted fit (inverse-variance, by representative method-level errors) ----------
print()
print("=" * 70)
print("WEIGHTED FIT (inverse-variance, representative fractional errors by method)")
print("=" * 70)
print("Adopted fractional distance errors:", METHOD_FRAC_ERROR)

frac_err = dfm.method.map(METHOD_FRAC_ERROR).values

print("\n--- Naive version: weight by distance error alone (sigma_pec = 0) ---")
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0o_naive, se_o_naive, r2_o_naive, chi2dof_naive = irls_origin(r_m, v, frac_err, sigma_pec=0.0)
    print(f"  {label}: H0 = {H0o_naive:5.2f} +/- {se_o_naive:4.2f} km/s/Mpc (chi2/dof={chi2dof_naive:.1f}) "
          f"-- pulled toward the low-r, peculiar-velocity-dominated regime")

print("\n--- Combined version: distance error + peculiar-velocity floor sigma_pec ---")
print("(sigma_pec is a distance-independent scatter floor from galaxy peculiar motions;")
print(" adopted as a representative order-of-magnitude, not a per-galaxy measurement)")
weighted_results = {}
for sigma_pec_test in (150.0, 200.0, 300.0):
    print(f"\n  sigma_pec = {sigma_pec_test:.0f} km/s:")
    for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
        H0o_w, se_o_w, r2_o_w, chi2dof_o = irls_origin(r_m, v, frac_err, sigma_pec=sigma_pec_test)
        H0f_w, se_f_w, b_f_w, r2_f_w, chi2dof_f = irls_free(r_m, v, frac_err, sigma_pec=sigma_pec_test)
        print(f"    {label}: forced-origin  H0 = {H0o_w:5.2f} +/- {se_o_w:4.2f} km/s/Mpc (R2={r2_o_w:.4f}, chi2/dof={chi2dof_o:.2f})")
        print(f"    {label}: free-intercept H0 = {H0f_w:5.2f} +/- {se_f_w:4.2f} km/s/Mpc (b={b_f_w:+.1f}, R2={r2_f_w:.4f}, chi2/dof={chi2dof_f:.2f})")
        if sigma_pec_test == SIGMA_PEC:
            weighted_results[label] = dict(H0f=H0f_w, se_f=se_f_w, b_f=b_f_w, r2_f=r2_f_w, chi2dof_f=chi2dof_f,
                                            H0o=H0o_w, se_o=se_o_w, r2_o=r2_o_w, chi2dof_o=chi2dof_o)

print(f"\nAdopted sigma_pec = {SIGMA_PEC:.0f} km/s, results used in the report:")
for label in ("v_helio", "v_CMB"):
    wr = weighted_results[label]
    unw = results[label]
    print(f"  {label}: weighted H0 = {wr['H0o']:.2f} +/- {wr['se_o']:.2f}  "
          f"(unweighted was {unw['H0o']:.2f} +/- {unw['se_o']:.2f})")

# ---------- ODR: errors-in-variables check ----------
print()
print("=" * 70)
print("ODR CHECK: accounting for errors on r AND v simultaneously")
print("=" * 70)
print(f"(sigma_r = f_method * r; sigma_v = sqrt(sigma_pec^2 + sigma_meas^2), "
      f"sigma_pec={SIGMA_PEC:.0f} km/s, sigma_meas={SIGMA_V_MEAS:.0f} km/s)")
print("(the distance error enters only through sigma_r: ODR already propagates it")
print(" onto v, so it must not be repeated in sigma_v)")

odr_results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    sigma_r = frac_err * r_m
    sigma_v = np.full_like(r_m, np.hypot(SIGMA_PEC, SIGMA_V_MEAS), dtype=float)
    H0_odr, se_odr = odr_origin(r_m, v, sigma_r, sigma_v)
    odr_results[label] = dict(H0=H0_odr, se=se_odr)
    wls_H0, wls_se = weighted_results[label]["H0o"], weighted_results[label]["se_o"]
    print(f"  {label}: ODR H0 = {H0_odr:5.2f} +/- {se_odr:4.2f} km/s/Mpc  "
          f"(weighted OLS was {wls_H0:.2f} +/- {wls_se:.2f})")
    # For transparency: the v1.1.0 version of this check also put the distance
    # term H0*f*r into sigma_v, counting the distance error twice.
    sigma_v_old = np.sqrt((wls_H0 * frac_err * r_m) ** 2 + SIGMA_PEC ** 2)
    H0_odr_old, _ = odr_origin(r_m, v, sigma_r, sigma_v_old)
    print(f"  {label}: (with the v1.1.0 double-counted sigma_v it would be {H0_odr_old:5.2f})")

# Residuals by method (checking for a method-dependent systematic offset)
print("\nResiduals from the unweighted v_helio origin-forced fit, grouped by method")
print("(fractional residual (v - H0 r) / (H0 r); std = scatter of the group,")
print(" sem = standard error of the mean = std / sqrt(count), z = mean / sem):")
H0o_ref, _, _ = fit_origin(r_m, v_helio)
dfm["resid_helio"] = v_helio - H0o_ref * r_m
dfm["frac_resid_helio"] = dfm["resid_helio"] / (H0o_ref * r_m)
resid_by_method = dfm.groupby("method")["frac_resid_helio"].agg(["mean", "std", "count"])
resid_by_method["sem"] = resid_by_method["std"] / np.sqrt(resid_by_method["count"])
resid_by_method["z"] = resid_by_method["mean"] / resid_by_method["sem"]
print(resid_by_method.to_string())
ceph = dfm[dfm.method == "Cepheids"].sort_values("frac_resid_helio", ascending=False)
print("  Largest Cepheid residuals:")
for _, row in ceph.head(3).iterrows():
    print(f"    {row.object:10s} r={row.r_Mpc:6.2f} Mpc: {row.frac_resid_helio:+.2f}")

# ---------- Method dependence vs. sampling noise (paired bootstrap) ----------
# The ten estimators above (2 frames x {OLS origin, OLS free, weighted origin,
# weighted free, ODR}) are all computed on the same 39 galaxies, so their
# differences are not independent of sampling noise. Here every estimator is
# recomputed on the SAME bootstrap resamples, which gives the noise of each
# estimator and of each difference between estimators.
print()
print("=" * 70)
print("METHOD DEPENDENCE vs. SAMPLING NOISE (paired bootstrap)")
print("=" * 70)

sigma_v_odr_const = np.hypot(SIGMA_PEC, SIGMA_V_MEAS)


def all_estimators(r, v_h, v_c, f):
    out = {}
    for fr, v in (("v_helio", v_h), ("v_CMB", v_c)):
        out[f"{fr} OLS origin"] = fit_origin(r, v)[0]
        out[f"{fr} OLS free"] = fit_free(r, v)[0]
        out[f"{fr} weighted origin"] = irls_origin(r, v, f, sigma_pec=SIGMA_PEC)[0]
        out[f"{fr} weighted free"] = irls_free(r, v, f, sigma_pec=SIGMA_PEC)[0]
        out[f"{fr} ODR"] = odr_origin(r, v, f * r,
                                      np.full_like(r, sigma_v_odr_const, dtype=float))[0]
    return out


est_obs = all_estimators(r_m, v_helio, v_cmb, frac_err)
est_names = list(est_obs)
est_se = {}
for label in ("v_helio", "v_CMB"):
    est_se[f"{label} OLS origin"] = results[label]["se_o"]
    est_se[f"{label} OLS free"] = results[label]["se_f"]
    est_se[f"{label} weighted origin"] = weighted_results[label]["se_o"]
    est_se[f"{label} weighted free"] = weighted_results[label]["se_f"]
    est_se[f"{label} ODR"] = odr_results[label]["se"]

N_BOOT_PAIRED = 4000
rng_paired = np.random.default_rng(2026)
n_m = len(r_m)
boot_est = np.full((N_BOOT_PAIRED, len(est_names)), np.nan)
for b in range(N_BOOT_PAIRED):
    s = rng_paired.integers(0, n_m, n_m)
    try:
        e = all_estimators(r_m[s], v_helio[s], v_cmb[s], frac_err[s])
        boot_est[b] = [e[k] for k in est_names]
    except Exception:
        pass
valid = ~np.isnan(boot_est).any(axis=1)
boot_est = boot_est[valid]
print(f"{valid.sum()} valid resamples out of {N_BOOT_PAIRED}\n")
print(f"  {'estimator':24s} {'H0':>6s} {'SE analytic':>11s} {'SD boot':>8s}   68% CI (bootstrap)")
for j, k in enumerate(est_names):
    lo, hi = np.percentile(boot_est[:, j], [16, 84])
    print(f"  {k:24s} {est_obs[k]:6.2f} {est_se[k]:11.2f} {boot_est[:, j].std(ddof=1):8.2f}   [{lo:.2f}, {hi:.2f}]")

ref_name = "v_helio OLS origin"
j_ref = est_names.index(ref_name)
print(f"\n  Shift of each estimator from the reference ({ref_name}),")
print("  compared with the bootstrap SD of that same difference:")
print(f"  {'estimator':24s} {'shift':>7s} {'SD(shift)':>9s} {'shift/SD':>8s}")
ratios = []
for j, k in enumerate(est_names):
    if j == j_ref:
        continue
    shift = est_obs[k] - est_obs[ref_name]
    sd_shift = (boot_est[:, j] - boot_est[:, j_ref]).std(ddof=1)
    ratios.append(abs(shift / sd_shift))
    print(f"  {k:24s} {shift:+7.2f} {sd_shift:9.2f} {shift / sd_shift:+8.1f}")
print(f"  Largest |shift|/SD: {max(ratios):.1f}")

vals = np.array([est_obs[k] for k in est_names])
print(f"\n  Spread of the ten estimates: SD {vals.std(ddof=1):.2f}, "
      f"range {vals.min():.2f}-{vals.max():.2f} km/s/Mpc")
for label in ("v_helio", "v_CMB"):
    v5 = np.array([est_obs[k] for k in est_names if k.startswith(label + " ")])
    print(f"  {label} only (5 estimators): SD {v5.std(ddof=1):.2f}, "
          f"range {v5.min():.2f}-{v5.max():.2f} km/s/Mpc")

# Distance of each estimator from the literature values, in units of the
# combined uncertainty (bootstrap SD of the estimator and quoted error of the
# literature value, in quadrature). Statistical error only: the dependence of
# the estimate on the choice of estimator is shown by the spread across rows.
LITERATURE = [
    ("Planck 2020", 67.4, 0.5),
    ("CCHP 2025", 70.39, np.sqrt(1.22**2 + 1.33**2 + 0.70**2)),
    ("SH0ES 2022", 73.3, 1.0),
]
print("\n  Distance from literature values, (H0 - H0_lit) / sqrt(SD_boot^2 + sigma_lit^2):")
print(f"  {'estimator':24s}" + "".join(f" {name:>12s}" for name, _, _ in LITERATURE))
for j, k in enumerate(est_names):
    sd = boot_est[:, j].std(ddof=1)
    row = "".join(f" {(est_obs[k] - val) / np.hypot(sd, err):+11.1f}s" for _, val, err in LITERATURE)
    print(f"  {k:24s}{row}")

# ---------- Validation against the published MCP result ----------
# The weighted origin-forced fit used above (method-level errors replaced
# here by each galaxy's published distance error, symmetrized, and the
# peculiar-velocity floor set to the 250 km/s used by Pesce et al. 2020)
# is applied to the six MCP galaxies with the distances and CMB-frame
# velocities from Table 1 of Pesce et al. (2020), and compared with their
# published H0 = 73.9 +/- 3.0 km/s/Mpc.
print()
print("=" * 70)
print("VALIDATION: weighted fit on the Pesce et al. (2020) maser sample")
print("=" * 70)
mcp = pd.read_csv("mcp_pesce2020.csv")
D_mcp = mcp.D_Mpc.values
eD_mcp = 0.5 * (mcp.D_err_plus.values + mcp.D_err_minus.values)
v_mcp = mcp.v_cmb_kms.values
H0_mcp, se_mcp_scaled, _, chi2dof_mcp = irls_origin(D_mcp, v_mcp, eD_mcp / D_mcp, sigma_pec=250.0)
sig_mcp = np.sqrt((H0_mcp * eD_mcp) ** 2 + 250.0 ** 2)
se_mcp = 1.0 / np.sqrt(np.sum(D_mcp ** 2 / sig_mcp ** 2))
print(f"  This method:           H0 = {H0_mcp:5.2f} +/- {se_mcp:4.2f} km/s/Mpc "
      f"(n={len(mcp)}, chi2/dof={chi2dof_mcp:.2f}; error from input uncertainties)")
print("  Pesce et al. (2020):   H0 = 73.90 +/- 3.00 km/s/Mpc")
print(f"  Difference: {H0_mcp - 73.9:+.2f} km/s/Mpc = "
      f"{(H0_mcp - 73.9) / np.hypot(se_mcp, 3.0):+.2f} sigma (errors combined in quadrature)")

# ---------- M31 check ----------
print()
print("=" * 70)
print("M31 (Andromeda) modern velocity check")
print("=" * 70)
print(f"Value quoted in report: v_helio(M31) ~= -297 km/s (not in the n={N_MODERN} table; NED modern measurement).")

# ---------- Figures ----------
print()
print("Generating figures...")

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# Figure 1: Hubble 1929, origin-forced fit, full sample vs excl. Local Group
H0o_all, _, _ = fit_origin(r29, v29)
H0o_nolg, _, _ = fit_origin(r29[~lg], v29[~lg])

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(r29[~lg], v29[~lg], c="#1f4e8c", s=32, label="Sample (n=18)", zorder=3)
ax.scatter(r29[lg], v29[lg], facecolors="none", edgecolors="#c0392b", s=45,
           linewidths=1.4, label="Local Group members (n=6)", zorder=3)
rr = np.linspace(0, r29.max() * 1.05, 100)
ax.plot(rr, H0o_all * rr, color="#555555", ls="--", lw=1.3,
        label=f"Fit, full sample (H$_0$={H0o_all:.0f})")
ax.plot(rr, H0o_nolg * rr, color="#1f4e8c", lw=1.6,
        label=f"Fit, excl. Local Group (H$_0$={H0o_nolg:.0f})")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Distance $r$ (Mpc)")
ax.set_ylabel("Radial velocity $v$ (km/s)")
ax.set_title("Hubble (1929) sample: velocity vs. distance")
ax.legend(fontsize=8.5, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("figures/Hubble.png", dpi=300)
plt.close(fig)

# Figures 2 & 3: modern NED sample, heliocentric and CMB frame
for label, v, fname, title in [
    (r"v_{\rm helio}", v_helio, "figures/Velocity (Heliocentric) vs. Distance - NED Sample.png",
     "Modern NED sample: heliocentric velocity vs. distance"),
    (r"v_{\rm CMB}", v_cmb, "figures/Velocity (3K CMB) vs. Distance - NED Sample.png",
     "Modern NED sample: CMB-frame velocity vs. distance"),
]:
    H0o, se_o, _ = fit_origin(r_m, v)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    method_colors = {"Cepheids": "#1f4e8c", "SBF": "#c0392b", "Maser": "#1a9850",
                      "SNIa": "#8e44ad", "SNII (optical)": "#e67e22", "TRGB": "#17a2b8"}
    for m, c in method_colors.items():
        sel = (dfm.method == m).values
        if sel.any():
            ax.scatter(r_m[sel], v[sel], color=c, s=34, label=m, zorder=3)
    rr = np.linspace(0, r_m.max() * 1.03, 100)
    ax.plot(rr, H0o * rr, color="black", lw=1.4, ls="--",
            label=f"Fit through origin (H$_0$={H0o:.1f}$\\pm${se_o:.1f})")
    ax.set_xlabel("Distance $r$ (Mpc)")
    ax.set_ylabel(f"Velocity ${label}$ (km/s)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# Figure 4: comparison with the literature ("forest plot")
H0o_helio_w = weighted_results["v_helio"]["H0o"]
se_o_helio_w = weighted_results["v_helio"]["se_o"]
H0o_cmb_w = weighted_results["v_CMB"]["H0o"]
se_o_cmb_w = weighted_results["v_CMB"]["se_o"]
H0o_helio_uw = results["v_helio"]["H0o"]
se_o_helio_uw = results["v_helio"]["se_o"]
H0o_cmb_uw = results["v_CMB"]["H0o"]
se_o_cmb_uw = results["v_CMB"]["se_o"]

rows = [
    ("Planck 2020 (CMB, $\\Lambda$CDM)", 67.4, 0.5, "#7f7f7f"),
    ("Freedman et al. 2025 (CCHP, TRGB)", 70.39, np.sqrt(1.22**2 + 1.33**2 + 0.70**2), "#7f7f7f"),
    ("Riess et al. 2022 (SH0ES, Cepheids)", 73.3, 1.0, "#7f7f7f"),
    ("This work -- $v_{\\rm helio}$, unweighted", H0o_helio_uw, se_o_helio_uw, "#1f4e8c"),
    ("This work -- $v_{\\rm helio}$, weighted", H0o_helio_w, se_o_helio_w, "#1a9850"),
    ("This work -- $v_{\\rm CMB}$, unweighted", H0o_cmb_uw, se_o_cmb_uw, "#1f4e8c"),
    ("This work -- $v_{\\rm CMB}$, weighted", H0o_cmb_w, se_o_cmb_w, "#1a9850"),
]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
ypos = np.arange(len(rows))[::-1]
for y, (name, val, err, color) in zip(ypos, rows):
    ax.errorbar(val, y, xerr=err, fmt="o", color=color, ecolor=color,
                capsize=3, markersize=6, elinewidth=1.4)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.axvspan(67.4 - 0.5, 73.3 + 1.0, color="#f0f0f0", zorder=0)
ax.set_xlabel("$H_0$ (km/s/Mpc)")
ax.set_title("Comparison with the literature", fontsize=13)
ax.set_xlim(60, 90)
fig.tight_layout()
fig.savefig("figures/H0_comparison.png", dpi=300)
plt.close(fig)

print("Done. Figures written to figures/.")
