"""
Analysis script for the Hubble constant project.
- Reproduces the fits quoted in the report (sanity check)
- Adds standard errors and R^2 for each fit
- Runs a jackknife sensitivity check on the modern sample, focused on the
  5 Megamaser Cosmology Project (MCP) galaxies, to quantify their leverage
  on the origin-forced fit.
- Regenerates the three figures used in the report.
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
METHOD_FRAC_ERROR = {
    "Cepheids": 0.07,
    "SBF": 0.10,
    "SNIa": 0.08,
    "SNII (optical)": 0.15,
    "Maser": 0.05,
    "FP": 0.20,
}


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
print("=" * 70)
print("PART 2: Modern NED sample (n=37)")
print("=" * 70)

dfm = pd.read_csv("ned_modern.csv")
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

# ---------- Sensitivity to the 5 MCP maser galaxies ----------
print()
print("=" * 70)
print("SENSITIVITY CHECK: influence of the 5 Megamaser (MCP) galaxies")
print("=" * 70)

for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0_all, se_all, r2_all = fit_origin(r_m, v)
    H0_nomaser, se_nomaser, r2_nomaser = fit_origin(r_m[~is_maser], v[~is_maser])
    shift = H0_all - H0_nomaser
    print(f"\n{label} (origin-forced fit):")
    print(f"  Full sample (n=37):        H0 = {H0_all:5.2f} +/- {se_all:4.2f} km/s/Mpc")
    print(f"  Excluding 5 masers (n=32): H0 = {H0_nomaser:5.2f} +/- {se_nomaser:4.2f} km/s/Mpc")
    print(f"  Shift from removing masers: {shift:+.2f} km/s/Mpc")

    jk = jackknife_origin(r_m, v)
    # Which single galaxy shifts H0 the most when removed?
    shifts_1 = H0_all - jk
    order = np.argsort(-np.abs(shifts_1))[:5]
    print("  Top-5 single-galaxy leverage (H0_all - H0_excl_i):")
    for i in order:
        print(f"    {dfm.object.iloc[i]:15s} ({dfm.method.iloc[i]:>8s}, r={dfm.r_Mpc.iloc[i]:6.1f} Mpc): {shifts_1[i]:+.2f} km/s/Mpc")

    boot = bootstrap_origin(r_m, v)
    lo, hi = np.percentile(boot, [16, 84])
    print(f"  Bootstrap 68% CI (percentile): [{lo:.2f}, {hi:.2f}] km/s/Mpc (median {np.median(boot):.2f})")

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
        print(f"    {label}: H0 = {H0o_w:5.2f} +/- {se_o_w:4.2f} km/s/Mpc (R2={r2_o_w:.4f}, chi2/dof={chi2dof_o:.2f})")
        if sigma_pec_test == 200.0:
            H0f_w, se_f_w, b_f_w, r2_f_w, chi2dof_f = irls_free(r_m, v, frac_err, sigma_pec=sigma_pec_test)
            weighted_results[label] = dict(H0f=H0f_w, se_f=se_f_w, b_f=b_f_w, r2_f=r2_f_w, chi2dof_f=chi2dof_f,
                                            H0o=H0o_w, se_o=se_o_w, r2_o=r2_o_w, chi2dof_o=chi2dof_o)

print("\nAdopted sigma_pec = 200 km/s, results used in the report:")
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
print("(sigma_r = f_method * r; sigma_v = sqrt((H0*f_method*r)^2 + sigma_pec^2), sigma_pec=200 km/s)")

odr_results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0_ref = weighted_results[label]["H0o"]
    sigma_r = frac_err * r_m
    sigma_v = np.sqrt((H0_ref * frac_err * r_m) ** 2 + 200.0 ** 2)
    H0_odr, se_odr = odr_origin(r_m, v, sigma_r, sigma_v)
    odr_results[label] = dict(H0=H0_odr, se=se_odr)
    wls_H0, wls_se = weighted_results[label]["H0o"], weighted_results[label]["se_o"]
    print(f"  {label}: ODR H0 = {H0_odr:5.2f} +/- {se_odr:4.2f} km/s/Mpc  "
          f"(weighted OLS was {wls_H0:.2f} +/- {wls_se:.2f})")

# Residuals by method (checking for a method-dependent systematic offset)
print("\nResiduals from the unweighted v_helio origin-forced fit, grouped by method:")
H0o_ref, _, _ = fit_origin(r_m, v_helio)
dfm["resid_helio"] = v_helio - H0o_ref * r_m
dfm["frac_resid_helio"] = dfm["resid_helio"] / (H0o_ref * r_m)
print(dfm.groupby("method")["frac_resid_helio"].agg(["mean", "std", "count"]).to_string())

# ---------- M31 check ----------
print()
print("=" * 70)
print("M31 (Andromeda) modern velocity check")
print("=" * 70)
print("Value quoted in report: v_helio(M31) ~= -297 km/s (not in the n=37 table; NED modern measurement).")

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
                      "SNIa": "#8e44ad", "SNII (optical)": "#e67e22", "FP": "#7f7f7f"}
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
    ("Freedman 2021 (TRGB)", 69.8, np.hypot(0.6, 1.6), "#7f7f7f"),
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
