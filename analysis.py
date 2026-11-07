"""
Analysis script for the Hubble constant project (version 1.3.0).
- Part 1: fits Hubble's (1929) data with v = H0*r, and reproduces Hubble's own
  solution, which also solved for the motion of the Sun (K, X, Y, Z).
- Part 2: fits the modern sample (39 galaxies) after converting every distance
  to comoving distance (the maser distances are angular-diameter distances,
  all the others luminosity distances) and every cz to H0*D_C, so that the
  linear Hubble law is exact.
- Repeats the fits on the nearby subsample (r <= 25 Mpc), runs jackknife and
  bootstrap checks focused on the 6 maser galaxies, a weighted fit whose
  peculiar-velocity floor is estimated from the data, and an orthogonal
  distance regression (errors on both axes).
- Compares the ten estimators with their own sampling noise (paired bootstrap)
  and with literature values.
- Validates the weighted method against the published Megamaser Cosmology
  Project result (Pesce et al. 2020, ApJL 891, L1; mcp_pesce2020.csv).
- Computes the Galactocentric velocities of M31 and M81 (reflex of the solar motion).
- Regenerates the four figures used in the report.
All fitting functions are in hubble_fits.py and are tested in tests/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hubble_fits import (
    C_KMS, OMEGA_M, fit_free, fit_origin, jackknife_origin, jackknife_se,
    bootstrap_origin, effective_sample_size, irls_origin, irls_free,
    sigma_pec_chi2, ml_origin, odr_origin, comoving_velocity,
    to_comoving_distance, fit_hubble_solar_motion, solar_reflex,
)

rng = np.random.default_rng(42)

# Representative FRACTIONAL distance uncertainties by method (typical ranges in
# Freedman & Madore 2010, ARA&A 48, 673), used only to set relative weights.
# The single TRGB distance (NGC 4594, McQuinn et al. 2016) is given 5%,
# slightly more than its own quoted uncertainty (about 3.5%).
METHOD_FRAC_ERROR = {
    "Cepheids": 0.07,
    "SBF": 0.10,
    "SNIa": 0.08,
    "SNII (optical)": 0.15,
    "TRGB": 0.05,
    "Maser": 0.05,
}
SIGMA_V_MEAS = 5.0          # measurement error on the velocities, km/s
SIGMA_PEC_TESTS = (150.0, 200.0, 300.0)

# Literature values: (label, H0, total 1-sigma uncertainty)
LITERATURE = [
    ("Planck 2020", 67.4, 0.5),
    ("CCHP 2025", 70.39, np.sqrt(1.22**2 + 1.33**2 + 0.70**2)),
    ("SH0ES 2022", 73.04, 1.04),
    ("H0DN 2026", 73.50, 0.81),
]


def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# PART 1: Hubble (1929)
# ======================================================================
header("PART 1: Hubble (1929) data")

df29 = pd.read_csv("hubble1929.csv")
r29 = df29.r_Mpc.values.astype(float)
v29 = df29.v_kms.values.astype(float)
lg = df29.local_group.values.astype(bool)

for label, mask in [("Full sample (24)", np.ones(len(df29), dtype=bool)),
                    ("Excluding Local Group (18)", ~lg)]:
    H0f, se_f, b_f, r2_f = fit_free(r29[mask], v29[mask])
    H0o, se_o, r2_o = fit_origin(r29[mask], v29[mask])
    print(f"\n{label}, n={mask.sum()}")
    print(f"  Free intercept:      H0 = {H0f:6.1f} +/- {se_f:4.1f} km/s/Mpc   (b={b_f:+.1f}, R2={r2_f:.3f})")
    print(f"  Forced thru origin:  H0 = {H0o:6.1f} +/- {se_o:4.1f} km/s/Mpc   (R2={r2_o:.3f})")

print("\nHubble's own model, v = K r + X cos(a)cos(d) + Y sin(a)cos(d) + Z sin(d)")
print("(K plus the motion of the Sun; J2000 coordinates, so X, Y, Z are rotated by")
print(" about 1.4 deg of precession with respect to Hubble's 1900 values; K is not affected)")
for label, mask in [("24 objects", np.ones(len(df29), dtype=bool)),
                    ("18 objects (no Local Group)", ~lg)]:
    s = fit_hubble_solar_motion(r29[mask], v29[mask], df29.ra_deg.values[mask], df29.dec_deg.values[mask])
    print(f"  {label}: K = {s['K']:.1f} +/- {s['se'][0]:.1f} km/s/Mpc, "
          f"X = {s['X']:+.0f}, Y = {s['Y']:+.0f}, Z = {s['Z']:+.0f} km/s,")
    print(f"      V0 = {s['V0']:.0f} km/s toward (A, D) = ({s['apex_A']:.0f}, {s['apex_D']:+.0f}) deg, "
          f"mean |residual| = {s['mean_abs_resid']:.0f} km/s")
print("  Hubble (1929), 24 objects: K = +465 +/- 50, X = -65, Y = +226, Z = -195,"
      " V0 = 306 km/s toward (286, +40), residuals average 150 km/s")

# ======================================================================
# PART 2: modern sample
# ======================================================================
dfm = pd.read_csv("ned_modern.csv")
N_MODERN = len(dfm)
header(f"PART 2: Modern sample (n={N_MODERN})")

r_obs = dfm.r_Mpc.values.astype(float)
v_helio_obs = dfm.v_helio_kms.values.astype(float)
v_cmb_obs = dfm.v_cmb_kms.values.astype(float)
is_maser = (dfm.method == "Maser").values
frac_err = dfm.method.map(METHOD_FRAC_ERROR).values

print(f"\nMaser galaxies (n={is_maser.sum()}); distances: Reid et al. 2019 (NGC 4258),"
      " Pesce et al. 2020 (others); velocities: Pesce et al. 2020 Table 1:")
print(dfm.loc[is_maser, ["object", "r_Mpc", "v_helio_kms", "v_cmb_kms"]].to_string(index=False))

w = r_obs**2
print(f"\nWeight of each galaxy in the origin-forced OLS fit is proportional to r^2:")
print(f"  share of sum(r^2) from the 6 masers: {w[is_maser].sum() / w.sum():.1%}; "
      f"from the {np.sum(r_obs > 25)} galaxies beyond 25 Mpc: {w[r_obs > 25].sum() / w.sum():.1%}")
print(f"  effective number of galaxies, (sum r^2)^2 / sum r^4 = {effective_sample_size(r_obs):.1f}")

# ---------- from measured quantities to the exact linear Hubble law ----------
# Maser distances are angular-diameter distances D_A (Pesce et al. 2020, Eq. 1);
# Cepheid, TRGB, SBF and supernova distances come from distance moduli and are
# luminosity distances D_L. v = H0 * D is exact for the comoving distance D_C,
# with v replaced by c * integral_0^z dz/E(z) (flat LambdaCDM, Omega_m = 0.315).
z_cmb = v_cmb_obs / C_KMS
r_m = to_comoving_distance(r_obs, z_cmb, angular=is_maser)
v_helio = comoving_velocity(v_helio_obs)
v_cmb = comoving_velocity(v_cmb_obs)

header("LOW-REDSHIFT CORRECTION: D_A or D_L -> comoving D_C, cz -> H0*D_C")
print(f"(flat LambdaCDM, Omega_m = {OMEGA_M}; maser distances are angular-diameter distances)")
print("  galaxy           method     z_CMB   v/r naive   v/r corrected   change")
for i in np.where(r_obs > 25)[0]:
    naive = v_cmb_obs[i] / r_obs[i]
    corr = v_cmb[i] / r_m[i]
    print(f"  {dfm.object.iloc[i]:15s} {dfm.method.iloc[i]:8s} {z_cmb[i]:7.4f}   {naive:8.2f}   "
          f"{corr:12.2f}   {corr / naive - 1:+7.2%}")
others = r_obs <= 25
ratio = (v_cmb / r_m) / (v_cmb_obs / r_obs)
print(f"  all {others.sum()} galaxies within 25 Mpc: |change| <= {np.max(np.abs(ratio[others] - 1)):.2%}")
for label, v_o in [("v_helio", v_helio_obs), ("v_CMB", v_cmb_obs)]:
    print(f"  {label}: origin-forced OLS without the correction would be {fit_origin(r_obs, v_o)[0]:.2f} km/s/Mpc")

# ---------- main fits ----------
header(f"MAIN FITS (n={N_MODERN}, comoving distances)")
results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0f, se_f, b_f, r2_f = fit_free(r_m, v)
    H0o, se_o, r2_o = fit_origin(r_m, v)
    results[label] = dict(H0f=H0f, se_f=se_f, b_f=b_f, r2_f=r2_f, H0o=H0o, se_o=se_o, r2_o=r2_o)
    print(f"\n{label}")
    print(f"  Free intercept:      H0 = {H0f:5.2f} +/- {se_f:4.2f} km/s/Mpc  (b={b_f:+.1f}, R2={r2_f:.4f})")
    print(f"  Forced thru origin:  H0 = {H0o:5.2f} +/- {se_o:4.2f} km/s/Mpc  (R2={r2_o:.4f})")

# ---------- nearby subsample ----------
R_CUT = 25.0
near = r_obs <= R_CUT
header(f"NEARBY SUBSAMPLE: r <= {R_CUT:.0f} Mpc (n={near.sum()}) vs. full sample (n={N_MODERN})")
near_results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0f, se_f, b_f, r2_f = fit_free(r_m[near], v[near])
    H0o, se_o, r2_o = fit_origin(r_m[near], v[near])
    near_results[label] = dict(H0f=H0f, H0o=H0o)
    print(f"\n{label}, r <= {R_CUT:.0f} Mpc")
    print(f"  Free intercept:      H0 = {H0f:5.2f} +/- {se_f:4.2f} km/s/Mpc  (b={b_f:+.1f}, R2={r2_f:.4f})")
    print(f"  Forced thru origin:  H0 = {H0o:5.2f} +/- {se_o:4.2f} km/s/Mpc  (R2={r2_o:.4f})")
print("\nFrame shift, H0(v_CMB) - H0(v_helio):")
for name, res in [(f"r <= {R_CUT:.0f} Mpc (n={near.sum()})", near_results),
                  (f"Full sample   (n={N_MODERN})", results)]:
    print(f"  {name}: free intercept {res['v_CMB']['H0f'] - res['v_helio']['H0f']:+6.2f}, "
          f"forced thru origin {res['v_CMB']['H0o'] - res['v_helio']['H0o']:+6.2f} km/s/Mpc")

# ---------- influence of the maser galaxies ----------
header(f"SENSITIVITY CHECK: influence of the {is_maser.sum()} maser-distance galaxies")
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0_all, se_all, _ = fit_origin(r_m, v)
    H0_nm, se_nm, _ = fit_origin(r_m[~is_maser], v[~is_maser])
    print(f"\n{label} (origin-forced fit):")
    print(f"  Full sample (n={N_MODERN}):        H0 = {H0_all:5.2f} +/- {se_all:4.2f} km/s/Mpc")
    print(f"  Excluding {is_maser.sum()} masers (n={(~is_maser).sum()}): H0 = {H0_nm:5.2f} +/- {se_nm:4.2f} km/s/Mpc")
    print(f"  Removing the masers raises H0 by {H0_nm - H0_all:+.2f} km/s/Mpc "
          f"= {(H0_nm - H0_all) / se_nm:+.2f} x the maser-free standard error; "
          f"standard error x {se_nm / se_all:.2f}")
    jk = jackknife_origin(r_m, v)
    shifts = H0_all - jk
    print("  Top-5 single-galaxy leverage (H0_all - H0_excl_i):")
    for i in np.argsort(-np.abs(shifts))[:5]:
        print(f"    {dfm.object.iloc[i]:15s} ({dfm.method.iloc[i]:>8s}, r={r_obs[i]:6.1f} Mpc): {shifts[i]:+.2f} km/s/Mpc")
    boot = bootstrap_origin(r_m, v, rng)
    lo, hi = np.percentile(boot, [16, 84])
    print(f"  Bootstrap 68% CI (percentile): [{lo:.2f}, {hi:.2f}] km/s/Mpc (median {np.median(boot):.2f})")
    print(f"  Standard error on H0: analytic {se_all:.2f}, jackknife {jackknife_se(jk):.2f}, "
          f"bootstrap SD {boot.std(ddof=1):.2f} km/s/Mpc")

# ---------- weighted fits ----------
header("WEIGHTED FIT (method-level distance errors + peculiar-velocity floor)")
print("Adopted fractional distance errors:", METHOD_FRAC_ERROR)
print("sigma_v^2 = (H0 f r)^2 + sigma_pec^2, iteratively reweighted")

print("\n--- Naive version: distance errors only (sigma_pec = 0) ---")
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    H0n, sen, _, c2n = irls_origin(r_m, v, frac_err, sigma_pec=0.0)
    print(f"  {label}: H0 = {H0n:5.2f} +/- {sen:4.2f} km/s/Mpc (chi2/dof={c2n:.1f})"
          " -- pulled toward the low-r, peculiar-velocity-dominated regime")

print("\n--- Peculiar-velocity floor estimated from the data ---")
sigma_pec_fit = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    sigma_pec_fit[label] = sigma_pec_chi2(r_m, v, frac_err)
    H0_ml, s_ml = ml_origin(r_m, v, frac_err)
    print(f"  {label}: sigma_pec = {sigma_pec_fit[label]:.0f} km/s (chi2/dof = 1 for the origin-forced fit); "
          f"maximum likelihood: sigma_pec = {s_ml:.0f} km/s, H0 = {H0_ml:.2f}")

weighted_results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    print(f"\n  {label}:")
    for s in SIGMA_PEC_TESTS + (sigma_pec_fit[label],):
        H0o, seo, r2o, c2o = irls_origin(r_m, v, frac_err, sigma_pec=s)
        H0f, sef, bf, r2f, c2f = irls_free(r_m, v, frac_err, sigma_pec=s)
        tag = " (fitted)" if s == sigma_pec_fit[label] else ""
        print(f"    sigma_pec = {s:4.0f}{tag:9s} origin H0 = {H0o:5.2f} +/- {seo:4.2f} (chi2/dof={c2o:.2f});"
              f"  free H0 = {H0f:5.2f} +/- {sef:4.2f} (b={bf:+.0f}, chi2/dof={c2f:.2f})")
        if s == sigma_pec_fit[label]:
            weighted_results[label] = dict(H0o=H0o, se_o=seo, H0f=H0f, se_f=sef, b_f=bf, c2f=c2f)
print("\nAdopted: fitted sigma_pec in each frame. Results used in the report:")
for label in ("v_helio", "v_CMB"):
    wr, un = weighted_results[label], results[label]
    print(f"  {label}: weighted origin H0 = {wr['H0o']:.2f} +/- {wr['se_o']:.2f}, "
          f"free H0 = {wr['H0f']:.2f} +/- {wr['se_f']:.2f}  (unweighted origin {un['H0o']:.2f} +/- {un['se_o']:.2f})")

# ---------- ODR ----------
header("ODR CHECK: errors on r AND v (effective-variance form of ODR)")
print("(sigma_r = f_method * r; sigma_v = sqrt(sigma_pec^2 + 5^2) with the fitted sigma_pec;")
print(" the distance error enters only through sigma_r)")
odr_results = {}
for label, v in [("v_helio", v_helio), ("v_CMB", v_cmb)]:
    sv = np.full_like(r_m, np.hypot(sigma_pec_fit[label], SIGMA_V_MEAS))
    H0_odr, se_odr = odr_origin(r_m, v, frac_err * r_m, sv)
    odr_results[label] = dict(H0=H0_odr, se=se_odr)
    print(f"  {label}: ODR H0 = {H0_odr:5.2f} +/- {se_odr:4.2f} km/s/Mpc  "
          f"(weighted fit {weighted_results[label]['H0o']:.2f} +/- {weighted_results[label]['se_o']:.2f})")

# ---------- Monte Carlo check of the estimators ----------
# Simulated samples with the real distances and distance methods, a true H0 of
# 72 km/s/Mpc, fractional distance errors f_method and Gaussian peculiar
# velocities with the fitted heliocentric sigma_pec: how biased is each estimator?
N_SIM = 1000
H0_TRUE = 72.0
rng_sim = np.random.default_rng(7)
s_sim = sigma_pec_fit["v_helio"]
sims = {"OLS origin": [], "weighted origin": [], "ODR": []}
for _ in range(N_SIM):
    r_sim = r_m * (1 + rng_sim.normal(0, frac_err))
    v_sim = H0_TRUE * r_m + rng_sim.normal(0, s_sim, r_m.size)
    sims["OLS origin"].append(fit_origin(r_sim, v_sim)[0])
    sims["weighted origin"].append(irls_origin(r_sim, v_sim, frac_err, sigma_pec=s_sim)[0])
    sims["ODR"].append(odr_origin(r_sim, v_sim, frac_err * r_sim,
                                  np.full_like(r_sim, np.hypot(s_sim, SIGMA_V_MEAS)))[0])
print(f"\nMonte Carlo check ({N_SIM} simulated samples, true H0 = {H0_TRUE:.0f}, sigma_pec = {s_sim:.0f} km/s):")
for k, x in sims.items():
    x = np.array(x)
    print(f"  {k:16s}: mean bias {x.mean() - H0_TRUE:+.2f} km/s/Mpc, SD {x.std(ddof=1):.2f}")

# ---------- residuals by method ----------
print("\nResiduals from the unweighted v_helio origin-forced fit, grouped by method")
print("(fractional residual (v - H0 r) / (H0 r); std = scatter of the group,")
print(" sem = standard error of the mean = std / sqrt(count), z = mean / sem):")
H0_ref = results["v_helio"]["H0o"]
dfm["frac_resid_helio"] = (v_helio - H0_ref * r_m) / (H0_ref * r_m)
grp = dfm.groupby("method")["frac_resid_helio"].agg(["mean", "std", "count"])
grp["sem"] = grp["std"] / np.sqrt(grp["count"])
grp["z"] = grp["mean"] / grp["sem"]
print(grp.to_string(float_format=lambda x: f"{x:.3f}"))
print("  Largest Cepheid residuals:")
for _, row in dfm[dfm.method == "Cepheids"].sort_values("frac_resid_helio", ascending=False).head(3).iterrows():
    print(f"    {row.object:10s} r={row.r_Mpc:6.2f} Mpc: {row.frac_resid_helio:+.2f}")

# ---------- paired bootstrap ----------
header("METHOD DEPENDENCE vs. SAMPLING NOISE (paired bootstrap)")
print("(every estimator, including the fit of sigma_pec, is recomputed on the same resamples)")


def all_estimators(r, v_h, v_c, f):
    out = {}
    for fr, v in (("v_helio", v_h), ("v_CMB", v_c)):
        s = sigma_pec_chi2(r, v, f)
        out[f"{fr} OLS origin"] = fit_origin(r, v)[0]
        out[f"{fr} OLS free"] = fit_free(r, v)[0]
        out[f"{fr} weighted origin"] = irls_origin(r, v, f, sigma_pec=s)[0]
        out[f"{fr} weighted free"] = irls_free(r, v, f, sigma_pec=s)[0]
        out[f"{fr} ODR"] = odr_origin(r, v, f * r, np.full_like(r, np.hypot(s, SIGMA_V_MEAS)))[0]
    return out


est_obs = all_estimators(r_m, v_helio, v_cmb, frac_err)
names = list(est_obs)
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
boot_est = np.full((N_BOOT_PAIRED, len(names)), np.nan)
for b in range(N_BOOT_PAIRED):
    s = rng_paired.integers(0, n_m, n_m)
    try:
        e = all_estimators(r_m[s], v_helio[s], v_cmb[s], frac_err[s])
        boot_est[b] = [e[k] for k in names]
    except (ValueError, np.linalg.LinAlgError):
        pass
valid = ~np.isnan(boot_est).any(axis=1)
boot_est = boot_est[valid]
print(f"{valid.sum()} valid resamples out of {N_BOOT_PAIRED}\n")
print(f"  {'estimator':24s} {'H0':>6s} {'SE analytic':>11s} {'SD boot':>8s}   68% CI (bootstrap)")
sd_boot = {}
for j, k in enumerate(names):
    lo, hi = np.percentile(boot_est[:, j], [16, 84])
    sd_boot[k] = boot_est[:, j].std(ddof=1)
    print(f"  {k:24s} {est_obs[k]:6.2f} {est_se[k]:11.2f} {sd_boot[k]:8.2f}   [{lo:.2f}, {hi:.2f}]")

ref = "v_helio OLS origin"
j_ref = names.index(ref)
print(f"\n  Shift of each estimator from the reference ({ref}),")
print("  compared with the bootstrap SD of that same difference:")
print(f"  {'estimator':24s} {'shift':>7s} {'SD(shift)':>9s} {'shift/SD':>8s}")
ratios = []
for j, k in enumerate(names):
    if j == j_ref:
        continue
    shift = est_obs[k] - est_obs[ref]
    sd_shift = (boot_est[:, j] - boot_est[:, j_ref]).std(ddof=1)
    ratios.append(abs(shift / sd_shift))
    print(f"  {k:24s} {shift:+7.2f} {sd_shift:9.2f} {shift / sd_shift:+8.1f}")
print(f"  Largest |shift|/SD: {max(ratios):.1f}")

vals = np.array([est_obs[k] for k in names])
print(f"\n  Spread of the ten estimates: SD {vals.std(ddof=1):.2f}, range {vals.min():.2f}-{vals.max():.2f} km/s/Mpc")
for label in ("v_helio", "v_CMB"):
    v5 = np.array([est_obs[k] for k in names if k.startswith(label + " ")])
    print(f"  {label} only (5 estimators): SD {v5.std(ddof=1):.2f}, range {v5.min():.2f}-{v5.max():.2f} km/s/Mpc")

print("\n  Distance from literature values, (H0 - H0_lit) / sqrt(SD_boot^2 + sigma_lit^2):")
print(f"  {'estimator':24s}" + "".join(f" {lab:>12s}" for lab, _, _ in LITERATURE))
for k in names:
    row = "".join(f" {(est_obs[k] - val) / np.hypot(sd_boot[k], err):+11.1f}s" for _, val, err in LITERATURE)
    print(f"  {k:24s}{row}")

# ---------- validation on the MCP sample ----------
header("VALIDATION: weighted fit on the Pesce et al. (2020) maser sample")
mcp = pd.read_csv("mcp_pesce2020.csv")
D_mcp = mcp.D_Mpc.values.astype(float)
eD_mcp = 0.5 * (mcp.D_err_plus.values + mcp.D_err_minus.values)
v_mcp_obs = mcp.v_cmb_kms.values.astype(float)
SIGMA_PEC_MCP = 250.0     # the fixed peculiar-velocity uncertainty of Pesce et al. (2020)


def mcp_fit(D, eD, v):
    H0, _, _, c2 = irls_origin(D, v, eD / D, sigma_pec=SIGMA_PEC_MCP)
    sig = np.sqrt((H0 * eD)**2 + SIGMA_PEC_MCP**2)
    return H0, 1.0 / np.sqrt(np.sum(D**2 / sig**2)), c2


z_mcp = v_mcp_obs / C_KMS
H0_mcp, se_mcp, c2_mcp = mcp_fit(D_mcp * (1 + z_mcp), eD_mcp * (1 + z_mcp), comoving_velocity(v_mcp_obs))
H0_mcp_naive, se_mcp_naive, _ = mcp_fit(D_mcp, eD_mcp, v_mcp_obs)
print(f"  This method:           H0 = {H0_mcp:5.2f} +/- {se_mcp:4.2f} km/s/Mpc "
      f"(n={len(mcp)}, chi2/dof={c2_mcp:.2f}; error from input uncertainties)")
print("  Pesce et al. (2020):   H0 = 73.90 +/- 3.00 km/s/Mpc")
print(f"  Difference: {H0_mcp - 73.9:+.2f} km/s/Mpc = {(H0_mcp - 73.9) / np.hypot(se_mcp, 3.0):+.2f} sigma")
print(f"  (without the D_A -> D_C correction: {H0_mcp_naive:.2f} +/- {se_mcp_naive:.2f}, "
      f"{(H0_mcp_naive - 73.9) / np.hypot(se_mcp_naive, 3.0):+.2f} sigma)")

# ---------- M31 and M81: heliocentric vs Galactocentric ----------
header("M31 and M81: reflex of the Sun's Galactocentric motion")
print("(Sun's velocity: Schoenrich et al. 2010 peculiar motion + 239 km/s circular speed)")
m31 = df29.object == "N.G.C. 224"
m81 = df29.object == "N.G.C. 3031"
for name, sel, v_mod in [("M31 (N.G.C. 224)", m31, -301.0), ("M81 (N.G.C. 3031)", m81, None)]:
    ra, dec = df29.ra_deg.values[sel][0], df29.dec_deg.values[sel][0]
    refl = solar_reflex(np.array([ra]), np.array([dec]))[0]
    v_h29 = v29[sel][0]
    line = (f"  {name}: solar reflex {refl:+.1f} km/s; Hubble's {v_h29:+.0f} km/s -> "
            f"{v_h29 + refl:+.0f} km/s Galactocentric")
    if v_mod is not None:
        line += f"; modern {v_mod:+.0f} km/s -> {v_mod + refl:+.0f} km/s"
    print(line)
print("  van der Marel et al. (2012): M31 Galactocentric radial velocity -109.3 +/- 4.4 km/s")

# ======================================================================
# Figures
# ======================================================================
print()
print("Generating figures...")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# Figure 1: Hubble 1929
H0o_all = fit_origin(r29, v29)[0]
H0o_nolg = fit_origin(r29[~lg], v29[~lg])[0]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(r29[~lg], v29[~lg], c="#1f4e8c", s=32, label="Sample (n=18)", zorder=3)
ax.scatter(r29[lg], v29[lg], facecolors="none", edgecolors="#c0392b", s=45,
           linewidths=1.4, label="Local Group members (n=6)", zorder=3)
rr = np.linspace(0, r29.max() * 1.05, 100)
ax.plot(rr, H0o_all * rr, color="#555555", ls="--", lw=1.3, label=f"Fit, full sample (H$_0$={H0o_all:.0f})")
ax.plot(rr, H0o_nolg * rr, color="#1f4e8c", lw=1.6, label=f"Fit, excl. Local Group (H$_0$={H0o_nolg:.0f})")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Distance $r$ (Mpc)")
ax.set_ylabel("Radial velocity $v$ (km/s)")
ax.set_title("Hubble (1929) sample: velocity vs. distance")
ax.legend(fontsize=8.5, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("figures/Hubble.png", dpi=300)
plt.close(fig)

# Figures 2 and 3: modern sample
method_colors = {"Cepheids": "#1f4e8c", "SBF": "#c0392b", "Maser": "#1a9850",
                 "SNIa": "#8e44ad", "SNII (optical)": "#e67e22", "TRGB": "#17a2b8"}
for lab, v, fname, title in [
    (r"v_{\rm helio}", v_helio, "figures/Velocity (Heliocentric) vs. Distance - NED Sample.png",
     "Modern sample: heliocentric velocity vs. distance"),
    (r"v_{\rm CMB}", v_cmb, "figures/Velocity (3K CMB) vs. Distance - NED Sample.png",
     "Modern sample: CMB-frame velocity vs. distance"),
]:
    H0o, se_o, _ = fit_origin(r_m, v)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for meth, col in method_colors.items():
        sel = (dfm.method == meth).values
        if sel.any():
            ax.scatter(r_m[sel], v[sel], color=col, s=34, label=meth, zorder=3)
    rr = np.linspace(0, r_m.max() * 1.03, 100)
    ax.plot(rr, H0o * rr, color="black", lw=1.4, ls="--",
            label=f"Fit through origin (H$_0$={H0o:.1f}$\\pm${se_o:.1f})")
    ax.set_xlabel("Comoving distance $D_C$ (Mpc)")
    ax.set_ylabel(f"Velocity ${lab}$ (km/s)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# Figure 4: comparison with the literature
lit_rows = [
    ("Planck 2020 (CMB, $\\Lambda$CDM)", *LITERATURE[0][1:]),
    ("Freedman et al. 2025 (CCHP, TRGB)", *LITERATURE[1][1:]),
    ("Riess et al. 2022 (SH0ES, Cepheids)", *LITERATURE[2][1:]),
    ("H0DN 2026 (Local Distance Network)", *LITERATURE[3][1:]),
]
rows = [(n, v, e, "#7f7f7f") for n, v, e in lit_rows] + [
    ("This work -- $v_{\\rm helio}$, unweighted", results["v_helio"]["H0o"], results["v_helio"]["se_o"], "#1f4e8c"),
    ("This work -- $v_{\\rm helio}$, weighted", weighted_results["v_helio"]["H0o"], weighted_results["v_helio"]["se_o"], "#1a9850"),
    ("This work -- $v_{\\rm CMB}$, unweighted", results["v_CMB"]["H0o"], results["v_CMB"]["se_o"], "#1f4e8c"),
    ("This work -- $v_{\\rm CMB}$, weighted", weighted_results["v_CMB"]["H0o"], weighted_results["v_CMB"]["se_o"], "#1a9850"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ypos = np.arange(len(rows))[::-1]
for y, (name, val, err, color) in zip(ypos, rows):
    ax.errorbar(val, y, xerr=err, fmt="o", color=color, ecolor=color, capsize=3, markersize=6, elinewidth=1.4)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
band_hi = max(v + e for _, v, e in LITERATURE[2:])
ax.axvspan(LITERATURE[0][1] - LITERATURE[0][2], band_hi, color="#f0f0f0", zorder=0)
ax.set_xlabel("$H_0$ (km/s/Mpc)")
ax.set_title("Comparison with the literature", fontsize=13)
ax.set_xlim(60, 90)
fig.tight_layout()
fig.savefig("figures/H0_comparison.png", dpi=300)
plt.close(fig)

print("Done. Figures written to figures/.")
