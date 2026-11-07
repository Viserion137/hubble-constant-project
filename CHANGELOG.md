# Changelog

All releases are archived on Zenodo under the concept DOI
[10.5281/zenodo.22260280](https://doi.org/10.5281/zenodo.22260280).

## v1.3.0 (November 2026)

- **Distances and redshifts on the same footing.** The maser distances (Reid et al. 2019; Pesce et al. 2020) are angular-diameter distances, the Cepheid, TRGB, SBF and supernova distances luminosity distances; the linear Hubble law holds exactly for the comoving distance. All distances are now converted to comoving distance and all velocities to H₀·D_C (flat ΛCDM, Ω_m = 0.315, as in Pesce et al. 2020, Eq. 1). The maser v/r change by −1.4% to −4.1%; the origin-forced estimates fall by about 2.1 km/s/Mpc (heliocentric: 74.6 → 72.5). The introduction's second-order distance–redshift formula had the wrong sign and is corrected.
- **Validation.** With the same conversion, the weighted fit of the six MCP galaxies gives 73.1 ± 3.0 km/s/Mpc against the published 73.9 ± 3.0 (−0.18σ; previously +0.36σ, attributed to the asymmetric likelihoods).
- **Hubble's own model.** Part 1 now also fits Hubble's equations of condition, v = K r + X cos α cos δ + Y sin α cos δ + Z sin δ, which include the motion of the Sun: K = 465.2 ± 55.7 km/s/Mpc, solar apex (286°, +39°), mean residual 150 km/s, as published by Hubble (465 ± 50, (286°, +40°), 150 km/s). `hubble1929.csv` now includes J2000 coordinates.
- **M31 and M81.** Their negative heliocentric velocities are mostly the reflex of the Sun's orbit in the Milky Way (+192 and +115 km/s along those lines of sight): M31's Galactocentric radial velocity is −109 km/s, M81 recedes. The report's interpretation is corrected.
- **Peculiar-velocity floor estimated from the data**: σ_pec = 368 km/s (heliocentric) and 528 km/s (CMB) give χ²/dof = 1 (maximum likelihood: 372 and 529 km/s), replacing the fixed 200 km/s, which gave χ²/dof = 3–6. 150, 200 and 300 km/s are kept as sensitivity tests.
- **Maser velocities.** The six maser galaxies now use the systemic velocities of Pesce et al. (2020, Table 1), also used in the validation (NED's values differed by up to 140 km/s). Heliocentric values are obtained with NED's frame correction for each direction. New column `v_source` in `ned_modern.csv`.
- **Monte Carlo check**: on 1000 simulated samples with the real distances and error model, OLS, weighted and ODR fits are unbiased to within 0.3 km/s/Mpc.
- **Literature.** SH0ES baseline value 73.04 ± 1.04 (Riess et al. 2022; 73.3 was the variant including high-redshift supernovae); H0DN 2026 (73.50 ± 0.81) added.
- **Code.** Functions moved to `hubble_fits.py`; ODR written out in its effective-variance form (identical results, no dependence on the deprecated `scipy.odr`); unit tests in `tests/`; the CI now runs the tests and checks that a fresh run reproduces `analysis_output.txt` exactly.
- **Text.** Effective number of galaxies (5.6) stated; minimum distance 1.88 Mpc; abstract, discussion, limitations and conclusion rewritten for the new numbers.

## v1.2.1 (October 2026)

- **Distances.** NGC 4594 (Sombrero) was listed as a Cepheid distance of 6.22 Mpc, a value that is actually a globular-cluster luminosity function estimate; it now uses the TRGB distance of 9.55 Mpc (McQuinn et al. 2016). NGC 300 (1.49 → 1.88 Mpc, Gieren et al. 2005) and NGC 2403, NGC 3351, NGC 3627 (→ 3.13, 9.33, 9.38 Mpc, HST Key Project, Freedman et al. 2001) were corrected after checking against the literature. The unweighted full-sample estimates change by at most 0.22 km/s/Mpc.
- **Cepheid residuals.** The mean residual is now quoted with its standard error: +0.16 ± 0.10 (1.6σ), previously +0.24 ± 0.11 (2.2σ) and described as "marginal".
- **Report.** The history of Hubble's overestimate is corrected: Baade's Cepheid recalibration accounts for only about a factor of 2; the rest came from the later revision of the brightest-star distances (HII regions mistaken for stars: Humason, Mayall & Sandage 1956, H₀ ≈ 180; Sandage 1958, 50–100). Hubble used Cepheids only for the nearest objects. Minor wording fixes.

## v1.2.0 (October 2026)

- **ODR fix.** The velocity error passed to the ODR check also contained the distance term `H0·f·r`, so the distance error was counted twice (ODR already propagates `sigma_r` onto the velocity axis). `sigma_v` now contains only the peculiar-velocity floor and a 5 km/s measurement error.
- **Maser distances.** The five Megamaser Cosmology Project galaxies now use the distances of Pesce et al. (2020, ApJL 891, L1), which supersede the earlier individual determinations; CGCG 074-064 uses its geometric maser distance (87.6 Mpc) instead of the Fundamental Plane one. This raises the origin-forced estimates by 2.4–2.5 km/s/Mpc relative to v1.1.0, and changes the maser sensitivity result: removing the six masers no longer shifts H₀ significantly (the earlier ~10 km/s/Mpc shift came largely from CGCG 074-064's Fundamental Plane distance), but more than doubles its uncertainty.
- **New checks.** Jackknife and bootstrap standard errors printed next to the analytic ones; a paired bootstrap comparing the ten estimators with their own sampling noise; distances from Planck, CCHP and SH0ES in units of σ; a validation of the weighted method on the Pesce et al. (2020) maser sample (75.4 ± 3.1 vs the published 73.9 ± 3.0).
- **Report.** Introduction updated with the CCHP 2025 TRGB result and a neutral statement of the tension; discussion and conclusion rewritten for the new numbers.

## v1.1.0 (September 2026)

- Concept DOI used everywhere; NGC 4258 maser distance 7.58 Mpc (Reid et al. 2019); nearby-subsample fits (r ≤ 25 Mpc) computed in the script; sample sizes computed from the data (n = 39); `.gitignore`, `requirements.txt`, `CITATION.cff` and CI workflow added.

## v1.0.0 / v1.0.1 (September 2026)

- First public release of the analysis, report and data.
