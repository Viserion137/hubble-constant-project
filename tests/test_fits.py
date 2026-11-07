"""Checks that the fitting functions recover known answers on synthetic data."""
import numpy as np
import pytest

import hubble_fits as hf


def test_origin_and_free_fits_recover_exact_line():
    r = np.linspace(1, 100, 20)
    H0, se, r2 = hf.fit_origin(r, 70.0 * r)
    assert H0 == pytest.approx(70.0, abs=1e-10)
    assert se == pytest.approx(0.0, abs=1e-8)
    H0f, _, b, _ = hf.fit_free(r, 70.0 * r + 150.0)
    assert H0f == pytest.approx(70.0, abs=1e-8)
    assert b == pytest.approx(150.0, abs=1e-6)


def test_origin_fit_unbiased_on_noisy_data():
    rng = np.random.default_rng(1)
    r = rng.uniform(5, 150, 2000)
    v = 72.0 * r + rng.normal(0, 300, r.size)
    H0, se, _ = hf.fit_origin(r, v)
    assert abs(H0 - 72.0) < 3 * se


def test_weighted_fits_reduce_to_ols_for_equal_weights():
    rng = np.random.default_rng(2)
    r = rng.uniform(5, 100, 50)
    v = 70 * r + rng.normal(0, 200, r.size)
    w = np.ones_like(r)
    assert hf.fit_origin_weighted(r, v, w)[0] == pytest.approx(hf.fit_origin(r, v)[0])
    assert hf.fit_free_weighted(r, v, w)[0] == pytest.approx(hf.fit_free(r, v)[0])


def test_sigma_pec_recovered_from_scatter():
    rng = np.random.default_rng(3)
    r = rng.uniform(5, 150, 3000)
    f = np.full_like(r, 0.05)
    v = 70 * r * (1 + rng.normal(0, 0.05, r.size)) + rng.normal(0, 350, r.size)
    assert hf.sigma_pec_chi2(r, v, f) == pytest.approx(350, rel=0.1)
    H0_ml, s_ml = hf.ml_origin(r, v, f)
    assert s_ml == pytest.approx(350, rel=0.1)
    assert H0_ml == pytest.approx(70, rel=0.02)


def test_odr_matches_weighted_fit_without_distance_errors():
    rng = np.random.default_rng(4)
    r = rng.uniform(5, 100, 40)
    v = 70 * r + rng.normal(0, 200, r.size)
    sv = np.full_like(r, 200.0)
    H0_odr, _ = hf.odr_origin(r, v, np.zeros_like(r), sv)
    assert H0_odr == pytest.approx(hf.fit_origin(r, v)[0], abs=1e-6)


def test_odr_corrects_attenuation_bias():
    rng = np.random.default_rng(5)
    r_true = rng.uniform(1, 40, 4000)
    r_obs = r_true * (1 + rng.normal(0, 0.3, r_true.size))
    v = 70 * r_true + rng.normal(0, 50, r_true.size)
    H0_ols = hf.fit_origin(r_obs, v)[0]
    # errors computed from the true distances: an error model scaled by the
    # *observed* r is itself correlated with the noise when errors are this large
    H0_odr, _ = hf.odr_origin(r_obs, v, 0.3 * r_true, np.full_like(r_obs, 50.0))
    assert H0_ols < 66          # attenuated
    assert abs(H0_odr - 70) < abs(H0_ols - 70)


def test_comoving_velocity_low_z_expansion():
    # c * int_0^z dz/E = cz [1 - (1 + q0) z / 2 + ...], q0 = 3 Om / 2 - 1
    q0 = 1.5 * hf.OMEGA_M - 1
    for z in (1e-4, 0.01, 0.034):
        v = hf.C_KMS * z
        expected = v * (1 - (1 + q0) * z / 2)
        assert hf.comoving_velocity(v)[0] == pytest.approx(expected, rel=2e-4)


def test_comoving_distance_conversion():
    r = np.array([100.0, 100.0])
    out = hf.to_comoving_distance(r, np.array([0.03, 0.03]), angular=np.array([True, False]))
    assert out[0] == pytest.approx(103.0)
    assert out[1] == pytest.approx(100 / 1.03)


def test_solar_motion_fit_recovers_synthetic_parameters():
    rng = np.random.default_rng(6)
    n = 60
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    r = rng.uniform(0.1, 2.0, n)
    K, X, Y, Z = 500.0, -60.0, 220.0, -190.0
    v = K * r + hf.unit_vectors(ra, dec) @ np.array([X, Y, Z])
    s = hf.fit_hubble_solar_motion(r, v, ra, dec)
    assert (s["K"], s["X"], s["Y"], s["Z"]) == pytest.approx((K, X, Y, Z), abs=1e-6)


def test_solar_reflex_toward_galactic_centre_and_m31():
    # toward the Galactic centre the reflex is U = 11.1 km/s
    assert hf.solar_reflex(np.array([266.405]), np.array([-28.936]))[0] == pytest.approx(11.1, abs=0.1)
    # van der Marel et al. (2012) quote 191.9 km/s at the position of M31
    assert hf.solar_reflex(np.array([10.6848]), np.array([41.2691]))[0] == pytest.approx(191.9, abs=0.2)
