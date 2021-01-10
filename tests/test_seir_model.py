"""
Basic tests for the SEIR model code.

Run with: pytest (from the project root, with src/ on the path)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seir_model import run_sir, run_seir, basic_reproduction_number, peak_infection_day


def test_population_stays_constant_sir():
    # S + I + R should always add up to N, since nobody enters or leaves
    # the population in this model
    N = 100_000
    t, S, I, R = run_sir(N, beta=0.3, gamma=0.1, days=100)
    total = S + I + R
    assert np.allclose(total, N, rtol=1e-4)


def test_population_stays_constant_seir():
    N = 100_000
    t, S, E, I, R = run_seir(N, beta=0.3, sigma=0.2, gamma=0.1, days=100)
    total = S + E + I + R
    assert np.allclose(total, N, rtol=1e-4)


def test_r0_formula():
    # R0 should just be beta/gamma
    assert basic_reproduction_number(0.4, 0.1) == pytest.approx(4.0)
    assert basic_reproduction_number(0.2, 0.1) == pytest.approx(2.0)


def test_no_outbreak_when_r0_below_one():
    # if R0 < 1 the infection should die out quickly and not spread
    # to a large fraction of the population
    N = 100_000
    t, S, I, R = run_sir(N, beta=0.05, gamma=0.1, days=200)
    assert R[-1] < 0.05 * N  # fewer than 5% ever get infected


def test_large_outbreak_when_r0_above_one():
    # with a high R0 most of the population should eventually be infected
    N = 100_000
    t, S, I, R = run_sir(N, beta=0.5, gamma=0.1, days=300)
    assert R[-1] > 0.8 * N


def test_seir_peak_later_than_sir_peak():
    # because of the extra exposed/incubation stage, SEIR should peak
    # later than SIR with the same beta and gamma
    N = 500_000
    beta, gamma, sigma = 0.4, 0.1, 1 / 5
    t_sir, S_sir, I_sir, R_sir = run_sir(N, beta, gamma, days=200)
    t_seir, S_seir, E_seir, I_seir, R_seir = run_seir(N, beta, sigma, gamma, days=200)

    peak_day_sir, _ = peak_infection_day(t_sir, I_sir)
    peak_day_seir, _ = peak_infection_day(t_seir, I_seir)

    assert peak_day_seir > peak_day_sir


def test_peak_infection_day_finds_max():
    t = np.array([0, 1, 2, 3, 4])
    I = np.array([10, 50, 100, 60, 20])
    day, value = peak_infection_day(t, I)
    assert day == 2
    assert value == 100
