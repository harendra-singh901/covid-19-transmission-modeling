"""
Fitting the SEIR model to case data.

This uses scipy's curve_fit (least squares) to estimate beta and gamma
from cumulative case counts. sigma (incubation rate) is fixed from
published estimates rather than fitted, because with only cumulative
case totals available there isn't enough information in the data to
pin down all three parameters at once (they trade off against each
other too easily).

This is a simpler approach than a full likelihood-based fit -- it just
minimises sum of squared errors between the model's cumulative cases
and the observed cumulative cases.
"""

import numpy as np
from scipy.optimize import curve_fit

from seir_model import seir_model
from scipy.integrate import odeint


def cumulative_cases_model(t, beta, gamma, N, sigma, E0, I0):
    """Run the SEIR model and return cumulative cases (E + I + R) at each
    time point, since that's what's usually reported as "total cases"."""
    S0 = N - E0 - I0
    y0 = [S0, E0, I0, 0]
    result = odeint(seir_model, y0, t, args=(beta, sigma, gamma, N))
    S, E, I, R = result.T
    cumulative = N - S  # everyone who has left the susceptible pool
    return cumulative


def fit_beta_gamma(t_data, cases_data, N, sigma, E0=1, I0=1,
                    beta_guess=0.4, gamma_guess=0.1):
    """Fit beta and gamma to observed cumulative case data.

    Returns the fitted (beta, gamma) and their estimated standard errors
    (from the covariance matrix curve_fit gives back).
    """

    def model_wrapper(t, beta, gamma):
        return cumulative_cases_model(t, beta, gamma, N, sigma, E0, I0)

    popt, pcov = curve_fit(
        model_wrapper,
        t_data,
        cases_data,
        p0=[beta_guess, gamma_guess],
        bounds=([0.01, 0.01], [2.0, 1.0]),
    )
    perr = np.sqrt(np.diag(pcov))
    beta_fit, gamma_fit = popt
    return beta_fit, gamma_fit, perr


def make_synthetic_case_data(N, beta_true, sigma, gamma_true, days, noise_level=0.05, seed=42):
    """Generate fake but realistic case data by running the model and
    adding some random noise, so the fitting function has something to
    be tested against with a known answer.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, days)
    S0 = N - 2
    y0 = [S0, 1, 1, 0]
    result = odeint(seir_model, y0, t, args=(beta_true, sigma, gamma_true, N))
    S, E, I, R = result.T
    cumulative = N - S
    noisy = cumulative * (1 + rng.normal(0, noise_level, size=cumulative.shape))
    noisy = np.clip(noisy, 0, N)
    return t, noisy
