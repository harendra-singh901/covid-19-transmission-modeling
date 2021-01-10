"""
SEIR model of COVID-19 spread.

This module sets up the classic SEIR compartmental model (Susceptible,
Exposed, Infected, Recovered) and solves it numerically with scipy's
odeint. The SIR model is also included since it's the simpler special
case (no exposed/incubation stage) and is useful as a sanity check.

References used while building this:
- Kermack & McKendrick (1927) for the original SIR idea
- Standard epidemiology course notes on the SEIR extension
- scipy documentation for odeint
"""

import numpy as np
from scipy.integrate import odeint


def sir_model(y, t, beta, gamma, N):
    """Basic SIR model equations.

    S: susceptible, I: infected, R: recovered
    beta: infection rate, gamma: recovery rate, N: total population
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def seir_model(y, t, beta, sigma, gamma, N):
    """SEIR model equations.

    Adds an Exposed compartment E for people who have been infected but
    are not yet infectious themselves (the incubation period).

    sigma is the rate at which exposed people become infectious
    (1/sigma = average incubation period in days).
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def run_sir(N, beta, gamma, I0=1, R0_init=0, days=160):
    """Solve the SIR model over a number of days and return t, S, I, R arrays."""
    S0 = N - I0 - R0_init
    y0 = [S0, I0, R0_init]
    t = np.linspace(0, days, days + 1)
    result = odeint(sir_model, y0, t, args=(beta, gamma, N))
    S, I, R = result.T
    return t, S, I, R


def run_seir(N, beta, sigma, gamma, E0=1, I0=1, R0_init=0, days=200):
    """Solve the SEIR model over a number of days and return t, S, E, I, R arrays."""
    S0 = N - E0 - I0 - R0_init
    y0 = [S0, E0, I0, R0_init]
    t = np.linspace(0, days, days + 1)
    result = odeint(seir_model, y0, t, args=(beta, sigma, gamma, N))
    S, E, I, R = result.T
    return t, S, E, I, R


def basic_reproduction_number(beta, gamma):
    """R0 = beta / gamma. Same formula for SIR and SEIR since the exposed
    stage doesn't change how many people an infectious person eventually
    infects, just delays it."""
    return beta / gamma


def peak_infection_day(t, I):
    """Find the day the infection curve peaks, and the peak value."""
    peak_idx = np.argmax(I)
    return t[peak_idx], I[peak_idx]
