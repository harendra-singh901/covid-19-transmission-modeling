"""
Simple one-at-a-time (OAT) sensitivity analysis for the SEIR model.

The idea: take the baseline parameters, change one parameter at a time
by a fixed percentage (keeping the others at baseline), rerun the
model, and see how much the outputs of interest (peak infected count,
day of peak, and total cases at the end) change.

This is a much simpler approach than global sensitivity methods (like
Sobol indices or PRCC) that would need a lot more simulation runs, but
it's a reasonable starting point for seeing which parameters matter
most.
"""

import numpy as np

from seir_model import run_seir, peak_infection_day


def run_scenario(N, beta, sigma, gamma, days=200):
    """Run the model once and pull out the three summary numbers we care about."""
    t, S, E, I, R = run_seir(N, beta, sigma, gamma, days=days)
    peak_day, peak_value = peak_infection_day(t, I)
    total_cases = N - S[-1]
    return {
        "peak_day": peak_day,
        "peak_value": peak_value,
        "total_cases": total_cases,
    }


def one_at_a_time_sensitivity(N, base_beta, base_sigma, base_gamma, days=200, change=0.2):
    """Vary each parameter up and down by `change` fraction (default +/-20%)
    and record the effect on the summary outputs.

    Returns a dictionary keyed by parameter name, each containing the
    baseline, low, and high scenario results.
    """
    baseline = run_scenario(N, base_beta, base_sigma, base_gamma, days)

    params = {
        "beta": base_beta,
        "sigma": base_sigma,
        "gamma": base_gamma,
    }

    results = {"baseline": baseline}

    for name, value in params.items():
        low_value = value * (1 - change)
        high_value = value * (1 + change)

        test_params_low = params.copy()
        test_params_low[name] = low_value
        low_result = run_scenario(N, test_params_low["beta"], test_params_low["sigma"],
                                   test_params_low["gamma"], days)

        test_params_high = params.copy()
        test_params_high[name] = high_value
        high_result = run_scenario(N, test_params_high["beta"], test_params_high["sigma"],
                                    test_params_high["gamma"], days)

        results[name] = {"low": low_result, "high": high_result}

    return results


def percent_change(baseline_value, new_value):
    """Helper to express a change as a percentage of the baseline."""
    return 100 * (new_value - baseline_value) / baseline_value
