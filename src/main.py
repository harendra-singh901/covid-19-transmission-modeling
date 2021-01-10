"""
Main script: runs the SEIR model, fits it to (synthetic) case data, does
a basic sensitivity analysis, and saves plots to the figures/ folder.

Run with:  python src/main.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from seir_model import run_sir, run_seir, basic_reproduction_number, peak_infection_day
from fitting import fit_beta_gamma, make_synthetic_case_data, cumulative_cases_model
from sensitivity import one_at_a_time_sensitivity, percent_change

# figures/ lives next to src/, so build the path relative to this file
# rather than assuming what folder the script is run from
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def part1_compare_sir_seir():
    N = 1_000_000
    beta = 0.4
    gamma = 1 / 10       # ~10 day infectious period
    sigma = 1 / 5.2      # ~5.2 day incubation period (commonly cited early COVID estimate)

    t_sir, S_sir, I_sir, R_sir = run_sir(N, beta, gamma, days=200)
    t_seir, S_seir, E_seir, I_seir, R_seir = run_seir(N, beta, sigma, gamma, days=200)

    R0 = basic_reproduction_number(beta, gamma)
    print(f"R0 = {R0:.2f}")

    peak_day_sir, peak_val_sir = peak_infection_day(t_sir, I_sir)
    peak_day_seir, peak_val_seir = peak_infection_day(t_seir, I_seir)
    print(f"SIR peak: day {peak_day_sir:.0f}, {peak_val_sir:.0f} infected")
    print(f"SEIR peak: day {peak_day_seir:.0f}, {peak_val_seir:.0f} infected")

    plt.figure(figsize=(8, 5))
    plt.plot(t_sir, I_sir, label="SIR infected")
    plt.plot(t_seir, I_seir, label="SEIR infected")
    plt.plot(t_seir, E_seir, "--", label="SEIR exposed")
    plt.xlabel("Day")
    plt.ylabel("Number of people")
    plt.title("SIR vs SEIR infection curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/1_sir_vs_seir.png")
    plt.close()


def part2_fit_to_data():
    N = 500_000
    true_beta = 0.35
    true_gamma = 1 / 8
    sigma = 1 / 5.2
    days = 120

    t_data, cases_data = make_synthetic_case_data(N, true_beta, sigma, true_gamma, days)

    fitted_beta, fitted_gamma, errors = fit_beta_gamma(
        t_data, cases_data, N, sigma, beta_guess=0.3, gamma_guess=0.15
    )

    print(f"\nTrue beta={true_beta}, gamma={true_gamma:.3f}")
    print(f"Fitted beta={fitted_beta:.3f} (+/-{errors[0]:.3f}), "
          f"gamma={fitted_gamma:.3f} (+/-{errors[1]:.3f})")
    print(f"Fitted R0 = {basic_reproduction_number(fitted_beta, fitted_gamma):.2f} "
          f"(true R0 = {basic_reproduction_number(true_beta, true_gamma):.2f})")

    model_cases = cumulative_cases_model(t_data, fitted_beta, fitted_gamma, N, sigma, 1, 1)

    plt.figure(figsize=(8, 5))
    plt.scatter(t_data, cases_data, s=10, color="gray", label="synthetic 'observed' cases")
    plt.plot(t_data, model_cases, color="red", label="fitted model")
    plt.xlabel("Day")
    plt.ylabel("Cumulative cases")
    plt.title("Fitting SEIR to case data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/2_model_fit.png")
    plt.close()


def part3_sensitivity_analysis():
    N = 1_000_000
    base_beta = 0.4
    base_sigma = 1 / 5.2
    base_gamma = 1 / 10

    results = one_at_a_time_sensitivity(N, base_beta, base_sigma, base_gamma)
    baseline = results["baseline"]

    print("\nSensitivity analysis (+/-20% on each parameter):")
    print(f"Baseline peak infected: {baseline['peak_value']:.0f} on day {baseline['peak_day']:.0f}")

    param_names = ["beta", "sigma", "gamma"]
    low_changes = []
    high_changes = []

    for name in param_names:
        low_pct = percent_change(baseline["peak_value"], results[name]["low"]["peak_value"])
        high_pct = percent_change(baseline["peak_value"], results[name]["high"]["peak_value"])
        low_changes.append(low_pct)
        high_changes.append(high_pct)
        print(f"  {name}: -20% -> peak changes {low_pct:+.1f}%,  "
              f"+20% -> peak changes {high_pct:+.1f}%")

    x = np.arange(len(param_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, low_changes, width, label="-20%")
    plt.bar(x + width / 2, high_changes, width, label="+20%")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, param_names)
    plt.ylabel("% change in peak infected")
    plt.title("Sensitivity of peak infections to each parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/3_sensitivity.png")
    plt.close()


if __name__ == "__main__":
    print("Part 1: Comparing SIR and SEIR models")
    part1_compare_sir_seir()

    print("\nPart 2: Fitting the model to case data")
    part2_fit_to_data()

    print("\nPart 3: Sensitivity analysis")
    part3_sensitivity_analysis()

    print("\nDone. Figures saved to the figures/ folder.")
