import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

from assignments.config import *
from assignments.load_data import load_data


def calculate_waiting_time(data: np.ndarray) -> np.ndarray:
    """
    Calculate the waiting time for each neuron.

    Params
    ------
    - data (np.ndarray): The input data.

    Returns
    -------
    - np.ndarray: The waiting time for each neuron.
    """
    waiting_time = np.diff(data)
    return waiting_time


def calculate_tau0(waiting_time: np.ndarray) -> float:
    """
    Calculate the minimum waiting time.

    Params
    ------
    - waiting_time (np.ndarray): The waiting time for each neuron.

    Returns
    -------
    - float: The minimum waiting time.
    """
    tau0 = np.min(waiting_time)
    return tau0


def fit_exponential_distribution(waiting_time: np.ndarray) -> tuple:
    """
    Fit an exponential distribution to the waiting time data.

    Params
    ------
    - waiting_time (np.ndarray): The waiting time for each neuron.

    Returns
    -------
    - tuple: The parameters of the fitted exponential distribution.
    """
    params = expon.fit(waiting_time, floc=0)
    return params


def fit_pdf(waiting_time: np.ndarray, tau: np.ndarray) -> tuple:
    """
    Get the probability density function (PDF) of the fitted exponential distribution.

    Params
    ------

    - waiting_time (np.ndarray): The waiting time for each neuron.
    - tau (np.ndarray): The x values for the PDF.

    Returns
    -------
    - tuple: The x values and the PDF values.
    """
    params = fit_exponential_distribution(waiting_time)
    pdf_fitted = expon.pdf(tau, *params)
    return pdf_fitted


def exponensial_pdf(lambda_param: float, tau0: float, tau: np.ndarray) -> np.ndarray:
    """
    Calculate the PDF of the exponential distribution.

    Params
    ------
    - lambda_param (float): The lambda parameter for the exponential distribution.
    - tau0 (float): The minimum waiting time.
    - tau (np.ndarray): The x values for the PDF.

    Returns
    -------
    - np.ndarray: The PDF values of the exponential distribution.
    """
    return np.where(tau < tau0, 0, lambda_param * np.exp(-lambda_param * (tau - tau0)))


def calculate_spike_times(
    lambda_param: float, tau0: float, size: int = DATA_POINTS
) -> np.ndarray:
    """
    Calculate the spike times based on the interspike intervals (ISI).

    Params
    ------
    - lambda_param (float): The lambda parameter for the exponential distribution.
    - tau0 (float): The minimum waiting time.
    - size (int): The number of spike times to generate. Default is DATA_POINTS.

    Returns
    -------
    - spike_times (np.ndarray): The spike times.
    """
    isi = tau0 + np.random.exponential(scale=lambda_param, size=size)
    spike_times = np.cumsum(isi)
    return spike_times


def calculate_average_firing_rate(spike_times: np.ndarray) -> float:
    """
    Calculate the average firing rate of the neuron.

    Params
    ------
    - spike_times (np.ndarray): The spike times of the neuron.

    Returns
    - f (float): The average firing rate of the neuron.
    """
    f = len(spike_times) / (spike_times[-1] - spike_times[0])
    return f


def plot_waiting_time_distribution(
    waiting_time: np.ndarray,
    tau: np.ndarray,
    tau0: float,
    lambda_estimate: float,
    pdf_fitted: np.ndarray,
    pdf_values: np.ndarray,
    bins: int = DATA_BINS,
    enable_tau0: bool = False,
    enable_exponential_fit: bool = False,
    enable_exponential_distribution: bool = False,
    save: bool = False,
) -> None:
    """
    Plot the waiting time distribution.

    Params
    ------
    - waiting_time (np.ndarray): The waiting time for each neuron.
    - tau (np.ndarray): The x values for the PDF.
    - tau0 (float): The minimum waiting time.
    - lambda_estimate (float): The lambda parameter for the exponential distribution.
    - pdf_fitted (np.ndarray): The PDF values of the fitted exponential distribution.
    - pdf_values (np.ndarray): The PDF values of the exponential distribution.
    - bins (int): The number of bins for the histogram. Default is DATA_BINS.
    - enable_tau0 (bool): Whether to show the tau0 line. Default is False.
    - enable_exponential_fit (bool): Whether to show the fitted exponential distribution. Default is False.
    - enable_exponential_distribution (bool): Whether to show the exponential distribution. Default is False.
    - save (bool): Whether to save the plot. Default is False.
    """
    plt.hist(
        waiting_time,
        bins=bins,
        density=True,
    )

    if enable_tau0:
        plt.axvline(
            tau0,
            color="red",
            linestyle="--",
            label=f"$\\tau_0$ = {tau0:.2f} ms",
        )

    if enable_exponential_fit:
        plt.plot(
            tau,
            pdf_fitted,
            color="red",
            linestyle=":",
            label=f"Fitted Exponential Distribution $\\lambda={lambda_estimate:.2f}$",
        )

    if enable_exponential_distribution:
        plt.plot(
            tau,
            pdf_values,
            color="red",
            linestyle="-",
            label=f"Exponential Distribution $\\lambda={lambda_estimate:.2f}$, $\\tau_0={tau0:.2f}$",
        )

    plt.xlabel(r"Interspike interval $\tau$ (ms)")
    plt.ylabel(r"P($\tau$)")
    plt.title("Neuronal activity waiting time distribution")
    if True in [enable_tau0, enable_exponential_fit, enable_exponential_distribution]:
        plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(RESULT_DIR, "waiting_time_distribution.png"), dpi=FIG_DPI
        )
        plt.close()
    else:
        plt.show()


def assignment_1():
    ####################  Assignment 1 #####################################
    data = load_data("Data_neuron.txt")
    waiting_time = calculate_waiting_time(data)
    waiting_time = np.sort(waiting_time)[::-1]

    tau0 = calculate_tau0(waiting_time)
    params = fit_exponential_distribution(waiting_time)
    lambda_estimate = 1 / params[1]

    tau = np.linspace(0, max(waiting_time), DATA_POINTS)
    pdf_fitted = fit_pdf(waiting_time, tau)
    pdf_values = exponensial_pdf(lambda_estimate, tau0, tau)

    plot_waiting_time_distribution(
        waiting_time,
        tau,
        tau0,
        lambda_estimate,
        pdf_fitted,
        pdf_values,
        save=True,
        # enable_exponential_distribution=True,
        # enable_exponential_fit=True,
        # enable_tau0=True,
    )

    spike_times = calculate_spike_times(lambda_estimate, tau0)
    f = calculate_average_firing_rate(spike_times)
    print(f"Average firing rate: {f:.2f} Hz")
    ######################################################################


if __name__ == "__main__":
    assignment_1()
