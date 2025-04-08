import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

from config import *


def load_data(file: str, dtype: str = "float") -> np.ndarray:
    """
    Load data from a file and return it as a numpy array.

    Params
    ------
    - file (str): The name of the file to load.
    - dtype (str): The data type of the file. Can be "float" or "int".

    Returns
    -------
    - np.ndarray: The loaded data as a numpy array.
    """
    file_path = os.path.join(DATA_DIR, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    assert dtype in ["float", "int"], "dtype must be 'float' or 'int'"

    if dtype == "float":
        data = np.loadtxt(file_path, dtype=float)
    elif dtype == "int":
        data = np.loadtxt(file_path, dtype=int)
    else:
        raise ValueError("Unsupported data type")
    return data


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


def calculate_lambda(waiting_time: np.ndarray) -> float:
    """
    Calculate the lambda parameter for the exponential distribution.

    Params
    ------
    - waiting_time (np.ndarray): The waiting time for each neuron.

    Returns
    -------
    - float: The lambda parameter.
    """
    params = fit_exponential_distribution(waiting_time)
    mean = params[1]
    lambda_estimate = 1 / mean
    return lambda_estimate


def fit_pdf(waiting_time: np.ndarray) -> tuple:
    """
    Get the probability density function (PDF) of the fitted exponential distribution.

    Params
    ------
    - waiting_time (np.ndarray): The waiting time for each neuron.

    Returns
    -------
    - tuple: The x values and the PDF values.
    """
    x = np.linspace(0, max(waiting_time), len(waiting_time))
    params = fit_exponential_distribution(waiting_time)
    pdf_fitted = expon.pdf(x, *params)
    return x, pdf_fitted


def plot_waiting_time_distribution(
    waiting_time: np.ndarray,
    tau0: float,
    lambda_estimate: float,
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
    - tau0 (float): The minimum waiting time.
    - tau (np.ndarray): The x values for the PDF.
    - lambda_estimate (float): The lambda parameter for the exponential distribution.
    - pdf_fitted (np.ndarray): The PDF values of the fitted exponential distribution.
    - pdf_values (np.ndarray): The PDF values of the exponential distribution.

    - save (bool): Whether to save the plot.
    """
    plt.hist(
        waiting_time,
        bins=50,
        density=True,
    )

    if enable_tau0:
        plt.axvline(
            tau0, color="red", linestyle="--", label=f"$\\tau_0$ = {tau0:.2f} ms"
        )

    if enable_exponential_fit:
        tau, pdf_fitted = fit_pdf(waiting_time)
        plt.plot(
            tau,
            pdf_fitted,
            color="red",
            linestyle=":",
            label=f"Fitted Exponential Distribution $\\lambda={lambda_estimate:.2f}$",
        )

    if enable_exponential_distribution:
        tau = np.linspace(0, max(waiting_time), 100)
        pdf_values = pdf(lambda_estimate, tau0, tau)

        plt.plot(
            tau,
            pdf_values,
            color="green",
            linestyle="-",
            label=f"Exponential Distribution $\\lambda={lambda_estimate:.2f}$, $\\tau_0={tau0:.2f}$",
        )

    plt.xlabel(r"Interspike interval $\tau$ (ms)")
    plt.ylabel(r"P($\tau$)")
    plt.title("Neuronal activity waiting time distribution")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(RESULT_DIR, "waiting_time_distribution.png"), dpi=FIG_DPI
        )
        plt.close()
    else:
        plt.show()


def pdf(lambda_param, tau0, tau):
    value = lambda_param * np.exp(-lambda_param * (tau * tau0))
    return value


def main():
    data = load_data("Data_neuron.txt")
    waiting_time = calculate_waiting_time(data)
    waiting_time = np.sort(waiting_time)[::-1]

    tau0 = calculate_tau0(waiting_time)
    lambda_estimate = calculate_lambda(waiting_time)
    plot_waiting_time_distribution(
        waiting_time,
        tau0,
        lambda_estimate,
        save=True,
        enable_exponential_distribution=True,
    )


if __name__ == "__main__":
    main()

np.sort
