import os

import matplotlib.pyplot as plt
import numpy as np

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


def plot_waiting_time_distribution(
    waiting_time: np.ndarray, tau0: float, save: bool = False
) -> None:
    """
    Plot the waiting time distribution.

    Params
    ------
    - waiting_time (np.ndarray): The waiting time for each neuron.
    - tau0 (float): The minimum waiting time.
    - save (bool): Whether to save the plot.
    """
    plt.hist(
        waiting_time,
        bins=50,
    )
    plt.axvline(tau0, color="red", linestyle="--", label=f"$\\tau_0$ = {tau0:.2f} ms")
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


def main():
    data = load_data("Data_neuron.txt")
    waiting_time = calculate_waiting_time(data)
    waiting_time = np.sort(waiting_time)[::-1]
    tau0 = calculate_tau0(waiting_time)
    plot_waiting_time_distribution(waiting_time, tau0, save=True)


if __name__ == "__main__":
    main()

np.sort
