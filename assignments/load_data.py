import os

import numpy as np

from assignments.config import DATA_DIR


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
