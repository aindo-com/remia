import json
import time
import random
import numpy as np
import torch
import GPUtil
from scipy.stats import binom, beta
import os
import fnmatch
import pandas as pd


def printc(message, color):
    """
    Print a message to the terminal in the specified color.

    color: one of "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    """
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }
    color_code = colors.get(color.lower(), colors["reset"])

    if isinstance(message, dict):
        message = json.dumps(message, indent=4)
    print(f"{color_code}{message}{colors['reset']}")


def store_json(d, *, file: str):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


def load_json(file: str) -> dict:
    with open(file, "r") as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_available_device(
    mem_required: float = 0.05, verbose: bool = False, stop_if_no_free_gpu: bool = True
):
    if not torch.cuda.is_available():
        return "cpu"

    try:
        devices = GPUtil.getGPUs()

        device_usages = [
            (device.id, device.memoryUsed / device.memoryTotal) for device in devices
        ]

        device_usages.sort(key=lambda x: x[1])

        if device_usages[0][1] > 1 - mem_required:
            if stop_if_no_free_gpu:
                raise RuntimeError("No GPU with sufficient free memory is available.")
            return "cpu"

        out = "cuda:" + str(device_usages[0][0])
        if verbose:
            print("\033[92m" + f"Using {out}" + "\033[0m")
        return out
    except ValueError as e:
        print(e)
        if stop_if_no_free_gpu:
            raise RuntimeError("Failed to retrieve GPU information.") from e
        return "cpu"


def balanced_acc_p_value(acc: float, n: int):
    return 1 - binom.cdf(k=int(acc * n), n=n, p=0.5)


def bayesian_accuracy_significance(acc: float, n: int):
    successes = round(acc * n)
    failures = n - successes
    # Using a uniform prior (alpha=1, beta=1)
    return 1 - beta.cdf(0.5, successes + 1, failures + 1)


def find_files(*, starting_folder: str = ".", pattern: str):
    """
    find all files that match the given pattern, starting from the given folder and going down the directory tree
    """
    matches = []
    for root, _, files in os.walk(starting_folder):
        for filename in files:
            full_name = os.path.join(root, filename)
            if fnmatch.fnmatch(full_name, pattern):
                matches.append(full_name)
    return matches


def pop_data(
    df: pd.DataFrame, n: int, random: bool = False, random_state: int | None = None
) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(
            f"Cannot pop {n} records from a dataset with only {len(df)} records"
        )
    if random:
        extracted_data = df.sample(n, random_state=random_state, replace=False)
    else:
        extracted_data = df.iloc[:n].copy()
    df.drop(extracted_data.index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return extracted_data.reset_index(drop=True)
