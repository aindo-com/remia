import sys

sys.path.append(".")
from src.util import load_json, find_files
import tqdm
import numpy as np
from run_experiments import OUTPUT_FOLDER


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*.json"
    files = find_files(starting_folder=OUTPUT_FOLDER, pattern=pattern)
    times = []
    for file in tqdm.tqdm(files):
        if "time" in load_json(file):
            times.append(load_json(file)["time"])

    times = np.array(times)
    total_time = float(times.sum())

    # print total time in seconds, in minutes, in hours
    print(f"Total experiments: {len(files)}")
    print(f"Total time: {total_time:,.2f} seconds")
    print(f"Total time: {total_time / 60:,.2f} minutes")
    print(f"Total time: {total_time / 3600:,.2f} hours")
    print(
        f"Time per experiment: {float(times.mean()):.2f}+-{float(times.std(ddof=1)):.2f} seconds"
    )
    print(
        f"Time per experiment range: {float(times.min()):.2f} - {float(times.max()):.2f} seconds"
    )


if __name__ == "__main__":
    main()
