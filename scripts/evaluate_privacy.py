import sys

sys.path.append(".")
from src.experiment import metric_experiment
import argparse

DEBUG = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        "-M",
        type=str,
        required=not DEBUG,
        default="remia_1.0",
    )
    parser.add_argument(
        "--dataset",
        "-D",
        type=str,
        required=not DEBUG,
        default="adult",
    )
    parser.add_argument(
        "--generator", "-G", type=str, required=not DEBUG, default="ctgan"
    )
    parser.add_argument("--seed", "-S", type=int, default=0)
    parser.add_argument("--training_size", "-T", type=int, default=1000)
    args = parser.parse_args()

    metric_experiment(**vars(args), verbose=True)


if __name__ == "__main__":
    main()
