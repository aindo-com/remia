import torch
import random
import numpy as np

from synthcity.plugins import Plugins
import argparse
import pandas as pd
import json
from joblib import Memory
import sys

sys.path.append(".")

from src.env_constants import CACHE_LOCATION

memory = Memory(location=CACHE_LOCATION, verbose=0)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@memory.cache
def cached_generator(
    *,
    generator_name,
    config: dict = {},
    dataset: pd.DataFrame,
    size: int,
    seed: int = 0,
) -> pd.DataFrame:
    set_seed(seed)
    model = Plugins().get(generator_name, **config)
    model.fit(dataset)
    return model.generate(count=size).dataframe()


def generate_with_cache_info(*args, **kwargs):
    """Wrapper that prints cache status."""
    # Check if result is cached by calling with check_cache_only
    cache_key = memory.cache(cached_generator).check_call_in_cache(*args, **kwargs)

    if cache_key:
        print("\033[92mLoading synthetic data from cache...\033[0m")
    else:
        print("\033[93mGenerating new synthetic data...\033[0m")

    return cached_generator(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-D", type=str)
    parser.add_argument("--output_file", "-O", type=str, default="synthetic.csv")
    parser.add_argument(
        "--model",
        "-M",
        type=str,
        choices=[
            "nflow",
            "aim",
            "arf",
            "ddpm",
            "bayesian_network",
            "radialgan",
            "fflows",
            "adsgan",
            "image_adsgan",
            "timegan",
            "dpgan",
            "survival_ctgan",
            "marginal_distributions",
            "survival_gan",
            "rtvae",
            "decaf",
            "survival_nflow",
            "ctgan",
            "great",
            "privbayes",
            "dummy_sampler",
            "timevae",
            "survae",
            "pategan",
            "image_cgan",
            "uniform_sampler",
            "tvae",
        ],
        help="benchmarking generative model used for synthesis",
    )
    parser.add_argument(
        "--size", "-S", type=int, default=0, help="size of generated dataset"
    )
    parser.add_argument(
        "--config",
        "-C",
        type=str,
        default=None,
        help="JSON config for model hyperparameters",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducibility"
    )
    args = parser.parse_args()
    # Parse config if provided
    config = {}
    if args.config:
        config = json.loads(args.config)

    if args.model in ["privbayes", "pategan", "dpgan"] and "epsilon" not in config:
        config = {"epsilon": 1000.0}

    if args.model == "ddpm":
        config = {"n_iter": 1_000, "batch_size": 1000}

    df = pd.read_csv(args.dataset_path)

    if args.size == 0:
        args.size = len(df)

    df_synth = generate_with_cache_info(
        generator_name=args.model,
        config=config,
        dataset=df,
        size=args.size,
        seed=args.seed,
    )

    df_synth.to_csv(args.output_file, index=False)
    # print(f"Synthetic data generated and saved to {args.output_file}")
