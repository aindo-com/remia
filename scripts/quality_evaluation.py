import sys

sys.path.append(".")
import os
import itertools
from src.util import printc, store_json, load_json, pop_data, Timer
from src.xgboost_discriminator import xgboost_discriminator_metrics_with_kfold
from src.xgboost_utility import xgboost_utility_with_kfold
from src.generators import get_generator, LeakyGenerator
from scipy.stats import wasserstein_distance_nd
import pandas as pd
from joblib import Memory
from experiments_constants import *
from src.env_constants import CACHE_LOCATION

memory = Memory(location=CACHE_LOCATION, verbose=0)


def load_metadata(dataset: str) -> dict:
    return load_json(f"data/{dataset}/metadata.json")


def load_df(dataset: str) -> pd.DataFrame:
    return pd.read_csv(f"data/{dataset}/data.csv")


def encode(df, metadata):
    df_mod = df.copy()
    for col in df.columns:
        if metadata[col] == "categorical":
            df_mod[col] = df_mod[col].astype("category")
        else:
            df_mod[col] = df_mod[col].astype(float)
    df_ohe = pd.get_dummies(
        df_mod,
        columns=df_mod.select_dtypes(include=["object", "category", "bool"]).columns,
        dtype=float,
    )
    return df_ohe.values


def wasserstein(df1: pd.DataFrame, df2: pd.DataFrame, metadata: dict) -> float:
    x = encode(df1, metadata)
    y = encode(df2, metadata)
    n = len(x)
    baseline = wasserstein_distance_nd(x[: n // 2], x[n // 2 :])
    distance = wasserstein_distance_nd(x[: n // 2], y[: n // 2])
    return distance - baseline


# @memory.cache
def quality_metrics(
    *, df_original: pd.DataFrame, df_synthetic: pd.DataFrame, metadata: dict
) -> dict:
    return {
        "xgboost_discriminator": xgboost_discriminator_metrics_with_kfold(
            df_original.copy(), df_synthetic.copy()
        ),
        "xgboost_utility": xgboost_utility_with_kfold(
            df_original.copy(), df_synthetic.copy()
        ),
        # compare with hold out set
        # "wasserstein_distance": wasserstein_distance_nd(
        #     encode(df_original, metadata), encode(df_synthetic, metadata)
        # ),
    }


def evaluate_quality(
    *, dataset: str, generator_name: str, seed: int = 0, size: int
) -> dict:

    metadata = load_metadata(dataset)
    df_shuffled = (
        load_df(dataset).sample(frac=1, random_state=seed).reset_index(drop=True)
    )
    df_train = pop_data(df_shuffled, size)
    df_test = pop_data(df_shuffled, size)

    if generator_name.startswith("leak"):
        df_additional = pop_data(df_shuffled, size)
        assert len(df_additional) == size, "Not enough data for additional leak dataset"
        generator = LeakyGenerator(
            leak_frac=float(generator_name.split("_")[-1]),
            additional_data=df_additional,
            seed=seed,
        )
    else:
        generator = get_generator(generator_name, metadata=metadata, seed=seed)

    df_synthetic = generator.fit_generate(train_dataset=df_train, n=len(df_train))

    return quality_metrics(
        df_original=df_test, df_synthetic=df_synthetic, metadata=metadata
    )


def quality_experiment(
    *, dataset: str, generator: str, size: int, seed: int, output_file: str
):
    with Timer() as t:
        quality_metrics = evaluate_quality(
            dataset=dataset, generator_name=generator, seed=seed, size=size
        )
    setting = {"dataset": dataset, "generator": generator, "size": size, "seed": seed}
    result = {"setting": setting, "quality_metrics": quality_metrics, "time": t.elapsed}
    store_json(result, file=output_file)
    printc(result, color="green")


def main(
    *,
    dataset: str,
    generator: str,
    size: int,
    seed: int,
    overwrite: bool = False,
):
    output_file = f"experiments/quality_evaluation/{generator}_{dataset}_size{size}_seed{seed}.json"
    if not overwrite and os.path.exists(output_file):
        print(f"Experiment already exists: {output_file}")
    else:
        printc(
            f"Running quality evaluation for {generator} on {dataset} with size {size} and seed {seed}",
            color="yellow",
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            quality_experiment(
                dataset=dataset,
                generator=generator,
                size=size,
                seed=seed,
                output_file=output_file,
            )
            printc("Done!", color="yellow")
        except Exception as e:
            printc(f"Error running experiment: {e}", color="red")


def main_multi(
    *,
    generators: list[str],
    datasets: list[str],
    size: int,
    seeds: list[int] = [0],
    overwrite: bool = False,
):
    for seed, generator, dataset in itertools.product(seeds, generators, datasets):
        main(
            dataset=dataset,
            generator=generator,
            size=size,
            seed=seed,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    all_generators_but_leaks = [g for g in ALL_GENERATORS if not g.startswith("leak")]
    main_multi(
        generators=all_generators_but_leaks,
        datasets=ALL_DATASETS,
        size=5_000,
        seeds=[0, 1, 2, 3],
        overwrite=False,
    )
    print("All experiments completed!")
