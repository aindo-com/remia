import sys

sys.path.append(".")
from src.experiment import metric_experiment
import os
import itertools
from src.util import printc
from scripts.quality_evaluation import main as quality_experiment
from datetime import datetime
import random
from experiments_constants import (
    ALL_METRICS,
    ALL_GENERATORS,
    ALL_DATASETS,
)

OUTPUT_FOLDER = "experiments/privacy_evaluation"


def experiment_exists(
    *, metric: str, dataset: str, generator: str, seed: int, training_size: int
) -> bool:
    if not os.path.exists(f"{OUTPUT_FOLDER}/{metric}/{generator}"):
        return False

    return (
        len(
            list(
                filter(
                    lambda x: (
                        x.endswith(".json")
                        and x.startswith(f"{dataset}_{training_size}_{seed}")
                    ),
                    os.listdir(f"{OUTPUT_FOLDER}/{metric}/{generator}"),
                )
            )
        )
        > 0
    )


def single_experiment(
    *,
    metric: str,
    dataset: str,
    generator: str,
    training_size: int,
    seed: int = 0,
    repeat: bool = False,
):
    has_run = False
    if (
        experiment_exists(
            metric=metric,
            generator=generator,
            dataset=dataset,
            seed=seed,
            training_size=training_size,
        )
        and not repeat
    ):
        printc("Experiment already exists, skipping...", color="yellow")
    else:
        print("\n" * 3)
        print("-" * 60)
        printc(f"Running {metric}, {generator}, {dataset}\n", color="yellow")
        try:
            os.makedirs(f"{OUTPUT_FOLDER}/{metric}/{generator}", exist_ok=True)
            metric_experiment(
                metric=metric,
                dataset=dataset,
                generator=generator,
                training_size=training_size,
                seed=seed,
                verbose=True,
                output_file=f"{OUTPUT_FOLDER}/{metric}/{generator}/{dataset}_{training_size}_{seed}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S_%f')}.json",
            )
            has_run = True
            printc("Done!", color="yellow")
        except Exception as e:
            printc(f"Error running experiment: {e}", color="red")

    return has_run


def multiple_experiments(
    *,
    metrics: list[str],
    generators: list[str],
    datasets: list[str],
    training_sizes: list[int] = [1_000],
    seeds: list[int] = [0],
    repeat: bool = False,
    random_order: bool = False,
    quality: bool = False,
):
    experiments_to_run = list(
        itertools.product(seeds, metrics, generators, datasets, training_sizes)
    )

    if not repeat:
        experiments_to_run = list(
            filter(
                lambda x: (
                    not experiment_exists(
                        metric=x[1],
                        generator=x[2],
                        dataset=x[3],
                        training_size=x[4],
                        seed=x[0],
                    )
                ),
                experiments_to_run,
            )
        )

    if random_order:
        random.shuffle(experiments_to_run)
    printc(f"Total experiments to run: {len(experiments_to_run)}", color="yellow")

    for seed, metric, generator, dataset, training_size in experiments_to_run:
        has_run = single_experiment(
            metric=metric,
            generator=generator,
            dataset=dataset,
            training_size=training_size,
            seed=seed,
            repeat=repeat,
        )

        if has_run and quality:
            quality_experiment(
                dataset=dataset,
                generator=generator,
                size=10_000,
                seed=seed,
                overwrite=True,
            )


if __name__ == "__main__":
    multiple_experiments(
        metrics=ALL_METRICS,
        generators=ALL_GENERATORS,
        datasets=ALL_DATASETS,
        repeat=False,
        quality=False,
        random_order=False,
        seeds=[0, 1, 2, 3],
        training_sizes=[1_000],
    )

    print("All experiments completed!")
