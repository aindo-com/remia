import sys

sys.path.append(".")
from scripts.experiments_constants import (
    FAST_METRICS,
    ALL_DATASETS,
    ALL_GENERATORS,
    SHADOW_MODELING_METRICS,
    SHADOW_MODELING_GENERATORS,
)
from scripts.run_experiments import multiple_experiments as privacy_experiments
from scripts.quality_evaluation import main_multi as quality_experiments

SEEDS = [0, 1, 2, 3]
TRAINING_SIZES = [1000]

if __name__ == "__main__":
    all_generators_but_leaks = [g for g in ALL_GENERATORS if not g.startswith("leak")]

    print("Evaluating quality of generators...")
    quality_experiments(
        generators=all_generators_but_leaks,
        datasets=ALL_DATASETS,
        size=10_000,
        seeds=SEEDS,
    )

    print(f"Running fast metrics experiments ({FAST_METRICS})...")
    privacy_experiments(
        metrics=FAST_METRICS,
        generators=ALL_GENERATORS,
        datasets=ALL_DATASETS,
        training_sizes=TRAINING_SIZES,
    )

    print(f"Running shadow modeling experiments ({SHADOW_MODELING_METRICS})...")
    privacy_experiments(
        metrics=SHADOW_MODELING_METRICS,
        generators=SHADOW_MODELING_GENERATORS,
        datasets=ALL_DATASETS,
        training_sizes=TRAINING_SIZES,
        seeds=SEEDS,
    )

    print("All experiments completed!")
