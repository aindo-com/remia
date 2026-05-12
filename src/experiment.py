from datetime import datetime
import os
from src.util import printc, store_json, Timer, set_seed
from src.metrics import *

EXPERIMENTS_FOLDER = "experiments/output"


def resolve_metric(metric_name: str):
    if metric_name.startswith("remia_") and "." in metric_name.split("_")[-1]:
        target_proportion = float(metric_name.split("_")[-1])
        return partial(remia, target_proportion=target_proportion)
    return eval(metric_name)


def metric_experiment(
    *,
    metric: str,
    dataset: str,
    generator: str,
    seed: int = 0,
    training_size: int = 1_000,
    verbose: bool = True,
    output_file: str | None = None,
    name: str = "",
) -> dict:

    metric_evaluator = resolve_metric(metric)

    set_seed(seed)

    input_args = {
        "dataset": dataset,
        "generator_name": generator,
        "training_size": training_size,
        "seed": seed,
    }

    with Timer() as timer:
        output = metric_evaluator(**input_args)

    result = {
        "input": input_args,
        "output": output,
        "time": timer.elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    result["input"]["metric"] = metric

    if verbose:
        printc(result, color="yellow")

    postfix = "_" + name if len(name) > 0 else ""

    if output_file is None:
        file_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f") + postfix
        os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)
        store_json(result, file=f"{EXPERIMENTS_FOLDER}/{file_name}.json")
    else:
        store_json(result, file=output_file)
    return result
