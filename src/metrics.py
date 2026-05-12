import os
import subprocess
import pandas as pd
import numpy as np
from src.util import store_json, get_available_device, load_json, pop_data, Timer
from src.data import load_data
from src.generators import get_generator, LeakyGenerator
import random
from functools import partial
from scipy.stats import binom, beta
from tempfile import TemporaryDirectory
from src.dcr import dcr_mia as run_dcr_mia
from src.dcr import dcr_comparison as run_dcr_comparison
from src.dcr import dcr_quantile as run_dcr_quantile
from src.remia import remia as run_remia
from src.env_constants import (
    ACHILLES_ENV_PATH,
    DOMIAS_ENV_PATH,
)
import math


def accuracy_significance(acc: float, n: int):
    successes = round(acc * n)
    failures = n - successes
    percentile = 0.01
    return {
        "p_value": 1 - binom.cdf(k=int(acc * n), n=n, p=0.5),
        "bayesian_p(acc)>0.5": 1 - beta.cdf(0.5, successes + 1, failures + 1),
        f"bayesian_quantile({percentile})": beta.ppf(
            percentile, successes + 1, failures + 1
        ),
        "n": n,
        "successes": successes,
        # "failures": failures,
    }


def output_summary(
    test_auc: float,
    n: int,
    test_successes: int | None = None,
    test_acc: float | None = None,
) -> dict:
    return dict(
        test_auc=test_auc,
        n=n,
        test_successes=test_successes
        if test_successes is not None
        else round(test_acc * n),
    )


def get_dataset_path(dataset_name: str) -> str:
    return f"data/{dataset_name}"


def dummy(dataset: str, generator_name: str, training_size: int, seed: int = 0):
    return {"message": "test!"}


def to_reprosyn_metadata(
    metadata: dict, df: pd.DataFrame, processing: bool = False
) -> list:
    reprosyn_metadata = []
    for col_name, col_type in metadata.items():
        if col_type == "float":
            c = {
                "name": col_name,
                "type": "Float",
                "representation": "number",
                "min": 0.0 if processing else float(df[col_name].min()),
                "max": 1.0 if processing else float(df[col_name].max()),
            }
        elif col_type == "int":
            c = {
                "name": col_name,
                "type": "Integer",
                "representation": "number",
                "min": int(df[col_name].min()),
                "max": int(df[col_name].max()),
            }
        elif col_type == "categorical":
            c = {
                "name": col_name,
                "type": "finite",
                "representation": [str(i) for i in range(len(df[col_name].unique()))]
                if processing
                else df[col_name].unique().astype(str).tolist(),
            }
        else:
            raise ValueError(f"Unsupported column type: {col_type}")
        reprosyn_metadata.append(c)
    return reprosyn_metadata


def get_achilles_discretized_metadata(dataset_path: str):
    df, metadata = load_data(dataset_path)
    return to_reprosyn_metadata(metadata, df, processing=True)


def shadow_modeling(
    dataset: str,
    generator_name: str,
    training_size: int = 1_000,
    seed: int = 0,
    *,
    target_selection: str,
):
    """
    Run the shadow_modeling attack via shell command.
    Assumes 'dataset' is the path to the data file and 'generator' is the generator name.
    target_selection options: "achilles_heels", "achilles_median", "random"
    """

    n_pos_train = 500
    n_pos_test = 100

    dataset_path = get_dataset_path(dataset)
    csv_path = dataset_path + "/data.csv"

    with TemporaryDirectory() as temp_output_dir:
        achilles_discretized_metadata = get_achilles_discretized_metadata(
            dataset_path=dataset_path
        )
        metadata_path = os.path.join(temp_output_dir, "metadata.json")
        store_json(achilles_discretized_metadata, file=metadata_path)

        if target_selection in ("achilles_heels", "achilles_median"):
            command = [
                ACHILLES_ENV_PATH,
                "src/find_vulnerable_records.py",
                f"--path_to_data={csv_path}",
                f"--path_to_metadata={metadata_path}",
                "--k_neighbors=5",
                "--distance_method=cosine",
                f"--output_path={temp_output_dir}",
            ]
            subprocess.run(command, capture_output=False, check=True)

            vulnerable_records = np.load(
                os.path.join(temp_output_dir, "vulnerable_records.npy")
            )
            if target_selection == "achilles_heels":
                target_record = int(vulnerable_records[0])
            elif target_selection == "achilles_median":
                target_record = int(vulnerable_records[len(vulnerable_records) // 2])
        elif target_selection == "random":
            df, _ = load_data(dataset_path)
            target_record = random.randint(0, len(df) - 1)
            # target_record = int(df.sample(1, random_state=seed).index[0])
        else:
            raise ValueError(f"Unsupported target selection method: {target_selection}")

        print(f"selected target record: {target_record} with method {target_selection}")
        df, _ = load_data(dataset_path)
        n_test = (
            min(25_000, len(df) // 3)  # 25,000 as the original work
            if not generator_name.startswith("leak")
            else min(25_000, len(df) // 4)
        )
        n_aux = n_test * 2
        if n_aux + n_test == len(df):
            n_test -= 1  # excluding the target from the total count

        command = [
            ACHILLES_ENV_PATH,
            "submodules/achilles_heels/Achilles_main.py",
            f"--path_to_data={os.path.realpath(csv_path)}",
            f"--path_to_metadata={os.path.realpath(metadata_path)}",
            f"--target_record_id={target_record}",
            f"--output_dir={temp_output_dir}",
            f"--name_generator={generator_name}",
            f"--n_aux={n_aux}",
            f"--n_test={n_test}",
            f"--n_original={training_size}",
            f"--n_synthetic={training_size}",
            f"--n_pos_train={n_pos_train}",
            f"--n_pos_test={n_pos_test}",
            f"--seed={seed}",
        ]

        subprocess.run(
            command,
            capture_output=False,
            check=True,
        )

        output_files = [
            f
            for f in os.listdir(temp_output_dir)
            if f.startswith("output_") and f.endswith(".csv")
        ]
        output = {}
        for file in output_files:
            attack = file.split("_")[-1].split(".")[0]
            data = (
                pd.read_csv(os.path.join(temp_output_dir, file), index_col=0)
                .iloc[0]
                .to_dict()
            )
            output[attack] = data
            output[attack]["train_acc_significance"] = accuracy_significance(
                data["train_acc"], n_pos_train * 2
            )

            output[attack]["test_acc_significance"] = accuracy_significance(
                data["test_acc"], n_pos_test * 2
            )

            times = load_json(os.path.join(temp_output_dir, "times.json"))

    output["info"] = {
        "target_record": target_record,
        "n_test": n_test,
        "n_aux": n_aux,
        "n_original": training_size,
        "n_synthetic": training_size,
        "n_pos_train": n_pos_train,
        "n_pos_test": n_pos_test,
    }

    output["summary"] = output_summary(
        test_auc=output["query"]["test_auc"],
        n=n_pos_test * 2,
        test_acc=output["query"]["test_acc"],
    )

    output["summary"]["generation_time"] = times["generation_time"]
    output["summary"]["generation_calls"] = times["generation_calls"]
    output["summary"]["overhead_time"] = times["overhead_time"]
    output["summary"]["total_data_usage"] = n_test + n_aux

    return output


def shadow_modeling_theoretical(
    dataset: str, generator_name: str, training_size: int = 1_000, seed: int = 0
):
    """
    IDEAL PREDICTOR FOR LEAK GENERATOR (with leak fraction f):
        p(x in T | x in S) = 1
        p(x in T | x not in S) = (1-f) / (2-f) <= 0.5

        classifier(S) = 1 if S contains x, 0 otherwise

    ACCURACY (does not assume attacker knowledge of leak fraction f):
        correct classification: x in T and x in S, or x not in T and x not in S
        p(correct) = p(x in T, x in S) + p(x not in T, x not in S)
                = p(x in S | x in T)p(x in T) + p(x not in S | x not in T)p(x not in T)
                = f * 0.5 + (1-0) * 0.5
                = (1+f)/2

    AUC?

    proof:
        T: training set
        S: synthetic data
        x: target record

        p(x in T | S = leak_generator(train=T, leak_fraction=f) )

        p(x in T | x in S) = p(x in S | x in T)p(x in T)/p(x in S)
                = p(x in S | x in T) p(x in T) / [ p(x in S | x in T)p(x in T) +  p(x in S | x notin T)p(x in notin T)]
                = p(x in S | x in T) / [ p(x in S | x in T) +  p(x in S | x notin T)]
                = f / [ f +  ~0 ]
                = ~1

        p(x in T | x not in S) = p(x not in S | x in T)p(x in T) / p(x not in S)
                    = p(x not in S | x in T) / [ p(x not in S | x in T) + p(x not in S | x not in T) ]
                    = (1-f) / [(1-f) + 1]
                    = (1-f) / (2-f)
    """
    if not generator_name.startswith("leak"):
        raise ValueError("This metric is only applicable for leak generators")
    p = float(generator_name.split("_")[-1])

    # accuracy is known analitically
    accuracy = (1 + p) / 2

    output = {
        "accuracy": accuracy,
        "test_auc": accuracy,
    }

    output["summary"] = {"test_auc": accuracy, "test_acc": accuracy}

    return output


def domias(
    dataset: str, generator_name: str, training_size: int = 1_000, seed: int = 0
):
    dataset_path = get_dataset_path(dataset)

    # parameters
    TRAIN_DATA_SIZE = training_size
    SYNTH_DATA_SIZE = TRAIN_DATA_SIZE
    AUXILIARY_DATA_SIZE = (
        TRAIN_DATA_SIZE * 5  # with bnaf there are issues if the size is not the same
    )

    DENSITY_ESTIMATOR = ["kde"]
    OHE = True
    DIM_REDUCTION = True

    # loading data
    df, metadata = load_data(dataset_path)

    # re shuffle the rows order of the dataset to avoid any bias in the split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # build train, auxiliary and control datasets
    train_data = pop_data(df, TRAIN_DATA_SIZE)
    unseen_half_test_data = pop_data(df, TRAIN_DATA_SIZE)
    auxiliary_data = pop_data(df, AUXILIARY_DATA_SIZE)

    with TemporaryDirectory() as temp_output_dir:
        domias_output_path = os.path.join(temp_output_dir, "domias_output.json")

        # generating synthetic data
        if generator_name.startswith("leak"):
            generator = LeakyGenerator(
                leak_frac=float(generator_name.split("_")[-1]),
                additional_data=pop_data(df, SYNTH_DATA_SIZE),
                seed=seed,
            )
        else:
            generator = get_generator(generator_name, metadata=metadata, seed=seed)

        print("generating synthetic data...")
        with Timer() as generation_timer:
            synthetic_data = generator.fit_generate(train_data, n=SYNTH_DATA_SIZE)
        print("Synthetic data generated!")

        # temporary saving the datasets as csv files to be used as input for domias
        path_to_train = os.path.join(temp_output_dir, "train_data.csv")
        path_to_syn = os.path.join(temp_output_dir, "synthetic_data.csv")
        path_to_aux = os.path.join(temp_output_dir, "auxiliary_data.csv")
        path_to_control = os.path.join(temp_output_dir, "control_data.csv")
        train_data.to_csv(path_to_train, index=False)
        synthetic_data.to_csv(path_to_syn, index=False)
        auxiliary_data.to_csv(path_to_aux, index=False)
        unseen_half_test_data.to_csv(path_to_control, index=False)

        # converting metadata
        names_map = {
            "int": "numeric",  # Data.NUMERIC,
            "float": "numeric",  # Data.NUMERIC,
            "categorical": "categorical",  # Data.CATEGORICAL,
        }
        schema = {k: names_map[v] for k, v in metadata.items()}
        temp_metadata_path = os.path.join(temp_output_dir, "metadata.json")
        store_json(schema, file=temp_metadata_path)

        # running domias via shell command
        command = [
            f"{DOMIAS_ENV_PATH}",
            "src/domias_main.py",
            f"--path_to_train={path_to_train}",
            f"--path_to_syn={path_to_syn}",
            f"--path_to_aux={path_to_aux}",
            f"--path_to_control={path_to_control}",
            f"--path_to_metadata={temp_metadata_path}",
            f"--output_file={domias_output_path}",
            f"--density_estimators={','.join(DENSITY_ESTIMATOR)}",
            f"--device={get_available_device()}",
            f"--path_to_metadata={temp_metadata_path}",
            f"--seed={seed}",
        ]

        if OHE:
            command += ["--ohe"]
        if DIM_REDUCTION:
            command += ["--dim_reduction"]

        with Timer() as domias_timer:
            subprocess.run(
                command,
                capture_output=False,
                check=True,
            )

        output = load_json(domias_output_path)
    output["info"] = {
        "train_data_size": TRAIN_DATA_SIZE,
        "synthetic_data_size": SYNTH_DATA_SIZE,
        "auxiliary_data_size": AUXILIARY_DATA_SIZE,
        "density_estimator": DENSITY_ESTIMATOR,
        "ohe": OHE,
        "dim_reduction": DIM_REDUCTION,
    }

    n = len(unseen_half_test_data) + len(train_data)
    for k, v in output["scores"].items():
        n = len(unseen_half_test_data) + len(train_data)
        if "outlier" in k:
            n = round(n * 0.2)
        v["significance"] = accuracy_significance(v["acc"], n)

    output["summary"] = output_summary(
        test_auc=output["scores"]["kde"]["auc"],
        n=len(unseen_half_test_data) + len(train_data),
        test_acc=output["scores"]["kde"]["acc"],
    )
    output["summary"]["generation_time"] = generation_timer.elapsed
    output["summary"]["generation_calls"] = 1
    output["summary"]["overhead_time"] = domias_timer.elapsed
    output["summary"]["total_data_usage"] = (
        len(unseen_half_test_data) + len(train_data) + len(auxiliary_data)
    )

    return output


shadow_modeling_achilles_heels = partial(
    shadow_modeling, target_selection="achilles_heels"
)
shadow_modeling_achilles_median = partial(
    shadow_modeling, target_selection="achilles_median"
)
shadow_modeling_random = partial(shadow_modeling, target_selection="random")


def get_remia_gen_config(generator_name: str, seed: int = 42) -> dict:

    if generator_name.startswith("leak"):
        p = float(generator_name.split("_")[-1])
        return {
            "synth_type": "leak",
            "p": p,
            "seed": seed,
        }

    return {
        "synth_type": "external",
        "generator_name": generator_name,
        "cfg_test": {
            "seed": seed,
        },
    }


def dcr_mia(
    dataset: str, generator_name: str, training_size: int = 1000, seed: int = 0
):

    # loading data and reshuffle
    dataset_path = get_dataset_path(dataset)
    df, metadata = load_data(dataset_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    used_data_index = 0

    df_train = df.iloc[used_data_index : used_data_index + training_size]
    used_data_index += len(df_train)

    # generating synthetic data
    if generator_name.startswith("leak"):
        additional_data = df.iloc[used_data_index : used_data_index + training_size]
        used_data_index += len(additional_data)
        generator = LeakyGenerator(
            leak_frac=float(generator_name.split("_")[-1]),
            additional_data=additional_data,
            seed=seed,
        )
    else:
        generator = get_generator(generator_name, metadata=metadata, seed=seed)

    print("generating synthetic data...")
    with Timer() as generation_timer:
        df_syn = generator.fit_generate(df_train, n=training_size)
    print("Synthetic data generated!")
    df_control = df.iloc[used_data_index : used_data_index + training_size]
    used_data_index += len(df_control)

    with Timer() as dcr_timer:
        output = run_dcr_mia(
            train=df_train,
            synth=df_syn,
            control=df_control,
            device=get_available_device(verbose=True),
            seed=seed,
            k=5,
        )

    output["summary"] = output_summary(
        test_auc=output["auc"],
        n=output[
            "n"
        ],  # Not exact since the accuracy is computed over an average of splits of size n
        test_acc=output["acc"],
    )

    output["summary"]["generation_time"] = generation_timer.elapsed
    output["summary"]["generation_calls"] = 1
    output["summary"]["overhead_time"] = dcr_timer.elapsed
    output["summary"]["total_data_usage"] = len(df_train) + len(df_control)

    return output


def dcr_comparison(
    dataset: str, generator_name: str, training_size: int = 1000, seed: int = 0
):

    # loading data and reshuffle
    dataset_path = get_dataset_path(dataset)
    df, metadata = load_data(dataset_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_train = pop_data(df, n=training_size)
    df_control = pop_data(df, n=training_size)

    # generating synthetic data
    if generator_name.startswith("leak"):
        additional_data = pop_data(df, n=training_size)
        generator = LeakyGenerator(
            leak_frac=float(generator_name.split("_")[-1]),
            additional_data=additional_data,
            seed=seed,
        )
    else:
        generator = get_generator(generator_name, metadata=metadata, seed=seed)

    print("generating synthetic data...")
    with Timer() as generation_timer:
        df_syn = generator.fit_generate(df_train, n=training_size)
    print("Synthetic data generated!")

    with Timer() as dcr_timer:
        output = run_dcr_comparison(
            train=df_train,
            synth=df_syn,
            control=df_control,
            device=get_available_device(verbose=True),
        )

    output["summary"] = {
        "score": output["fraction"],
        "n": output["n"],
        "test_successes": output["n_records_closer_to_synth"],
    }

    output["summary"]["generation_time"] = generation_timer.elapsed
    output["summary"]["generation_calls"] = 1
    output["summary"]["overhead_time"] = dcr_timer.elapsed
    output["summary"]["total_data_usage"] = len(df_train) + len(df_control)

    return output


def dcr_quantile(
    dataset: str,
    generator_name: str,
    training_size: int = 1000,
    seed: int = 0,
    k: int = 1,
):

    # loading data and reshuffle
    dataset_path = get_dataset_path(dataset)
    df, metadata = load_data(dataset_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_train = pop_data(df, n=training_size)

    # generating synthetic data
    if generator_name.startswith("leak"):
        additional_data = pop_data(df, n=training_size)
        generator = LeakyGenerator(
            leak_frac=float(generator_name.split("_")[-1]),
            additional_data=additional_data,
            seed=seed,
        )
    else:
        generator = get_generator(generator_name, metadata=metadata, seed=seed)

    print("generating synthetic data...")
    with Timer() as generation_timer:
        df_syn = generator.fit_generate(df_train, n=training_size)
    print("Synthetic data generated!")
    df_control = pop_data(df, n=training_size)

    with Timer() as dcr_timer:
        output = run_dcr_quantile(
            train=df_train,
            synth=df_syn,
            control=df_control,
            device=get_available_device(verbose=True),
            percentiles=[0.05, 0.02],
        )

    output["summary"] = {"score": output["control_to_train_quantile"]["0.05"]}

    output["summary"]["generation_time"] = generation_timer.elapsed
    output["summary"]["generation_calls"] = 1
    output["summary"]["overhead_time"] = dcr_timer.elapsed
    output["summary"]["total_data_usage"] = len(df_train) + len(df_control)

    return output


def remia(
    dataset: str,
    generator_name: str,
    training_size: int = 1000,
    seed: int = 0,
    target_proportion: float = 0.5,
):
    necessary_rows = math.ceil(training_size * (1 + target_proportion))

    dataset_path = get_dataset_path(dataset)
    df, metadata = load_data(dataset_path)

    data = (
        df.iloc[:necessary_rows]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    if generator_name.startswith("leak"):
        leak_necessary_rows = training_size * 2
        generator = LeakyGenerator(
            leak_frac=float(generator_name.split("_")[-1]),
            additional_data=df.iloc[
                necessary_rows : necessary_rows + leak_necessary_rows
            ],
            seed=seed,
        )
    else:
        generator = get_generator(generator_name, metadata=metadata, seed=seed)

    config = {
        "training_size": training_size,
        "target_proportion": target_proportion,
        "max_epochs": 1,
        "patience": 0,
        "val_each": 2,
        "seed": seed,
    }

    remia_output = run_remia(
        data=data,
        generator=generator,
        device=get_available_device(verbose=True),
        generator_name=generator_name,
        dataset_name=dataset,
        **config,
    )

    output = remia_output
    output["config"] = config
    # output["summary"] = output_summary(
    #     test_auc=remia_output["best_target_ema"]["auc"],
    #     n=remia_output["best_target_ema"]["n"],
    #     test_acc=remia_output["best_target_ema"]["accuracy"],
    # )
    output["summary"] = output_summary(
        test_auc=remia_output["target_smoothed_on_best_syn_capped"]["auc"],
        n=remia_output["target_smoothed_on_best_syn_capped"]["n"],
        test_acc=remia_output["target_smoothed_on_best_syn_capped"]["accuracy"],
    )

    output["summary"]["generation_time"] = remia_output["generation_time"]
    output["summary"]["generation_calls"] = 2
    output["summary"]["overhead_time"] = remia_output["overhead_time"]
    output["summary"]["total_data_usage"] = training_size + int(
        target_proportion * training_size
    )

    return output
