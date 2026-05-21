import sys

sys.path.append(".")
from src.util import load_json, find_files
import pandas as pd
import tqdm
from scipy.stats import binom
from joblib import Memory
from scripts.run_experiments import OUTPUT_FOLDER
from src.env_constants import CACHE_LOCATION

memory = Memory(location=CACHE_LOCATION, verbose=0)


def compute_significance(n, successes) -> dict:
    return {"acc_p_value": 1 - binom.cdf(k=successes, n=n, p=0.5)}


@memory.cache
def results_to_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    input_columns = ["metric", "generator_name", "dataset"]

    for d in tqdm.tqdm(results):
        row = {
            **d["input"],
            **d["output"]["summary"],
            "time": d["time"],
            "timestamp": d["timestamp"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # move the input columns to the front
    df = df[input_columns + [col for col in df.columns if col not in input_columns]]

    # compute significance column
    significance_dict = compute_significance(n=df["n"], successes=df["test_successes"])
    for name, column in significance_dict.items():
        df[name] = column

    return df


def quality_results_to_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for d in tqdm.tqdm(results):
        row = {
            **d["setting"],
            "xgboost_discr_auc": float(
                d["quality_metrics"]["xgboost_discriminator"]["auc"].split(" ")[0]
            ),
        }
        if "xgboost_utility" in d["quality_metrics"]:
            utility_score = d["quality_metrics"]["xgboost_utility"]["relative"]["mean"]
            utility_score = (
                utility_score["accuracy"]
                if "accuracy" in utility_score
                else -utility_score["rmse"]
            )
            row["xgboost_utility_score"] = utility_score
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["generator", "dataset", "size"])


def get_pivoted_results(df: pd.DataFrame) -> pd.DataFrame:
    # create pivot table with generator and dataset as index, <metric>_test_auc and <metric>_significance_dict as columns
    df_pivot = df.pivot_table(
        index=["generator_name", "dataset"],
        columns="metric",
        values=["score", "test_auc", "acc_p_value"],
        aggfunc="first",
    )

    # drop time column
    df_pivot = df_pivot.drop(columns=["time"], errors="ignore")

    # "flatten" the pivot table
    df_pivot.columns = [f"{metric}_{col}" for col, metric in df_pivot.columns]
    df_processed = df_pivot.reset_index()

    # reintroduce the time and timestamp columns by taking the max timestamp for each generator and dataset
    time_timestamp = (
        df.groupby(["generator_name", "dataset"])[["time", "timestamp"]]
        .max()
        .reset_index()
    )
    df_processed = df_processed.merge(
        time_timestamp, on=["generator_name", "dataset"], how="left"
    )

    return df_processed


def get_results(aggregation_type: str | None = None) -> pd.DataFrame:
    # gather all json files
    files = find_files(starting_folder=OUTPUT_FOLDER, pattern="*.json")

    # load each json file and append it to a list
    results = [load_json(file) for file in files]

    # write the results to a csv file
    df = results_to_csv(results)

    # group by dataset, metric, generator and take the row with the last timestamp
    input_columns = ["metric", "generator_name", "dataset", "training_size", "seed"]

    if aggregation_type == "last":
        df = df.sort_values("timestamp").groupby(input_columns, as_index=False).last()
    elif aggregation_type == "mean":
        df = df.groupby(input_columns, as_index=False).mean()
    elif aggregation_type == "max":
        df = df.groupby(input_columns, as_index=False).max()
    elif aggregation_type == "min":
        df = df.groupby(input_columns, as_index=False).min()

    df.sort_values(input_columns + ["seed"], inplace=True)

    return df


def get_quality_results() -> pd.DataFrame:
    # gather all json files
    files = find_files(
        starting_folder="experiments/quality_evaluation", pattern="*.json"
    )

    # load each json file and append it to a list
    results = [load_json(file) for file in files]

    # write the results to a csv file
    df = quality_results_to_csv(results)

    return df


def main():
    get_results(aggregation_type=None).to_csv(
        "experiments/privacy_results.csv", index=False, float_format="%.4f"
    )
    get_quality_results().to_csv("experiments/quality_results.csv", index=False)


if __name__ == "__main__":
    main()
