import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import sys
from joblib import Memory

sys.path.append(".")
from src.env_constants import CACHE_LOCATION

sys.path.insert(0, "submodules/achilles_heels")

from src.data_prep import (
    read_data,
    read_metadata,
    discretize_dataset,
    normalize_cont_cols,
)
from src.feature_extractors import fit_ohe, apply_ohe
from src.distance import compute_distances

memory = Memory(location=CACHE_LOCATION, verbose=0)


def top_k_cosine_distance_matrix(
    df: pd.DataFrame,
    cat_columns: list,
    batch_size: int,
    device="cpu",
    standardize: bool = False,
    k: int = 5,
) -> np.ndarray:

    # convert categorical variables (string, object) to integers codes
    for col in cat_columns:
        df[col] = df[col].astype("category").cat.codes

    # convert to tensors
    x_continuous = torch.tensor(
        df.drop(columns=cat_columns).values, dtype=torch.float32
    ).to(device)
    x_categorical = torch.tensor(df[cat_columns].values, dtype=torch.int64).to(device)

    if standardize:
        x_continuous = (x_continuous - x_continuous.mean(dim=0)) / x_continuous.std(
            dim=0
        )

    distance_matrix = torch.zeros((len(df), k)).to(device)

    for i in tqdm(range(0, len(df), batch_size)):
        start = i
        end = min((i + 1) + batch_size, len(df))

        batch_continuous = x_continuous[start:end]
        batch_categorical = x_categorical[start:end]

        distance = torch.zeros((end - start, len(df))).to(device)
        distance += 1 - torch.nn.functional.cosine_similarity(
            batch_continuous.unsqueeze(1), x_continuous.unsqueeze(0), dim=-1
        )
        distance += (batch_categorical.unsqueeze(1) != x_categorical.unsqueeze(0)).sum(
            -1
        )

        distance_matrix[start:end] = torch.topk(distance, k=k, largest=False)[0]
    return distance_matrix.cpu().numpy()


@memory.cache
def achilles_outliers(
    df: pd.DataFrame,
    meta_data_og,
    categorical_cols,
    continuous_cols,
    k_neighbors=5,
    distance_method="cosine",
    legacy: bool = False,
):

    df = discretize_dataset(df, categorical_cols)
    df = normalize_cont_cols(df, meta_data_og, df_aux=df, types=("Float", "Integer"))

    if legacy:
        ohe, ohe_column_names = fit_ohe(df, categorical_cols, meta_data_og)
        df_ohe = apply_ohe(
            df.copy(), ohe, categorical_cols, ohe_column_names, continuous_cols
        )

        # Using the whole dataset as the subset for distance computation, as per the original Achilles heels attack
        sub_df_ohe = df_ohe

        # get the right indices for the distance computation
        all_columns = list(sub_df_ohe.columns)
        ohe_cat_indices = [all_columns.index(col) for col in ohe_column_names]
        continous_indices = [all_columns.index(col) for col in continuous_cols]

        # make sure to save the computed distances
        ALL_DISTANCES = dict()

        df_ohe_values = sub_df_ohe.values
        targets = sub_df_ohe.index

        for i in tqdm(range(len(targets))):
            target_id = targets[i]
            target_record = sub_df_ohe.loc[target_id].values
            distances_target = compute_distances(
                record=target_record,
                values=df_ohe_values,
                ohe_cat_indices=ohe_cat_indices,
                continous_indices=continous_indices,
                n_cat_cols=len(categorical_cols),
                n_cont_cols=len(continuous_cols),
                method=distance_method,
            )

            # let's sort it already
            ALL_DISTANCES[target_id] = np.sort(distances_target)
    else:
        sorted_distances = top_k_cosine_distance_matrix(
            df,
            categorical_cols,
            batch_size=100,
            k=k_neighbors,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        ALL_DISTANCES = {
            target_id: sorted_distances[i] for i, target_id in enumerate(df.index)
        }

    mean_distance_top_k = dict()
    for target_id in ALL_DISTANCES.keys():
        mean_distance_top_k[target_id] = np.mean(ALL_DISTANCES[target_id][:k_neighbors])

    # Identify N vulnerable records by taking the N records with the greatest value
    sorted_key_vals = sorted(
        mean_distance_top_k.items(), key=lambda x: x[1], reverse=True
    )

    vulnerable_records = [k[0] for k in sorted_key_vals]

    return vulnerable_records


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Find vulnerable records with Achilles outlier detection"
    )
    argparser.add_argument(
        "--path_to_data",
        type=str,
        default="submodules/achilles_heels/data/2011 Census Microdata Teaching File_OG.csv",
    )
    argparser.add_argument(
        "--path_to_metadata",
        type=str,
        default="submodules/achilles_heels/data/2011 Census Microdata Teaching Discretized.json",
    )
    argparser.add_argument("--k_neighbors", type=int, default=5)
    argparser.add_argument("--distance_method", type=str, default="cosine")
    argparser.add_argument("--output_path", type=str, default="./output")
    argparser.add_argument("--legacy", type=bool, default=False)
    args = argparser.parse_args()

    meta_data_og, categorical_cols, continuous_cols = read_metadata(
        args.path_to_metadata
    )
    df = read_data(args.path_to_data, categorical_cols, continuous_cols)

    vulnerable_records = achilles_outliers(
        df=df,
        meta_data_og=meta_data_og,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        k_neighbors=args.k_neighbors,
        distance_method=args.distance_method,
        legacy=args.legacy,
    )

    file = f"{args.output_path}/vulnerable_records.npy"
    np.save(file, vulnerable_records)
    print(f"Vulnerable records saved to {file}")
