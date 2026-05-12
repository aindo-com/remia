import pandas as pd
import numpy as np
from typing import Dict, Tuple
from typing import List
from itertools import islice
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from tqdm import tqdm
from enum import Enum
import random
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import Memory

from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
import torch

from scipy import stats
import sys

sys.path.append(".")
from src.env_constants import CACHE_LOCATION

memory = Memory(location=CACHE_LOCATION, verbose=0)


class Data(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    CATEGORICAL_ORD = "categorical_ord"
    DATE = "date"
    DATETIME = "datetime"
    INTEGER = "integer"


def raiser():
    raise ValueError(
        'attribute type must be in ["numeric", "categorical", "ordinal", "already_preprocessed"]'
    )


class StandardScalerNAN:
    def __init__(self):
        self.preproc = StandardScaler()

    def fit(self, data: pd.Series) -> None:
        data = data.to_frame()
        self.preproc.fit(data)

    def transform(self, data: pd.Series):
        data = data.to_frame()
        data = data.astype(float)
        data.fillna(data.mean(), inplace=True)  # TODO: add other methods to handle NANs
        # print(self.preproc.transform(data).shape)
        return self.preproc.transform(data)


class LabelEncoderUnseenCategories:
    def __init__(
        self,
        min_freq: int = 2,
        max_num_categories: int = 100,
    ):
        self.preproc = LabelEncoder()
        self.min_freq = min_freq
        self.max_num_categories = max_num_categories
        self.category_to_index = {}
        self.index_to_category = {}
        self.unknown_token = "__unknown__"

    def _find_common_categories(self, data: pd.Series) -> List:
        categories_count = data.value_counts().to_dict()
        selected_categories = list(
            islice(
                [
                    k
                    for k in categories_count.keys()
                    if categories_count[k] > self.min_freq
                ],
                self.max_num_categories,
            )
        )
        return selected_categories

    def fit(self, data: pd.Series) -> None:
        selected_categories = self._find_common_categories(data)
        all_categories = selected_categories + [self.unknown_token]
        self.category_to_index = {cat: idx for idx, cat in enumerate(all_categories)}
        self.index_to_category = {
            idx: cat for cat, idx in self.category_to_index.items()
        }

    def transform(self, data: pd.Series) -> pd.Series:
        return data.apply(
            lambda x: self.category_to_index.get(
                x, self.category_to_index[self.unknown_token]
            )
        ).values


class Preprocessor:
    def __init__(
        self,
        schema: Dict,
        ordered_categories: Dict = None,
        verbose: bool = False,
    ):
        self.schema = schema
        self.preprocessor = {
            name: StandardScalerNAN()
            if attribute_type == Data.NUMERIC
            else OrdinalEncoder(
                handle_unknown="use_encoded_value", categories=ordered_categories[name]
            )
            if attribute_type == Data.CATEGORICAL_ORD
            else LabelEncoderUnseenCategories()
            if attribute_type == Data.CATEGORICAL
            else None
            if attribute_type == "already_preprocessed"
            else raiser()
            for name, attribute_type in schema.items()
        }
        self.verbose = verbose

    def fit(self, data: pd.DataFrame) -> None:
        for name, preproc in tqdm(
            self.preprocessor.items(),
            desc="Fitting preprocessor",
            disable=not self.verbose,
        ):
            if preproc is None:
                continue
            preproc.fit(data[name])

    def transform(self, data: pd.DataFrame) -> Dict:
        d = {}
        for name, preproc in tqdm(
            self.preprocessor.items(),
            desc="Transforming data",
            disable=not self.verbose,
        ):
            if preproc is None:
                continue
            d[name] = preproc.transform(data[name])
        return d


def DOMIAS(
    release: pd.DataFrame,
    device: str,
    train_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    control_data: pd.DataFrame,
    schema: Dict[str, Data],
    ohe: bool,
    density_estimators: list[str],
    dim_reduction: bool,
):
    """DOMIAS proposed in: https://arxiv.org/pdf/2302.12580,
            original code: https://github.com/vanderschaarlab/DOMIAS/tree/main,

    Args:
        release: pd.DataFrame. Holdout set that allows DOMIAS computations.
        device: str. Device to use.
        density_estimator: str | list. Density estimation method (possible methods: 'bnaf' and 'kde').

    Returns:
        (float, np.array): DOMIAS score and the ground truth for classification.
    """

    train = train_data.copy()
    synthetic = synthetic_data.copy()
    control = control_data.copy()

    preprocessor = Preprocessor(schema)
    preprocessor.fit(train)
    train = preprocessor.transform(train)
    synthetic = preprocessor.transform(synthetic)
    control = preprocessor.transform(control)
    release = preprocessor.transform(release)

    if ohe:
        from sklearn.preprocessing import OneHotEncoder

        for c, dtype in schema.items():
            if dtype == Data.CATEGORICAL:
                ohe = OneHotEncoder(handle_unknown="ignore")
                ohe.fit(train[c].reshape(-1, 1))
                train[c] = ohe.transform(train[c].reshape(-1, 1)).toarray()
                synthetic[c] = ohe.transform(synthetic[c].reshape(-1, 1)).toarray()
                control[c] = ohe.transform(control[c].reshape(-1, 1)).toarray()
                release[c] = ohe.transform(release[c].reshape(-1, 1)).toarray()

                # add dequantization noise to the one-hot encoded features to avoid singularity of the matrix for density estimation
                # std = 0.1
                # train[c] += np.random.normal(0, std, size=train[c].shape)
                # synthetic[c] += np.random.normal(0, std, size=synthetic[c].shape)
                # control[c] += np.random.normal(0, std, size=control[c].shape)
                # release[c] += np.random.normal(0, std, size=release[c].shape)

    def add_dimension_if_needed(arr):
        if len(arr.shape) == 1:
            return arr.reshape(-1, 1)
        return arr

    train = np.concatenate(
        [add_dimension_if_needed(arr) for arr in train.values()], axis=1
    )
    release = np.concatenate(
        [add_dimension_if_needed(arr) for arr in release.values()], axis=1
    )
    synthetic = np.concatenate(
        [add_dimension_if_needed(arr) for arr in synthetic.values()], axis=1
    )
    control = np.concatenate(
        [add_dimension_if_needed(arr) for arr in control.values()], axis=1
    )

    if dim_reduction:
        if train.shape[1] != np.linalg.matrix_rank(train):
            print("Matrix is singular, applying dimensionality reduction...")
            # do pca to reduce the dimension to the rank of the matrix
            from sklearn.decomposition import PCA

            pca = PCA(n_components=np.linalg.matrix_rank(train))
            real_dataset = np.concatenate((train, release, control), axis=0)
            pca.fit(real_dataset)
            pca_threshold = 0.99
            mask = (
                np.cumsum(pca.explained_variance_ratio_) < pca_threshold
            )  # & (pca.explained_variance_ratio_ > 1e-3)
            train = pca.transform(train)[:, mask]
            synthetic = pca.transform(synthetic)[:, mask]
            release = pca.transform(release)[:, mask]
            control = pca.transform(control)[:, mask]

    X_test = np.concatenate([train, control])

    gt = np.concatenate([np.ones(len(train)), np.zeros(len(control))])

    rand_permutation = np.random.permutation(len(X_test))
    X_test = X_test[rand_permutation]
    gt = gt[rand_permutation]

    scores = {}
    gts = {}

    assert all(estimator in ["bnaf", "kde"] for estimator in density_estimators), (
        "Density estimator must be in ['bnaf', 'kde']"
    )

    if "kde" in density_estimators:
        scores["kde"] = kde_scores(release, synthetic, X_test)
        gts["kde"] = gt

    if "bnaf" in density_estimators:
        scores["bnaf"] = bnaf_scores(release, synthetic, X_test, device)
        gts["bnaf"] = gt

    return scores, gts


def bnaf_scores(release, synthetic, X_test, device):
    # assert release.shape[0] == synthetic.shape[0], (
    #     "For bnaf density estimation, there could be a bias in the estimated density depending on the size of the datasets."
    # )
    _, real_model = density_estimator_trainer(
        release, device=device, save=True, epochs=100
    )
    real_model.to(device)
    real_log_density = (
        compute_log_p_x(real_model, torch.as_tensor(X_test).float().to(device))
        .cpu()
        .detach()
        .numpy()
    )
    # real_density = np.exp(real_log_density)

    _, synth_model = density_estimator_trainer(
        synthetic, device=device, save=True, epochs=100
    )
    synth_model.to(device)
    synth_log_density = (
        compute_log_p_x(synth_model, torch.as_tensor(X_test).float().to(device))
        .cpu()
        .detach()
        .numpy()
    )
    # synth_density = np.exp(synth_log_density)

    # only the order of the scores matter for the computation of the AUC, so we can return the log density instead of the density to avoid numerical issues with the exponentiation
    return synth_log_density - real_log_density


def kde_scores(release, synthetic, X_test):
    if np.linalg.matrix_rank(synthetic) < synthetic.shape[1]:
        print("Synthetic matrix is singular...")

        # remove columns with zero variance
        synthetic = synthetic + np.random.normal(0, 1e-7, size=synthetic.shape)

    density_gen = stats.gaussian_kde(synthetic.transpose(1, 0))
    density_data = stats.gaussian_kde(release.transpose(1, 0))
    synth_density = density_gen(X_test.transpose(1, 0))
    real_density = density_data(X_test.transpose(1, 0))

    return synth_density / (real_density + 1e-8)


def compute_metrics_baseline(y_scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Function from domias library"""
    y_pred = y_scores > np.median(y_scores)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    return {
        "acc": float(acc),
        "auc": float(auc),
    }


def DOMIAS_metrics(
    score: np.ndarray,
    ground_truth: np.ndarray,
) -> Tuple[float, float]:
    """Computes the AUC for DOMIAS scores.

    Args:
        score: np.ndarray. DOMIAS scores.
        ground_truth: np.ndarray. Ground truth labels.

    Returns:
        float: AUC score.
    """

    y_pred = score > 0.5
    acc = accuracy_score(ground_truth, y_pred)
    auc = roc_auc_score(ground_truth, score)
    return float(acc), float(auc)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def domias_main():

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_train", type=str, required=True)
    parser.add_argument("--path_to_syn", type=str, required=True)
    parser.add_argument("--path_to_aux", type=str, required=True)
    parser.add_argument("--path_to_control", type=str, required=True)
    parser.add_argument("--path_to_metadata", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--density_estimators", type=str, required=True)
    parser.add_argument("--ohe", action="store_true")
    parser.add_argument("--dim_reduction", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    def store_json(d, *, file: str):
        with open(file, "w") as f:
            json.dump(d, f, indent=4)

    def load_json(file: str) -> dict:
        with open(file, "r") as f:
            return json.load(f)

    train_data = pd.read_csv(args.path_to_train)
    synthetic_data = pd.read_csv(args.path_to_syn)
    auxiliary_data = pd.read_csv(args.path_to_aux)
    control_data = pd.read_csv(args.path_to_control)
    schema = load_json(args.path_to_metadata)

    # running domias
    scores, ground_truth = DOMIAS(
        release=auxiliary_data,  # the auxiliary dataset
        device=args.device,
        train_data=train_data,
        synthetic_data=synthetic_data,  # the generated dataset
        control_data=control_data,  # the unseen half of the test dataset (the other half is the train dataset)
        schema=schema,
        density_estimators=args.density_estimators.split(","),
        ohe=args.ohe,
        dim_reduction=args.dim_reduction,
    )

    out = {}
    out["scores"] = {}

    for density_estimator, score in scores.items():
        out["scores"][density_estimator] = compute_metrics_baseline(
            score, ground_truth[density_estimator]
        )

    out["info"] = {
        "train_size": len(train_data),
        "synthetic_size": len(synthetic_data),
        "auxiliary_size": len(auxiliary_data),
    }

    store_json(out, file=args.output_file)


if __name__ == "__main__":
    domias_main()
