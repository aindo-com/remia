import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch


def get_dcr(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    k: int = 1,
    device: str = "cpu",
    standardize: bool = True,
    batch_size: int = 1000,
) -> pd.DataFrame:

    source_df = source_df.copy()
    target_df = target_df.copy()

    cat_columns = source_df.select_dtypes(include=["category"]).columns.tolist()

    # Convert object columns to categorical
    for col in cat_columns:
        source_df[col + "_number"] = source_df[col].cat.codes
        target_df[col + "_number"] = target_df[col].cat.codes
        source_df = source_df.drop(columns=[col])
        target_df = target_df.drop(columns=[col])
        source_df = source_df.rename(columns={col + "_number": col})
        target_df = target_df.rename(columns={col + "_number": col})

    # convert to tensors
    source_continuous = torch.tensor(
        source_df.drop(columns=cat_columns).values, dtype=torch.float32
    ).to(device)
    target_continuous = torch.tensor(
        target_df.drop(columns=cat_columns).values, dtype=torch.float32
    ).to(device)
    source_categorical = torch.tensor(
        source_df[cat_columns].values, dtype=torch.int64
    ).to(device)
    target_categorical = torch.tensor(
        target_df[cat_columns].values, dtype=torch.int64
    ).to(device)

    if standardize:
        # using the mean and std of the target dataset
        mean = target_continuous.mean(dim=0)
        std = target_continuous.std(dim=0)
        source_continuous = (source_continuous - mean) / std
        target_continuous = (target_continuous - mean) / std

    distance_matrix = torch.zeros((len(source_df), k)).to(device)

    for i in tqdm(range(0, len(source_df), batch_size)):
        start = i
        end = min((i + 1) + batch_size, len(source_df))

        batch_continuous = source_continuous[start:end]
        batch_categorical = source_categorical[start:end]

        distance = torch.zeros((end - start, len(target_df))).to(device)

        distance += 1 - torch.nn.functional.cosine_similarity(
            batch_continuous.unsqueeze(1), target_continuous.unsqueeze(0), dim=-1
        )
        distance += (
            batch_categorical.unsqueeze(1) != target_categorical.unsqueeze(0)
        ).sum(-1)

        distance_matrix[start:end] = torch.topk(distance, k=k, largest=False)[0]

    distances = distance_matrix.cpu().numpy()
    dcr_df = pd.DataFrame(distances, columns=[f"dcr_{i}" for i in range(k)])

    # add mean and std of other columns as well
    if k > 1:
        dcr_df["dcr_mean"] = distances.mean(axis=1)
        dcr_df["dcr_std"] = distances.std(axis=1)

    return dcr_df


def dcr_mia(
    *,
    train: pd.DataFrame,
    synth: pd.DataFrame,
    control: pd.DataFrame,
    device: str = "cpu",
    seed: int = 0,
    k=1,
) -> dict:
    """
    The attack consists in training a binary classifier to distinguish between samples from
    the training set and samples from a control set (not used for training and generating
    the synthetic data).
    """

    # Convert all object columns to categorical (with common codes)
    data_temp = pd.concat([train, synth, control], ignore_index=True)
    for col in data_temp.select_dtypes(include=["object"]).columns:
        data_temp[col] = data_temp[col].astype("category")

    train_df = data_temp.iloc[: len(train)]
    synth_df = data_temp.iloc[len(train) : len(train) + len(synth)]
    control_df = data_temp.iloc[len(train) + len(synth) :]

    # first featurize both train and control data (currently, only the dcr)
    train_features = get_dcr(source_df=train_df, target_df=synth_df, device=device, k=k)
    control_features = get_dcr(
        source_df=control_df, target_df=synth_df, device=device, k=k
    )

    # Add labels
    train_features = train_features.assign(label=1)
    control_features = control_features.assign(label=0)

    # Combine the data
    data = pd.concat([train_features, control_features])

    # Convert object columns to categorical
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].astype("category")

    # Split features and labels
    X = data.drop(columns=["label"])
    y = data["label"]

    # Drop labels from original datasets
    del train_features["label"]
    del control_features["label"]

    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Metrics to track
    accuracies = []
    aucs = []

    # Perform Stratified K-Fold Cross-Validation with tqdm progress bar
    for train_index, test_index in tqdm(
        skf.split(X, y), total=n_splits, desc="Cross-Validation Progress"
    ):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the XGBoost classifier
        model = xgb.XGBClassifier(
            eval_metric="logloss", enable_categorical=True, random_state=seed
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_pred_proba))
        n = len(X_test)

    # Compute average and standard deviation for metrics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Features importance features
    importance = [
        round(float(x), 4) for x in model.feature_importances_
    ]  # for some reason I need to do this
    feature_names = X.columns
    feature_importance = (
        pd.Series(importance, index=feature_names)
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "acc": avg_accuracy,
        "acc_std": std_accuracy,
        "auc": avg_auc,
        "auc_std": std_auc,
        "n": n,
        "feature_importance": feature_importance,
    }


def dcr_comparison(
    train: pd.DataFrame,
    synth: pd.DataFrame,
    control: pd.DataFrame,
    device: str = "cpu",
    seed: int = 0,
    k: int = 1,
) -> dict:

    # Convert all object columns to categorical (with common codes)
    data_temp = pd.concat([train, synth, control], ignore_index=True)
    for col in data_temp.select_dtypes(include=["object"]).columns:
        data_temp[col] = data_temp[col].astype("category")

    train_df = data_temp.iloc[: len(train)]
    synth_df = data_temp.iloc[len(train) : len(train) + len(synth)]
    control_df = data_temp.iloc[len(train) + len(synth) :]

    dcr_train_to_synth = get_dcr(
        source_df=train_df, target_df=synth_df, device=device, k=1
    )["dcr_0"]
    dcr_train_to_control = get_dcr(
        source_df=train_df, target_df=control_df, device=device, k=k
    )["dcr_0"]
    dcr_train_to_train = get_dcr(
        source_df=train_df, target_df=train_df, device=device, k=2
    )["dcr_1"]

    comparison = float(
        (dcr_train_to_synth < dcr_train_to_control).mean()
    )  # fraction of samples that are closer to synth than control
    comparison_no_control = float((dcr_train_to_synth < dcr_train_to_train).mean())

    return {
        "fraction": comparison,
        "fraction_no_control": comparison_no_control,
        "mean_dcr_train_to_synth": float(dcr_train_to_synth.mean()),
        "mean_dcr_train_to_control": float(dcr_train_to_control.mean()),
        "mean_dcr_train_to_train": float(dcr_train_to_train.mean()),
        "n": len(dcr_train_to_synth),
        "n_records_closer_to_synth": int(
            (dcr_train_to_synth < dcr_train_to_control).sum()
        ),
    }


def dcr_quantile(
    train: pd.DataFrame,
    synth: pd.DataFrame,
    control: pd.DataFrame,
    device: str = "cpu",
    percentiles: list = [0.02, 0.05],
) -> dict:

    # Convert all object columns to categorical (with common codes)
    data_temp = pd.concat([train, synth, control], ignore_index=True)
    for col in data_temp.select_dtypes(include=["object"]).columns:
        data_temp[col] = data_temp[col].astype("category")

    train_df = data_temp.iloc[: len(train)]
    synth_df = data_temp.iloc[len(train) : len(train) + len(synth)]
    control_df = data_temp.iloc[len(train) + len(synth) :]

    dcr_train_to_synth = get_dcr(
        source_df=train_df, target_df=synth_df, device=device, k=1
    )["dcr_0"]
    dcr_control_to_train = get_dcr(
        source_df=control_df, target_df=train_df, device=device, k=1
    )["dcr_0"]
    dcr_train_to_train = get_dcr(
        source_df=train_df, target_df=train_df, device=device, k=2
    )["dcr_1"]

    output = {}
    output["mean_dcr_train_to_synth"] = float(dcr_train_to_synth.mean())
    output["mean_dcr_control_to_train"] = float(dcr_control_to_train.mean())
    output["mean_dcr_train_to_train"] = float(dcr_train_to_train.mean())

    output["control_to_train_quantile"] = {}
    output["train_to_train_quantile"] = {}

    for p in percentiles:
        control_to_train_quantile = float(dcr_control_to_train.quantile(p))
        output["control_to_train_quantile"][str(p)] = float(
            (dcr_train_to_synth < control_to_train_quantile).mean()
        )

        train_to_train_quantile = float(dcr_train_to_train.quantile(p))
        output["train_to_train_quantile"][str(p)] = float(
            (dcr_train_to_synth < train_to_train_quantile).mean()
        )

    return output
