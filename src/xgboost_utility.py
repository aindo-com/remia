import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    root_mean_squared_error,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from joblib import Memory
from src.env_constants import CACHE_LOCATION
import tqdm

memory = Memory(location=CACHE_LOCATION, verbose=0)

PRECISION_DIGITS = 3


# @memory.cache
def xgboost_performance(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state=1234,
) -> dict:
    is_classification = y_train.dtype in (object, str, "category")
    is_multiclass = is_classification and y_train.nunique() > 2
    if is_classification:
        model = xgb.XGBClassifier(
            eval_metric="logloss", enable_categorical=True, random_state=random_state
        )
    else:
        model = xgb.XGBRegressor(enable_categorical=True, random_state=random_state)
    model.fit(X_train, y_train)

    if is_classification:
        output = {"accuracy": accuracy_score(y_test, model.predict(X_test))}
        y_proba = model.predict_proba(X_test)
        if is_multiclass:
            output["cross_entropy"] = log_loss(y_test, y_proba)
        else:
            output["auc"] = roc_auc_score(y_test, y_proba[:, 1])
            output["cross_entropy"] = log_loss(y_test, y_proba[:, 1])
        return output
    else:
        return {
            "rmse": root_mean_squared_error(y_test, model.predict(X_test)),
        }


def xgboost_utility(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame,
    random_state=1234,
) -> dict:

    label = real_data.columns[-1]  # Assuming the last column is the label

    # Convert all object columns to categorical (with common codes)
    data_temp = pd.concat([real_data, generated_data], ignore_index=True)

    if data_temp[label].nunique() <= 10:
        data_temp[label] = data_temp[label].astype("category")

    for col in data_temp.select_dtypes(include=["object"]).columns:
        data_temp[col] = data_temp[col].astype("category")

    if data_temp[label].dtype == "category":
        data_temp[label] = data_temp[label].cat.codes.astype("category")

    real_data = data_temp.iloc[: len(real_data)].reset_index(drop=True)
    generated_data = data_temp.iloc[len(real_data) :].reset_index(drop=True)

    # Split features and labels
    X_real = real_data.drop(columns=[label])
    X_generated = generated_data.drop(columns=[label])
    y_real = real_data[label]
    y_generated = generated_data[label]

    utility_performance = xgboost_performance(
        X_train=X_generated,
        y_train=y_generated,
        X_test=X_real,
        y_test=y_real,
        random_state=random_state,
    )

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.1, random_state=random_state
    )
    original_performance = xgboost_performance(
        X_train=X_real_train,
        y_train=y_real_train,
        X_test=X_real_test,
        y_test=y_real_test,
        random_state=random_state,
    )
    relative_performance = (
        pd.Series(utility_performance) - pd.Series(original_performance)
    ).to_dict()

    return {
        "utility_performance": utility_performance,
        "original_performance": original_performance,
        "relative_performance": relative_performance,
    }


def xgboost_utility_with_kfold(
    real_data: pd.DataFrame,
    generated_data: pd.DataFrame,
    random_state=1234,
    n_splits=5,
) -> dict:
    # This function can be implemented similarly to xgboost_utility but using KFold cross-validation instead of a single train-test split.
    label = real_data.columns[-1]  # Assuming the last column is the label

    # Convert all object columns to categorical (with common codes)
    data_temp = pd.concat([real_data, generated_data], ignore_index=True)

    if data_temp[label].nunique() <= 10:
        data_temp[label] = data_temp[label].astype("category")

    for col in data_temp.select_dtypes(include=["object"]).columns:
        data_temp[col] = data_temp[col].astype("category")

    if data_temp[label].dtype == "category":
        data_temp[label] = data_temp[label].cat.codes.astype("category")

    real_data = data_temp.iloc[: len(real_data)].reset_index(drop=True)
    generated_data = data_temp.iloc[len(real_data) :].reset_index(drop=True)

    X_real = real_data.drop(columns=[label])
    y_real = real_data[label]
    X_generated = generated_data.drop(columns=[label])
    y_generated = generated_data[label]

    is_classification = y_generated.dtype in (object, str, "category")

    if is_classification:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    performance = {"utility": [], "original": [], "relative": []}

    for train_index, test_index in tqdm.tqdm(
        skf.split(X_real, y_real), total=n_splits, desc="K-Fold Cross-Validation"
    ):
        X_real_train, X_real_test = X_real.iloc[train_index], X_real.iloc[test_index]
        y_real_train, y_real_test = y_real.iloc[train_index], y_real.iloc[test_index]

        utility_performance = xgboost_performance(
            X_train=X_generated,
            y_train=y_generated,
            X_test=X_real_test,
            y_test=y_real_test,
            random_state=random_state,
        )

        original_performance = xgboost_performance(
            X_train=X_real_train,
            y_train=y_real_train,
            X_test=X_real_test,
            y_test=y_real_test,
            random_state=random_state,
        )

        relative_performance = (
            pd.Series(utility_performance) - pd.Series(original_performance)
        ).to_dict()

        performance["utility"].append(utility_performance)
        performance["original"].append(original_performance)
        performance["relative"].append(relative_performance)

    for key in performance:
        df = pd.DataFrame(performance[key])
        performance[key] = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "n": n_splits,
        }
        pd.DataFrame(performance[key]).mean().to_dict()

    return performance
