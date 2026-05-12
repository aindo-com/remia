import sys
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_uci_dataset(*, dataset_id: int, path: str, include_y: bool = True):
    uci_data = fetch_ucirepo(id=dataset_id)

    if uci_data.data is None:
        raise ValueError("Dataset not found")

    X = uci_data.data.features
    y = uci_data.data.targets
    df = pd.concat([X, y], axis=1) if include_y else X

    os.makedirs("data", exist_ok=True)

    df.to_csv(path, index=False)


sys.path.append(".")


ID = 2

if __name__ == "__main__":
    path = "data/adult/data.csv"
    download_uci_dataset(dataset_id=ID, include_y=True, path=path)
    df = pd.read_csv(path)
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df.to_csv(path, index=False)
