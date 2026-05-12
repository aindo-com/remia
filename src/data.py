import pandas as pd
from src.util import load_json


def load_data(dataset: str) -> tuple[pd.DataFrame, dict]:
    """
    Load the dataset from the given path.
    """
    dataset_path = dataset + "/data.csv"
    metadata_path = dataset + "/metadata.json"
    df = pd.read_csv(dataset_path)
    metadata = load_json(metadata_path)
    return df, metadata
