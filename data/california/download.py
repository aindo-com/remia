import sys
from pathlib import Path
from sklearn.datasets import fetch_california_housing


def download_california_housing(*, path: str) -> None:
    """Download California Housing and save it as CSV."""
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


sys.path.append(".")


if __name__ == "__main__":
    download_california_housing(path="data/california/data.csv")
