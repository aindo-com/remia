### add generators
from abc import ABC
from sklearn.calibration import LabelEncoder
import numpy as np
import torch
import pandas as pd
import json

from reprosyn.methods import DS_PRIVBAYES, CTGAN, SYNTHPOP, DS_BAYNET, DS_INDHIST
from joblib import Memory
import sys
import os

sys.path.append(".")
from src.env_constants import CACHE_LOCATION

memory = Memory(location=CACHE_LOCATION, verbose=0)


def store_json(d, *, file: str):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


def to_reprosyn_metadata(
    *, metadata: dict | None = None, df: pd.DataFrame, processing: bool = False
) -> list:
    reprosyn_metadata = []

    if metadata is None:
        metadata = {col_name: str(col_type) for col_name, col_type in df.dtypes.items()}

    for col_name, col_type in metadata.items():
        col_type = str(col_type).lower()
        if col_type in ("float", "float64", "float32", "numeric", float):
            c = {
                "name": col_name,
                "type": "Float",
                "representation": "number",
                "min": 0.0 if processing else float(df[col_name].min()),
                "max": 1.0 if processing else float(df[col_name].max()),
            }
        elif col_type in ("int", "int64", "int32", "integer", int):
            c = {
                "name": col_name,
                "type": "Integer",
                "representation": "number",
                "min": int(df[col_name].min()),
                "max": int(df[col_name].max()),
            }
        elif col_type in (
            "categorical",
            "object",
            "category",
            "string",
            "alphanumeric",
            str,
        ):
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


class Generator(ABC):
    """Base class for generators"""

    def __init__(self):
        self.trained = False

    @property
    def label(self):
        return "Unnamed Generator"

    def __str__(self):
        return self.label

    def discretize(self, data: pd.DataFrame, metadata: dict, n_bins: int = 100):
        data = data.copy()
        encoders = {}

        for m in metadata:
            col = m["name"]
            if m["type"] == "finite":
                continue
            if len(data[col].unique()) < n_bins or data[col].dtype.name not in [
                "object",
                "category",
            ]:
                encoders[col] = {
                    "type": "categorical",
                    "model": LabelEncoder().fit(data[col]),
                }
                data[col] = encoders[col]["model"].transform(data[col])

            else:
                col_data = pd.cut(data[col], bins=n_bins)
                encoders[col] = {
                    "type": "continuous",
                    "model": LabelEncoder().fit(col_data),
                }
                data[col] = encoders[col]["model"].transform(col_data)

        self.encoders = encoders

        discretized_metadata = []
        for col_name, col_type in data.dtypes.items():
            c = {
                "name": col_name,
                "type": "finite",
                "representation": data[col_name]
                .sort_values()
                .unique()
                .astype(str)
                .tolist(),
            }
            discretized_metadata.append(c)

        return data.astype(str), discretized_metadata

    def undiscretize(self, data: pd.DataFrame):
        for col in data.columns:
            if col not in self.encoders:
                continue
            inversed = self.encoders[col]["model"].inverse_transform(
                data[col].astype(int)
            )
            if self.encoders[col]["type"] == "categorical":
                data[col] = inversed
            elif self.encoders[col]["type"] == "continuous":
                output = []
                for interval in inversed:
                    output.append(np.random.uniform(interval.left, interval.right))

                data[col] = output
            else:
                raise RuntimeError(f"Invalid encoder {self.encoders[col]}")

        return data


class baynet(Generator):
    """This generator is based on BAYNET."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        discretized_dataset, discretized_metadata = self.discretize(dataset, metadata)
        baynet = DS_BAYNET(
            dataset=discretized_dataset,
            metadata=discretized_metadata,
            size=size,
            seed=seed,
        )
        baynet.run()
        return self.undiscretize(baynet.output)

    @property
    def label(self):
        return "BAYNET"


class privbayes(Generator):
    """This generator is based on privbayes."""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        discretized_dataset, discretized_metadata = self.discretize(dataset, metadata)
        pbayes = DS_PRIVBAYES(
            dataset=discretized_dataset,
            metadata=discretized_metadata,
            size=size,
            epsilon=self.epsilon,
            seed=seed,
        )
        pbayes.run()
        return self.undiscretize(pbayes.output)

    @property
    def label(self):
        return "privbayes"


class ctgan(Generator):
    """This generator is based on CTGAN."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed, epochs=50):
        torch.manual_seed(seed)
        ctgan = CTGAN(dataset=dataset, metadata=metadata, size=size, epochs=epochs)
        ctgan.run()
        return ctgan.output

    @property
    def label(self):
        return "CTGAN"


class synthpop(Generator):
    """This generator is based on SYNTHPOP."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        spop = SYNTHPOP(dataset=dataset, metadata=metadata, size=size, seed=seed)
        spop.run()
        return spop.output

    @property
    def label(self):
        return "SYNTHPOP"


class indhist(Generator):
    """This generator is based on INDHIST."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        indhist = DS_INDHIST(dataset=dataset, metadata=metadata, size=size)
        indhist.run()
        return indhist.output

    @property
    def label(self):
        return "INDHIST"


def get_generator(name_generator: str, epsilon: float):

    name = name_generator.lower()

    if name == "baynet":
        return baynet()
    elif name == "privbayes":
        return privbayes(epsilon)
    elif name == "ctgan":
        return ctgan()
    elif name == "synthpop":
        return synthpop()
    elif name == "indhist":
        return indhist()
    else:
        raise ValueError(
            f"Generator {name_generator} not recognized. Available generators are: identity, baynet, privbayes, ctgan, synthpop, indhist."
        )


@memory.cache
def generate(
    df: pd.DataFrame,
    metadata: list,
    generator_name: str,
    size: int,
    seed: int = 0,
    epsilon: float = 0.0,
) -> pd.DataFrame:
    generator = get_generator(name_generator=generator_name, epsilon=epsilon)
    df_synth = generator.fit_generate(
        dataset=df, metadata=metadata, size=size, seed=seed
    )
    df_synth = df_synth.astype(df.dtypes.to_dict())
    assert df_synth.dtypes.equals(df.dtypes), (
        "Generated data has different dtypes than original data"
    )
    return df_synth


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-D", type=str, required=True)
    parser.add_argument("--output_file", "-O", type=str, default="synthetic.csv")
    parser.add_argument(
        "--model",
        "-M",
        type=str,
        choices=["baynet", "privbayes", "ctgan", "synthpop", "indhist"],
        help="benchmarking generative model used for synthesis",
        required=True,
    )
    parser.add_argument(
        "--size", "-S", type=int, default=0, help="size of generated dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1000.0, help="privacy budget for DP generators"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="path to metadata json file for Rsyn generators",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)

    if args.size == 0:
        args.size = len(df)
    metadata = None
    if args.metadata is not None:
        with open(args.metadata, "r") as f:
            metadata = json.load(f)

    metadata = to_reprosyn_metadata(metadata=metadata, df=df, processing=False)

    df_synth = generate(
        df=df,
        metadata=metadata,
        generator_name=args.model,
        size=args.size,
        seed=args.seed,
        epsilon=args.epsilon,
    )
    os.remove("output.csv")  # remove temporary file created by rsyn
    df_synth.to_csv(args.output_file, index=False)
    # print(f"Synthetic data generated and saved to {args.output_file}")
