### add generators
from abc import ABC
from sklearn.calibration import LabelEncoder
import numpy as np
import torch
import pandas as pd
from reprosyn.methods import DS_PRIVBAYES, CTGAN, SYNTHPOP, DS_BAYNET, DS_INDHIST
from .aindo_perturbation import PerturbationNumerical, PerturbationCategorical
import subprocess
from tempfile import TemporaryDirectory
import os

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


class identity(Generator):
    """This generator is the identity generator: just return the input dataset."""

    def __init__(self):
        super().__init__()

    def fit_generate(self, dataset, metadata, size, seed):
        return dataset

    @property
    def label(self):
        return "identity"


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


class leak(Generator):
    def __init__(
        self,
        additional_data: pd.DataFrame,
        leak_frac: float,
        train_replace: bool = False,
        test_replace: bool = False,
    ):
        super().__init__()
        self.additional_data = additional_data
        self.leak_frac = leak_frac
        self.train_replace = train_replace
        self.test_replace = test_replace

    def fit_generate(self, dataset, metadata, size, seed) -> pd.DataFrame:
        if self.leak_frac < 0 or self.leak_frac > 1:
            raise ValueError(
                "The value of the parameter 'leak_frac' must be between 0 and 1"
            )

        leak_records = int(size * self.leak_frac)
        new_records = size - leak_records
        synth_df = pd.concat(
            [
                dataset.sample(leak_records, replace=self.train_replace),
                self.additional_data.sample(new_records, replace=self.test_replace),
            ]
        )

        # shuffle the synthetic dataset
        synth_df = synth_df.sample(frac=1).reset_index(drop=True)

        return synth_df

    @property
    def label(self):
        return "leaky"


class perturbation(Generator):
    """This generator is based on INDHIST."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def fit_generate(self, dataset, metadata, size, seed):
        anonymized_cols = {}
        technique_cat = PerturbationCategorical(alpha=self.alpha**2, seed=seed)
        technique_num = PerturbationNumerical(alpha=self.alpha**3, seed=seed+1)
        for col_info in metadata:
            col = col_info["name"]
            if col_info["type"].lower() == 'finite':
                technique = technique_cat
            else:
                technique = technique_num
            anonymized_cols[col] = technique.anonymize_column(dataset[col])
            assert len(anonymized_cols[col]) == len(dataset), f"Anonymized column {col} has a different number of records ({len(anonymized_cols[col])}) than the original dataset ({len(dataset)})"
        df_anonymized = pd.DataFrame({k: v.reset_index(drop=True) for k, v in anonymized_cols.items()})

        assert len(df_anonymized) == size, f"The generated dataset has a different number of records ({len(df_anonymized)}) than the specified size ({size})"
        return df_anonymized

    @property
    def label(self):
        return "perturbation"


class SynthcityGenerator(Generator):
    available_generators = [
        "nflow",
        "aim",
        "arf",
        "ddpm",
        "bayesian_network",
        "radialgan",
        "fflows",
        "adsgan",
        "image_adsgan",
        "timegan",
        "dpgan",
        "survival_ctgan",
        "marginal_distributions",
        "survival_gan",
        "rtvae",
        "decaf",
        "survival_nflow",
        "ctgan",
        "great",
        "privbayes",
        "dummy_sampler",
        "timevae",
        "survae",
        "pategan",
        "image_cgan",
        "uniform_sampler",
        "tvae",
    ]

    def __init__(self, generator_name: str, epsilon: float | None = None) -> None:
        super().__init__()
        self.generator_name = generator_name
        self.epsilon = epsilon
        self.seed_counter = 0

    def fit_generate(self, dataset, metadata, size, seed) -> pd.DataFrame:
        import sys
        sys.path.append('.')
        from src.env_constants import SYNTHCITY_ENV_PATH, SYNTHCITY_SCRIPT

        with TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.csv")
            train_dataset_path = os.path.join(tmpdir, "train_dataset.csv")
            dataset.to_csv(train_dataset_path, index=False)

            command = [
                SYNTHCITY_ENV_PATH,
                SYNTHCITY_SCRIPT,
                f"--model={self.generator_name}",
                f"--dataset_path={train_dataset_path}",
                f"--output_file={output_file}",
                f"--size={size}",
                f"--seed={seed + self.seed_counter}",
            ]
            if self.epsilon is not None and self.generator_name.lower() in [
                "privbayes",
                "pategan",
                "dpgan",
                "adsgan",
            ]:
                command.append(f"--config={{\"epsilon\": {self.epsilon}}}")

            subprocess.run(command, capture_output=False, check=True)

            df = pd.read_csv(output_file)
            self.seed_counter += 1
        return df


def get_generator(name_generator: str, epsilon: float):

    name = name_generator.lower()
    
    if name.startswith("privbayes") or name.startswith("rsyn_privbayes"):
        epsilon = float(name.split("_")[-1])
        return privbayes(epsilon)

    if name == "identity":
        return identity()
    elif name == "baynet":
        return baynet()
    elif 'privbayes' in name:
        return privbayes(epsilon)
    elif 'ctgan' in name:
        return ctgan()
    elif name == "synthpop":
        return synthpop()
    elif name == "indhist":
        return indhist()
    elif name in SynthcityGenerator.available_generators:
        return SynthcityGenerator(generator_name=name, epsilon=epsilon)
    elif name.startswith("perturbation_"):
        alpha = float(name[len("perturbation_"):])
        return perturbation(alpha=alpha)
    else:
        print("Not a valid generator.")
