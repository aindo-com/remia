from abc import ABC, abstractmethod
import pandas as pd
import os
from tempfile import TemporaryDirectory
import json
import subprocess
from aindo.anonymize.techniques import PerturbationNumerical, PerturbationCategorical
from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer
from sdv.metadata import Metadata
from functools import partial

from src.env_constants import (
    REPROSYN_ENV_PATH,
    REPROSYN_SCRIPT,
    SYNTHCITY_ENV_PATH,
    SYNTHCITY_SCRIPT,
)


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def store_json(d, *, file: str):
    with open(file, "w") as f:
        json.dump(d, f, indent=4)


class Generator(ABC):
    @abstractmethod
    def fit_generate(self, train_dataset: pd.DataFrame, n: int) -> pd.DataFrame:
        pass

    def __eval__(
        self, train_dataset: pd.DataFrame | str, n: int | None = None, **kwargs
    ) -> pd.DataFrame:
        if isinstance(train_dataset, str):
            train_dataset = pd.read_csv(train_dataset)
        if n is None:
            n = len(train_dataset)
        return self.fit_generate(train_dataset, n, **kwargs)

    def use_seed(self) -> int:
        if not hasattr(self, "seed"):
            raise ValueError("The generator does not have a seed attribute")
        out = self.seed
        self.seed += 1
        return out


class ExternalGenerator(Generator):
    @abstractmethod
    def shell_fit_generate(
        self, train_dataset_path: str, output_file: str, n: int, tmpdir: str
    ):
        pass

    def fit_generate(
        self, train_dataset: pd.DataFrame, n: int | None = None
    ) -> pd.DataFrame:
        if n is None:
            n = len(train_dataset)
        with TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.csv")
            train_dataset_path = os.path.join(tmpdir, "train_dataset.csv")
            train_dataset.to_csv(train_dataset_path, index=False)
            self.shell_fit_generate(train_dataset_path, output_file, n, tmpdir)
            df = pd.read_csv(output_file)
        return df


class RsynGenerator(ExternalGenerator):
    available_generators = ["ctgan", "synthpop", "privbayes", "baynet"]

    def __init__(
        self,
        generator_name: str,
        seed: int,
        metadata: dict | None = None,
        epsilon: float = 1000,
    ):
        if generator_name not in self.available_generators:
            raise ValueError(
                f"Unsupported generator: {generator_name}, supported generators are: {self.available_generators}"
            )
        self.generator_name = generator_name
        self.metadata = metadata
        self.epsilon = epsilon
        self.seed = seed

    def shell_fit_generate(
        self, train_dataset_path: str, output_file: str, n: int, tmpdir: str
    ):
        command = [
            REPROSYN_ENV_PATH,
            REPROSYN_SCRIPT,
            f"--model={self.generator_name}",
            f"--dataset_path={train_dataset_path}",
            f"--output_file={output_file}",
            f"--size={n}",
            f"--epsilon={self.epsilon}",
            f"--seed={self.use_seed()}",
        ]
        if self.metadata is not None:
            metadata_path = os.path.join(tmpdir, "metadata.json")
            store_json(self.metadata, file=metadata_path)
            command.append(f"--metadata={metadata_path}")
        subprocess.run(command, capture_output=False, check=True)


class SynthcityGenerator(ExternalGenerator):
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

    def __init__(self, generator_name: str, seed: int) -> None:
        super().__init__()
        if generator_name not in self.available_generators:
            raise ValueError(
                f"Unsupported generator: {generator_name}, supported generators are: {self.available_generators}"
            )
        self.generator_name = generator_name
        self.seed = seed

    def shell_fit_generate(
        self, train_dataset_path: str, output_file: str, n: int, tmpdir: str
    ):
        command = [
            SYNTHCITY_ENV_PATH,
            SYNTHCITY_SCRIPT,
            f"--model={self.generator_name}",
            f"--dataset_path={train_dataset_path}",
            f"--output_file={output_file}",
            f"--size={n}",
            f"--seed={self.use_seed()}",
        ]
        subprocess.run(command, capture_output=False, check=True)


def random_pop_data(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(
            f"Cannot sample {n} records from a dataset with only {len(df)} records without replacement"
        )
    extracted_data = df.sample(n, replace=False, random_state=seed)
    df.drop(extracted_data.index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return extracted_data.reset_index(drop=True)


class LeakyGenerator(Generator):
    def __init__(
        self,
        additional_data: pd.DataFrame,
        leak_frac: float,
        seed: int,
        train_replace: bool = False,
        test_replace: bool = False,
    ):
        super().__init__()
        self.additional_data = additional_data
        self.leak_frac = leak_frac
        self.train_replace = train_replace
        self.test_replace = test_replace
        self.seed = seed

    def sample_from_additional_data(self, n: int) -> pd.DataFrame:
        if self.test_replace:
            return self.additional_data.sample(n, replace=True).reset_index(drop=True)
        else:
            return random_pop_data(self.additional_data, n, self.use_seed())

    def fit_generate(self, train_dataset: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Create a mock synthetic dataset consisting of records from the independent dataframe release_df where a
        leak_frac percentage has been replaced with records from train_df.
        Code adapted from by Nicolas Milton Plasencia Palacios (https://github.com/nickplas/privacy_evaluation_metrics).
        """

        if self.leak_frac < 0 or self.leak_frac > 1:
            raise ValueError(
                "The value of the parameter 'leak_frac' must be between 0 and 1"
            )

        leak_records = int(n * self.leak_frac)
        new_records = n - leak_records
        synth_df = pd.concat(
            [
                train_dataset.sample(
                    leak_records,
                    replace=self.train_replace,
                    random_state=self.use_seed(),
                ),
                self.sample_from_additional_data(new_records),
            ]
        )

        # shuffle the synthetic dataset
        synth_df = synth_df.sample(frac=1, random_state=self.use_seed()).reset_index(
            drop=True
        )

        return synth_df


def quantize_as(source: pd.DataFrame, target: pd.DataFrame) -> None:
    """
    Quantize the numerical columns of the source dataframe to have the same number of decimal places as the corresponding columns in the target dataframe.
    """
    for col in source.columns:
        if pd.api.types.is_numeric_dtype(source[col]) and pd.api.types.is_numeric_dtype(
            target[col]
        ):
            max_decimals = int(
                target[col]
                .apply(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
                .max()
            )
            source[col] = source[col].round(max_decimals)


class Perturbation(Generator):
    def __init__(self, alpha: float, metadata: dict, seed: int = 0) -> None:
        super().__init__()
        self.alpha = alpha
        self.seed = seed
        self.metadata = metadata

    def fit_generate(self, train_dataset: pd.DataFrame, n: int) -> pd.DataFrame:

        anonymized_cols = {}
        technique_cat = PerturbationCategorical(
            alpha=self.alpha**2, seed=self.use_seed()
        )
        technique_num = PerturbationNumerical(alpha=self.alpha**3, seed=self.use_seed())

        for col in train_dataset.columns:
            if (
                "cat" in self.metadata[col].lower()
                or "alphanumeric" in self.metadata[col].lower()
            ):
                technique = technique_cat
            elif (
                "float" in self.metadata[col].lower()
                or "int" in self.metadata[col].lower()
                or "numeric" in self.metadata[col].lower()
            ):
                technique = technique_num
            else:
                raise ValueError(
                    f"Unsupported column type: {self.metadata[col]} for column {col}"
                )
            anonymized_cols[col] = technique.anonymize_column(train_dataset[col])

        df_anonymized = pd.DataFrame(
            {k: v.reset_index(drop=True) for k, v in anonymized_cols.items()}
        )

        quantize_as(source=df_anonymized, target=train_dataset)

        assert len(df_anonymized) == n, (
            f"The generated dataset has a different number of records ({len(df_anonymized)}) than the specified size ({n})"
        )
        return df_anonymized


class Identity(Generator):
    def fit_generate(self, train_dataset: pd.DataFrame, n: int) -> pd.DataFrame:
        if n > len(train_dataset):
            raise ValueError(
                f"Cannot generate {n} records from a dataset with only {len(train_dataset)} records without replacement"
            )
        return train_dataset.copy().reset_index(drop=True)


class SDV(Generator):
    available_generators_map = {
        "gaussian_copula": GaussianCopulaSynthesizer,
        "copula_gan": partial(CopulaGANSynthesizer, verbose=True),
    }

    def __init__(self, generator_name: str, seed: int = 0) -> None:
        super().__init__()
        self.generator_name = generator_name
        self.seed = seed

    def fit_generate(self, train_dataset: pd.DataFrame, n: int) -> pd.DataFrame:
        set_seed(self.seed)
        metadata = Metadata.detect_from_dataframe(train_dataset)
        synthesizer = self.available_generators_map[self.generator_name](
            metadata=metadata
        )
        synthesizer.fit(train_dataset)
        synthetic_data = synthesizer.sample(num_rows=n)
        return synthetic_data.reset_index(drop=True)

    @staticmethod
    def available_generators() -> list[str]:
        return list(SDV.available_generators_map.keys())


def get_generator(
    generator_name: str, seed: int, metadata: dict | None = None, **kwargs
) -> Generator:
    """
    Factory function that returns a Generator instance based on the generator name.
    Examples:
        generator = get_generator("rsyn_ctgan", metadata=metadata)
        generator = get_generator("synthcity_ctgan", seed=42)
        generator = get_generator("ctgan", seed=42) # synthcity's ctgan by default
        generator = get_generator("ddpm")

    Metadata example for Rsyn generators (othwerwise inferred from pandas types):
        metadata = {
            "age": "int",
            "workclass": "categorical",
            "fnlwgt": "float",
            "education": "categorical",
            "education-num": "categorical",
            "marital-status": "categorical",
            "occupation": "categorical",
            "relationship": "categorical",
            "race": "categorical",
            "sex": "categorical",
            "capital-gain": "float",
            "capital-loss": "float",
            "hours-per-week": "float",
            "native-country": "categorical",
            "income": "categorical"
        }
    """

    if generator_name in SDV.available_generators():
        return SDV(generator_name=generator_name, seed=seed)

    if generator_name.startswith("perturbation_"):
        if metadata is None:
            raise ValueError("Metadata must be provided for AindoAnonymizer")
        alpha = float(generator_name[len("perturbation_") :])
        return Perturbation(alpha=alpha, metadata=metadata, seed=seed)

    if generator_name == "perturbation":
        if metadata is None:
            raise ValueError("Metadata must be provided for AindoAnonymizer")
        return Perturbation(alpha=kwargs["alpha"], metadata=metadata, seed=seed)

    if generator_name.startswith("privbayes_") or generator_name.startswith(
        "rsyn_privbayes_"
    ):
        epsilon = float(generator_name.split("_")[-1])
        return RsynGenerator(
            generator_name="privbayes",
            metadata=metadata,
            epsilon=epsilon,
            seed=seed,
            **kwargs,
        )

    if generator_name == "identity":
        return Identity()

    # First check if the generator name starts with "rsyn" or "synthcity" to avoid ambiguity with the available generators list
    if generator_name.startswith("rsyn"):
        return RsynGenerator(
            generator_name=generator_name[len("rsyn_") :],
            metadata=metadata,
            seed=seed,
            **kwargs,
        )
    if generator_name.startswith("synthcity"):
        return SynthcityGenerator(
            generator_name=generator_name[len("synthcity_") :], seed=seed, **kwargs
        )

    if generator_name in RsynGenerator.available_generators:
        return RsynGenerator(
            generator_name=generator_name, metadata=metadata, seed=seed, **kwargs
        )
    elif generator_name in SynthcityGenerator.available_generators:
        return SynthcityGenerator(generator_name=generator_name, seed=seed, **kwargs)
    else:
        return eval(generator_name)(**kwargs)
