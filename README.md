# ReMIA: a Powerful and Efficient Alternative to Membership Inference Attacks against Synthetic Data Generators

This is the code repository for the article __ReMIA: a Powerful and Efficient Alternative to Membership Inference Attacks against Synthetic Data Generators__ by Davide Scassola, Andrea Coser, and Sebastiano Saccani.

## Installation
This project requires multiple environments to run. We recommend using Conda to manage them.

### Installing the main environment
```bash
make .venv
source .venv/bin/activate
make install
```

To install the additional environments, we recommend using `conda`. If you don't have Conda, you can install `micromamba` with:
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
alias conda=micromamba
```

### Installing Achille's Heels environment
The environment is needed to run a modified version of the [original code](https://github.com/imperial-aisp/MIA-synthetic) from the article __Achilles' Heels: Vulnerable Record Identification in Synthetic Data Publishing__.

```bash
cd submodules/achilles_heels
conda create --name achilles_heels python=3.10
conda activate achilles_heels
git clone https://github.com/alan-turing-institute/reprosyn
cd reprosyn
curl -sSL https://install.python-poetry.org | python3 -
~/.local/bin/poetry install -E ektelo
cd ..
git clone git@github.com:imperial-aisp/querysnout.git
cd querysnout/src/optimized_qbs
python setup.py install
cd ../../..
pip install torch==2.0.0
```

### Installing DOMIAS environment
```bash
conda create --name domias python=3.10
conda activate domias
pip install domias==0.0.5
pip install torch==2.2.2
pip install tqdm
pip install pykeops==2.3
pip install numpy==1.26.4
pip install pandas==2.3.3
```

### Installing Synthcity environment
```bash
conda create --name synthcity python=3.12
conda activate synthcity
pip install synthcity==0.2.12
pip install torch==2.2.2
pip install numpy==1.26.4
pip install pandas==2.3.3
pip install transformers==4.44.2
pip install opacus==1.4.0
```

We also provide the full list of dependencies in the `envs_pip_list` folder in order to improve reproducibility.

## Getting the Data
Metadata and download scripts are already provided in the `data` folder.
In order to download a dataset, activate the main environment and run
```bash
python data/<name-of-the-dataset>/download.py
```
the data will be stored as `data/<name-of-the-dataset>/data.csv`.
Sometimes the download script will only print instructions to download the data manually; in that case, you would have to store the data as `data/<name-of-the-dataset>/data.csv` yourself.

## Running experiments
In order to run experiments, you have to first activate the main environment:
```bash
source .venv/bin/activate
```

You can run a single privacy evaluation experiment in the following way:
```bash
python scripts/evaluate_privacy.py --metric <metric> --dataset <dataset> --generator <generator> --seed <seed> --training_size <size>
```

The arguments are:
- `--metric` / `-M` (string): Privacy metric to evaluate.
- `--dataset` / `-D` (string): Dataset to use for evaluation.
- `--generator` / `-G` (string): Data generator/synthesis method.
- `--seed` / `-S` (integer): Random seed for reproducibility. Default: `0`
- `--training_size` / `-T` (integer): Size of the training dataset. Default: `1000`

The available options are:
- Metric: `remia`, `domias`, `shadow_modeling_achilles_heels`
- Dataset: `adult`, `california`, `uk_census`
- Generator: `synthpop`, `ctgan`, `tvae`, `baynet`, `arf`, `ddpm`, `adsgan`, `pategan`, `privbayes_<epsilon>`, `leak_<fraction>`, `perturbation_<alpha>`


When an experiment is completed, the result will be printed and stored in the `experiments/output` folder.

#### Example:
```bash
python scripts/evaluate_privacy.py --metric remia --dataset adult --generator synthpop
```

## Reproducing paper experiments
In order to run all the experiments, run
```bash
python scripts/reproduce_experiments.py
```
Results will be stored in the folder `experiments/privacy_evaluation` and `experiments/quality_evaluation`

You can then obtain the relative plots by running
```bash
python scripts/article_tables_and_plots.py
```
these will be stored in the `article/figures` folder


## Getting stored article experiments
We provide the results of the experiments that we included in the article in the `experiments.tar.xz` file. You can extract it with the following command:
```bash
tar -xJf experiments.tar.xz
```
or simply run
```bash
make experiments
```
when the experiments folder is not present in the repository.