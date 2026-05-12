linspace = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PERTURBATIONS = [f"perturbation_{alpha}" for alpha in linspace]

LEAKS = [f"leak_{p}" for p in linspace]

PRIVBAYES_GENERATORS = [f"privbayes_{eps}" for eps in (10, 1e3)]

GENERATIVE_MODELS = PRIVBAYES_GENERATORS + [
    "synthpop",
    "ctgan",
    "tvae",
    "baynet",
    "arf",
    "ddpm",
    "adsgan",
    "pategan",
]

ALL_GENERATORS = LEAKS + PERTURBATIONS + GENERATIVE_MODELS + PRIVBAYES_GENERATORS

SHADOW_MODELING_GENERATORS = (
    PERTURBATIONS
    + PRIVBAYES_GENERATORS
    + [
        "synthpop",
        "ctgan",
        "baynet",
        "arf",
    ]
)

ALL_DATASETS = ["california", "adult", "uk_census"]

SHADOW_MODELING_METRICS = [
    "shadow_modeling_achilles_heels",
    "shadow_modeling_achilles_median",
]

FAST_METRICS = ["domias", "remia_1.0", "dcr_comparison"]

ALL_METRICS = FAST_METRICS + SHADOW_MODELING_METRICS
