import sys

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, ttest_rel, binom

import numpy as np

sys.path.append(".")
from experiments_to_csv import get_results, get_quality_results
from df_to_latex import df_to_latex

SIGNIFICANCE_LEVEL = 1e-3
DIGITS = 3

METRICS = {
    "dcr_comparison": "DCR",
    "domias": "DOMIAS",
    "remia_1.0": "ReMIA",
    "shadow_modeling_achilles_heels": "smMIA(out)",
    "shadow_modeling_achilles_median": "smMIA(med)",
}

DATASETS = {
    "adult": "Adult",
    "uk_census": "UK Census",
    "california": "California",
}

GENERATIVE_MODELS = {
    "aindo": "Aindo",
    "synthpop": "SynthPop",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "baynet": "BayNet",
    "arf": "ARF",
    "ddpm": "TabDDPM",
    "adsgan": "ADSGAN",
    "pategan": "PateGAN",
    "privbayes_10": "PrivBayes(ε=10)",
    "privbayes_1000.0": "PrivBayes(ε=1000)",
}


def binomial_test(n, successes):
    return 1 - binom.cdf(k=successes, n=n, p=0.5)


# def fix_names(s: pd.Series):
#     # substitute "rsyn_" or "synthcity_" with empty string
#     s = s.str.replace(r"^(rsyn|synthcity)_", "", regex=True)

#     # for privbayes, substitute "privbayes_<number>" with "privbayes(ε=<number>)"
#     # and normalize trailing .0 (e.g., 1000.0 -> 1000)
#     s = s.str.replace(
#         r"^privbayes_(\d+(?:\.\d+)?)$",
#         lambda m: (
#             f"privbayes(ε={m.group(1).rstrip('0').rstrip('.')})"
#             if "." in m.group(1)
#             else f"privbayes(ε={m.group(1)})"
#         ),
#         regex=True,
#     )
#     return s


def get_privacy_results(
    average_over_seeds: bool = False,
    only_first_seed: bool = False,
    preselect_metrics: bool = True,
) -> pd.DataFrame:
    df = get_results()

    # rename generators with GENERATIVE_MODELS
    df["generator_name"] = df["generator_name"].replace(GENERATIVE_MODELS)

    # keep only most training size = 1000
    df = df[df["training_size"] == 1000]

    # filter only last experiment (by timestamp) for each metric, generator, dataset, seed
    df = df.loc[
        df.groupby(["metric", "generator_name", "dataset", "seed"])[
            "timestamp"
        ].idxmax()
    ].drop(columns=["timestamp"])

    # rename datasets
    df["dataset"] = df["dataset"].replace(DATASETS)

    # select only relevant metrics
    if preselect_metrics:
        df = df[df["metric"].isin(METRICS.keys())]
        # rename metrics
        df["metric"] = df["metric"].replace(METRICS)

    # unify scores from columns "score" and "test_auc" in the same column "score"
    df["score"] = df["score"].fillna(df["test_auc"])

    # sort by generator, metric, dataset
    df = df.sort_values(["generator_name", "dataset", "metric"])

    if average_over_seeds:
        print(
            "Warning: aggregating p-values by using the binomtest on the sum of successes and n"
        )
        # average over seeds
        df_std = (
            df.groupby(["metric", "generator_name", "dataset", "training_size"])
            .agg(
                {
                    "test_auc": "std",
                    "score": "std",
                    "overhead_time": "std",
                }
            )
            .reset_index()
        )
        df = (
            df.groupby(["metric", "generator_name", "dataset", "training_size"])
            .agg(
                {
                    "test_successes": "sum",
                    "n": "sum",
                    "test_auc": "mean",
                    "score": "mean",
                    "generation_calls": "mean",
                    "total_data_usage": "mean",
                    "overhead_time": "mean",
                    "seed": "count",
                }
            )
            .reset_index()
        ).rename(columns={"seed": "num_seeds"})
        df = pd.merge(
            df,
            df_std,
            on=["metric", "generator_name", "dataset", "training_size"],
            suffixes=("", "_std"),
        )
    elif only_first_seed:
        # keep first seed only
        df = df.loc[
            df.groupby(["metric", "generator_name", "dataset", "training_size"])[
                "seed"
            ].idxmin()
        ]

    # add the p-value column for each metric
    df["acc_p_value"] = binomial_test(n=df["n"], successes=df["test_successes"])
    df["significant"] = df["acc_p_value"] <= SIGNIFICANCE_LEVEL

    return df


def privacy_scores_tables(df_privacy: pd.DataFrame):
    """
    table: PRIVACY SCORES OVERVIEW
        only for adult
        rows: generators (all but leaks, perturbations)
        columns: metric (ours, domias, smMIA (outlier), smMIA (median))
        content: attacks score.
        underline values that are not statistically significant?
    """

    # rename generator_name into Generator
    df_privacy = df_privacy.rename(columns={"generator_name": "Generator"})

    # filter only generators in GENERATIVE_MODELS and keep leak_0.0
    df = df_privacy[
        # df_privacy["Generator"].isin(list(GENERATIVE_MODELS.values()) + ["leak_0.0"])
        df_privacy["Generator"].isin(list(GENERATIVE_MODELS.values()))
    ]

    # rename leak_0.0 to "perfect generator"
    # df["Generator"] = df["Generator"].replace("leak_0.0", "perfect generator")

    values = ["score", "significant"]
    if "score_std" in df_privacy.columns:
        values += ["score_std"]

    # pivot table
    df_pivot = (
        df.pivot_table(
            index=["Generator", "dataset"],
            columns="metric",
            values=values,
        )
        * 100
    )  # convert to percentage

    # sort columns by METRICS order
    df_pivot = df_pivot.reindex(columns=list(METRICS.values()), level=1)

    for dataset in df["dataset"].unique():
        df_pivot_dataset = df_pivot.xs(dataset, level="dataset")
        df_score = df_pivot_dataset["score"]
        df_significant = df_pivot_dataset["significant"]

        df_to_latex(
            df=pd.DataFrame(df_score),
            df_std=pd.DataFrame(df_pivot_dataset["score_std"])
            if "score_std" in df_pivot_dataset.columns
            else None,
            highlight=pd.DataFrame(df_significant),
            filename=f"article/tables/privacy_scores_{dataset}.tex",
            digits=1,
        )


def quality_utility_table(df_quality: pd.DataFrame):
    """
    table: QUALITY AND UTILITY OVERVIEW
        rows: generators (all but leaks, perturbations)
        columns: dataset
        content: (detection, ml efficacy).
    """
    df_quality = df_quality.rename(columns={"generator": "Generator"})

    # only keep generators in GENERATIVE_MODELS
    df_quality = df_quality[
        df_quality["Generator"].isin(list(GENERATIVE_MODELS.values()))
    ]

    # renaming columns
    COLUMNS = {
        "xgboost_discr_auc": "Detection",
        "xgboost_utility_score": "ML Efficacy",
        "xgboost_discr_auc_std": "Detection_std",
        "xgboost_utility_score_std": "ML Efficacy_std",
    }
    df_quality = df_quality.rename(columns=COLUMNS)

    values = ["Detection", "ML Efficacy"]
    if "Detection_std" in df_quality.columns:
        values += ["Detection_std", "ML Efficacy_std"]

    # pivot table with generator as index, dataset as columns, and xgboost_discr_auc and xgboost_utility_score as values
    df_pivot = (
        df_quality.pivot_table(
            index="Generator",
            columns="dataset",
            values=values,
        )
        * 100
    )

    df_pivot["ML Efficacy"] = -df_pivot[
        "ML Efficacy"
    ]  # invert ML Efficacy so that higher is better for both metrics

    df_to_latex(
        df=df_pivot["Detection"],
        df_std=df_pivot["Detection_std"].rename(
            columns={
                "Detection_std": "Detection",
            }
        )
        if "Detection_std" in df_pivot.columns
        else None,
        filename="article/tables/detection.tex",
        digits=2,
    )

    df_to_latex(
        df=-df_pivot["ML Efficacy"],
        df_std=df_pivot["ML Efficacy_std"].rename(
            columns={
                "ML Efficacy_std": "ML Efficacy",
            }
        )
        if "ML Efficacy_std" in df_pivot.columns
        else None,
        filename="article/tables/ml_efficacy.tex",
        digits=2,
    )


def metrics_power_comparison_table():

    df = get_privacy_results(
        average_over_seeds=True, only_first_seed=False, preselect_metrics=False
    )
    df_seeds = get_privacy_results(
        average_over_seeds=False, only_first_seed=False, preselect_metrics=False
    )

    # rename generator_name into Generator
    df = df.rename(columns={"generator_name": "Generator"})
    df_seeds = df_seeds.rename(columns={"generator_name": "Generator"})

    # keep only GENERATIVE_MODELS
    df = df[df["Generator"].isin(list(GENERATIVE_MODELS.values()))]
    df_seeds = df_seeds[df_seeds["Generator"].isin(list(GENERATIVE_MODELS.values()))]

    metrics = {
        "dcr_comparison": "DCR",
        "domias": "DOMIAS",
        "remia_1.0": "ReMIA(f=1.0)",
        "remia_0.5": "ReMIA(f=0.5)",
        "shadow_modeling_achilles_heels": "smMIA(out)",
        "shadow_modeling_achilles_median": "smMIA(med)",
    }

    # subsetting metrics and renaming them in one step
    df = df[df["metric"].isin(metrics.keys())].copy()
    df["metric"] = df["metric"].replace(metrics)

    df_seeds = df_seeds[df_seeds["metric"].isin(metrics.keys())].copy()
    df_seeds["metric"] = df_seeds["metric"].replace(metrics)

    # setting as minimum the score of 0.5 for all scores
    df["score"] = df["score"].clip(lower=0.5)
    df_seeds["score"] = df_seeds["score"].clip(lower=0.5)
    df_seeds["acc"] = df_seeds["test_successes"] / df_seeds["n"]

    # pivot table, with generator, dataset, seed as index, metric as columns, and score as values
    df_pivot = df.pivot_table(
        index=["Generator", "dataset"],
        columns="metric",
        values=["score", "score_std", "significant"],
    )
    df_pivot_seeds = df_seeds.pivot_table(
        index=["Generator", "dataset", "seed"],
        columns="metric",
        values="acc",
    )

    # sort columns by METRICS order
    df_pivot = df_pivot.reindex(columns=list(metrics.values()), level=1)
    df_pivot_seeds = df_pivot_seeds.reindex(columns=list(metrics.values()), level=1)
    # correlation with significance
    correlation_type = "spearman"
    corr_method = spearmanr if correlation_type == "spearman" else pearsonr
    rho = df_pivot["score"].corr(correlation_type)
    pval = df_pivot["score"].corr(method=lambda x, y: corr_method(x, y)[1]) - np.eye(
        *rho.shape
    )
    p = pval.map(lambda x: "".join(["*" for t in [0.05, 1e-2, 1e-3] if x <= t]))
    corr_significant = pval < SIGNIFICANCE_LEVEL
    corr = rho.round(2).astype(str) + p
    # empty the diagonal
    np.fill_diagonal(rho.values, None)
    np.fill_diagonal(corr.values, None)
    np.fill_diagonal(corr_significant.values, False)

    # rename index to Metric
    corr.index.name = "Metric"

    # drop DCR row and column
    corr = corr.drop(index="DCR", columns="DCR")

    # drop last corr row
    #corr = corr.drop(index=corr.index[-1])

    df_to_latex(
        df=corr,
        # highlight=corr_significant,
        filename="article/tables/metrics_power_comparison_corr.tex",
        digits=2,
    )

    # how many times method is statistically greater than 0.5
    def count_significant(metric):
        total_significant = df_pivot["significant", metric].sum()
        total = len(df_pivot["significant", metric].dropna())
        return {
            "total_significant": int(total_significant),
            "total": int(total),
            "fraction_significant": float(total_significant / total),
        }

    df_pivot_minimum = (
        df_pivot_seeds.groupby(["Generator", "dataset"]).min().reset_index()
    )

    def count_significant_min(metric):
        total_significant = (df_pivot_minimum[metric] > 0.5).sum()
        total = len(df_pivot_minimum[metric].dropna())
        return {
            "total_significant": int(total_significant),
            "total": int(total),
            "fraction_significant": float(total_significant / total),
        }

    # how many times method is greater than other method
    def count_greater_than(metric1, metric2, significance_level=SIGNIFICANCE_LEVEL):
        df_temp = df_pivot_seeds.unstack(level=2)[[metric1, metric2]]
        greater_count = 0
        smaller_count = 0
        total_count = len(df_temp)
        for scores in df_temp.iloc:
            greater = (
                ttest_rel(scores[metric1], scores[metric2], alternative="greater")[1]
                < significance_level
            )
            smaller = (
                ttest_rel(scores[metric1], scores[metric2], alternative="less")[1]
                < significance_level
            )
            if greater:
                greater_count += 1
            if smaller:
                smaller_count += 1
        return {
            "total_significantly_greater": int(greater_count),
            "total_significantly_smaller": int(smaller_count),
            "total": int(total_count),
            "fraction_significantly_greater": float(greater_count / total_count),
            "fraction_significantly_smaller": float(smaller_count / total_count),
        }

    # mean and std difference of scores between two methods
    def score_difference(metric1, metric2):
        diff = (df_pivot["score", metric1] - df_pivot["score", metric2]) * 100
        return {
            "mean_diff": round(float(diff.mean()), 2),
            "std_diff": round(float(diff.std()), 2),
        }

    summary_table_rows = {}
    for metric in (
        "ReMIA(f=1.0)",
        "ReMIA(f=0.5)",
        "DCR",
        "DOMIAS",
        "smMIA(out)",
        "smMIA(med)",
    ):
        sig = count_significant_min(metric)
        score_diff = score_difference(metric, "ReMIA(f=1.0)")
        mean_score_diff = f"{score_diff['mean_diff']:.1f}"
        std_score_diff = f"{score_diff['std_diff']:.1f}"

        summary_table_rows[metric] = {
            "Risk Detected (%)": f"{sig['fraction_significant'] * 100:.1f}% ({sig['total_significant']}/{sig['total']})",
            # score should be reported as $29.7\footnotesize{\pm16.4}
            "Score Difference vs ReMIA (%)": rf"\num{{{mean_score_diff} +- {std_score_diff}}}"
            if metric != "ReMIA(f=1.0)"
            else "-",
        }
    df_summary = pd.DataFrame.from_dict(summary_table_rows, orient="index")

    # # store summary_dictionary in a json file
    # import json
    # with open(f"article/tables/metrics_power_comparison_summary.json", "w") as f:
    #     json.dump(summary_dictionary, f, indent=4)

    # store summary table in a latex file
    df_to_latex(
        df=df_summary,
        filename="article/tables/metrics_power_comparison_summary.tex",
        digits=2,
    )


def leak_fraction_alpha_vs_privacy_plot(
    df_privacy: pd.DataFrame, legend_on_right: bool = True
):
    """
    plot: PRIVACY RISK VS LEAK FRACTION/NOISE LEVEL
        x: leak fraction
        y: privacy risk
        one line for each method
        one plot for each dataset

    Args:
        df_privacy: DataFrame with privacy results
        legend_outside: if True, place legend on the right outside plots; if False, place below
    """

    # filter only leak models and perturbation models
    df = df_privacy[
        df_privacy["generator_name"].str.startswith("leak_")
        | df_privacy["generator_name"].str.startswith("perturbation_")
    ]
    df["x"] = df["generator_name"].str.extract(r"_(0\.\d+|1\.0)")[0].astype(float)

    # cap lower bound of score to 0.5 (random guess) for better visualization
    df["score"] = df["score"].clip(lower=0.5)

    # remove smMIA (median)
    df = df[df["metric"] != "smMIA(med)"]

    def line_plot(df_generator, ax, x_label):
        if x_label == "Leak Fraction":
            # remove smMIA (both)
            df_generator = df_generator[~df_generator["metric"].str.contains("smMIA")]

            # add theoretical line for smMIA (0.5 + leak_fraction / 2)
            leak_fraction = df_generator["x"].values
            theoretical_sm_mia = 0.5 + leak_fraction / 2
            fourth_color = sns.color_palette("tab10")[3]
            ax.plot(
                leak_fraction,
                theoretical_sm_mia,
                label="smMIA(out)",
                linestyle="-",
                # marker=".",
                alpha=0.6,
                color=fourth_color,  # get fourth color from palette
            )

        sns.lineplot(
            data=df_generator,
            x="x",
            y="score",
            hue="metric",
            marker=".",
            ax=ax,
            alpha=0.6,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Privacy Score")
        ax.grid(True, alpha=0.3)

    for dataset in df["dataset"].unique():
        if legend_on_right:
            fig, (ax_leak, ax_perturbation) = plt.subplots(
                1, 2, sharey=True, figsize=(6, 1.5)
            )
        else:
            fig, (ax_leak, ax_perturbation) = plt.subplots(
                1, 2, sharey=True, figsize=(8, 4)
            )
        df_dataset = df[df["dataset"] == dataset]
        line_plot(
            df_dataset[df_dataset["generator_name"].str.startswith("leak_")],
            ax_leak,
            "Leak Fraction",
        )
        line_plot(
            df_dataset[df_dataset["generator_name"].str.startswith("perturbation_")],
            ax_perturbation,
            "Noise Level",
        )

        # Remove individual legends and create a single shared legend
        ax_leak.get_legend().remove()
        ax_perturbation.get_legend().remove()

        # Get handles and labels from both axes
        handles1, labels1 = ax_leak.get_legend_handles_labels()
        handles2, labels2 = ax_perturbation.get_legend_handles_labels()

        # Combine, preserving order and avoiding duplicates
        seen = set()
        combined_handles = []
        combined_labels = []
        for h, l in zip(handles1 + handles2, labels1 + labels2):
            if l not in seen:
                combined_handles.append(h)
                combined_labels.append(l)
                seen.add(l)

        if legend_on_right:
            fig.legend(
                combined_handles,
                combined_labels,
                loc="center left",
                bbox_to_anchor=(0.85, 0.5),
                frameon=True,
            )
            plt.subplots_adjust(right=0.82, wspace=0.15)
        else:
            fig.legend(
                combined_handles,
                combined_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(combined_handles),
                frameon=True,
            )
            plt.subplots_adjust(bottom=0.18)

        plt.savefig(
            f"article/figures/privacy_risk_vs_leak_fraction_and_noise_level_{dataset}.pdf".lower().replace(
                " ", "_"
            ),
            bbox_inches="tight",
        )


def pq_scatter_plot(
    dataset: str,
    privacy_metric: str,
    quality_metric: str,
    df: pd.DataFrame,
    file: str,
):
    # select only generators in GENERATIVE_MODELS
    df_models = df[df["generator"].isin(list(GENERATIVE_MODELS.values()))]

    # select perturbations for separate line
    df_perturbations = df[df["generator"].str.startswith("perturbation_")]

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.lineplot(
        data=df_perturbations,
        x=quality_metric,
        y="score",
        marker="o",
        label="Noise-based Anonymization",
        color="gray",
        alpha=0.75,
        ax=ax,
    )
    sns.scatterplot(
        data=df_models,
        x=quality_metric,
        y="score",
        hue="generator",
        legend="full",
        ax=ax,
    )

    quality_direction = "lower" if quality_metric == "Detection" else "higher"
    # sns.scatterplot(data=df_perturbations, x="score", y="x", hue="generator", style="generator")
    ax.set_xlabel(f"{quality_metric} ({quality_direction} is better)")
    ax.set_ylabel(f"{privacy_metric} Privacy Score (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(
        file,
        bbox_inches="tight",
    )
    plt.close(fig)


def appendix_quality_vs_privacy_plot(
    *, df_quality: pd.DataFrame, df_privacy: pd.DataFrame, detection: bool = True
):
    # remove leak experiments
    df_quality = df_quality[~df_quality["generator"].str.startswith("leak")]

    # join quality and privacy dataframes on generator and dataset
    df_privacy = df_privacy.rename(columns={"generator_name": "generator"})
    df_privacy_quality = pd.merge(
        df_quality, df_privacy, on=["generator", "dataset"], how="inner"
    )

    new_names = {
        "xgboost_discr_auc": "Detection",
        "xgboost_utility_score": "ML Efficacy",
    }
    df_privacy_quality = df_privacy_quality.rename(columns=new_names)

    metrics = ["DCR", "DOMIAS", "ReMIA", "smMIA(out)", "smMIA(med)"]
    datasets = [
        d for d in DATASETS.values() if d in df_privacy_quality["dataset"].unique()
    ]
    if not datasets:
        datasets = sorted(df_privacy_quality["dataset"].unique())

    quality_metric = "Detection" if detection else "ML Efficacy"
    quality_direction = "lower" if quality_metric == "Detection" else "higher"

    df_plot = df_privacy_quality[df_privacy_quality["metric"].isin(metrics)].dropna(
        subset=["score", quality_metric]
    )
    generator_labels = [
        g for g in GENERATIVE_MODELS.values() if g in df_plot["generator"].unique()
    ]
    palette = sns.color_palette("tab20", n_colors=max(1, len(generator_labels)))
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "d", "*", "h", "8"]
    color_map = {g: palette[i] for i, g in enumerate(generator_labels)}
    marker_map = {
        g: marker_cycle[i % len(marker_cycle)] for i, g in enumerate(generator_labels)
    }

    if not df_plot.empty:
        x_ranges = (
            df_plot.groupby("dataset")[quality_metric]
            .agg(["min", "max"])
            .to_dict(orient="index")
        )
        y_ranges = (
            df_plot.groupby("metric")["score"]
            .agg(["min", "max"])
            .to_dict(orient="index")
        )
    else:
        x_ranges = {}
        y_ranges = {}

    def add_pq_plot(ax, dataset, metric):
        df_metric_dataset = df_plot[
            (df_plot["dataset"] == dataset) & (df_plot["metric"] == metric)
        ]

        if df_metric_dataset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            ax.grid(False)
            return

        df_models = df_metric_dataset[
            df_metric_dataset["generator"].isin(generator_labels)
        ]
        df_perturbations = df_metric_dataset[
            df_metric_dataset["generator"].str.startswith("perturbation_")
        ].copy()

        if not df_perturbations.empty:
            df_perturbations["perturbation_level"] = pd.to_numeric(
                df_perturbations["generator"].str.extract(r"_(0\.\d+|1\.0)$")[0],
                errors="coerce",
            )
            sort_key = (
                "perturbation_level"
                if df_perturbations["perturbation_level"].notna().any()
                else quality_metric
            )
            df_perturbations = df_perturbations.sort_values(sort_key)
            ax.plot(
                df_perturbations[quality_metric],
                df_perturbations["score"],
                color="gray",
                marker="o",
                linestyle="-",
                linewidth=1.2,
                markersize=3,
                alpha=0.7,
                label="_nolegend_",
            )

        for g in generator_labels:
            df_g = df_models[df_models["generator"] == g]
            if df_g.empty:
                continue
            ax.scatter(
                df_g[quality_metric],
                df_g["score"],
                color=color_map[g],
                marker=marker_map[g],
                s=35,
                alpha=0.8,
                label="_nolegend_",
            )

        ax.grid(True, alpha=0.3)
        if metric == "DCR":
            ax.axhline(
                0.5,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                zorder=-1,
            )
            current_ticks = ax.get_yticks()
            if not np.isclose(current_ticks, 0.5).any():
                ax.set_yticks(np.sort(np.append(current_ticks, 0.5)))
        dataset_range = x_ranges.get(dataset)
        if dataset_range:
            x_min = float(dataset_range["min"])
            x_max = float(dataset_range["max"])
            x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.02
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

        metric_range = y_ranges.get(metric)
        if metric_range:
            y_min = float(metric_range["min"])
            y_max = float(metric_range["max"])
            y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.02
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # make a (metric x datasets) grid of plots with shared x and y axes, shared legend on the bottom
    os.makedirs("article/figures/quality_vs_privacy", exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(datasets),
        figsize=(8.27, 11.69),
        sharex="col",
        sharey="row",
    )
    axes = np.array(axes).reshape(len(metrics), len(datasets))

    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            add_pq_plot(ax, dataset, metric)

            if i == 0:
                ax.set_title(dataset)
            if j == 0:
                ax.set_ylabel(f"{metric} score")
            else:
                ax.set_ylabel("")
            if i == len(metrics) - 1:
                ax.set_xlabel(f"{quality_metric} ({quality_direction} is better)")
            else:
                ax.set_xlabel("")

    legend_handles = []
    legend_labels = []

    if df_plot["generator"].str.startswith("perturbation_").any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                linestyle="-",
                linewidth=1.2,
                markersize=4,
                alpha=0.7,
            )
        )
        legend_labels.append("Noise Anon.")

    for g in generator_labels:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color_map[g],
                marker=marker_map[g],
                linestyle="",
                markersize=6,
            )
        )
        legend_labels.append(g)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.52, 0.02),
            ncol=min(5, len(legend_handles)),
            frameon=True,
        )
        fig.tight_layout(rect=[0.05, 0.08, 0.98, 0.98])
    else:
        fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.98])

    name = "detection" if detection else "ml_efficacy"

    fig.savefig(
        f"article/figures/quality_vs_privacy/{name}_vs_privacy_all.pdf",
        bbox_inches="tight",
    )


def quality_vs_privacy_plot(*, df_quality: pd.DataFrame, df_privacy: pd.DataFrame):

    # remove leak experiments
    df_quality = df_quality[~df_quality["generator"].str.startswith("leak")]

    # join quality and privacy dataframes on generator and dataset by generator_name and dataset
    df_privacy = df_privacy.rename(columns={"generator_name": "generator"})
    df_privacy_quality = pd.merge(
        df_quality, df_privacy, on=["generator", "dataset"], how="inner"
    )

    # Dedicated comparison plot for Adult: Detection vs ReMIA / smMIA.
    adult_quality = df_quality[
        (df_quality["dataset"] == "Adult")
        & (df_quality["generator"].isin(list(GENERATIVE_MODELS.values())))
    ][["generator", "xgboost_discr_auc"]].copy()
    adult_quality = adult_quality.rename(columns={"xgboost_discr_auc": "Detection"})

    adult_privacy = df_privacy[
        (df_privacy["dataset"] == "Adult")
        & (df_privacy["metric"].isin(["ReMIA", "smMIA(out)"]))
    ][["generator", "metric", "score"]].copy()

    adult_privacy = (
        adult_privacy.pivot_table(
            index="generator", columns="metric", values="score", aggfunc="mean"
        )
        .reset_index()
        .copy()
    )

    adult_plot_df = pd.merge(adult_quality, adult_privacy, on="generator", how="inner")

    adult_perturb_quality = df_quality[
        (df_quality["dataset"] == "Adult")
        & (df_quality["generator"].str.startswith("perturbation_"))
    ][["generator", "xgboost_discr_auc"]].copy()
    adult_perturb_quality = adult_perturb_quality.rename(
        columns={"xgboost_discr_auc": "Detection"}
    )

    adult_perturb_privacy = df_privacy[
        (df_privacy["dataset"] == "Adult")
        & (df_privacy["generator"].str.startswith("perturbation_"))
        & (df_privacy["metric"].isin(["ReMIA", "smMIA(out)"]))
    ][["generator", "metric", "score"]].copy()
    adult_perturb_privacy = (
        adult_perturb_privacy.pivot_table(
            index="generator", columns="metric", values="score", aggfunc="mean"
        )
        .reset_index()
        .copy()
    )
    adult_perturb_plot_df = pd.merge(
        adult_perturb_quality, adult_perturb_privacy, on="generator", how="inner"
    )
    adult_perturb_plot_df["perturbation_level"] = pd.to_numeric(
        adult_perturb_plot_df["generator"].str.extract(r"_(0\.\d+|1\.0)$")[0],
        errors="coerce",
    )
    adult_perturb_plot_df = adult_perturb_plot_df.sort_values("perturbation_level")

    required_columns = {"Detection", "ReMIA", "smMIA(out)"}
    if required_columns.issubset(adult_plot_df.columns) and not adult_plot_df.empty:
        generator_labels = sorted(adult_plot_df["generator"].unique())
        palette = sns.color_palette("tab10", n_colors=len(generator_labels))
        marker_cycle = ["o", "s", "*", "D", "P", "X", "d", "v", "^", "p", "s"]

        color_map = {g: palette[i] for i, g in enumerate(generator_labels)}
        marker_map = {
            g: marker_cycle[i % len(marker_cycle)]
            for i, g in enumerate(generator_labels)
        }

        fig, (ax_remia, ax_sm) = plt.subplots(1, 2, figsize=(5, 2.5))

        if not adult_perturb_plot_df.empty:
            sns.lineplot(
                data=adult_perturb_plot_df,
                x="Detection",
                y="ReMIA",
                marker="o",
                color="gray",
                alpha=0.75,
                markersize=3,
                label="_nolegend_",
                ax=ax_remia,
            )
            sns.lineplot(
                data=adult_perturb_plot_df,
                x="Detection",
                y="smMIA(out)",
                marker="o",
                color="gray",
                alpha=0.75,
                markersize=3,
                label="_nolegend_",
                ax=ax_sm,
            )

        for _, row in adult_plot_df.iterrows():
            g = row["generator"]
            color = color_map[g]
            marker = marker_map[g]

            ax_remia.scatter(
                row["Detection"],
                row["ReMIA"],
                color=color,
                marker=marker,
                s=45,
                alpha=0.75,
                label="_nolegend_",
            )
            ax_sm.scatter(
                row["Detection"],
                row["smMIA(out)"],
                color=color,
                marker=marker,
                s=45,
                alpha=0.8,
                label="_nolegend_",
            )

        # ax_remia.set_xlabel("Detection (lower is better)")
        # ax_sm.set_xlabel("Detection (lower is better)")

        # only one x label:
        # plt.xlabel("Detection (lower is better)")
        ax_remia.set_ylabel("ReMIA score")
        ax_sm.set_ylabel("smMIA(out) score")

        ax_remia.grid(True, alpha=0.3)
        ax_sm.grid(True, alpha=0.3)

        # Ensure no axis-level legend is shown; keep a single shared legend outside.
        for ax in (ax_remia, ax_sm):
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        legend_handles = []
        legend_labels = []

        if not adult_perturb_plot_df.empty:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    markersize=5,
                    alpha=0.75,
                )
            )
            legend_labels.append("Noise Anon.")

        for g in generator_labels:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color_map[g],
                    marker=marker_map[g],
                    linestyle="",
                    markersize=6,
                )
            )
            legend_labels.append(g)

        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.98, 0.51),
            frameon=True,
            # title="Generator",
        )
        fig.tight_layout(pad=0.75)

        fig.savefig(
            "article/figures/quality_vs_privacy/adult_detection_vs_remia_sm_mia.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def metrics_efficiency_comparison_table(df_privacy: pd.DataFrame):
    """
    table: METRICS EFFICIENCY COMPARISON
        rows: metrics (except smMIA)
        columns: overhead time, number of models trained, excess data required
        content: value
    """

    df_privacy = get_privacy_results(
        average_over_seeds=True, only_first_seed=False, preselect_metrics=False
    )

    metrics = {
        "dcr_comparison": "DCR",
        "domias": "DOMIAS",
        "remia_1.0": "ReMIA(f=1.0)",
        "remia_0.5": "ReMIA(f=0.5)",
        "shadow_modeling_achilles_heels": "smMIA(out)",
        "shadow_modeling_achilles_median": "smMIA(med)",
    }

    # subsetting metrics and renaming them in one step
    df_privacy = df_privacy[df_privacy["metric"].isin(metrics.keys())].copy()
    df_privacy["metric"] = df_privacy["metric"].replace(metrics)

    # treat smMIA(out) and smMIA(med) as the same metric for efficiency comparison, since they are the same method with different thresholds
    df_privacy["metric"] = df_privacy["metric"].replace(
        {"smMIA(out)": "smMIA", "smMIA(med)": "smMIA"}
    )

    def show_range(values):
        min_value = values.min()
        max_value = values.max()

        if pd.isna(min_value) or pd.isna(max_value):
            return "N/A"
        elif min_value == max_value:
            return f"{min_value:.1f}"
        elif min_value.round(1) == 0:
            return f"<{max_value:.1f}"
        else:
            return f"{min_value:.1f} - {max_value:.1f}"

    rows = []
    for metric in df_privacy["metric"].unique():
        df_metric = df_privacy[df_privacy["metric"] == metric]

        additional_data_usage = (
            df_metric["total_data_usage"] / df_metric["training_size"] - 1
        )
        rows.append(
            {
                "Metric": metric,
                "SDG Calls": str(int(df_metric["generation_calls"].mean())),
                "Additional Data Usage": show_range(additional_data_usage),
                "Overhead Time (s)": show_range(df_metric["overhead_time"]),
            }
        )
    df_efficiency = pd.DataFrame(rows).set_index("Metric")

    df_to_latex(
        df=df_efficiency,
        filename="article/tables/metrics_efficiency_comparison.tex",
        digits=1,
    )


def get_quality_results_updated(average_over_seeds: int) -> pd.DataFrame:
    df_quality = get_quality_results()

    # rename generators with GENERATIVE_MODELS
    df_quality["generator"] = df_quality["generator"].replace(GENERATIVE_MODELS)

    # only keep the most frequent size for each dataset-generation combination
    size_mode = df_quality.groupby(["generator", "dataset"])["size"].transform(
        lambda s: s.mode().iloc[0]
    )
    df_quality = df_quality[df_quality["size"] == size_mode].drop(columns=["size"])

    # most_frequent_size = df_quality["size"].mode().iloc[0]
    # df_quality = df_quality[df_quality["size"] == most_frequent_size].drop(
    #     columns=["size"]
    # )

    # rename datasets
    df_quality["dataset"] = df_quality["dataset"].replace(DATASETS)

    # if average_over_seeds is True, average over seeds
    if average_over_seeds:
        df_quality_mean = (
            df_quality.groupby(["generator", "dataset"])
            .agg(
                {
                    "xgboost_discr_auc": "mean",
                    "xgboost_utility_score": "mean",
                    "seed": "count",
                }
            )
            .reset_index()
        ).rename(columns={"seed": "num_seeds"})
        df_quality_std = (
            df_quality.groupby(["generator", "dataset"])
            .agg(
                {
                    "xgboost_discr_auc": "std",
                    "xgboost_utility_score": "std",
                }
            )
            .reset_index()
        )
        df_quality = pd.merge(
            df_quality_mean,
            df_quality_std,
            on=["generator", "dataset"],
            suffixes=("", "_std"),
        )
    else:  # only keep first seed
        df_quality = df_quality.loc[
            df_quality.groupby(["generator", "dataset"])["seed"].idxmin()
        ]

    return df_quality


def sdgs_runtime():

    df = get_privacy_results(
        average_over_seeds=False, only_first_seed=False, preselect_metrics=False
    )

    # filter only GENERATIVE_MODELS
    df = df[df["generator_name"].isin(list(GENERATIVE_MODELS.values()))]

    # generation_time / generation_calls
    df["sdg_time"] = df["generation_time"] / df["generation_calls"]

    # take the maximum for every generator (index: generator_name) and column sdg_time
    df = df.groupby("generator_name").agg({"sdg_time": "max"}).reset_index()

    df_to_latex(
        df=df.set_index("generator_name"),
        filename="article/tables/sdg_runtime.tex",
        digits=1,
    )


if __name__ == "__main__":
    os.makedirs("article/tables", exist_ok=True)
    os.makedirs("article/figures", exist_ok=True)
    
    df_privacy = get_privacy_results(average_over_seeds=True)
    df_quality = get_quality_results_updated(average_over_seeds=True)

    print("privacy_scores_tables...")
    privacy_scores_tables(df_privacy.copy())
    print("quality_utility_table...")
    quality_utility_table(df_quality.copy())
    print("metrics_power_comparison_table...")
    metrics_power_comparison_table()
    print("leak_fraction_alpha_vs_privacy_plot...")
    leak_fraction_alpha_vs_privacy_plot(df_privacy.copy())
    print("quality_vs_privacy_plot...")
    quality_vs_privacy_plot(df_quality=df_quality.copy(), df_privacy=df_privacy.copy())
    print("metrics_efficiency_comparison_table...")
    metrics_efficiency_comparison_table(df_privacy.copy())

    print("appendix_quality_vs_privacy_plot...")
    appendix_quality_vs_privacy_plot(
        df_quality=df_quality.copy(), df_privacy=df_privacy.copy(), detection=True
    )
    appendix_quality_vs_privacy_plot(
        df_quality=df_quality.copy(), df_privacy=df_privacy.copy(), detection=False
    )

    print("sdgs_runtime...")
    sdgs_runtime()
