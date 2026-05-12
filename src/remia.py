import os

import pandas as pd
import torch
from src.generators import Generator
from src.tab_mlp import TabularMLP
from src.util import Timer
import numpy as np
from sklearn.metrics import roc_auc_score


def pop_data(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    extracted_data = df.sample(n=n, random_state=seed)
    df.drop(extracted_data.index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return extracted_data.reset_index(drop=True)


class ValidationHook:
    def __init__(self, discriminator: TabularMLP, df: pd.DataFrame):
        self.discriminator = discriminator
        self.df = df
        self._perf_records = []
        self._best_ll = 0
        self.X = None
        self.y_true = np.array(self.df["label"].values)

    def __call__(self, n: int):
        if self.X is None:
            self.X = (
                self.discriminator.preprocessor_x.transform(
                    self.df.drop(columns=["label"])
                )
                .float()
                .to(self.discriminator.device)
            )

        y_probs = self.discriminator.predict(self.X).cpu().numpy()
        ll = np.where(
            self.y_true == 1, np.log(y_probs + 1e-15), np.log(1 - y_probs + 1e-15)
        ).mean()
        perf = {
            "step": n,
            "ll": ll,
            "accuracy": (y_probs.round() == self.y_true).mean(),
            "auc": roc_auc_score(self.y_true, y_probs),
        }
        self._perf_records.append(perf)

    def perf_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._perf_records).set_index("step")


def remia(
    *,
    data: pd.DataFrame,
    generator: Generator,
    device: str,
    training_size: int,
    target_proportion: float,
    max_epochs: int,
    patience: int,
    lr: float = 5e-3,
    mlp_units: list[int] = [100, 50],
    batch_size: int = 500,
    stop_threshold: float = 0.99,
    val_each: int,
    seed: int,
    generator_name: str | None = None,
    dataset_name: str | None = None,
    use_val_target: bool = False,
) -> dict:
    output = {}

    with Timer() as total_time:
        data_shuffled = (
            data.copy().sample(frac=1, random_state=seed).reset_index(drop=True)
        )
        n_target = round(target_proportion * training_size)

        df_target_0 = pop_data(data_shuffled, n=n_target, seed=seed + 1)
        df_target_1 = pop_data(data_shuffled, n=n_target, seed=seed + 2)
        df_common_train = pop_data(
            data_shuffled, n=training_size - n_target, seed=seed + 3
        )

        df_train_0 = (
            pd.concat([df_target_0, df_common_train], ignore_index=True)
            .sample(frac=1, random_state=seed + 4)
            .reset_index(drop=True)
        )
        df_train_1 = (
            pd.concat([df_target_1, df_common_train], ignore_index=True)
            .sample(frac=1, random_state=seed + 5)
            .reset_index(drop=True)
        )

        with Timer() as generation_timer:
            df_synth_0 = generator.fit_generate(
                train_dataset=df_train_0, n=len(df_train_0)
            )
            df_synth_1 = generator.fit_generate(
                train_dataset=df_train_1, n=len(df_train_1)
            )
        output["generation_time"] = generation_timer.elapsed

        df_syn = (
            pd.concat(
                [df_synth_0.assign(label=0), df_synth_1.assign(label=1)],
                ignore_index=True,
            )
            .sample(frac=1, random_state=seed + 6)
            .reset_index(drop=True)
        )

        df_targets = (
            pd.concat(
                [df_target_0.assign(label=0), df_target_1.assign(label=1)],
                ignore_index=True,
            )
            .sample(frac=1, random_state=seed + 7)
            .reset_index(drop=True)
        )

        discriminator = TabularMLP(
            units=mlp_units, dropout=0.0, weight_decay=0.0, activation=torch.nn.SiLU()
        )

        hooks = {
            "targets": ValidationHook(discriminator=discriminator, df=df_targets),
            "syn": ValidationHook(discriminator=discriminator, df=df_syn),
        }

        if use_val_target:
            t = len(df_targets)
            df_val_targets = df_targets[: t // 2].copy()
            df_test_targets = df_targets[t // 2 :].copy()
            hooks["val_targets"] = ValidationHook(
                discriminator=discriminator, df=df_val_targets
            )
            hooks["test_targets"] = ValidationHook(
                discriminator=discriminator, df=df_test_targets
            )

        discriminator.train(
            df=df_syn,
            target_col="label",
            epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            hooks=list(hooks.values()),
            hook_period=val_each,
            lr=lr,
        )

        window_size = len(hooks["targets"].perf_df()) // 10

        perf_df = {
            subset: {
                "df": hook.perf_df(),
                "df_smoothed": hook.perf_df()
                .rolling(window=window_size, center=True, min_periods=2)
                .mean(),
            }
            for subset, hook in hooks.items()
        }

        metric = "auc"

        best_syn_capped = (
            perf_df["syn"]["df"][metric].index[
                perf_df["syn"]["df"][metric] > stop_threshold
            ][0]
            if (perf_df["syn"]["df"][metric] > stop_threshold).any()
            else perf_df["syn"]["df"][metric].idxmax()
        )

        # CRITERIA:
        stopping_criterion = {
            "best_syn": perf_df["syn"]["df"][metric].idxmax(),
            "best_syn_capped": best_syn_capped,
            "best_target_smoothed": perf_df["targets"]["df_smoothed"][metric].idxmax(),
        }
        if use_val_target:
            stopping_criterion["best_val_target_smoothed"] = perf_df["val_targets"][
                "df_smoothed"
            ][metric].idxmax()

        # VALUES TO EXTRACT:
        output_value = {
            "target": (perf_df["targets"]["df"], len(df_targets)),
            "target_smoothed": (perf_df["targets"]["df_smoothed"], len(df_targets)),
        }
        if use_val_target:
            output_value["test_target"] = (
                perf_df["test_targets"]["df"],
                len(df_test_targets),
            )
            output_value["test_target_smoothed"] = (
                perf_df["test_targets"]["df_smoothed"],
                len(df_test_targets),
            )

        pairs = [
            ("best_syn", "target_smoothed"),
            ("best_syn_capped", "target_smoothed"),
            ("best_syn", "target"),
            ("best_syn_capped", "target"),
            ("best_target_smoothed", "target_smoothed"),
        ]

        if use_val_target:
            pairs.append(("best_val_target_smoothed", "test_target_smoothed"))
            pairs.append(("best_val_target_smoothed", "test_target"))

        reference_step = int(stopping_criterion["best_target_smoothed"])
        reference_auc = float(
            perf_df["targets"]["df_smoothed"].loc[reference_step, "auc"]
        )

        for c, v in pairs:
            idx = stopping_criterion[c]
            df, n = output_value[v]
            output[f"{v}_on_{c}"] = df.loc[idx, ["auc", "accuracy"]].to_dict()
            output[f"{v}_on_{c}"]["n"] = n
            output[f"{v}_on_{c}"]["step"] = int(idx)
            output[f"{v}_on_{c}"]["step_diff"] = int(idx - reference_step)
            output[f"{v}_on_{c}"]["auc_diff"] = float(
                df.loc[idx, "auc"] - reference_auc
            )

        DEBUG = True
        if DEBUG:
            import matplotlib.pyplot as plt
            from scipy.stats import binom

            def significance_threshold(n: int, p_value: float = 1e-3):
                print(
                    f"Significance threshold for n={n}: {binom.ppf(1 - p_value, n=n, p=0.5) / n}"
                )
                return binom.ppf(1 - p_value, n=n, p=0.5) / n

            def metric_plot(
                ax,
                metric: str,
                subset: str,
                color: str,
                smooth: bool = True,
                significance_threshold: bool = True,
                label: str | None = None,
                ylabel: str | None = None,
            ):
                steps = perf_df[subset]["df"].index
                ax.plot(
                    steps,
                    perf_df[subset]["df"][metric],
                    alpha=0.3,
                    label=subset if label is None else label,
                    color=color,
                )
                if smooth:
                    ax.plot(
                        steps,
                        perf_df[subset]["df_smoothed"][metric],
                        alpha=0.8,
                        color=color,
                    )
                    if subset in hooks and significance_threshold:
                        ax.axhline(
                            significance_threshold(len(hooks[subset].df)),
                            color=color,
                            linestyle="--",
                            alpha=0.8,
                        )
                df = perf_df[subset]["df_smoothed"] if smooth else perf_df[subset]["df"]
                idx_max = df[metric].idxmax()
                ax.scatter(idx_max, df.loc[idx_max, metric], color=color, alpha=0.8)
                ax.set_xlabel("Step")
                ax.set_ylabel(metric if ylabel is None else ylabel)

            # increase resolution of the plot
            if not use_val_target:
                fig, ax1 = plt.subplots(dpi=200, figsize=(6, 3))
            else:
                fig, (ax1, ax2) = plt.subplots(
                    dpi=200, nrows=2, sharex=True, figsize=(5, 7)
                )

            # ALL TARGETS PLOT
            metric_plot(
                ax1,
                metric=metric,
                subset="targets",
                color="blue",
                significance_threshold=False,
                label="target (test)",
                ylabel="AUROC (test)",
            )
            ax1_r = ax1.twinx()
            metric_plot(
                ax1_r,
                metric="auc",
                subset="syn",
                color="black",
                smooth=False,
                label="synthetic (train)",
                ylabel="AUROC (train)",
            )
            ax1_r.axvline(
                stopping_criterion["best_syn_capped"],
                color="black",
                linestyle="--",
                alpha=0.8,
                label=f"{stop_threshold}% threshold step",
            )

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_r.get_legend_handles_labels()

            legend = ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="lower right",
                framealpha=0.9,
            )
            legend.set_zorder(10)
            ax1.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)

            # VAL AND TEST TARGETS PLOT
            if use_val_target:
                for subset in ["val", "test"]:
                    color = "blue" if subset == "val" else "red"
                    metric_plot(
                        ax2, metric=metric, subset=f"{subset}_targets", color=color
                    )

                ax2_r = ax2.twinx()
                metric_plot(
                    ax2_r, metric=metric, subset="syn", color="black", smooth=False
                )

                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_r.get_legend_handles_labels()

                legend = ax2.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc="lower right",
                    framealpha=0.9,
                )
                legend.set_zorder(10)

            folder = "remia_plots"
            os.makedirs(folder, exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                f"{folder}/{generator_name}_{dataset_name}_{training_size}_{seed}_{metric}_{target_proportion}.pdf"
            )
            plt.clf()

    output["total_time"] = total_time.elapsed
    output["overhead_time"] = total_time.elapsed - generation_timer.elapsed
    return output
