import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem


def plot_uplift_bins(
    t: pd.Series,
    y: pd.Series,
    tau: pd.Series,
    control_name: int | str = 0,
    treatment_name: int | str = 1,
    ax: plt.Axes = None,
    n_bins: int = 10,
    name: str = "Uplift Bins",
    figsize: tuple[int] = (10, 6),
) -> plt.Axes:
    """
    Plot the uplift bins.

    Parameters
    ----------
    t: Series
        The treatment Series.
    y: Series
        The target Series.
    tau: Series
        The uplift Series.
    control_name: str or int
        The name of the control recipe.
    treatment_name: str or int
        The name of the treatment recipe.
    ax: plt.Axes
        The axes of the plot.
    n_bins: int
        The number of bins.
    name: str
        The name of the plot.
    figsize: tuple[int]
        The size of the figure.

    Returns
    -------
    plt.Axes
        The axes of the plot.
    """
    bin_df = pd.DataFrame({"t": t, "y": y, "tau": tau})
    bin_df = bin_df.sort_values("tau", ascending=False).reset_index(drop=True)
    bin_df["t"] = bin_df["t"].map(
        {control_name: "control", treatment_name: "treatment"}
    )
    bin_df["bin"] = pd.qcut(bin_df.index, q=n_bins, labels=range(1, n_bins + 1))

    grouped = (
        bin_df.groupby(["bin", "t"], observed=True)["y"]
        .agg(mean="mean", sem=lambda x: sem(x, nan_policy="omit"), count="count")
        .reset_index()
    )
    bin_df = (
        grouped.pivot(index="bin", columns="t", values=["mean", "sem"])
        .sort_index()
        .astype(np.float32)
    )

    bin_df[("mean", "actual_uplift")] = (
        bin_df["mean"]["treatment"] - bin_df["mean"]["control"]
    )
    bin_df[("sem", "actual_uplift")] = np.sqrt(
        bin_df["sem"]["treatment"] ** 2 + bin_df["sem"]["control"] ** 2
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        pass

    colors = ["red" if x < 0 else "green" for x in bin_df["mean"]["actual_uplift"]]
    ax.bar(
        bin_df.index.astype(str),
        bin_df["mean"]["actual_uplift"],
        yerr=bin_df["sem"]["actual_uplift"],
        capsize=4,
        color=colors,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=1)
    ax.set_xlabel("Bin")
    ax.set_ylabel("Actual Uplift Mean")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax
