from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem
from sklift.metrics import qini_curve, perfect_qini_curve, qini_auc_score, uplift_curve, perfect_uplift_curve, \
    uplift_auc_score


class UpliftCurveDisplay:
    """Qini and Uplift curve visualization.

    Args:
        x_actual, y_actual (array (shape = [>2]), array (shape = [>2])): Points on a curve
        x_baseline, y_baseline (array (shape = [>2]), array (shape = [>2])): Points on a random curve
        x_perfect, y_perfect (array (shape = [>2]), array (shape = [>2])): Points on a perfect curve
        random (bool): Plotting a random curve
        perfect (bool): Plotting a perfect curve
        estimator_name (str): Name of estimator. If None, the estimator name is not shown.
    """

    def __init__(self, x_actual, y_actual, x_baseline=None,
                 y_baseline=None, x_perfect=None, y_perfect=None,
                 random=None, perfect=None, estimator_name=None):
        self.x_actual = x_actual
        self.y_actual = y_actual
        self.x_baseline = x_baseline
        self.y_baseline = y_baseline
        self.x_perfect = x_perfect
        self.y_perfect = y_perfect
        self.random = random
        self.perfect = perfect
        self.estimator_name = estimator_name

    def plot(self, auc_score, ax=None, name=None, title=None, **kwargs):
        """Plot visualization

        Args:
            auc_score (float): Area under curve.§
            ax (matplotlib axes): Axes object to plot on. If `None`, a new figure and axes is created. Default is None.
            name (str): Name of ROC Curve for labeling. If `None`, use the name of the estimator. Default is None.
            title (str): Title plot. Default is None.

        Returns:
            Object that stores computed values
        """

        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if auc_score is not None and name is not None:
            line_kwargs["label"] = f"{name} ({title} = {auc_score:0.2f})"
        elif auc_score is not None:
            line_kwargs["label"] = f"{title} = {auc_score:0.2f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        if ax is None:
            fig, ax = plt.subplots()

        self.line_, = ax.plot(self.x_actual, self.y_actual, **line_kwargs)

        if self.random:
            ax.plot(self.x_baseline, self.y_baseline, label="Random")
            ax.fill_between(self.x_actual, self.y_actual, self.y_baseline, alpha=0.2)

        if self.perfect:
            ax.plot(self.x_perfect, self.y_perfect, label="Perfect")

        ax.set_xlabel('Number targeted')
        ax.set_ylabel('Number of incremental outcome')

        if self.random == self.perfect:
            variance = False
        else:
            variance = True

        if len(ax.lines) > 4:
            ax.lines.pop(len(ax.lines) - 1)
            if variance == False:
                ax.lines.pop(len(ax.lines) - 1)

        if "label" in line_kwargs:
            ax.legend(loc=u'upper left', bbox_to_anchor=(1, 1))

        self.ax_ = ax
        self.figure_ = ax.figure

        return self


def plot_qini_curve(y_true, uplift, treatment,
                    random=True, perfect=True, negative_effect=True, ax=None, name=None, **kwargs):
    """Plot Qini curves from predictions.

    Args:
        y_true (1d array-like): Ground truth (correct) binary labels.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        random (bool): Draw a random curve. Default is True.
        perfect (bool): Draw a perfect curve. Default is True.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not
            contain the negative effects. Default is True.
        ax (object): The graph on which the function will be built. Default is None.
        name (string): The name of the function. Default is None.

    Returns:
        Object that stores computed values.

    Example::

        from sklift.viz import plot_qini_curve


        qini_disp = plot_qini_curve(
            y_test, uplift_predicted, trmnt_test,
            perfect=True, name='Model name'
        );

        qini_disp.figure_.suptitle("Qini curve");
    """

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    x_actual, y_actual = qini_curve(y_true, uplift, treatment)

    if random:
        x_baseline, y_baseline = x_actual, x_actual * y_actual[-1] / len(y_true)
    else:
        x_baseline, y_baseline = None, None

    if perfect:
        x_perfect, y_perfect = perfect_qini_curve(
            y_true, treatment, negative_effect)
    else:
        x_perfect, y_perfect = None, None

    viz = UpliftCurveDisplay(
        x_actual=x_actual,
        y_actual=y_actual,
        x_baseline=x_baseline,
        y_baseline=y_baseline,
        x_perfect=x_perfect,
        y_perfect=y_perfect,
        random=random,
        perfect=perfect,
        estimator_name=name,
    )

    auc = qini_auc_score(y_true, uplift, treatment, negative_effect)

    return viz.plot(auc, ax=ax, title="AUC", **kwargs)


def plot_uplift_curve(y_true, uplift, treatment,
                      random=True, perfect=True, ax=None, name=None, **kwargs):
    """Plot Uplift curves from predictions.

    Args:
        y_true (1d array-like): Ground truth (correct) binary labels.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        random (bool): Draw a random curve. Default is True.
        perfect (bool): Draw a perfect curve. Default is True.
        ax (object): The graph on which the function will be built. Default is None.
        name (string): The name of the function. Default is None.

    Returns:
        Object that stores computed values.

    Example::

        from sklift.viz import plot_uplift_curve


        uplift_disp = plot_uplift_curve(
            y_test, uplift_predicted, trmnt_test,
            perfect=True, name='Model name'
        );

        uplift_disp.figure_.suptitle("Uplift curve");
    """
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)

    if random:
        x_baseline, y_baseline = x_actual, x_actual * y_actual[-1] / len(y_true)
    else:
        x_baseline, y_baseline = None, None

    if perfect:
        x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    else:
        x_perfect, y_perfect = None, None

    viz = UpliftCurveDisplay(
        x_actual=x_actual,
        y_actual=y_actual,
        x_baseline=x_baseline,
        y_baseline=y_baseline,
        x_perfect=x_perfect,
        y_perfect=y_perfect,
        random=random,
        perfect=perfect,
        estimator_name=name,
    )

    auc = uplift_auc_score(y_true, uplift, treatment)

    return viz.plot(auc, ax=ax, title="AUC", **kwargs)


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
    bin_df = grouped.pivot(index="bin", columns="t", values=["mean", "sem"]).sort_index().astype(np.float32)

    bin_df[("mean", "actual_uplift")] = bin_df["mean"]["treatment"] - bin_df["mean"]["control"]
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
