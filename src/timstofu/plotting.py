from numpy.typing import NDArray

import itertools
import math
import pandas as pd


def plot_discrete_marginals(
    marginals: dict[str, NDArray],
    show: bool = True,
    n: int | None = None,
    m: int | None = None,
    imshow_kwargs: dict = {},
    aspect: str = "auto",
):
    """
    Plot a collection of 2D discrete marginal distributions in a grid of subplots.

    Each entry in the `marginals` dictionary is expected to be a 2D array representing
    the joint distribution of a pair of discrete variables. The keys of the dictionary
    are tuples of strings (colA, colB) indicating the variable names.

    Parameters
    ----------
    marginals : dict[str, NDArray]
        A dictionary where each key is a tuple of variable names (colA, colB) and the value
        is a 2D NumPy array representing the joint probability distribution of colA and colB.
    show : bool, default=True
        Whether to display the plot immediately using `plt.show()`.
    n : int or None, optional
        Number of columns in the subplot grid. If not provided, it will be inferred automatically.
    m : int or None, optional
        Number of rows in the subplot grid. If not provided, it will be inferred automatically.

    Raises
    ------
    AssertionError
        If the number of inferred subplots is not sufficient to plot all marginals.

    Notes
    -----
    Unused subplots (if any) will be hidden.
    """
    import matplotlib.pyplot as plt

    N = len(marginals)
    if n is None and m is None:
        n = math.ceil(math.sqrt(N))
        m = n if n**2 >= N else n + 1
    elif n is None:
        n = math.ceil(N / m)
    elif m is None:
        m = math.ceil(N / n)
    assert n * m >= N, "Error in getting the number of subplots."
    fig, axes = plt.subplots(m, n)
    for plotting_inputs, idx in itertools.zip_longest(
        marginals.items(),
        itertools.product(range(m), range(n), repeat=1),
        fillvalue=None,
    ):
        ax = axes[idx if m > 1 and n > 1 else max(idx)]
        if plotting_inputs is not None:
            ((colA, colB), data) = plotting_inputs
            ax.imshow(marginals[(colA, colB)][0], **imshow_kwargs)
            ax.set_ylabel(colA)
            ax.set_xlabel(colB)
            if aspect:
                ax.set_aspect(aspect=aspect)
        else:
            ax.axis("off")
    if show:
        plt.show()
    return fig, axes


def df_to_plotly_scatterplot3D(
    df: pd.DataFrame,
    show: bool = True,
    **kwargs,
):
    import plotly.graph_objects as go

    xl, yl, zl, weights = df.columns

    fig = go.Figure(
        data=go.Scatter3d(
            x=df[xl],
            y=df[yl],
            z=df[zl],
            mode="markers",
            marker=dict(
                size=kwargs.get("s", 5),
                color=df[weights],
                colorscale=kwargs.get("cmap", "Viridis"),
                colorbar=dict(title="Intensity"),
                opacity=kwargs.get("alpha", 0.8),
            ),
        )
    )

    fig.update_layout(
        autosize=True,
        scene=dict(
            xaxis_title=xl,
            yaxis_title=yl,
            zaxis_title=zl,
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    if show:
        fig.show()

    return fig
