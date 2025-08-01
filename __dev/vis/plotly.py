import pandas as pd
import plotly.graph_objects as go


def df_to_plotly_scatterplot3D(
    df: pd.DataFrame,
    show: bool = True,
    **kwargs,
):
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


fig = df_to_plotly_scatterplot3D(X_pd, s=1, cmap="Inferno", alpha=0.5)
