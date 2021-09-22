import plotly.express as px
from plotly.graph_objects import Figure as PlotlyFigure
from pandas import DataFrame, Series
from typing import Optional
from numpy import ndarray
import numpy as np
from typing import List, Union
from scipy.stats import norm


def boxplot(data: DataFrame, y: str, x: Optional[str] = None, **kwargs) -> PlotlyFigure:

    try:
        order = {x: data[x].sort_values().unique()}
    except TypeError:
        order = None

    fig = px.box(data_frame=data, x=x, y=y, category_orders=order, **kwargs)

    return fig


def qqplot(residuals: Union[List, ndarray]) -> PlotlyFigure:
    probabilities: ndarray = (
        np.array([1.0, *np.arange(start=5, stop=96, step=1), 99.0]) / 100
    )
    observed_quantiles: ndarray = np.quantile(np.sort(residuals), probabilities)
    theoretical_quantiles: ndarray = norm.ppf(probabilities, loc=10, scale=50)

    fig: PlotlyFigure = px.scatter(
        x=observed_quantiles, y=theoretical_quantiles, title="qq-plot (Standard Normal)"
    )

    return fig


def save_plots_to_html(figures: List[PlotlyFigure], file: str) -> None:

    with open(file, "w") as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    return None


def save_plots_to_pdf(figures: List[PlotlyFigure], file: str) -> None:

    for fig in figures:
        fig.write_image(file)

    return None
