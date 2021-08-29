import plotly.express as px
from plotly.graph_objects import Figure as PlotlyFigure
from pandas import DataFrame
from typing import Optional
import numpy as np
from typing import List

def boxplot(
    data: DataFrame,
    y: str,
    x: Optional[str] = None,
    **kwargs) -> PlotlyFigure:

    order = { x: np.sort(data[x].unique()) }
    fig = \
        px.box(
            data_frame=data, 
            x=x, 
            y=y, 
            category_orders=order, 
            **kwargs)

    return fig

def save_plots_to_html(
    figures: List[PlotlyFigure], 
    file: str) -> None:

    with open(file, 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    return None

def save_plots_to_pdf(
    figures: List[PlotlyFigure], 
    file: str) -> None:

    for fig in figures:
        fig.write_image(file)
    
    return None