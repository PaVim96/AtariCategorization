import pandas as pd
import altair as alt 
from altair import datum
import numpy as np
from vega_datasets import data


game = pd.DataFrame(["A", "B", "C", "A", "B", "C", "A", "B", "C"], columns=["Game"])
algo = pd.DataFrame(["AlgA", "AlgA", "AlgA", "AlgB", "AlgB", "AlgB", "AlgC", "AlgC", "AlgC"], columns=["Algorithm"])
performance = pd.DataFrame([0, 0.5, 1, 0.5, 1.0, 0, 1.0, 0, 0.5], columns=["Performance"])
categories = pd.DataFrame(["Any, CatA", "Any, CatA, CatB", "Any, CatC","Any, CatA", "Any, CatA, CatB", "Any, CatC", "Any, CatA", "Any, CatA, CatB", "Any, CatC"], columns=["Categories"])
heatmapDF = pd.concat([game,algo,performance, categories], axis=1)

#make selection out of binding
input_dropdown = alt.binding_select(options=["Any", "CatA", "CatB", "CatC"], name="Game Category: ")
selection = alt.selection_single(fields=['Category'], bind=input_dropdown)

base = alt.Chart(heatmapDF).encode(
    alt.X('Game:O', scale=alt.Scale(paddingInner=0)),
    alt.Y('Algorithm:O', scale=alt.Scale(paddingInner=0)),
).add_selection(
    selection
).transform_filter(
    selection
)

#make scale for coloring 
myScale = alt.Scale(domain=[0, 0.25, 0.5, 0.75, 0.985, 1.0], range=['darkred', 'orange', 'white', 'darkgreen', 'green', 'green'], type='linear')
myColor = alt.Color('Performance:Q', scale=myScale)

heatmap = base.mark_rect().encode(
    color= myColor
)
heatmap