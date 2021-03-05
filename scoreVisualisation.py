import pandas as pd
import altair as alt 
from altair import datum
import numpy as np
from vega_datasets import data
import math 


def calculateGradient(performanceOfGame):
    convertedToNumbers = performanceOfGame.str.replace(',', '.', regex=False).astype(float).to_numpy()
    convertedToNumbers[:] = (convertedToNumbers[:] - convertedToNumbers.min() ) / (convertedToNumbers.max() - convertedToNumbers.min()).round(1)

    return convertedToNumbers

def convertToColorGradient(performanceData):
    colorGradientFrame = calculateGradient(performanceData.loc[0, :])
    for i in range(1, performanceData.shape[0]):
        colorGradientFrame = np.vstack((colorGradientFrame, calculateGradient(performanceData.loc[i, :])))

    colorGradientSortedFrame = list()
    #conver to a list where the columns are the performances of every algorithm on every game (so first #game columns performance of DQN) 
    for i in range(colorGradientFrame.shape[1]):
        colorGradientSortedFrame.append(colorGradientFrame[:, i])

    #this contains in the i'th row the performance of the i'th algorithm of every game
    colorGradientSortedFrame = np.array(colorGradientSortedFrame)
    result = pd.Series(pd.DataFrame(colorGradientSortedFrame).values.ravel())
    return result

def makeEasyUniqueCats(categoriesGames):
    categories = list() 
    categoriesGames = categoriesGames.to_numpy()
    for cat in categoriesGames:
        if len(categories) == 0:
            categories.append(cat)
        else:
            if not (cat in categories):
                categories.append(cat)
    categories.append("Any")
    return categories



def makeUniqueCategories(categoriesGames):
    categories = list() 
    categoriesGames = categoriesGames.to_numpy() 
    for categoriesString in categoriesGames:
        if type(categoriesString) == float and np.isnan(categoriesString):
            continue
        categoriesString = categoriesString.replace(" ", "")
        listOfCategories = categoriesString.split(",")
        for category in listOfCategories:
            if len(categories) == 0:
                categories.append(category)
            else:
                if not (category in categories):
                    categories.append(category)
    categories.append('Any')
    return categories

def convertRepeatedAlgorithms(algorithms):
    algorithms = algorithms.to_numpy()
    stackedPerformancesOfAlgos = list() 
    for i in range(algorithms.shape[1]):
        stackedPerformancesOfAlgos.append(algorithms[:, i])
    stackedPerformancesOfAlgos = np.array(stackedPerformancesOfAlgos)
    result = pd.Series(pd.DataFrame(stackedPerformancesOfAlgos).values.ravel())
    return result

def resetInfo(self):
    self.info = []

def catVis(normalizedScores, needsNorm, hasSeveral, minVal = 0.0, maxVal = 1.0):
    gameNames = normalizedScores['Game']
    categoriesGames = normalizedScores['Categories']
    algorithms = normalizedScores.iloc[:, 1:-1]
    algorithmNames = list(algorithms.columns)
    if hasSeveral:
        uniqueCategories = makeUniqueCategories(categoriesGames)
    #this should only be called when categories where made from k-Means
    #since it assumes every cell is not zero
    else:
        uniqueCategories = makeEasyUniqueCats(categoriesGames)
    
    #get number of times we need to repeat the game names (for every algo one time)
    numberOfAlgorithms = len(algorithms.columns)
    numberOfGames = len(gameNames)
    gameColumn = pd.concat([gameNames] * numberOfAlgorithms).reset_index(drop=True)

    #convert the algorithm performances of every game for the different algorithms into color gradients 
    if needsNorm:
        algorithmPerformances = convertToColorGradient(algorithms)
    else:
        algorithmPerformances = convertRepeatedAlgorithms(algorithms)

    #make selection out of binding
    input_dropdown = alt.binding_select(options=uniqueCategories, name="Game Category: ")
    selection = alt.selection_single(fields=['Categories'], bind=input_dropdown, name="selector", init={"Categories": "Any"})

    #create stacked version of categories repeated for every algorithm (so #Games x #Algos)
    categoriesRepeated = pd.concat([categoriesGames] * numberOfAlgorithms).reset_index(drop=True)

    algorithmsRepeated = pd.DataFrame()
    #create stacked algorithm names for every game (so #Games x #Algos)
    for algo in algorithmNames:
        algorithmsRepeated = pd.concat([algorithmsRepeated, pd.DataFrame([[algo] * numberOfGames])])

    algorithmsRepeated = algorithmsRepeated.stack().reset_index(drop=True)


    #make dataframe out of those 4 columns 
    heatmapDF = pd.concat([gameColumn, algorithmsRepeated, algorithmPerformances, categoriesRepeated], keys=["Game", "Algorithm", "Performance", "Categories"], axis=1)
    if hasSeveral:
        base = alt.Chart(heatmapDF).encode(
            alt.X('Game:O', scale=alt.Scale(paddingInner=0)),
            alt.Y('Algorithm:O', scale=alt.Scale(paddingInner=0)),
        ).add_selection(
            selection
        ).transform_filter(
            "selector.Categories[0] == 'Any' || indexof(split(datum.Categories, ','), selector.Categories[0]) >= 0"
        )
    else:
        base = alt.Chart(heatmapDF).encode(
            alt.X('Game:O', scale=alt.Scale(paddingInner=0)),
            alt.Y('Algorithm:O', scale=alt.Scale(paddingInner=0)),
        ).add_selection(
            selection
        ).transform_filter(
            "selector.Categories[0] == 'Any' || selector.Categories[0] == datum.Categories"
        )

    #make scale for coloring 
    myScale = alt.Scale(domain=[0, 0.25, 0.5, 0.75, 0.985, 1.0], range=['darkred', 'orange', 'white', 'darkgreen', 'green', 'green'], type='linear')
    myColor = alt.Color('Performance:Q', scale=myScale)

    heatmap = base.mark_rect().encode(
        color= myColor
    )

    return heatmap 
