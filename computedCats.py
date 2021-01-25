import pandas as pd 
import scipy as scp 
import numpy as np 
import scoreVisualisation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.special import binom
import json
import sys
from math import floor
import random
import webbrowser
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


"""
IMPORTANT: Everything that does fit_predict(X) takes X as shape (n_samples, n_features)
in our case: n_features -> algorithms
"""
class computedCategories:

    def __init__(self, csvPath, isSearch, nameToSave, n = 5, evaluationCSV = ""):
        if not isSearch:
            games, performances, indexes = self.prepData(csvPath, False)
            cats, score = self.calculateCategories(games, performances, n)
            self.csv = self.makeCsv(csvPath, cats)
            self.csv = self.modPerfOfCsv(self.csv, performances)
            self.heatmap = scoreVisualisation.catVis(self.csv, False, False)
        else:
            games, dataForCats, indexes = self.prepData(csvPath, isSearch)
            self.games = games
            #normalize data that is used for categorisation
            normalizedData, normalizationInfo = self.normalizeData(dataForCats, 2, alongGame=1)


            #pca after or game vector normalization?
            n, visData = self.findPCACategories(games, normalizedData, indexes)


            #cats, _ = self.searchByDropping(games, normalizedData, indexes, n)
            cats = self.elbowMethod(games, normalizedData, indexes, n)

            constraintScore = "No constraint matrix used"
            #if evaluationMatrix not empty, calculate performance
            if evaluationCSV != "":
                constraintMatrix = self.mkConstraintMatrix(evaluationCSV)
                constraintScore = self.calculatePerformance(cats, constraintMatrix)

            self.info = self.setInformation(n, constraintScore, normalizationInfo, nameToSave, csvPath, evaluationCSV)
            self.calculatedCats = cats


            #self.visualizeClusters(visData, cats, games)
            self.csv = self.makeCsv(csvPath, cats)
            #performance scaling to 0-1 only for visalisation
            performances, _ = self.normalizeData(dataForCats, normType = 0, alongGame = 1)

            self.csv = self.modPerfOfCsv(self.csv, performances)
            self.heatmap = scoreVisualisation.catVis(self.csv, False, False)

    def writeCatsToCSV(self, filePath):
        cats = self.calculatedCats
        games = self.games
        dataCsv = {"games": games, "calculatedCats": cats}
        pd.DataFrame(data=dataCsv).to_csv(filePath, index=False)



    def setInformation(self, n, constraintScore, normalizationInfo, nameToSave, csvPath, evaluationCSV):
        cats = f"Number of categories is {n}"
        usedFile = f"The categorization was done on {csvPath}"
        savePlace = f"The categorization is saved with filename {nameToSave}.html"
        goodness = constraintScore
        if evaluationCSV != "":
            goodness = f"Categorization achieved {constraintScore}% on the constraint matrix from {evaluationCSV}"
        return [cats, usedFile, savePlace, normalizationInfo, goodness]


    def getInformation(self):
        return self.info



    def buildCategoryMatrix(self, matrix):
        #first check if matrix contains zero, if yes just shift by one 
        hasZero = not np.all(matrix)
        if hasZero:
            matrix = matrix + 1


        matrix = matrix.reshape((-1,1))

        catMatrix = np.dot(matrix, matrix.T)
        catTruthMatrix = np.ones((catMatrix.shape)).astype(bool)

        #for every row do:
        #get ith-element, since it is the base element
        #take square root, since we now that it is a square
        #divide row by base element
        #check whether elements in row are equal to base element, if yes -> same category
        for i, row in enumerate(catMatrix):
            baseCategory = np.sqrt(row[i])
            catMatrix[i] = row / baseCategory  

            catTruthMatrix[i] = (row == baseCategory)
        return catTruthMatrix

    def mkConstraintMatrix(self, evaluationCSV):
        """Converts csv into constraint matrix

        Args:
            evaluationCSV ([file]): [filepath to csv, that contains a single column]
        
        Returns:
            constraintMatrix ([numpy array]): [(n,) numpy array that contains a constraint for n games in form of categories]
        """
        csvConstraint = pd.read_csv(evaluationCSV)
        constraintMatrix = csvConstraint.iloc[:, 1].to_numpy().astype(int)

        return constraintMatrix


    def extractPerformanceValues(self, performanceMatrix):
        m = performanceMatrix.shape[0]
        r = np.arange(m)
        mask = r[:, None] < r
        return performanceMatrix[mask]



    def calculatePerformance(self, calcCats, constraintMatrix):
        """
        Calculates a performance metric based on a constraintMatrix

        Args: 
            calcCats: 1d-numpy Array: the calculated categories
            constraintMatrix: 1d-numpy Array: constraints in form of desired categories

        Returns: 
            performance: float: performance of the categorization with respect to the constraintMatrix
            
        """
        #TODO: Think about what to do with don't cares in constraintMatrix
        #Possible solution: dont care's should be -1 and after calculating truth matrix set every position of dont cares to true



        #first build square boolean matrix for calculated categories, matrix[i][j] is True iff i in category with j
        gameCategoryCorrespondences = self.buildCategoryMatrix(calcCats)
        #build constraint boolean matrix, matrix[i][j] True iff i and j should be in the same category
        constraintCorrespondences = self.buildCategoryMatrix(constraintMatrix)

        #check whether games are categorised together AND fulfill constraint 
        performanceMatrix = np.equal(gameCategoryCorrespondences, constraintCorrespondences).astype(int)

        #take into account that we have (n over 2) possibilities 
        n = gameCategoryCorrespondences.shape[0]
        maxNumberPerformances = binom(n, 2) 
        actualPerformance = np.sum(self.extractPerformanceValues(performanceMatrix))

        performance = (actualPerformance / maxNumberPerformances)
        return performance

        


    def calculateKMeans(self, games, dataForClustering, indexes, n):
        clusterPerformances = np.copy(dataForClustering)
        #we assume that data is already normalized
        #clusterPerformances = self.normalizeData(clusterPerformances, normType = 4)
        cats, score = self.calculateCategories(games, clusterPerformances, n)
        return cats, score



    #implements elbow method to find good number of categories 
    #n is pre defined set of categories -> do elbow method on [n-floor(n/3),n+floor(n/3)]
    def elbowMethod(self, games, dataForCats, indexes, n):
        finalN = n
        print(f"Proposed number of categories is {n}")
        nStart = 1 
        nEnd = 3*n+1
        nList = list(range(nStart, nEnd))

        bestN = -1
        scores = list()
        catList = list()
        for n in range(nStart, nEnd):
            cats, score = self.calculateKMeans(games, dataForCats, indexes, n)
            scores.append(score)
            catList.append(cats)


        for n in nList:
            plt.axvline(x=n)
        plt.xticks(nList)
        plt.plot(nList, scores)
        plt.show()


        return catList[finalN]   


    def visualizeClusters(self, visData, cats, games):
        n = np.unique(cats)
        visData = visData.T
        
        plt.rcParams["figure.figsize"] = (10,6)
        for cat in n:
            plt.scatter(visData[0, cats == cat], visData[1, cats == cat], label=cat)

        #annotate all points
        for i in range(games.shape[0]):
            plt.annotate(games[i], (visData[0, i], visData[1,i]))

        plt.legend()
        plt.show()




    """
        performances: (n,m) array -> n: number of games, m: number of features
    """
    #calculates PCA to find number of categories 
    def findPCACategories(self, games, performances, perfIndexes):
        assert performances.shape[0] > performances.shape[1], "wrong ordering, or maybe check if really less games then information"




        #transpose the performances vector
        #https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
        #we assume already normalized data
        gameInformations = np.copy(performances)
        #gameInformations = np.copy(performances).T
        #gameInformations = self.normalizeData(gameInformations.T, normType=4).T


        #base for PCA
        pcaOfData = PCA(n_components=0.95, svd_solver="full")


        pcaVis = PCA(n_components=2)

        #we do the pca also on the informations and only keep 2 dimensions for visualisation of clusters
        visualizeClusters = np.copy(np.transpose(gameInformations))
        visData = pcaVis.fit_transform(visualizeClusters)
    


        categorizedData = pcaOfData.fit_transform(gameInformations)
        n = categorizedData.shape[1]
        
        return n, visData



    """
        performances: (n,m) array -> n: number of games, m: features
    """
    #this method tries to search for good categories by dropping some algorithms that make for obvious correlations
    #for example noisy and dqn is in most of the cases bad and good
    def searchByDropping(self, games, performances, indexes,n):
        numInfos = performances.shape[1]


        #we drop a max of 1/3% of information from games for clustering
        maxColsDropped = floor(numInfos * 0.50)


        #Scaling
        clusterPerformances = np.copy(performances)
        #we assume data is already normalized 
        #clusterPerformances = self.normalizeData(clusterPerformances, normType=0)


        #best winner approach
        #for every category in cats make list of possible categories [0,n]
        winnerCategories = [[0 for k in range(n)] for j in range(clusterPerformances.shape[0])]
        #we randomly drop a amount of informations out of the game vectors (max 33%)
        #we then calculate the clustering with the dropped informations

        repetitions = 100
        score = 0
        #return self.calculateCategories(games, clusterPerformances, n)

        #assign at position j of categories [0,..., #games] category with most votes 
        for _ in range(repetitions):
            #calculate the random number of information we want to drop
            numberCols = random.sample(range(1, maxColsDropped), 1)[0]
            #calculate numberCols cols to drop
            droppedInfos = random.sample(range(0, numInfos), numberCols)
            currentPerf = np.delete(clusterPerformances, droppedInfos, axis=1)
            currentCat, currScore = self.calculateCategories(games, currentPerf, n)
            score += currScore
            for i, cat in enumerate(currentCat):
                winnerCategories[i][cat] += 1
            

        score /= repetitions
        #interpret the current categories as probabilites of being in a category and cluster based off of that
        winnerCategories = np.array(winnerCategories) / repetitions
        winnerCategories, score = self.calculateCategories(games, winnerCategories, n)

        return winnerCategories, score
        
                    

    #make CSV like NormalizedScores0112.csv format 
    #arg cats is numpy list of categories for every game
    def makeCsv(self, csvPath, cats):
        csv = pd.read_csv(csvPath)
        countCol = len(csv.columns)
        csv.insert(countCol, "Categories", cats, True)
        return csv

    def makeListOfGames(self, games, cats):
        n = np.max(cats)+1
        listOfGames = {k: [] for k in range(n)}
        for i, cat in enumerate(cats):
            currValue = listOfGames[cat]
            if currValue == None:
                listOfGames[cat] = games[i]
            else:
                listOfGames[cat].append(games[i])
        return listOfGames, dict(zip(games, cats))
        


    def getHeatmap(self):
        return self.heatmap
        
    #modifies a csv to contain normalized performances instead of raw scores for visualisation
    def modPerfOfCsv(self, csv, normPerf):
        numAlgos = normPerf.shape[1]
        csv.iloc[:, 1:1+numAlgos] = normPerf
        return csv
    
    """
        Converts the performances of the csv of csvPath into numpy float array
        Also returns all included games and indexes (names) of available performances
    """
    def prepData(self, csvPath, isSearch):
        data = pd.read_csv(csvPath)
        performances = data.iloc[:, 1:].to_numpy()
        perfIndex = data.iloc[:, 1:].columns.to_numpy()
        games = data.iloc[:, 0].to_numpy()
        #convert commas to points for float
        #but only convert if type isn't already float
        isFloat = performances.dtype == float
        if not isFloat:
            convertFloat = lambda x: [s.replace(",", ".") for s in x]
            performances = np.apply_along_axis(convertFloat, 0, performances).astype(float)


        return games, performances, perfIndex
    


    def normalizeData(self, performances, normType=0, alongGame=0):
        """Normalizes data

        Args:
            performances ([numpy array (n,m)]): [A numpy array containing n games and m information-values]
            normType (int, optional): [type of normalization
                                       0: min-max normalization
                                       1: softmax normalization
                                       2: Standard Scaling normalization]. Defaults to 0.
            alongGame (int, optional): [which axis to normalize
                                        0: features = information
                                        1: features = games]. Defaults to 0.

        Returns:
            normalizedPerf [numpy array (n,m)]: [Normalized data]
            type [string]: [The type of normalization used]
        """
        assert performances.shape[0] > performances.shape[1], "wrong ordering, or maybe check if really less games then information"

        if alongGame == 1:
            performances = np.transpose(performances)

        if normType == 0:
            #code is ugly, need to transpose again so it stays consistent with alongGame
            performances = np.transpose(performances)
            normString = "Normalization: min-max along information"
            maxPerformances = np.apply_along_axis(np.max, axis=1, arr=performances)
            minPerformances = np.apply_along_axis(np.min, axis=1, arr=performances)
            #normalization
            maxPerformances = np.array([[maxPerformances],]*performances.shape[1]).transpose().squeeze()
            minPerformances = np.array([[minPerformances],]*performances.shape[1]).transpose().squeeze()
            normalizedPerf = (performances - minPerformances) / (maxPerformances - minPerformances)

            if alongGame == 1:
                normString = "Normalization: min-max along game"
            elif alongGame == 0:
                normalizedPerf = np.transpose(normalizedPerf) 
            return normalizedPerf, normString
        elif normType == 1:
            performances = np.transpose(performances)
            normString = "Normalization: softmax along information"
            #for me: specifying the axis makes that axis disappear -> so (X,Y) => (X,) 
            gameInfoSum = np.sum(performances, axis=1).reshape((-1,1))
            normalizedPerf = performances / gameInfoSum
            if alongGame == 1:
                normString = "Normalization: softmax along game"
            elif alongGame == 0:
                normalizedPerf = np.transpose(normalizedPerf)
            return normalizedPerf, normString
        elif normType == 2:
            normString = "Normalization: Standard scaler along information"
            scaler = StandardScaler()
            normalizedPerf = scaler.fit_transform(performances)
            if alongGame == 1:
                normString = "Normalization: Standard scaler along game"
                normalizedPerf = np.transpose(normalizedPerf)
            return normalizedPerf, normString

                
            
    
    def calculateCategories(self, games, dataForClustering, n):
        kmeans = KMeans(n, random_state=0)
        kmeansData = kmeans.fit_predict(dataForClustering)
        score = kmeans.inertia_
        return kmeansData, score


if __name__ == '__main__':
    numberArgs = len(sys.argv)
    assert numberArgs >= 4, "Too Few Args, need: File, {search, vis}, fileNameToSave, #Categories (if vis),  evaluationMatrix (if search)"
    categData = sys.argv[1] 
    saveFileName = sys.argv[3]
    searchOrVis = sys.argv[2] 
    assert (searchOrVis == "search" or searchOrVis == "vis"), "Third arg should be {search, vis}"
    isSearch = False
    if searchOrVis == "search":
        isSearch = True
    if not isSearch:
        if numberArgs == 5:
            n = int(sys.argv[4])
            visualizer = computedCategories(categData, saveFileName, isSearch, n=n)
        else:
            visualizer = computedCategories(categData, isSearch, saveFileName)
    else:
        if numberArgs == 5:
            evaluationCSV = sys.argv[4]
            visualizer = computedCategories(categData, isSearch, saveFileName, evaluationCSV=evaluationCSV)
        else:
            visualizer = computedCategories(categData, isSearch, saveFileName)
        
    heatmap = visualizer.getHeatmap()
    print("-----------------")
    print(saveFileName)
    print(f"Saving the chart in: ./Visualisations/{saveFileName}.html")
    heatmap.save(f"./Visualisations/{saveFileName}.html")
    print(f"Writing information for categorization in ./CategInfo/{saveFileName}.txt")
    info = visualizer.getInformation()
    f = open(f"./CategInfo/{saveFileName}.txt", "w")
    for s in info:
        f.write(f"{s}\n")
    f.close()

    catsPath = f"./CalculatedCats/{saveFileName}.csv"
    print(f"Writing the calculated categories into {catsPath}")
    visualizer.writeCatsToCSV(catsPath)
    print("Opening the visualisation for you")
    webbrowser.open(f"./Visualisations/{saveFileName}.html")
