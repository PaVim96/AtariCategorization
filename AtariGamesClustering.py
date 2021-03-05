import pandas as pd
import numpy as np
import scoreVisualisation
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import jaccard_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class AtariGamesClustering:
    """A class for Atari Games clustering methods. The csv should have for every game a single row, with features 
    as columns
    Args:
        dataPath (filePath): [A path to a csv file with data to cluster with]
    """

    def __init__(self, dataPath):
        """ Constructor

        Args:
            dataPath (filePath): [path to CSV-File, used as data]
            saveFileName ([type]): [filename to save work in]
        """
        self.info = []
        self.numberNormalizations = 0
        self.numberCategorizations = 0
        
        #first set info of possible normalization types and 
        self.normalizationTypes: list = self.__setNormalizationTypes()
        self.categorizationAlgos: list = self.__setCatAlgos()

        #prepare data to numpy array
        self.listOfGameNames, self.originalData, self.listOfFeatureNames = self.__prepData(dataPath)


    
        

    
    def __normalizeData(self, dataToNormalize, normMethod, alongGame, writeInfo=False):
        """Normalizes data

        Args:
            dataToNormalize ((g,f) numpy-Array): [The data to normalize: g games and for every game f features]
            normMethod (String): [The type of normalization to use: [Min-Max, Softmax, Standard, XY-Softmax, NoNorm]]
            alongGame (Boolean): [Whether to normalize along games (True) or not]
            writeInfo (Boolean, Defaults to False): [Whether to add info or not]
        Returns:
            normalizedData ((g,f) numpy-Array): [Normalized data]
        """

        self.__checkNormVal(normMethod)
        self.numberNormalizations += 1 

        #add info of methods used to information
        if writeInfo:
            alongGameInfo = f"Normalization {self.numberNormalizations} was done along games: {alongGame}"
            normMethodInfo = f"Normalization {self.numberNormalizations} was done using {normMethod}"
            self.__addInfo(alongGameInfo)
            self.__addInfo(normMethodInfo)


        #if we want to do stuff along the game, transpose data
        if alongGame:
            dataToNormalize = np.transpose(dataToNormalize)
        
        if normMethod == "Min-Max":
            maxPerformances = np.max(dataToNormalize, axis=0)
            minPerformances = np.min(dataToNormalize, axis=0) 
            normalizedData = (dataToNormalize - minPerformances)/ (maxPerformances - minPerformances)
        elif normMethod == "Standard":
            scaler = StandardScaler()
            normalizedData = scaler.fit_transform(dataToNormalize)
        elif normMethod == "NoNorm":
            normalizedData = dataToNormalize
        
        #if we wanted alongGame we transposed data before, so undo that
        if alongGame:
            normalizedData = np.transpose(normalizedData)

        return normalizedData

    def __getListOfGames(self):
        return self.listOfGameNames

    def __getInfo(self):
        return self.info


    def getData(self):
        return self.originalData


    def writeInfo(self, fileName):
        info = self.__getInfo()
        f = open(f"./CategInfo/{fileName}.txt", "w")
        for infoString in info:
            if not infoString.isspace():
                f.write(f"{infoString}\n")
        f.close()


    def saveClustering(self, labels, saveFileName):
        """Saves a Clustering under CalculatedCats with the name specified

        Args:
            labels ((g,) numpy-Array): [1d numpy-Array containing the calculated clusters]
            saveFileName (String): [The name to save the csv under in CalculatedCats]
        """
        gameNames = self.__getListOfGames()
        if labels.size != gameNames.size:
            raise ValueError("The calculated labels doesn't match the number of games of the original data")

        csvData = {'Game': gameNames, 'Categories': labels}
        dataFrame = pd.DataFrame(data=csvData)
        dataFrame.to_csv(f"./CalculatedCats/{saveFileName}.csv", index=False)

    
    def saveHeatmap(self, labels, saveFileName):
        """Saves a Heatmap under Visualisations with name specified. 
        The data is the original data without feature selection

        Args:
            labels ((g,) numpy-Array): [1d numpy-Array containing the calculated clusters]
            saveFileName (String): [The name to save the csv under in CalculatedCats]
        """

        #first make csv containing games, original data, categories

        #(g,f) numpy array
        data = self.originalData

        #convert data to appropriate format for heatmap 
        data = self.__normalizeData(data, "Min-Max", alongGame=True)

        games = self.listOfGameNames
        featureNames = self.listOfFeatureNames

        #build dataframe
        startData = {'Game': games}
        startIndex = 1
        dataFrame = pd.DataFrame(data=startData)

        #insert featureNames
        for i in range(featureNames.size):
            dataFrame.insert(startIndex, featureNames[i], data[:, i])
            startIndex += 1 

        #insert labels
        dataFrame.insert(startIndex, "Categories", labels, allow_duplicates=True)


        #convert to csv needed for heatmap 
        heatmap = scoreVisualisation.catVis(dataFrame, False, False)

        #save heatmap
        heatmap.save(f"./Visualisations/{saveFileName}.html")







    def __checkCategMethodVal(self, categMethod):
        if categMethod not in self.getPossibleCatAlgorithms():
            raise ValueError(f"Clustering Algorithm {categMethod} is not specified, possible Values are: {self.getPossibleCatAlgorithms()}")
    
    def __checkNormVal(self, normMethod):
        if normMethod not in self.getPossibleNormalizations():
            raise ValueError(f"This type of Normalization {normMethod} is not supported, supported is: {self.getPossibleNormalizations()}")


    def optimalKSilhouette(self, data, normMethod, alongGame, categMethod, writeInfo = False):
        """Calculates optimal number of clusters k with respect to the silhouette score and the number k. 
        If scores for two k1 and k2 are similar but k2 has way more clusters we prefer k2.

        Args:
            data ((g,f) numpy-Array): [The data used to cluster games]
            normMethod (String): [Method to normalize data with]
            alongGame (Boolean): [Whether to normalize along game or not]
            categType (Int): [Type of clustering used. Only relevant for the metric used to calculate the silhouette score]
            writeInfo (Bool, defaults to False): [Whether to write info in later info file]

        Returns:
            k (Int): [Optimal number of clusters k, regarding silhouette score and number of clusters k]
        """

        self.__checkCategMethodVal(categMethod)

        normalizedData = self.__normalizeData(data, normMethod, alongGame, writeInfo)

        silhouetteScores = list()

        upperBound = int(normalizedData.shape[0] / 2)
        
        for i in range(2, upperBound):
            _, currScore = self.calculateCategories(normalizedData, normMethod, alongGame, i, categMethod)
            silhouetteScores.append(currScore)
        
        silhouetteScores = np.array(silhouetteScores).squeeze()
        #get best N (we assume descending sorting)
        bestN = np.argsort(silhouetteScores)[::-1] + 2
        bestN = bestN[:5]
        bestNScores = np.take(silhouetteScores, bestN-2)
        #we do own scoring, because having more categories is better but we penalize with score 
        bestNPenalized = (bestNScores) * (bestN / np.max(bestN))

        n = bestN[np.argmax(bestNPenalized)]
        correspondingScore = bestNScores[np.argmax(bestNPenalized)]
        startStringInfo = "Silhouette Component Info:"
        silhouetteInfo = f"Weighted Silhouette scoring proposes: {bestN}, with best k = {n} and corresponding score is {correspondingScore}"
        if writeInfo:
            self.__addInfo(startStringInfo)
            self.__addInfo(silhouetteInfo)
        return n


    #TODO: ADD DBSCAN
    def calculateCategories(self, data, normMethod, alongGame, n, catMethod, writeInfo=False):
        """Clusters normedData by the specified clustering method

        Args:
            data ((g,f) numpy-Array): [Feature Matrix of data for g games]
            normMethod (String): [The method to normalize the data with]
            alongGame (Boolean): [Whether to normalize along games or not]
            n (Int): [Number of clusters to use]
            catMethod (String): [Clustering Algorithm to use]
            writeInfo (Boolean): [Whether to add information to later information file]
        Returns:
            labels ((g,) numpy-Array): [Labels for the data]
            score (Float): [Mean silhouette coefficient of the clustering]
        """
        self.__checkCategMethodVal(catMethod)
        self.numberCategorizations += 1
        normedData = self.__normalizeData(data, normMethod, alongGame)

        if writeInfo:
            infoStart = "Clustering Info:"
            infoString = f"Categorization {self.numberCategorizations} was done with {catMethod}, with k: {n}"
            self.__addInfo(infoStart)
            self.__addInfo(infoString)

        metric = "euclidian"
        if catMethod == "KMeans":
            model = KMeans(n)
        elif catMethod == "GMM":
            model = GaussianMixture(n, random_state=0, covariance_type="tied")
        elif catMethod == "KMedoids":
            metric = "mahalanobis"
            model = KMedoids(n, metric=metric)
        
        labels = model.fit_predict(normedData)
        score = silhouette_score(normedData, labels)
        return labels, score


    def featureEngineerer(self, features, featureNames, method="pearson"):
        """Calculates feature selected data from the original data

        Args:
            features ((g,f) numpy array): [Numpy array containing as rows number of games and per feature a column]
            featureNames (f numpy array): [Numpy array containing names of features]
            method (string): [Method to use when doing feature selection]
        Returns:
            featureSelected ((g, f') Numpy-array): [Numpy array containing only selected features]
        """


        if method == "pca":
            featureSelector = PCA(n_components=0.95, svd_solver="full")
            featureSelected = featureSelector.fit_transform(features)
            leftFeatures = f"Features still left are {featureNames}"
            self.__addInfo(leftFeatures)
            return featureSelected
        elif method == "pearson":
            dataframe = pd.DataFrame(data=features, columns=featureNames)
            correlation = np.abs(dataframe.corr(method="pearson").to_numpy())

            #feature indexes to remove
            indexes = list()

            #remove features with correlation > 0.80
            for i in range(correlation.shape[0]):
                for j in range(correlation.shape[1]):
                    if j in indexes:
                        continue 
                    #if same index we have 1 and pass
                    if i == j:
                        continue
                    if correlation[i][j] > 0.8:
                        #i throw away myself
                        indexes.append(i)
            indexes = np.unique(np.array(indexes))
        
            #drop data that we dont need anymore
            newFeatures = np.delete(features, indexes, axis=1)
            newFeatureNames = np.delete(featureNames, indexes, axis=0)
            leftFeatures = f"Features still left: {newFeatureNames}"
            self.__addInfo(leftFeatures)
            return newFeatures
        else:
            raise ValueError(f"{method} as Method for feature selection not known")


    #https://people.eecs.berkeley.edu/~jordan/sail/readings/luxburg_ftml.pdf
    #http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=9128DE558BAA6F3B88A36718F4FD858D?doi=10.1.1.331.1569&rep=rep1&type=pdf
    #second paper p.6-7
    def calcRobustness(self, data, normMethod, alongGame, n, catMethod):
        """Calculates the robustness of a clustering algorithm

        Args:
            data ((g,f) np-array): [The data used for clustering]
            normMethod (String): [The method to normalize the data with]
            alongGame (Boolean): [Whether to normalize along games or not]
            n (int): [number of clusters k to use]
            categType (String): [type of clustering to use]
        Returns:
            performance (float): [Robustness score of clustering, between [0,1]]
        """


        #we keep 75% information for every iteration
        sampleSize = int(3*data.shape[0]/4)

        #if number k is greater than samples we have, we cant calculate clustering -> bad clustering
        if n >= sampleSize:
            return 0
        
        wholeSim = 0
        maxAmount = 100

        groundTruth, _ = self.calculateCategories(data, normMethod, alongGame, n, catMethod)

        for _ in range(maxAmount):
            #generate bootstrap sample indices, without replacement for numerical stability
            x_star = random.sample(range(0, data.shape[0]), sampleSize)
            samples = data[x_star, :]
            assert samples.shape[1] == data.shape[1], "sampling on stability testing is wrong"
            compareLabels, _, = self.calculateCategories(samples, normMethod, alongGame, n, catMethod) 
            #taken from paper
            #indices that were used in the to compare clustering
            #take indices ground truth clustering were indices were used for bootstrapped clustering
            groundTruthCompare = groundTruth[x_star]
            sim = self.calcSimilarity(groundTruthCompare, compareLabels)
            wholeSim += sim
        
        #calculate average jaccard similarity
        performance = float(wholeSim / maxAmount)

        return performance


    def calcSimilarity(self, groundTruth, compareValues):
        """
        Calculates jaccard similarity between a ground Truth and calculated clustering. Used for robustness calculation

        Args: 
            groundTruth: (1d-numpy Array): [ground Truth categories]
            compareCats: (1d-numpy Array): [categories to compare]

        Returns: 
            performance: float: performance of the categorization with respect to the constraintMatrix
        """
        return jaccard_score(groundTruth, compareValues, average="weighted")


    def calculateInstanceScore(self, labels):
        """Calculates a goodness of clustering depending on how many games are in the different categories. 
        A good clustering is a clustering with clusters with more than 1 game but not too many games

        Args:
            categories (g numpy-Array): [For each game in the original data the corresponding label]

        Returns:
            [float]: [Instance score]
        """

        numberGames = np.size(labels)
        numberCats = np.size(np.unique(labels))
        
        maxScore = numberCats
        currScore = 0
        for i in range(numberCats):
            boolArray = labels == i
            currNumberCats = np.size(labels[boolArray])
            if currNumberCats == 1 or currNumberCats >= int(0.75 * numberGames):
                continue
            else:
                currScore += 1
        return float(currScore / maxScore)


    def __addInfo(self, infoString):
        self.info.append(infoString)

    def __prepData(self, csvPath):
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


    def __setNormalizationTypes(self):
        normalizationTypes = ["Min-Max", "Standard", "NoNorm"]
        return normalizationTypes

    def __setCatAlgos(self):
        categorizationAlgos = ["KMeans", "KMedoids", "GMM", "DBSCAN"]
        return categorizationAlgos


    def getPossibleNormalizations(self):
        return self.normalizationTypes
    
    def getPossibleCatAlgorithms(self):
        return self.categorizationAlgos