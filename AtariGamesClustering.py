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
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


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
        self.numberFeatureSelections = 0
        self.DBSCANEpsilon = None
        self.needEpsilon = True
        self.AlgoType = None
        self.NormType = None
        self.AlongGame = None
        self.FeatureEngineerer = None


        #first set info of possible normalization types and 
        self.normalizationTypes: list = self.__setNormalizationTypes()
        self.categorizationAlgos: list = self.__setCatAlgos()

        #prepare data to numpy array
        self.listOfGameNames, self.originalData, self.listOfFeatureNames = self.__prepData(dataPath)
        print(self.listOfGameNames)

        self.data = np.copy(self.originalData)

    def setClusterAlgorithm(self, methodName):
        self.__checkCategMethodVal(methodName)
        self.AlgoType = methodName

    def getClusterAlgorithm(self):
        algo = self.AlgoType
        if algo is None:
            raise ValueError("Please make sure that you have set the Clustering Algorithm with setClusterAlgorithm(...)")
        else:
            return algo

    
    def setNormalization(self, normType, alongGame):
        self.__checkNormVal(normType)
        self.NormType = normType
        self.AlongGame = alongGame

    def getNormalization(self):
        normalization = (self.NormType, self.AlongGame)
        if normalization[0] is None or normalization[1] is None:
            raise ValueError("Please make sure that you have set the Normalization type with setNormalization(...)")
        else:
            return normalization

    def convertDRLScores(self, startIndexes, endIndexes, normalizationCSVPath):
        """Converts Raw DRL-Scores contained in the original data to normalized data (human-normalized for example)

        Args:
            startIndexes (List): [List of start indexes. Every start index indicates the first column of DRL-Agent scores of some evaluation type to normalize on. Minimum possible index is always 1 (since Games are 0)]
            endIndexes ([type]): [List of end indexes (! inclusive !). Analogous to start indexes]
            normalizationCSVPath (String): [A csv file in ./BaselineData. Number of columns needs to be equal to 2*size(startIndexes)+1]
        """

        #first a lot of checks 

        #check lists 

        if (not isinstance(startIndexes, list)) or (not isinstance(endIndexes, list)):
            raise TypeError("Indexes need to be lists")

        #check matching list sizes 
        if len(startIndexes) != len(endIndexes):
            raise ValueError("Lists don't match")


        baselineFrame = pd.read_csv(normalizationCSVPath)
        #check that size is right for data 
        indexSize = len(baselineFrame.columns)
        
        #2*listSize + 1 because CSV contains column with game names
        if indexSize != 2*len(startIndexes)+1:
            raise ValueError("CSV isn't in right format")


        baselineData = baselineFrame.iloc[:, 1:].to_numpy()
    

        #number baselines and random agent scores
        #first check that number cols is correct
        sizeBaseline = baselineData.shape[1]
        if sizeBaseline % 2 != 0:
            raise ValueError("Format incorrect: Columns need to be [Baseline1, Randomscore1, (Baseline2, Randomscore2, ...)]")

        baselineScores = np.zeros((baselineData.shape[0], int(sizeBaseline / 2)))
        randomScores = np.zeros((baselineData.shape[0], int(sizeBaseline / 2)))

        for i in range(int(sizeBaseline/2)):
            baselineScores[:, i] = baselineData[:, i]
            randomScores[:, i] = baselineData[:, i+1]

        data = self.getData()
        #now normalize original data

        sizeListIndexes = len(startIndexes)

        for i in range(sizeListIndexes):
            startIndex = startIndexes[i]
            endIndex = endIndexes[i]

            currRandomScore = randomScores[:, i].reshape(-1, 1)
            currBaselineScore = baselineScores[:, i].reshape(-1, 1)

            currData = data[:, startIndex:endIndex+1]
            newData = (currData - currRandomScore) / (currBaselineScore - currRandomScore)
            self.setData(newData, startIndex, endIndex)
        
            

    
    def __normalizeData(self, dataToNormalize, writeInfo=False):
        """Normalizes data

        Args:
            dataToNormalize ((g,f) numpy-Array): [The data to normalize: g games and for every game f features]
            normMethod (String): [The type of normalization to use: [Min-Max, Softmax, Standard, XY-Softmax, NoNorm]]
            alongGame (Boolean): [Whether to normalize along games (True) or not]
            writeInfo (Boolean, Defaults to False): [Whether to add info or not]
        Returns:
            normalizedData ((g,f) numpy-Array): [Normalized data]
        """


        normMethod, alongGame = self.getNormalization()

        self.numberNormalizations += 1 

        #add info of methods used to information
        if writeInfo:
            alongGameInfo = f"Normalization {self.numberNormalizations} was done along games: {alongGame}"
            normMethodInfo = f"Normalization {self.numberNormalizations} was done using {normMethod}"
            self.__addInfo(alongGameInfo)
            self.__addInfo(normMethodInfo)

        #if we have alongInformation scaling and number of rows == 1 -> can't normalize
        if not alongGame and dataToNormalize.shape[0] == 1:
            return dataToNormalize


        #if we want to do stuff along the game, transpose data
        if alongGame:
            #if #columns == 1 we cant do normalization along games, so just return value
            if dataToNormalize.shape[1] == 1:
                return dataToNormalize
            dataToNormalize = np.transpose(dataToNormalize)
        
        if normMethod == "Min-Max":
            #min-max only makes sense with at least 3 samples
            if dataToNormalize.shape[0] < 3:
                normalizedData = dataToNormalize
            else:
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

    def __resetData(self):
        self.data = self.originalData


    def __getListOfGames(self):
        return self.listOfGameNames

    def __getInfo(self):
        return self.info

    def __getFeatureNames(self):
        return self.listOfFeatureNames


    def getData(self):
        return np.copy(self.data)

    def setData(self, data, startIndex=0, endIndex=-1):
        """Sets the original data to the input data at the specified indexes

        Args:
            data ((g,f) numpy-Array): [Data to override with]
            startIndex (int, optional): [Start index of overwriting]. Defaults to 0.
            endIndex (int, optional): [End index of overwriting]. Defaults to -1.
        """


        #check that indexes aren't weird
        if (startIndex >= endIndex and endIndex != -1) or startIndex < 0 or endIndex > self.data.shape[1]:
            raise ValueError("Either start index or end index is non valid")
        #check that size of data and indexes match 
        if endIndex+1 != data.shape[1] or data.shape[0] != self.data.shape[0]:
            raise ValueError("Non matching sizes")

        #make index inclusive not exclusive
        if endIndex != -1:
            endIndex += 1

        self.data[:, startIndex:endIndex] = data


    def makeFileName(self, clusterDataName):
        norm = self.getNormalization()
        cluster = self.getClusterAlgorithm()
        featuremethod = self.__getFeatureMethod()
        along = None
        if norm[1]:
            along = "AlongGame"
        else:
            along = "AlongInfo"

        fileName = f"{clusterDataName}_{cluster}_{along}_{featuremethod}"
        return fileName




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
        #set normalization to min-max, alonGame, keep old normalization for afterwards
        currentNorm = self.getNormalization()
        self.setNormalization("Min-Max", True)
        data = self.__normalizeData(data)

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
        self.setNormalization(currentNorm[0], currentNorm[1])






    def __checkCategMethodVal(self, categMethod):
        if categMethod not in self.getPossibleCatAlgorithms():
            raise ValueError(f"Clustering Algorithm {categMethod} is not specified, possible Values are: {self.getPossibleCatAlgorithms()}")
    
    def __checkNormVal(self, normMethod):
        if normMethod not in self.getPossibleNormalizations():
            raise ValueError(f"This type of Normalization {normMethod} is not supported, supported is: {self.getPossibleNormalizations()}")


    def optimalKSilhouette(self, data, writeInfo = False):
        """Calculates optimal number of clusters k with respect to the silhouette score and the number k. 
        If scores for two k1 and k2 are similar but k2 has way more clusters we prefer k2.

        Args:
            data ((g,f) numpy-Array): [The data used to cluster games]
            normMethod (String): [Method to normalize data with]
            alongGame (Boolean): [Whether to normalize along game or not]
            categType (String): [Type of clustering used. Only relevant for the metric used to calculate the silhouette score]
            writeInfo (Bool, defaults to False): [Whether to write info in later info file]

        Returns:
            k (Int): [Optimal number of clusters k, regarding silhouette score and number of clusters k]
        """


        normalizedData = self.__normalizeData(data, writeInfo)

        silhouetteScores = list()

        upperBound = int(normalizedData.shape[0] / 2)
        
        for i in range(2, upperBound):
            _, currScore = self.calculateCategories(normalizedData, i)
            silhouetteScores.append(currScore)
        
        silhouetteScores = np.array(silhouetteScores).squeeze()
        #get best N (we assume ascending sorting)
        #+2 because we start with k=2
        bestN = np.argsort(silhouetteScores)[::-1] + 2
        bestN = bestN[:5]
        bestNScores = np.take(silhouetteScores, bestN-2)
        #we do own scoring, because having more categories is better but we penalize with score 
        bestNPenalized = 80*((bestNScores + 1)/2) + 20*(bestN / np.max(bestN))

        n = bestN[np.argmax(bestNPenalized)]
        correspondingScore = bestNScores[np.argmax(bestNPenalized)]
        startStringInfo = "Silhouette Component Info:"
        silhouetteInfo = f"Weighted Silhouette scoring proposes: {bestN}, with best k = {n} and corresponding score is {correspondingScore}"
        if writeInfo:
            self.__addInfo(startStringInfo)
            self.__addInfo(silhouetteInfo)
        return n


    #TODO: ADD DBSCAN
    def calculateCategories(self, data, n, writeInfo=False):
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
        self.numberCategorizations += 1
        normedData = self.__normalizeData(data)
        catMethod = self.getClusterAlgorithm()

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
        elif catMethod == "DBSCAN":
            needEpsilon = self.__needDBSCANEpsilon() 
            model = self.DBSCANAlgo(normedData, needElbow=needEpsilon)
        
        labels = model.fit_predict(normedData)

        #if #labels == 1 -> score is -1, because we have no information from clustering 
        if np.unique(labels).size == 1:
            score = -1
        else:
            score = silhouette_score(normedData, labels)
        return labels, score

    def __incrementSelections(self):
        self.numberFeatureSelections += 1

    def __getCountFeatureSelection(self):
        return self.numberFeatureSelections


    def __setFeatureMethod(self, method):
        self.featureEngineerer = method

    def __getFeatureMethod(self):
        return self.featureEngineerer

    def featureEngineerer(self, method="pearson", writeInfo = False):
        """Calculates feature selected data from the original data

        Args:
            method (string): [Method to use when doing feature selection]
        """
        features = self.getData()
        featureNames = self.__getFeatureNames()
        if self.__getCountFeatureSelection() > 0:
            self.__resetData()
        self.__incrementSelections()
        self.__setFeatureMethod(method)
        if method == "pca":
            featureSelector = PCA(n_components=0.95, svd_solver="full")
            featureSelected = featureSelector.fit_transform(features)
            if writeInfo:
                leftFeatures = f"Features still left are {featureNames}"
                self.__addInfo(leftFeatures)
            self.data = featureSelected
        elif method == "pearson":
            dataframe = pd.DataFrame(data=features, columns=featureNames)
            correlation = np.abs(dataframe.corr(method="pearson").to_numpy())

            #feature indexes to remove
            indexes = list()

            #remove features with correlation > 0.9
            for i in range(correlation.shape[0]):
                for j in range(correlation.shape[1]):
                    if j in indexes:
                        continue 
                    #if same index we have 1 and pass
                    if i == j:
                        continue
                    if correlation[i][j] > 0.9:
                        #i throw away myself
                        indexes.append(i)
            indexes = np.unique(np.array(indexes))
        
            #drop data that we dont need anymore
            newFeatures = np.delete(features, indexes, axis=1)
            newFeatureNames = np.delete(featureNames, indexes, axis=0)
            if writeInfo:
                leftFeatures = f"Features still left: {newFeatureNames}"
                self.__addInfo(leftFeatures)
            #not with setData() because we override the original data!
            self.data = newFeatures
        elif method == "sfs":
            newData = self.__sfsSelection(features, featureNames, writeInfo=writeInfo)
            self.data = newData
        else:
            raise ValueError(f"{method} as Method for feature selection not known")


    def __sfsSelection(self, data, featureNames, toKeep=5, writeInfo=False):
        """Feature Selection with Sequential Forward Selection

        Args:
            data ((g,f) numpy-Array): [Numpy Array containing the data]
            featureNames ((f,) numpy-Array): [Array containing the names of the features]
            toKeep (Int): [Number of Features to keep, defaults to 5]

        Returns:
            newData ((g,toKeep)): [Numpy array containing new features]
        """


        #keep count of optimal features to keep
        indicesToKeep = []
        bestScoreAll = -np.inf
        iterations = 0

        
        while len(indicesToKeep) < toKeep and iterations != 5:
            #ugly but needed: for DBSCAN we set for every iteration epsilon to None because for new features we need new epsilon
            self.__setDBSCANEpsilon(None)


            #every new iteration we keep track of bestScore
            iterations += 1
            bestScore = -np.inf
            indexToKeep = None
            for featureIndex in range(data.shape[1]):
                if featureIndex in indicesToKeep:
                    continue
                currentIndexArray = indicesToKeep + [featureIndex]
                #construct array out of indicestoKeep and current index
                #len == 0 -> just index
                currentData = data[:, currentIndexArray]
                k = self.optimalKSilhouette(currentData, writeInfo=writeInfo)
                _, score = self.calculateCategories(currentData, n=k, writeInfo=writeInfo)
                if score > bestScore:
                    bestScore = score
                    indexToKeep = featureIndex
            if bestScore > bestScoreAll:
                bestScoreAll = bestScore
                indicesToKeep.append(indexToKeep)
            #if the score didn't get better, we already have optimal amount of features
            else:
                break

        featureInfo = f"Features kept are: {featureNames[indicesToKeep]}" 
        print(featureInfo)



        if writeInfo:
            self.__addInfo(featureInfo) 
        return data[:, indicesToKeep]

    def DBSCANAlgo(self, normalizedData, writeInfo=False, needElbow=False):
        if needElbow:
            #get distance data
            neighbors = NearestNeighbors(n_neighbors=3)
            nbrs = neighbors.fit(normalizedData)
            distances, _ = nbrs.kneighbors(normalizedData)
            distances = np.sort(distances, axis=0)
            distances = distances[:, 2]
            
            print("Extract from the following plot the elbow-Point for the DBSCAN Algorithm")
            plt.plot(distances)
            plt.show()

            epsilon = input("Please Input the optimal Epsilon obtained by the Elbow-Plot:\n")
            epsilon = float(epsilon)
            self.__setDBSCANEpsilon(epsilon)
        else:
            epsilon = self.__getDBSCANEpsilon()
        

        model = DBSCAN(eps=epsilon, min_samples=2)

        if writeInfo:
            info = f"Best epsilon for DBSCAN is {epsilon}"
            self.__addInfo(info)
        return model


    def __needDBSCANEpsilon(self):
        return self.needEpsilon

    def __setDBSCANEpsilon(self, epsilon):
        if epsilon is None:
            self.needEpsilon = True
        else:
            self.needEpsilon = False
        self.DBSCANEpsilon = epsilon

    def __getDBSCANEpsilon(self):
        epsilon = self.DBSCANEpsilon

        if epsilon is None:
            raise ValueError("DBSCAN needs Epsilon to continue")

        return epsilon



    #https://people.eecs.berkeley.edu/~jordan/sail/readings/luxburg_ftml.pdf
    #http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=9128DE558BAA6F3B88A36718F4FD858D?doi=10.1.1.331.1569&rep=rep1&type=pdf
    #second paper p.6-7
    def calcRobustness(self, data, n):
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

        groundTruth, _ = self.calculateCategories(data, n)

        for _ in range(maxAmount):
            #generate bootstrap sample indices, without replacement for numerical stability
            x_star = random.sample(range(0, data.shape[0]), sampleSize)
            samples = data[x_star, :]
            assert samples.shape[1] == data.shape[1], "sampling on stability testing is wrong"
            compareLabels, _, = self.calculateCategories(samples, n) 
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