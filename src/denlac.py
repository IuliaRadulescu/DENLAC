__author__ = "Radulescu Iulia-Maria"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "iulia.radulescu@cs.pub.ro"
__status__ = "Production"

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort
import scipy.stats as st
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import estimate_bandwidth

from random import randint
import argparse
import math
import collections
import evaluation_measures
import time


class Denlac:

    def __init__(self, noClusters, noBins, expandFactor, noDims, aggMethod, debugMode):

        self.noClusters = noClusters
        self.noBins = noBins
        # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)
        self.expandFactor = expandFactor

        self.noDims = noDims
        self.debugMode = debugMode
        self.aggMethod = aggMethod

        self.idCluster = -1

        self.ALREADY_PARSED_FALSE = -1
        self.ALREADY_PARSED_TRUE = 1

    def rebuildDictIndexes(self, dictToRebuild, distBetweenPartitionsCache):

        newDict = dict()
        newCacheDict = dict()
        newDictIdx = 0

        oldNewIndexesCorrelation = {}

        for i in dictToRebuild:
            newDict[newDictIdx] = dictToRebuild[i]
            oldNewIndexesCorrelation[i] = newDictIdx
            newDictIdx = newDictIdx + 1

        for keyTuple in distBetweenPartitionsCache:
            if(keyTuple[0] in oldNewIndexesCorrelation and keyTuple[1] in oldNewIndexesCorrelation):
                newI = oldNewIndexesCorrelation[keyTuple[0]]
                newJ = oldNewIndexesCorrelation[keyTuple[1]]
                newCacheDict[(newI, newJ)] = distBetweenPartitionsCache[keyTuple]

        return newDict, newCacheDict, newDictIdx

    def computeDistanceIndices(self, partitions, distBetweenPartitionsCache):

        distances = []
        for i in range(len(partitions)):
            for j in range(len(partitions)):
                if (i == j):
                    distBetweenPartitions = -1
                else:
                    if (i, j) in distBetweenPartitionsCache:
                        distBetweenPartitions = distBetweenPartitionsCache[(i, j)]
                    else:
                        if (self.aggMethod == 2):
                            distBetweenPartitions = self.calculateCentroid(partitions[i], partitions[j])
                        else:
                            distBetweenPartitions = self.calculateSmallestPairwise(partitions[i], partitions[j])
                        distBetweenPartitionsCache[(i, j)] = distBetweenPartitions
                distances.append(distBetweenPartitions)

        # sort by distance
        distances = np.array(distances)
        indices = np.argsort(distances)

        finalIndices = [index for index in indices if distances[index] > 0]

        return finalIndices

    # i = index, x = amount of columns, y = amount of rows
    def indexToCoords(self, index, columns, rows):

        for i in range(rows):
            # check if the index parameter is in the row
            if (index >= columns * i and index < (columns * i) + columns):
                # return x, y
                return index - columns * i, i

    def sortAndDeduplicate(self, l):
        result = []
        for value in l:
            if value not in result:
                result.append(value)

        return result

    def sortAndDeduplicateDict(self, d):
        result = {}
        for key, value in d.items():
            if value not in result.values():
                result[key] = value

        return result

    def joinPartitions(self, adjacentComponents, finalNoClusters):

        partitionsList = [[element[0:self.noDims] for element in adjacentComponents[k]] for k in adjacentComponents] 
        partitions = dict(zip(range(len(partitionsList)), partitionsList))

        distBetweenPartitionsCache = {}
        distancesIndices = self.computeDistanceIndices(partitions, distBetweenPartitionsCache)

        while len(partitions) > finalNoClusters:

            smallestDistancesIndex = distancesIndices[0]

            (j, i) = self.indexToCoords(smallestDistancesIndex, len(partitions), len(partitions))
            partitionToAdd = partitions[i] + partitions[j]
            partitionToAdd = self.sortAndDeduplicate(partitionToAdd)

            if (i in partitions):
                del partitions[i]

            if (j in partitions):
                del partitions[j]

            if ((i, j) in distBetweenPartitionsCache):
                del distBetweenPartitionsCache[(i, j)]

            (partitions, distBetweenPartitionsCache, newDictIdx) = self.rebuildDictIndexes(partitions, distBetweenPartitionsCache)

            partitions[newDictIdx] = partitionToAdd

            distancesIndices = self.computeDistanceIndices(
                partitions, distBetweenPartitionsCache)

        if (self.noDims == 2 and self.debugMode == 1):
            for k in partitions:
                c = self.randomColorScaled()
                for element in partitions[k]:
                    plt.scatter(element[0], element[1], color=c)
            plt.show()

        return partitions

    def computePdfKdeScipy(self, eachDimensionValues):
        '''
            compute pdf and its values for elements in dataset
        '''
        kernel = st.gaussian_kde(eachDimensionValues, bw_method='scott')
        pdf = kernel.evaluate(eachDimensionValues)

        return pdf

    def computePdfKdeSklearn(self, dataset):
        '''
        compute pdf and its values for elements in dataset
        '''
        bwSklearn = estimate_bandwidth(dataset)
        print("bwSklearn este " + str(bwSklearn))
        kde = KernelDensity(kernel='gaussian',
                            bandwidth=bwSklearn).fit(dataset)
        logPdf = kde.score_samples(dataset)
        pdf = np.exp(logPdf)
        return pdf

    def computePdfKde(self, dataset, eachDimensionValues):
        try:
            pdf = self.computePdfKdeScipy(eachDimensionValues)
        except np.linalg.LinAlgError:
            pdf = self.computePdfKdeSklearn(dataset)
        finally:
            return pdf

    def evaluatePdfKdeScipy(self, eachDimensionValues):
        '''
            pdf evaluation scipy - only for two dimensions, it generates the blue density levels plot
        '''
        x = []
        y = []

        x = eachDimensionValues[0]
        y = eachDimensionValues[1]

        xmin = min(x) - 2
        xmax = max(x) + 2

        ymin = min(y) - 2
        ymax = max(y) + 2

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)  # bw_method=

        scottFact = kernel.scotts_factor()
        print("who is scott eval? " + str(scottFact))

        f = np.reshape(kernel(positions).T, xx.shape)
        return f, xmin, xmax, ymin, ymax, xx, yy

    def randomColorScaled(self):
        b = randint(0, 255)
        g = randint(0, 255)
        r = randint(0, 255)
        return [round(b / 255, 2), round(g / 255, 2), round(r / 255, 2)]

    def DistFunc(self, x, y):

        sum_powers = 0
        for dim in range(self.noDims):
            sum_powers = math.pow(x[dim] - y[dim], 2) + sum_powers
        return math.sqrt(sum_powers)

    def centroid(self, objects):

        sumEachDim = {}
        for dim in range(self.noDims):
            sumEachDim[dim] = 0

        for object in objects:
            for dim in range(self.noDims):
                sumEachDim[dim] = sumEachDim[dim] + object[dim]

        centroidCoords = []
        for sumId in sumEachDim:
            centroidCoords.append(round(sumEachDim[sumId] / len(objects), 2))

        centroidCoords = tuple(centroidCoords)

        return centroidCoords

    def outliersIqr(self, ys):
        '''
            Outliers detection with IQR
        '''
        quartile1, quartile3 = np.percentile(ys, [25, 75])
        iqr = quartile3 - quartile1
        lowerBound = quartile1 - (iqr * 1.5)
        outliersIqr = []
        outliersIqr = [idx for idx in range(len(ys)) if ys[idx] < lowerBound]
        return outliersIqr

    def calculateAveragePairwise(self, cluster1, cluster2):

        pairwiseDistances = []

        for element1 in cluster1:
            for element2 in cluster2:
                pairwiseDistances.append(self.DistFunc(element1, element2))

        avgPairwise = sum(pairwiseDistances) / len(pairwiseDistances)
        return avgPairwise

    def calculateSmallestPairwise(self, cluster1, cluster2):

        minPairwise = 999999
        for element1 in cluster1:
            for element2 in cluster2:
                if (element1 != element2):
                    distBetween = self.DistFunc(element1, element2)
                    if (distBetween < minPairwise):
                        minPairwise = distBetween
        return minPairwise

    def calculateCentroid(self, cluster1, cluster2):
        centroid1 = self.centroid(cluster1)
        centroid2 = self.centroid(cluster2)

        dist = self.DistFunc(centroid1, centroid2)

        return dist

    def computeDistanceMatrix(self, elements):

        elementsNr = len(elements)

        # compute distance matrix
        distanceMatrix = np.zeros((elementsNr, elementsNr))

        for elementId1 in range(elementsNr):
            for elementId2 in range(elementsNr):
                distanceMatrix[elementId1][elementId2] = self.DistFunc(elements[elementId1], elements[elementId2])
        return distanceMatrix

    def getDistancesToKthNeigh(self, kthNeigh, elements):

        distanceMatrix = self.computeDistanceMatrix(elements)

        distanceMatrix.sort(axis=1)

        # compute distances to closest K neigh
        distancesToClosestK = distanceMatrix[:, kthNeigh] 

        # return k distances, sorted descending
        return np.array(sorted(distancesToClosestK, reverse=True))

    def getCorrectRadius(self, elementsPartition):

        justRelevantDimensions = np.array([element[0:self.noDims] for element in elementsPartition])

        ns = 2 * self.noDims - 1

        # distances to the kth nearest neighbor, sorted
        distanceDec = self.getDistancesToKthNeigh(ns - 1, justRelevantDimensions)

        # get inflection element of the distanceDec plot
        maxSlopeIdx = np.argmax(distanceDec[:-1] - distanceDec[1:])

        return distanceDec[maxSlopeIdx]

    def getClosestKNeigh(self, elementId, elementsPartition, closestMean):
        '''
            Get one element's closest v neighbours
            v is not a constant!! for each element you keep adding neighbours
            untill the distance from the next neigbour and the element is larger than
            expand_factor * closestMean
        '''

        justUsefulDimensions = np.array([element[0:self.noDims] for element in elementsPartition])

        radius = self.expandFactor * closestMean

        neighIdsToDistances = {}

        for elementId2 in range(len(justUsefulDimensions)):
            if (elementId == elementId2):
                continue
            neighIdsToDistances[elementId2] = self.DistFunc(justUsefulDimensions[elementId], justUsefulDimensions[elementId2])

        closestKNeigh = []

        for key, distance in neighIdsToDistances.items():
            if distance <= radius:
                closestKNeigh.append(key)

        return closestKNeigh

    def expandKnn(self, elementId, elementsPartition):
        '''
            Extend current cluster
            Take the current element's nearest v neighbours
            Add them to the cluster
            Take the v neighbours of the v neighbours and add them to the cluster
            When you can't expand anymore start new cluster
        '''

        closestMean = self.getCorrectRadius(elementsPartition)
        neighIds = self.getClosestKNeigh(elementId, elementsPartition, closestMean)

        if (len(neighIds) > 0):
            elementsPartition[elementId][self.noDims] = self.idCluster
            elementsPartition[elementId][self.noDims + 1] = self.ALREADY_PARSED_TRUE

            for neighId in neighIds:
                if (elementsPartition[neighId][self.noDims + 1] == self.ALREADY_PARSED_FALSE):
                    self.expandKnn(neighId, elementsPartition)
        else:
            elementsPartition[elementId][self.noDims] = -1
            elementsPartition[elementId][self.noDims + 1] = self.ALREADY_PARSED_TRUE

    def splitDensityBins(self, densityBins):

        print("Expand factor " + str(self.expandFactor))

        noise = []
        noClustersPartition = 1
        partId = 0
        adjacentComponents = collections.defaultdict(list)

        for k in densityBins:

            # EXPANSION STEP
            self.idCluster = -1
            densityBin = densityBins[k]

            closestMean = self.getCorrectRadius(densityBin)

            for elementId in range(len(densityBin)):
                
                if (densityBin[elementId][self.noDims] == -1):
                    self.idCluster = self.idCluster + 1
                    noClustersPartition = noClustersPartition + 1
                    densityBin[elementId][self.noDims + 1] = self.ALREADY_PARSED_TRUE
                    densityBin[elementId][self.noDims] = self.idCluster
                    
                    neighIds = self.getClosestKNeigh(elementId, densityBin, closestMean)

                    for neighId in neighIds:
                        if (densityBin[neighId][self.noDims] == -1):
                            densityBin[neighId][self.noDims + 1] = self.ALREADY_PARSED_TRUE
                            densityBin[neighId][self.noDims] = self.idCluster
                            self.expandKnn(neighId, densityBin)

            # ARRANGE STEP
            # create partitions
            intermediaryAdjacentComponents = collections.defaultdict(list)
            clusterId = 0
            for clusterId in range(noClustersPartition):
                intermediaryAdjacentComponents[clusterId] = [element for element in densityBin if element[self.noDims] == clusterId]

            noise += [element for element in densityBin if element[self.noDims] == -1]

            # filter partitions - eliminate the ones with a single element and add them to the noise list
            keysToDelete = []
            for k in intermediaryAdjacentComponents:
                if (len(intermediaryAdjacentComponents[k]) <= 1):
                    keysToDelete.append(k)
                    # we save these elements and assign them to the closest cluster
                    if (len(intermediaryAdjacentComponents[k]) > 0):
                        noise += [element for element in intermediaryAdjacentComponents[k]]

            for k in keysToDelete:
                del intermediaryAdjacentComponents[k]

            # reindex dict
            intermediaryAdjacentComponentsFiltered = dict(zip(range(len(intermediaryAdjacentComponents)), list(intermediaryAdjacentComponents.values())))

            for partIdInner in intermediaryAdjacentComponentsFiltered:
                adjacentComponents[partId] = intermediaryAdjacentComponentsFiltered[partIdInner]
                partId = partId + 1

        return (adjacentComponents, noise)

    def addNoiseToFinalPartitions(self, noise, joinedPartitions):
        noiseToPartition = collections.defaultdict(list)
        # reassign the noise to the class that contains the nearest neighbor
        for noiseElement in noise:
            # determine which is the closest cluster to noiseElement
            closestPartitionIdx = 0
            minDist = 99999
            for k in joinedPartitions:
                dist = self.calculateSmallestPairwise(
                    [noiseElement], joinedPartitions[k])
                if (dist < minDist):
                    closestPartitionIdx = k
                    minDist = dist
            noiseToPartition[closestPartitionIdx].append(noiseElement)

        for joinedPartId in noiseToPartition:
            for noiseElement in noiseToPartition[joinedPartId]:
                joinedPartitions[joinedPartId].append(noiseElement)

    def evaluateCluster(self, dataset, clusterElements):

        evaluationDict = {}
        element2cluster = {}
        element2class = {}

        for element in dataset:
            clusterId = self.noDims
            element2class[tuple(element[0:clusterId])] = element[clusterId]

        for clusterId in element2class.values():
            evaluationDict[clusterId] = {}

        idx = 1
        for elem in clusterElements:
            for element in clusterElements[elem]:
                indexDict = []
                for dim in range(self.noDims):
                    indexDict.append(element[dim])
                element2cluster[tuple(indexDict)] = idx
            for c in evaluationDict:
                evaluationDict[c][idx] = 0
            idx += 1

        for element in element2cluster:
            evaluationDict[element2class[element]][element2cluster[element]] += 1

        print('Purity:  ', evaluation_measures.purity(evaluationDict))
        # perfect results have entropy == 0
        print('Entropy: ', evaluation_measures.entropy(evaluationDict))
        print('RI       ', evaluation_measures.rand_index(evaluationDict))
        print('ARI      ', evaluation_measures.adj_rand_index(evaluationDict))

        f = open("rezultate_evaluare.txt", "a")
        f.write('Purity:  ' + str(evaluation_measures.purity(evaluationDict)) + "\n")
        f.write('Entropy:  ' +
                str(evaluation_measures.entropy(evaluationDict)) + "\n")
        f.write('RI:  ' + str(evaluation_measures.rand_index(evaluationDict)) + "\n")
        f.write(
            'ARI:  ' + str(evaluation_measures.adj_rand_index(evaluationDict)) + "\n")
        f.close()

    def clusterDataset(self, datasetWithClusterIndex):

        # we want to use the dataset without cluster index in our processing
        dataset = np.array(datasetWithClusterIndex)[:, :-1]

        pdf = self.computePdfKde(dataset,
                                 list(np.array(dataset).transpose()))  # calculez functia densitate probabilitate utilizand kde

        '''
        Detect and eliminate outliers
        '''
        outliersIqrPdf = self.outliersIqr(pdf)
        print("We identified " + str(len(outliersIqrPdf)) +
              " outliers from " + str(len(dataset)) + " elements")

        # recompute dataset without outliers
        dataset = [dataset[q]
                   for q in range(len(dataset)) if q not in outliersIqrPdf]

        '''
         Compute dataset pdf
        '''
        pdf = self.computePdfKde(dataset,
                                 list(np.array(dataset).transpose()))  # calculez functia densitate probabilitate din nou

        if(self.noDims == 2 and self.debugMode == 1):
            # plot pdf contour plot
            f, _, _, _, _, xx, yy = self.evaluatePdfKdeScipy(
                list(np.array(dataset).transpose()))  # pentru afisare zone dense albastre
            # pentru afisare zone dense albastre
            plt.contourf(xx, yy, f, cmap='Blues')

        '''
		Split the dataset in density bins
		'''
        _, bins = np.histogram(pdf, bins=self.noBins)

        densityBins = collections.defaultdict(list)

        for idxBin in range((len(bins) - 1)):
            color = self.randomColorScaled()
            for idxElement in range(len(dataset)):
                if (pdf[idxElement] >= bins[idxBin] and pdf[idxElement] <= bins[idxBin + 1]):
                    binElement = []
                    for dim in range(self.noDims):
                        binElement.append(dataset[idxElement][dim])

                    # additional helpful values
                    # the split nearest-neighbour cluster the element belongs to
                    binElement.append(-1)
                    binElement.append(self.ALREADY_PARSED_FALSE)  # was the element already parsed?

                    densityBins[idxBin].append(binElement)

                    # scatter plot for 2d and 3d if debug mode is on
                    if (self.noDims == 2 and self.debugMode == 1):
                        plt.scatter(dataset[idxElement][0],
                                    dataset[idxElement][1], color=color)
                    elif (self.noDims == 3 and self.debugMode == 1):
                        plt.scatter(dataset[idxElement][0], dataset[idxElement][1], dataset[idxElement][2],
                                    color=color)
        if ((self.noDims == 2 or self.noDims == 3) and self.debugMode == 1):
            plt.show()

        '''
		Density levels bins distance split
		'''
        adjacentComponents, noise = self.splitDensityBins(densityBins)  # split the densityBins

        print('noise elements ' + str(len(noise)) +
              ' from ' + str(len(dataset)) + ' elements')

        if (self.noDims == 2 and self.debugMode == 1):
            for k in adjacentComponents:
                color = self.randomColorScaled()
                for element in adjacentComponents[k]:
                    plt.scatter(element[0], element[1], color=color)

            plt.show()

        '''
        Joining partitions based on distances
         '''
        joinedPartitions = self.joinPartitions(adjacentComponents, self.noClusters)

        '''
        Adding what was classified as noise to the corresponding partition
        '''
        self.addNoiseToFinalPartitions(noise, joinedPartitions)

        '''
        Evaluate performance
        '''
        self.evaluateCluster(datasetWithClusterIndex, joinedPartitions)
        print("Evaluation")
        print("==============================")

        if (self.noDims == 2 and self.debugMode == 1):
            for k in joinedPartitions:
                c = self.randomColorScaled()
                for element in joinedPartitions[k]:
                    plt.scatter(element[0], element[1], color=c)

            plt.show()

        return joinedPartitions


'''
=============================================
Denlac Algorithm
'''
if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename',
                        help="the filename which contains the dataset")
    parser.add_argument('-nclusters', '--nclusters', type=int,
                        help="the desired number of clusters")
    parser.add_argument('-nbins', '--nbins', type=int,
                        help="the number of density levels of the dataset")
    parser.add_argument('-expFactor', '--expansionFactor', type=float,
                        help="between 0.2 and 1.5 - the level of wideness of the density bins")
    parser.add_argument('-aggMethod', '--agglomerationMethod', type=int,
                        help="1 smallest pairwise (default) or 2 centroidclo", default=1)
    parser.add_argument('-dm', '--debugMode', type=int,
                        help="optional, set to 1 to show debug plots and comments for 2 dimensional datasets", default=0)
    args = parser.parse_args()

    filename = args.filename
    noClusters = int(args.nclusters)  # no clusters
    noBins = int(args.nbins)  # no bins
    # expansion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)
    expandFactor = float(args.expansionFactor)
    aggMethod = int(args.agglomerationMethod)
    debugMode = args.debugMode

    # read from file
    datasetWithClusterIndex = []

    with open(filename) as f:
        content = f.readlines()

    content = [l.strip() for l in content]

    noDims = 0
    for line in content:
        lineParts = line.split(',')
        noDims = len(lineParts) - 1
        datasetObject = []
        for dim in range(noDims):
            datasetObject.append(float(lineParts[dim]))
        # the last number on a line is the clusterId
        clusterId = int(lineParts[noDims])
        datasetObject.append(clusterId)
        datasetWithClusterIndex.append(datasetObject)

    denlacInstance = Denlac(
        noClusters, noBins, expandFactor, noDims, aggMethod, debugMode)
    clusterElements = denlacInstance.clusterDataset(datasetWithClusterIndex)

    end = time.time()
    print('It took ' + str(end - start))
