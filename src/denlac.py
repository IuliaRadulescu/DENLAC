__author__ = "Radulescu Iulia-Maria"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "iulia.radulescu@cs.pub.ro"
__status__ = "Production"

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.neighbors.kde import KernelDensity, KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import estimate_bandwidth

from random import randint
import argparse
import math
import collections
import evaluation_measures
import time

class Denlac:

    def __init__(self, noClusters, noBins, expandFactor, noDims, aggMethod, debugMode):

        self.no_clusters = noClusters
        self.no_bins = noBins
        self.expandFactor = expandFactor  # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)

        self.noDims = noDims
        self.debugMode = debugMode
        self.aggMethod = aggMethod

        self.id_cluster = -1

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

    def joinPartitions(self, final_partitions, finalNoClusters):

        partitions = dict()
        partId = 0

        for k in final_partitions:
            partitions[partId] = list()
            partId = partId + 1

        partId = 0

        for k in final_partitions:
            for pixel in final_partitions[k]:
                kDimensionalPoint = list()
                for kDim in range(self.noDims):
                    kDimensionalPoint.append(pixel[kDim])
                partitions[partId].append(kDimensionalPoint)
            partId = partId + 1

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

            if ((i,j) in distBetweenPartitionsCache):
                del distBetweenPartitionsCache[(i,j)]

            (partitions, distBetweenPartitionsCache, newDictIdx) = self.rebuildDictIndexes(partitions, distBetweenPartitionsCache)

            partitions[newDictIdx] = partitionToAdd

            distancesIndices = self.computeDistanceIndices(partitions, distBetweenPartitionsCache)


        if (self.noDims == 2 and self.debugMode == 1):
            for k in partitions:
                c = self.randomColorScaled()
                for point in partitions[k]:
                    plt.scatter(point[0], point[1], color=c)
            plt.show()

        return partitions

    def computePdfKdeScipy(self, eachDimensionValues):
        '''
		compute pdf and its values for points in dataset_xy
		'''
        kernel = st.gaussian_kde(eachDimensionValues, bw_method='scott')
        pdf = kernel.evaluate(eachDimensionValues)

        return pdf

    def computePdfKdeSklearn(self, datasetXY):
        '''
        compute pdf and its values for points in dataset_xy
        '''
        bwSklearn = estimate_bandwidth(datasetXY)
        print("bwSklearn este " + str(bwSklearn))
        kde = KernelDensity(kernel='gaussian', bandwidth=bwSklearn).fit(datasetXY)
        logPdf = kde.score_samples(datasetXY)
        pdf = np.exp(logPdf)
        return pdf

    def computePdfKde(self, dataset_xy, eachDimensionValues):

        values = np.vstack(eachDimensionValues)
        try: 
            pdf = self.computePdfKdeScipy(eachDimensionValues)
        except np.linalg.LinAlgError:
            pdf = self.computePdfKdeSklearn(dataset_xy)
        finally:
            return pdf

    def evaluatePdfKdeSklearn(self, datasetXY, eachDimensionValues):
        # pdf sklearn
        x = list()
        y = list()

        x = eachDimensionValues[0]
        y = eachDimensionValues[1]

        xmin = min(x) - 2
        xmax = max(x) + 2

        ymin = min(y) - 2
        ymax = max(y) + 2

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        xxRavel = xx.ravel()
        yyRavel = yy.ravel()
        datasetXXYY = list()
        for q in range(len(xxRavel)):
            datasetXXYY.append([xxRavel[q], yyRavel[q]])
        bw_scott = self.computeScipyBandwidth(datasetXY, eachDimensionValues)
        kde = KernelDensity(kernel='gaussian', bandwidth=bw_scott).fit(datasetXY)
        log_pdf = kde.score_samples(datasetXXYY)
        pdf = np.exp(log_pdf)
        f = np.reshape(pdf.T, xx.shape)
        return (f, xmin, xmax, ymin, ymax, xx, yy)

    def evaluatePdfKdeScipy(self, eachDimensionValues):
        '''
		pdf evaluation scipy - only for two dimensions, it generates the blue density levels plot
		'''
        x = list()
        y = list()

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

        centroidCoords = list()
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
        upperBound = quartile3 + (iqr * 1.5)
        outliersIqr = list()
        for idx in range(len(ys)):
            if ys[idx] < lowerBound:
                outliersIqr.append(idx)
        return outliersIqr

    def calculateAveragePairwise(self, cluster1, cluster2):

        average_pairwise = 0
        sum_pairwise = 0
        nr = 0

        for pixel1 in cluster1:
            for pixel2 in cluster2:
                distBetween = self.DistFunc(pixel1, pixel2)
                sum_pairwise = sum_pairwise + distBetween
                nr = nr + 1

        average_pairwise = sum_pairwise / nr
        return average_pairwise

    def calculateSmallestPairwise(self, cluster1, cluster2):

        minPairwise = 999999
        for pixel1 in cluster1:
            for pixel2 in cluster2:
                if (pixel1 != pixel2):
                    distBetween = self.DistFunc(pixel1, pixel2)
                    if (distBetween < minPairwise):
                        minPairwise = distBetween
        return minPairwise

    def calculateCentroid(self, cluster1, cluster2):
        centroid1 = self.centroid(cluster1)
        centroid2 = self.centroid(cluster2)

        dist = self.DistFunc(centroid1, centroid2)

        return dist

    def getCorrectRadius(self, pointsPartition):

        pointsRelevantDimensions = [point[0:self.noDims] for point in pointsPartition]
        pointsRelevantDimensions = np.array(pointsRelevantDimensions)

        ns = 3
        nbrs = NearestNeighbors(n_neighbors=ns).fit(pointsRelevantDimensions)
        distances, indices = nbrs.kneighbors(pointsRelevantDimensions)
        distanceDec = sorted(distances[:, ns - 1], reverse=True)

        distanceDec = np.array(distanceDec)
        maxSlopeIdx = np.argmax(distanceDec[:-1] - distanceDec[1:])

        return distanceDec[maxSlopeIdx]


    def getClosestKNeigh(self, point, id_point, pointsPartition):
        '''
		Get a point's closest v neighbours
		v is not a constant!! for each point you keep adding neighbours
		untill the distance from the next neigbour and the point is larger than
		expand_factor * closestMean (closestMean este calculata de functia anterioara)
		'''

        closestMean = self.getCorrectRadius(pointsPartition)

        pointsJustUsefulDimensions = [point[0:self.noDims] for point in pointsPartition]
        pointsJustUsefulDimensions = np.array(pointsJustUsefulDimensions)

        partitionTree = KDTree(pointsJustUsefulDimensions)

        radius = self.expandFactor * closestMean
        ind = partitionTree.query_radius(pointsJustUsefulDimensions[id_point:id_point+1], radius)

        return list(ind[0])

    def expandKnn(self, point_id, pointsPartition):
        '''
		Extend current cluster
		Take the current point's nearest v neighbours
		Add them to the cluster
		Take the v neighbours of the v neighbours and add them to the cluster
		When you can't expand anymore start new cluster
		'''

        point = pointsPartition[point_id]
        neigh_ids = self.getClosestKNeigh(point, point_id, pointsPartition)

        if (len(neigh_ids) > 0):
            pointsPartition[point_id][self.noDims] = self.id_cluster
            pointsPartition[point_id][self.noDims + 2] = 1
            for neigh_id in neigh_ids:

                if (pointsPartition[neigh_id][self.noDims + 2] == -1):
                    self.expandKnn(neigh_id, pointsPartition)
        else:
            pointsPartition[point_id][self.noDims] = -1
            pointsPartition[point_id][self.noDims + 2] = 1

    def splitPartitions(self, partition_dict):

        print("Expand factor " + str(self.expandFactor))
        noise = list()
        noClustersPartition = 1
        partId = 0
        finalPartitions = collections.defaultdict(list)

        for k in partition_dict:

            # EXPANSION STEP
            self.id_cluster = -1
            pointsPartition = partition_dict[k]

            for pixel_id in range(len(pointsPartition)):
                pixel = pointsPartition[pixel_id]

                if (pointsPartition[pixel_id][self.noDims] == -1):
                    self.id_cluster = self.id_cluster + 1
                    noClustersPartition = noClustersPartition + 1
                    pointsPartition[pixel_id][self.noDims + 2] = 1
                    pointsPartition[pixel_id][self.noDims] = self.id_cluster
                    neigh_ids = self.getClosestKNeigh(pixel, pixel_id, pointsPartition)

                    for neigh_id in neigh_ids:
                        if (pointsPartition[neigh_id][self.noDims] == -1):
                            pointsPartition[neigh_id][self.noDims + 2] = 1
                            pointsPartition[neigh_id][self.noDims] = self.id_cluster
                            self.expandKnn(neigh_id, pointsPartition)

            # ARRANGE STEP
            # create partitions
            innerPartitions = collections.defaultdict(list)
            partIdInner = 0
            for i in range(noClustersPartition):
                innerPartitions[partIdInner] = [pixel for pixel in pointsPartition if pixel[self.noDims] == i]
                partIdInner = partIdInner + 1

            noise += [pixel for pixel in pointsPartition if pixel[self.noDims] == -1]

            # filter partitions - eliminate the ones with a single point and add them to the noise list
            keysToDelete = list()
            for k in innerPartitions:
                if (len(innerPartitions[k]) <= 1):
                    keysToDelete.append(k)
                    # we save these points and assign them to the closest cluster
                    if (len(innerPartitions[k]) > 0):
                        noise += [pixel for pixel in innerPartitions[k]]

            for k in keysToDelete:
                del innerPartitions[k]

            # reindex dict
            innerPartitionsFiltered = dict(zip(range(0, len(innerPartitions)), list(innerPartitions.values())))

            for partIdInner in innerPartitionsFiltered:
                finalPartitions[partId] = innerPartitionsFiltered[partIdInner]
                partId = partId + 1

        return (finalPartitions, noise)

    def addNoiseToFinalPartitions(self, noise, joinedPartitions):
        noise_to_partition = collections.defaultdict(list)
        # reassign the noise to the class that contains the nearest neighbor
        for noise_point in noise:
            # determine which is the closest cluster to noise_point
            closest_partition_idx = 0
            minDist = 99999
            for k in joinedPartitions:
                dist = self.calculateSmallestPairwise([noise_point], joinedPartitions[k])
                if (dist < minDist):
                    closest_partition_idx = k
                    minDist = dist
            noise_to_partition[closest_partition_idx].append(noise_point)

        for joinedPartId in noise_to_partition:
            for noise_point in noise_to_partition[joinedPartId]:
                joinedPartitions[joinedPartId].append(noise_point)

    def evaluateCluster(self, dataset, clusterPoints):

        evaluationDict = {}
        point2cluster = {}
        point2class = {}

        for point in dataset:
            clusterId = self.noDims
            point2class[tuple(point[0:clusterId])] = point[clusterId]

        for clusterId in point2class.values():
            evaluationDict[clusterId] = {}

        idx = 1
        for elem in clusterPoints:
            for point in clusterPoints[elem]:
                indexDict = list()
                for dim in range(self.noDims):
                    indexDict.append(point[dim])
                point2cluster[tuple(indexDict)] = idx
            for c in evaluationDict:
                evaluationDict[c][idx] = 0
            idx += 1

        for point in point2cluster:
            evaluationDict[point2class[point]][point2cluster[point]] += 1

        print('Purity:  ', evaluation_measures.purity(evaluationDict))
        print('Entropy: ', evaluation_measures.entropy(evaluationDict))  # perfect results have entropy == 0
        print('RI       ', evaluation_measures.rand_index(evaluationDict))
        print('ARI      ', evaluation_measures.adj_rand_index(evaluationDict))

        f = open("rezultate_evaluare.txt", "a")
        f.write('Purity:  ' + str(evaluation_measures.purity(evaluationDict)) + "\n")
        f.write('Entropy:  ' + str(evaluation_measures.entropy(evaluationDict)) + "\n")
        f.write('RI:  ' + str(evaluation_measures.rand_index(evaluationDict)) + "\n")
        f.write('ARI:  ' + str(evaluation_measures.adj_rand_index(evaluationDict)) + "\n")
        f.close()

    def clusterDataset(self, dataset):

        # we want to use the dataset without cluster index in our processing
        datasetXY = np.array(dataset)[:,:-1]

        intermediaryPartitionsDict = collections.defaultdict(list)

        pdf = self.computePdfKde(datasetXY,
                                      list(np.array(datasetXY).transpose()))  # calculez functia densitate probabilitate utilizand kde

        '''
        Detect and eliminate outliers
        '''
        outliersIqrPdf = self.outliersIqr(pdf)
        print("We identified " + str(len(outliersIqrPdf)) + " outliers from " + str(len(datasetXY)) + " points")

        # recompute datasetXY, x and y
        datasetXY = [datasetXY[q] for q in range(len(datasetXY)) if q not in outliersIqrPdf]

        '''
         Compute dataset pdf
        '''
        pdf = self.computePdfKde(datasetXY,
                                      list(np.array(datasetXY).transpose()))  # calculez functia densitate probabilitate din nou

        if(self.noDims==2 and self.debugMode == 1):
            #plot pdf contour plot
            f,xmin, xmax, ymin, ymax, xx, yy = self.evaluatePdfKdeScipy(list(np.array(datasetXY).transpose())) #pentru afisare zone dense albastre
            plt.contourf(xx, yy, f, cmap='Blues') #pentru afisare zone dense albastre

        '''
		Split the dataset in density bins
		'''
        pixels_per_bin, bins = np.histogram(pdf, bins=self.no_bins)

        for idxBin in range((len(bins) - 1)):
            color = self.randomColorScaled()
            for idxPoint in range(len(datasetXY)):
                if (pdf[idxPoint] >= bins[idxBin] and pdf[idxPoint] <= bins[idxBin + 1]):
                    element_to_append = list()
                    for dim in range(self.noDims):
                        element_to_append.append(datasetXY[idxPoint][dim])
                    # additional helpful values
                    element_to_append.append(-1)  # the split nearest-neighbour cluster the point belongs to
                    element_to_append.append(pdf[idxPoint])
                    element_to_append.append(-1)  # was the point already parsed?

                    intermediaryPartitionsDict[idxBin].append(element_to_append)

                    # scatter plot for 2d and 3d if debug mode is on
                    if (self.noDims == 2 and self.debugMode == 1):
                        plt.scatter(datasetXY[idxPoint][0], datasetXY[idxPoint][1], color=color)
                    elif (self.noDims == 3 and self.debugMode == 1):
                        plt.scatter(datasetXY[idxPoint][0], datasetXY[idxPoint][1], datasetXY[idxPoint][2],
                                    color=color)
        if ((self.noDims == 2 or self.noDims == 3) and self.debugMode == 1):
            plt.show()

        '''
		Density levels bins distance split
		'''
        final_partitions, noise = self.splitPartitions(intermediaryPartitionsDict)  # functie care scindeaza partitiile

        print('noise points ' + str(len(noise)) + ' from ' + str(len(datasetXY)) + ' points')

        if (self.noDims == 2 and self.debugMode == 1):
            for k in final_partitions:
                color = self.randomColorScaled()
                for pixel in final_partitions[k]:
                    plt.scatter(pixel[0], pixel[1], color=color)

            plt.show()

        '''
        Joining partitions based on distances
         '''
        joinedPartitions = self.joinPartitions(final_partitions, self.no_clusters)

        '''
        Adding what was classified as noise to the corresponding partition
        '''
        self.addNoiseToFinalPartitions(noise, joinedPartitions)

        '''
        Evaluate performance
        '''
        self.evaluateCluster(dataset, joinedPartitions)
        print("Evaluation")
        print("==============================")

        if (self.noDims == 2 and self.debugMode == 1):
            for k in joinedPartitions:
                c = self.randomColorScaled()
                for point in joinedPartitions[k]:
                    plt.scatter(point[0], point[1], color=c)

            plt.show()

        return joinedPartitions

'''
=============================================
Denlac Algorithm
'''
if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', help = "the filename which contains the dataset")
    parser.add_argument('-nclusters', '--nclusters', type = int, help = "the desired number of clusters")
    parser.add_argument('-nbins', '--nbins', type = int, help = "the number of density levels of the dataset")
    parser.add_argument('-expFactor', '--expansionFactor', type = float, help = "between 0.2 and 1.5 - the level of wideness of the density bins")
    parser.add_argument('-aggMethod', '--agglomerationMethod', type=int,
                        help="1 smallest pairwise (default) or 2 centroidclo", default = 1)
    parser.add_argument('-dm', '--debugMode', type = int,
                        help = "optional, set to 1 to show debug plots and comments for 2 dimensional datasets", default = 0)
    args = parser.parse_args()

    filename = args.filename
    no_clusters = int(args.nclusters)  # no clusters
    no_bins = int(args.nbins)  # no bins
    expand_factor = float(args.expansionFactor)  # expansion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)
    aggMethod = int(args.agglomerationMethod)
    debugMode = args.debugMode

    # read from file
    dataset = list()

    with open(filename) as f:
        content = f.readlines()

    content = [l.strip() for l in content]

    noDims = 0
    for l in content:
        aux = l.split(',')
        noDims = len(aux) - 1
        listOfCoords = list()
        for dim in range(noDims):
            listOfCoords.append(float(aux[dim]))
        listOfCoords.append(int(aux[noDims]))
        dataset.append(listOfCoords)

    denlacInstance = Denlac(no_clusters, no_bins, expand_factor, noDims, aggMethod, debugMode)
    cluster_points = denlacInstance.clusterDataset(dataset)

    end = time.time()
    print('It took ' + str(end - start))
