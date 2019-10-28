from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import estimate_bandwidth

import sys
from random import randint
import argparse
import math
import collections
import evaluation_measures

'''
=============================================
FUNCTII AUXILIARE
'''


class Denlac:

    def __init__(self, noClusters, noBins, expandFactor, noDims, debugMode):

        self.no_clusters = noClusters
        self.no_bins = noBins
        self.expandFactor = expandFactor  # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)

        self.noDims = noDims
        self.debugMode = debugMode

        self.id_cluster = -1
        self.pdf = list()

    def upsertToJoinedPartitions(self, keys, partitionToAdd, joinedPartitions):

        upserted = False
        for joinedPartitionsKeys in joinedPartitions:
            if (keys[0] in joinedPartitionsKeys or keys[1] in joinedPartitionsKeys):
                resulting_list = list(joinedPartitions[joinedPartitionsKeys])
                resulting_list.extend(x for x in partitionToAdd if x not in resulting_list)

                joinedPartitions[joinedPartitionsKeys] = resulting_list
                upserted = True

        if (upserted == False):
            joinedPartitions[keys] = partitionToAdd

    def rebuildDictIndexes(self, dictToRebuild, joinedPartitions, mergedIndexes):

        newDict = dict()
        newDictIdx = 0

        for i in dictToRebuild:
            if (i not in mergedIndexes):
                newDict[newDictIdx] = dictToRebuild[i]
                newDictIdx = newDictIdx + 1

        for joinedPartitionsKeys in joinedPartitions:
            newDict[newDictIdx] = joinedPartitions[joinedPartitionsKeys]
            newDictIdx = newDictIdx + 1

        return newDict

    def computeDistanceIndices(self, partitions):

        distances = []

        for i in range(len(partitions)):
            for j in range(len(partitions)):
                if (i == j):
                    distBetweenPartitions = -1
                else:
                    distBetweenPartitions = self.calculateSmallestPairwise(partitions[i], partitions[j])
                distances.append(distBetweenPartitions)

        distances = np.array(distances)

        indicesNegative = np.where(distances < 0)
        distancesIndices = np.argsort(distances)

        finalIndices = [index for index in distancesIndices if index not in indicesNegative[0]]

        return finalIndices

    # i = index, x = amount of columns, y = amount of rows
    def indexToCoords(self, index, columns, rows):

        for i in range(rows):
            # check if the index parameter is in the row
            if (index >= columns * i and index < (columns * i) + columns):
                # return x, y
                return index - columns * i, i

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

        numberOfPartitions = len(partitions)

        distancesIndices = self.computeDistanceIndices(partitions)

        while numberOfPartitions > finalNoClusters:

            joinedPartitions = dict()
            mergedIndexes = list()

            for smallestDistancesIndex in distancesIndices:

                (j, i) = self.indexToCoords(smallestDistancesIndex, len(partitions), len(partitions))
                partitionToAdd = partitions[i] + partitions[j]

                self.upsertToJoinedPartitions((i, j), partitionToAdd, joinedPartitions)

                mergedIndexes.append(i)
                mergedIndexes.append(j)

                numberOfPartitions = numberOfPartitions - 1

                if numberOfPartitions <= finalNoClusters:
                    break
            mergedIndexes = set(mergedIndexes)
            partitions = self.rebuildDictIndexes(partitions, joinedPartitions, mergedIndexes)

            if (self.noDims == 2 and self.debugMode == 1):
                for k in partitions:
                    c = self.randomColorScaled()
                    for point in partitions[k]:
                        plt.scatter(point[0], point[1], color=c)
                plt.show()

            numberOfPartitions = len(partitions)
            distancesIndices = self.computeDistanceIndices(partitions)

        return partitions

    def computePdfKdeScipy(self, eachDimensionValues):
        '''
		compute pdf and its values for points in dataset_xy
		'''
        stackingList = list()
        for dimId in eachDimensionValues:
            stackingList.append(eachDimensionValues[dimId])
        values = np.vstack(stackingList)
        kernel = st.gaussian_kde(values)
        pdf = kernel.evaluate(values)

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

        stackingList = list()
        for dim_id in eachDimensionValues:
            stackingList.append(eachDimensionValues[dim_id])
        values = np.vstack(stackingList)
        kernel = st.gaussian_kde(values)
        print("norm_factor = " + str(kernel._norm_factor))

        if (kernel._norm_factor != 0):
            # not 0, use scipy
            pdf = self.computePdfKdeScipy(eachDimensionValues)
        else:
            # 0, use sklearn
            pdf = self.computePdfKdeSklearn(dataset_xy)
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

    def calculateSmallestPairwise(self, cluster1, cluster2):

        minPairwise = 999999
        for pixel1 in cluster1:
            for pixel2 in cluster2:
                if (pixel1 != pixel2):
                    distBetween = self.DistFunc(pixel1, pixel2)
                    if (distBetween < minPairwise):
                        minPairwise = distBetween
        return minPairwise

    def getClosestMean(self, pointsPartition):
        '''
		The mean of k pairwise distances
		'''

        justPdfs = [point[self.noDims + 1] for point in pointsPartition]
        justPdfs = list(set(justPdfs))

        mean_pdf = sum(justPdfs) / len(justPdfs)
        k = int(math.ceil(0.1 * len(pointsPartition)))
        distances = list()

        # get all points with density above mean
        # take all distances between each point and its closest neighbour
        for point in pointsPartition:
            deja_parsati = list()
            if (point[noDims + 1] > mean_pdf):
                while (k > 0):
                    neigh_id = 0
                    minDist = 99999
                    for id_point_k in range(len(pointsPartition)):
                        point_k = pointsPartition[id_point_k]
                        if (point_k not in deja_parsati):
                            dist = self.DistFunc(point, point_k)
                            if (dist < minDist and dist > 0):
                                minDist = dist
                                neigh_id = id_point_k
                    distances.append(minDist)
                    neigh = pointsPartition[neigh_id]
                    deja_parsati.append(neigh)
                    k = k - 1
        distances = list(set(distances))

        # if no point complies, do this instead (almost never useful)
        if (len(distances) == 0):
            print('UGLY FALLBACK')
            k = int(math.ceil(0.1 * len(pointsPartition)))
            distances = list()
            for point in pointsPartition:
                deja_parsati = list()
                while (k > 0):
                    neigh_id = 0
                    minDist = 99999
                    for id_point_k in range(len(pointsPartition)):
                        point_k = pointsPartition[id_point_k]
                        if (point_k not in deja_parsati):
                            dist = self.DistFunc(point, point_k)
                            if (dist < minDist and dist > 0):
                                minDist = dist
                                neigh_id = id_point_k
                    distances.append(minDist)
                    neigh = pointsPartition[neigh_id]
                    deja_parsati.append(neigh)
                    k = k - 1
            distances = list(set(distances))

        return sum(distances) / len(distances)

    def getClosestKNeigh(self, point, id_point, pointsPartition):
        '''
		Get a point's closest v neighbours
		v is not a constant!! for each point you keep adding neighbours
		untill the distance from the next neigbour and the point is larger than
		expand_factor * closestMean (closestMean este calculata de functia anterioara)
		'''

        neighIds = list()
        distances = list()
        alreadyParsed = list()
        canContinue = 1
        closestMean = self.getClosestMean(pointsPartition)
        while (canContinue == 1):
            minDist = 99999
            neighId = 0
            for idPointK in range(len(pointsPartition)):
                pointK = pointsPartition[idPointK]
                if (pointK not in alreadyParsed):
                    dist = self.DistFunc(point, pointK)
                    if (dist < minDist and dist > 0):
                        minDist = dist
                        neighId = idPointK

            if (minDist <= self.expandFactor * closestMean):
                neigh = pointsPartition[neighId]
                neighIds.append([neighId, neigh])
                distances.append(minDist)

                alreadyParsed.append(neigh)
            else:
                canContinue = 0

        neighIds.sort(key=lambda x: x[1])

        neighIdsFinal = [n_id[0] for n_id in neighIds]

        return neighIdsFinal

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

            innerPartitions = collections.defaultdict(list)
            innerPartitionsFiltered = collections.defaultdict(list)
            partIdInner = 0

            # ARRANGE STEP

            for i in range(noClustersPartition):
                for pixel in pointsPartition:
                    if (pixel[self.noDims] == i):
                        innerPartitions[partIdInner].append(pixel)
                partIdInner = partIdInner + 1

            # add noise too
            for pixel in pointsPartition:
                if (pixel[self.noDims] == -1):
                    innerPartitions[partIdInner].append(pixel)
                    partIdInner = partIdInner + 1

            # filter partitions - eliminate the ones with a single point and add them to the noise list
            keysToDelete = list()
            for k in innerPartitions:
                if (len(innerPartitions[k]) <= 1):
                    keysToDelete.append(k)
                    # we save these points and assign them to the closest cluster
                    if (len(innerPartitions[k]) > 0):
                        for pinner in innerPartitions[k]:
                            noise.append(pinner)
            for k in keysToDelete:
                del innerPartitions[k]

            partIdFiltered = 0
            for part_id_k in innerPartitions:
                innerPartitionsFiltered[partIdFiltered] = innerPartitions[part_id_k]
                partIdFiltered = partIdFiltered + 1

            for partIdInner in innerPartitionsFiltered:
                finalPartitions[partId] = innerPartitionsFiltered[partIdInner]
                partId = partId + 1

        return (finalPartitions, noise)

    def evaluateCluster(self, clasePoints, clusterPoints):

        evaluationDict = {}
        point2cluster = {}
        point2class = {}

        idx = 0
        for elem in clasePoints:
            evaluationDict[idx] = {}
            for points in clasePoints[elem]:
                point2class[points] = idx
            idx += 1

        idx = 0
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

    def clusterDataset(self, datasetXY, datasetXYvalidate, eachDimensionValues, pointClasses):

        partition_dict = collections.defaultdict(list)

        self.pdf = self.computePdfKde(datasetXY,
                                      eachDimensionValues)  # calculez functia densitate probabilitate utilizand kde

        # detect and eliminate outliers

        outliers_iqr_pdf = self.outliersIqr(self.pdf)
        print("We identified " + str(len(outliers_iqr_pdf)) + " outliers from " + str(len(datasetXY)) + " points")

        # recompute datasetXY, x and y
        datasetXY = [datasetXY[q] for q in range(len(datasetXY)) if q not in outliers_iqr_pdf]
        datasetXYvalidate = [datasetXY[q] for q in range(len(datasetXY))]
        for dim in range(noDims):
            eachDimensionValues[dim] = [datasetXY[q][dim] for q in range(len(datasetXY))]

        # recalculez pdf, ca altfel se produc erori

        self.pdf = self.computePdfKde(datasetXY,
                                      eachDimensionValues)  # calculez functia densitate probabilitate din nou

        if(self.noDims==2 and self.debugMode == 1):
            #plot pdf contour plot
            f,xmin, xmax, ymin, ymax, xx, yy = self.evaluatePdfKdeScipy(eachDimensionValues) #pentru afisare zone dense albastre
            plt.contourf(xx, yy, f, cmap='Blues') #pentru afisare zone dense albastre

        '''
		Split the dataset in density levels
		'''
        pixels_per_bin, bins = np.histogram(self.pdf, bins=self.no_bins)

        for idxBin in range((len(bins) - 1)):
            color = self.randomColorScaled()
            for idxPoint in range(len(datasetXY)):
                if (self.pdf[idxPoint] >= bins[idxBin] and self.pdf[idxPoint] <= bins[idxBin + 1]):
                    element_to_append = list()
                    for dim in range(self.noDims):
                        element_to_append.append(datasetXY[idxPoint][dim])
                    element_to_append.append(-1)  # clusterul nearest neighbour din care face parte punctul
                    element_to_append.append(self.pdf[idxPoint])
                    element_to_append.append(-1)  # daca punctul e deja parsta nearest neighbour
                    element_to_append.append(idxPoint)
                    element_to_append.append(datasetXYvalidate[idxPoint])
                    partition_dict[idxBin].append(element_to_append)
                    # scatter doar pentru 2 sau 3 dimensiuni
                    if (self.noDims == 2 and self.debugMode == 1):
                        plt.scatter(datasetXY[idxPoint][0], datasetXY[idxPoint][1], color=color)
                    elif (self.noDims == 3 and self.debugMode == 1):
                        plt.scatter(datasetXY[idxPoint][0], datasetXY[idxPoint][1], datasetXY[idxPoint][2],
                                    color=color)
        if ((self.noDims == 2 or self.noDims == 3) and self.debugMode == 1):
            plt.show()

        '''
		Density levels distance split
		'''

        final_partitions, noise = self.splitPartitions(partition_dict)  # functie care scindeaza partitiile

        print('noise points ' + str(len(noise)) + ' from ' + str(len(datasetXY)) + ' points')

        if (self.noDims == 2 and self.debugMode == 1):
            for k in final_partitions:
                color = self.randomColorScaled()
                for pixel in final_partitions[k]:
                    plt.scatter(pixel[0], pixel[1], color=color)

            plt.show()

        joinedPartitions = self.joinPartitions(final_partitions, self.no_clusters)

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

        self.evaluateCluster(pointClasses, joinedPartitions)
        print("Evaluation")
        print("==============================")

        if (self.noDims == 2 and self.debugMode == 1):
            # plt.contourf(xx, yy, f, cmap='Blues')
            # final plot
            for k in joinedPartitions:
                c = self.randomColorScaled()
                for point in joinedPartitions[k]:
                    plt.scatter(point[0], point[1], color=c)

            plt.show()

        return joinedPartitions

    def plot_clusters(self, cluster_points, set_de_date, color_list):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sorted_ids = list()
        cluster_ids_sorted = {}
        l = 0
        for k in sorted(cluster_points, key=lambda k: len(cluster_points[k]), reverse=True):
            sorted_ids.append(k)
            cluster_ids_sorted[k] = l
            l = l + 1

        for cluster_id in sorted_ids:
            color = color_list[cluster_ids_sorted[cluster_id]]
            for point in cluster_points[cluster_id]:
                ax.scatter(point[0], point[1], color=color)

        fig.savefig('F:\\IULIA\\GITHUB_denlac\\denlac\\results\\poze2\\denlac' + '_' + str(
            set_de_date) + '.png')  # save the figure to file
        plt.close(fig)

    def return_generated_colors(self):
        colors = [[0.4983913408111469, 0.9878468789867579, 0.6660097921680713],
                  [0.9744941631787404, 0.2832566337094712, 0.9879204118216028],
                  [0.2513270277379317, 0.2743083421568847, 0.24523147335002599],
                  [0.9449152611869482, 0.6829811084805801, 0.23098727325934598],
                  [0.2930994694413758, 0.4447870676048005, 0.9360225619487069],
                  [0.7573766048982865, 0.3564335977711406, 0.5156761252908519],
                  [0.7856267252783685, 0.8893618277470249, 0.9998901678967227],
                  [0.454408739644873, 0.6276300415432641, 0.44436302877623274],
                  [0.5960549019562876, 0.9169447263679981, 0.23343224756103573],
                  [0.5043076141852516, 0.24928662375540336, 0.783126632292948],
                  [0.9247167854639711, 0.8850738215338994, 0.5660824976182026],
                  [0.6968162201133189, 0.5394098658486699, 0.8777137989623846],
                  [0.24964251456446662, 0.8062739995395578, 0.7581261497155073],
                  [0.2575944036656022, 0.7915937407896246, 0.2960661699553983],
                  [0.6437636915214084, 0.4266693349653669, 0.23677001364815042],
                  [0.23112259938541102, 0.32175446177894845, 0.645224195428065],
                  [0.7243345083671118, 0.753389424009313, 0.6716029761309434],
                  [0.9722842730592992, 0.47349469240107894, 0.4282317021959992],
                  [0.9487569650924492, 0.6891786532046004, 0.9520338320784278],
                  [0.5051885381513164, 0.7452481002341962, 0.953601834451638],
                  [0.39319970873496335, 0.5150008439629207, 0.6894464075507598],
                  [0.9907888356008789, 0.3349550392437493, 0.6631372416723879],
                  [0.8941331011073401, 0.23083104173874827, 0.3338481968809],
                  [0.995585861422136, 0.9539037035322647, 0.8814571710677304],
                  [0.3229010345744149, 0.9929405485082905, 0.9199797840228496],
                  [0.8587274228303506, 0.23960128391504704, 0.7796299268247159],
                  [0.9755623661339603, 0.9736967761902182, 0.3368365287453637],
                  [0.26070353957125486, 0.6611108693105839, 0.5626778400435902],
                  [0.33209253309750436, 0.9376441530076292, 0.47506002838287176],
                  [0.8388207042685366, 0.6295035956243679, 0.5353583425079034],
                  [0.3222337347709434, 0.40224067198150343, 0.40979789009079776],
                  [0.6442372806094001, 0.26292344132349454, 0.9763078755323873],
                  [0.7668883074119105, 0.8486492161433142, 0.3841638241303332],
                  [0.5216210516737045, 0.27506979815845595, 0.39564388714836696],
                  [0.6036371225021209, 0.5467800941023466, 0.5990844069213549],
                  [0.5988470728143217, 0.8689413295622888, 0.5609526743224205],
                  [0.8935152630682563, 0.5596944902716602, 0.7784415487870969],
                  [0.686841264479984, 0.9412597573588116, 0.849613972582678],
                  [0.400134697318114, 0.5384071943290534, 0.24536921682148846],
                  [0.5304620100522262, 0.6770501903569319, 0.718601456418752]]

        return colors


'''
=============================================
Denlac Algorithm
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', help = "the filename which contains the dataset")
    parser.add_argument('-nclusters', '--nclusters', type = int, help = "the desired number of clusters")
    parser.add_argument('-nbins', '--nbins', type = int, help = "the number of density levels of the dataset")
    parser.add_argument('-expFactor', '--expansionFactor', type = float, help = "between 0.2 and 1.5 - the level of wideness of the density bins")
    parser.add_argument('-dm', '--debugMode', type = int,
                        help = "optional, set to 1 to show debug plots and comments", default = 0)
    args = parser.parse_args()

    filename = args.filename
    no_clusters = int(args.nclusters)  # no clusters
    no_bins = int(args.nbins)  # no bins
    expand_factor = float(args.expansionFactor)  # expansion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)
    debugMode = args.debugMode

    # read from file
    each_dimension_values = collections.defaultdict(list)
    datasetXY = list()
    datasetXYValidate = list()
    pointsClasses = collections.defaultdict(list)

    with open(filename) as f:
        content = f.readlines()

    content = [l.strip() for l in content]

    noDims = 0
    for l in content:
        aux = l.split(',')
        noDims = len(aux) - 1
        for dim in range(noDims):
            each_dimension_values[dim].append(float(aux[dim]))
        listOfCoords = list()
        for dim in range(noDims):
            listOfCoords.append(float(aux[dim]))
        datasetXY.append(listOfCoords)
        datasetXYValidate.append(int(aux[noDims]))
        pointsClasses[int(aux[noDims])].append(tuple(listOfCoords))

    denlacInstance = Denlac(no_clusters, no_bins, expand_factor, noDims, debugMode)
    cluster_points = denlacInstance.clusterDataset(datasetXY, datasetXYValidate, each_dimension_values, pointsClasses)
