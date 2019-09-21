from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import estimate_bandwidth

import sys
import os
from random import randint
from random import shuffle
import math
import collections
import evaluation_measures


'''
=============================================
FUNCTII AUXILIARE
'''

class Denlac:

	def __init__(self, no_clusters, no_bins, expand_factor):
		self.no_clusters = no_clusters 
		self.no_bins = no_bins
		self.expand_factor = expand_factor # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)

		self.id_cluster = -1
		self.points_partition = list()
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
				if (i==j):
					distBetweenPartitions = -1
				else:
					distBetweenPartitions = self.calculateSmallestPairwise(partitions[i], partitions[j])
				distances.append(distBetweenPartitions)

		distances = np.array(distances)

		indicesNegative = np.where(distances < 0)
		distancesIndices = np.argsort(distances)

		finalIndices = [index for index in distancesIndices if index not in indicesNegative[0]]

		return finalIndices

	#i = index, x = amount of columns, y = amount of rows
	def indexToCoords(self, index, columns, rows):

		for i in range(rows):
			print('i = '+ str(i) + 'columns = '+ str(columns) + 'index = ' + str(index))
    	#check if the index parameter is in the row
			if (index >= columns * i and index < (columns * i) + columns):
        		#return x, y
				print("iese")
				return (index - columns * i, i);

	def joinPartitions(self, finalPartitions, finalNoClusters):

		partitions = dict()
		partId = 0

		for k in finalPartitions:
			partitions[partId] = list()
			partId = partId + 1

		partId = 0

		for k in finalPartitions:
			for point in finalPartitions[k]:
				kDimensionalPoint = list()
				for kDim in range(self.no_dims):
					kDimensionalPoint.append(point[kDim])
				partitions[partId].append(kDimensionalPoint)
			partId = partId + 1

		print('initial len '+str(len(partitions)))

		print(partitions)

		numberOfPartitions = len(partitions)

		distancesIndices = self.computeDistanceIndices(partitions)

		while numberOfPartitions > finalNoClusters:
			
			joinedPartitions = dict()
			mergedIndexes = list()

			for smallestDistancesIndex in distancesIndices:

				(j, i) = self.indexToCoords(smallestDistancesIndex, len(partitions), len(partitions))
					
				print('i = '+str(i) + 'j = ' + str(j)+ ' len partitions '+str(len(partitions)))
				partitionToAdd = partitions[i] + partitions[j]

				self.upsertToJoinedPartitions((i, j), partitionToAdd, joinedPartitions)

				mergedIndexes.append(i)
				mergedIndexes.append(j)

				print("in if "+str(len(partitions)))

				numberOfPartitions = numberOfPartitions - 1

				if numberOfPartitions <= finalNoClusters:
					break;

			mergedIndexes = set(mergedIndexes)
			partitions = self.rebuildDictIndexes(partitions, joinedPartitions, mergedIndexes)

			if(self.no_dims==2):
				#plt.contourf(xx, yy, f, cmap='Blues')
				#final plot
				for k in partitions:
					c = self.randomColorScaled()
					for point in partitions[k]:
						plt.scatter(point[0], point[1], color=c)
				plt.show()

			numberOfPartitions = len(partitions)
			distancesIndices = self.computeDistanceIndices(partitions)

		return partitions
	
	def computeKDE(self, X, each_dimension_values):
		
		stacking_list = list()
		for dim_id in each_dimension_values:
			stacking_list.append(each_dimension_values[dim_id])
		values = np.vstack(stacking_list)
		kernel = st.gaussian_kde(values)
		print("norm_factor = "+str(kernel._norm_factor))
		pdf = []
		if(kernel._norm_factor!=0):
			#not 0, use scipy
			'''
			compute pdf and its values for points in dataset X
			'''
			stacking_list = list()
			for dim_id in each_dimension_values:
				stacking_list.append(each_dimension_values[dim_id])
			values = np.vstack(stacking_list)
			kernel = st.gaussian_kde(values) 
			pdf = kernel.evaluate(values)
		else:
			#0, use sklearn
			bw_sklearn = estimate_bandwidth(X)
			print("bw_sklearn este "+str(bw_sklearn))
			kde = KernelDensity(kernel='gaussian', bandwidth=bw_sklearn).fit(X)
			log_pdf = kde.score_samples(X)
			pdf = np.exp(log_pdf)
		return pdf

	def randomColorScaled(self):
		b = randint(0, 255)
		g = randint(0, 255)
		r = randint(0, 255)
		return [round(b/255,2), round(g/255,2), round(r/255,2)]

	def distanceFunction(self, x, y):
		# Euclidean distance
		sum_powers = 0
		for dim in range(self.no_dims):
			sum_powers += math.pow(x[dim]-y[dim], 2)
		return math.sqrt(sum_powers)
	

	def centroid(self, points):
		
		sum_each_dim = {}
		for dim in range(self.no_dims):
			sum_each_dim[dim] = 0

		for point in points:
			for dim in range(self.no_dims):
				sum_each_dim[dim] += point[dim]
		
		centroid_coords = list()
		for sum_id in sum_each_dim:
			centroid_coords.append(round(sum_each_dim[sum_id]/len(points), 2))

		centroid_coords = tuple(centroid_coords)

		return centroid_coords

		
	def computeOutliersIQR(self, ys):
		'''
		Outliers detection with IQR
		'''
		quartile_1, quartile_3 = np.percentile(ys, [25, 75])
		iqr = quartile_3 - quartile_1
		lower_bound = quartile_1 - (iqr * 1.5)
		upper_bound = quartile_3 + (iqr * 1.5)
		outliers_iqr = list()
		for idx in range(len(ys)):
			if ys[idx] < lower_bound:
				outliers_iqr.append(idx)
		return outliers_iqr

	def getClosestMean(self):
		'''
		The mean of k pairwise distances
		'''

		just_pdfs = [point[self.no_dims+1] for point in self.points_partition]
		just_pdfs = list(set(just_pdfs))

		mean_pdf = sum(just_pdfs)/len(just_pdfs)
		k=int(math.ceil(0.1*len(self.points_partition)))
		distances = list()

		for point in self.points_partition:
			deja_parsati = list()
			if(point[no_dims+1] > mean_pdf):
				while(k>0):
					neigh_id = 0
					minDist = 99999
					for id_point_k in range(len(self.points_partition)):
						point_k = self.points_partition[id_point_k]
						if(point_k not in deja_parsati):
							dist = self.distanceFunction(point, point_k)
							if(dist < minDist and dist > 0):
								minDist = dist
								neigh_id = id_point_k
					distances.append(minDist)
					neigh = self.points_partition[neigh_id]
					deja_parsati.append(neigh)
					k=k-1
		distances = list(set(distances))

		if(len(distances)==0):
			k=int(math.ceil(0.1*len(self.points_partition)))
			distances = list()
			for point in self.points_partition:
				deja_parsati = list()
				while(k>0):
					neigh_id = 0
					minDist = 99999
					for id_point_k in range(len(self.points_partition)):
						point_k = self.points_partition[id_point_k]
						if(point_k not in deja_parsati):
							dist = self.distanceFunction(point, point_k)
							if(dist < minDist and dist > 0):
								minDist = dist
								neigh_id = id_point_k
					distances.append(minDist)
					neigh = self.points_partition[neigh_id]
					deja_parsati.append(neigh)
					k=k-1
			distances = list(set(distances))


		return sum(distances)/len(distances)


	def getClosestKNeighbors(self, point, id_point):
		'''
		Get a point's closest v neighbours
		v is not a constant!! for each point you keep adding neighbours
		untill the distance from the next neigbour and the point is larger than
		expand_factor * closest_mean (closest_mean este calculata de functia anterioara)
		'''
		
		neigh_ids = list()
		distances = list()
		deja_parsati = list()
		pot_continua = 1
		closest_mean = self.getClosestMean()
		while(pot_continua==1):
			minDist = 99999
			neigh_id = 0
			for id_point_k in range(len(self.points_partition)):
				point_k = self.points_partition[id_point_k]
				if(point_k not in deja_parsati):
					dist = self.distanceFunction(point, point_k)
					if(dist < minDist and dist > 0):
						minDist = dist
						neigh_id = id_point_k
			

			if(minDist <= expand_factor*closest_mean):
				neigh = self.points_partition[neigh_id]
				neigh_ids.append([neigh_id, neigh])
				distances.append(minDist)
				
				deja_parsati.append(neigh)
			else:
				pot_continua = 0
			
		neigh_ids.sort(key=lambda x: x[1])

		neigh_ids_final = [n_id[0] for n_id in neigh_ids]

		return neigh_ids_final


	def expandKNN(self, point_id):
		'''
		Extend current cluster
		Take the current point's nearest v neighbours 
		Add them to the cluster
		Take the v neighbours of the v neighbours and add them to the cluster
		When you can't expand anymore start new cluster
		'''

		point = self.points_partition[point_id]
		neigh_ids = self.getClosestKNeighbors(point, point_id)
		
		if(len(neigh_ids)>0):
			self.points_partition[point_id][self.no_dims] = self.id_cluster
			self.points_partition[point_id][self.no_dims+2] = 1
			for neigh_id in neigh_ids:
				
				if(self.points_partition[neigh_id][self.no_dims+2]==-1):
					self.expandKNN(neigh_id)
		else:
			self.points_partition[point_id][self.no_dims] = -1
			self.points_partition[point_id][self.no_dims+2] = 1
			

	def calculateSmallestPairwise(self, cluster1, cluster2):

		min_pairwise = 999999
		for point1 in cluster1:
			for point2 in cluster2:
				if(point1!=point2):
					distBetween = self.distanceFunction(point1, point2)
					if(distBetween < min_pairwise):
						min_pairwise = distBetween
		return min_pairwise


	def splitPartitions(self, partition_dict):

		print("Expand factor "+str(self.expand_factor))
		noise = list()
		no_clusters_partition = 1
		part_id=0
		finalPartitions = collections.defaultdict(list)

		for k in partition_dict:
			self.points_partition = partition_dict[k]

			self.id_cluster = -1

			for point_id in range(len(self.points_partition)):
				point = self.points_partition[point_id]
				
				if(self.points_partition[point_id][self.no_dims]==-1):
					self.id_cluster = self.id_cluster + 1
					no_clusters_partition = no_clusters_partition + 1
					self.points_partition[point_id][self.no_dims+2] = 1
					self.points_partition[point_id][self.no_dims] = self.id_cluster
					neigh_ids = self.getClosestKNeighbors(point, point_id)
					
					for neigh_id in neigh_ids:
						if(self.points_partition[neigh_id][self.no_dims]==-1):
							self.points_partition[neigh_id][self.no_dims+2]=1
							self.points_partition[neigh_id][self.no_dims]=self.id_cluster
							self.expandKNN(neigh_id)
						
			inner_partitions = collections.defaultdict(list)
			inner_partitions_filtered = collections.defaultdict(list)
			part_id_inner = 0

			for i in range(no_clusters_partition):
				for point in self.points_partition:
					if(point[self.no_dims]==i):
						inner_partitions[part_id_inner].append(point)
				part_id_inner = part_id_inner+1
			#adaug si zgomotul
			for point in self.points_partition:
				if(point[self.no_dims]==-1):
					inner_partitions[part_id_inner].append(point)
					part_id_inner = part_id_inner+1
					

			#filter partitions - le elimin pe cele care contin un singur punct
			keys_to_delete = list()
			for k in inner_partitions:
				if(len(inner_partitions[k])<=1):
					keys_to_delete.append(k)
					#salvam aceste puncte si le reasignam la sfarsit celui mai apropiat cluster
					if(len(inner_partitions[k])>0):
						for pinner in inner_partitions[k]:
							noise.append(pinner)
			for k in keys_to_delete:
				del inner_partitions[k]

			part_id_filtered = 0
			for part_id_k in inner_partitions:
				inner_partitions_filtered[part_id_filtered] = inner_partitions[part_id_k]
				part_id_filtered = part_id_filtered + 1


			for part_id_inner in inner_partitions_filtered:
				finalPartitions[part_id] = inner_partitions_filtered[part_id_inner]
				part_id = part_id + 1

		return (finalPartitions, noise)


	def evaluateCluster(self, clase_points, cluster_points):
		
		evaluation_dict = {}
		point2cluster = {}
		point2class = {}

		idx = 0
		for elem in clase_points:
			evaluation_dict[idx] = {}
			for points in clase_points[elem]:
				point2class[points] = idx
			idx += 1

		idx = 0
		for elem in cluster_points:
			for point in cluster_points[elem]:
				index_dict = list()
				for dim in range(self.no_dims):
					index_dict.append(point[dim])
				point2cluster[tuple(index_dict)] = idx
			for c in evaluation_dict:
				evaluation_dict[c][idx] = 0
			idx += 1

		'''for point in point2class:		
			if point2cluster.get(point, -1) == -1:
				print("punct pierdut dupa clustering:", point)'''

		for point in point2cluster:
			evaluation_dict[point2class[point]][point2cluster[point]] += 1
				

		print('Purity:  ', evaluation_measures.purity(evaluation_dict))
		print('Entropy: ', evaluation_measures.entropy(evaluation_dict)) # perfect results have entropy == 0
		print('RI       ', evaluation_measures.rand_index(evaluation_dict))
		print('ARI      ', evaluation_measures.adj_rand_index(evaluation_dict))

		f = open("rezultate_evaluare.txt", "a")
		f.write('Purity:  '+str(evaluation_measures.purity(evaluation_dict))+"\n")
		f.write('Entropy:  '+str(evaluation_measures.entropy(evaluation_dict))+"\n")
		f.write('RI:  '+str(evaluation_measures.rand_index(evaluation_dict))+"\n")
		f.write('ARI:  '+str(evaluation_measures.adj_rand_index(evaluation_dict))+"\n")
		f.close()

	def fitPredict(self, X, y, each_dimension_values, clase_points):

		partition_dict = collections.defaultdict(list)			
		self.no_dims = len(X[0])
		self.pdf = self.computeKDE(X, each_dimension_values) #calculez functia densitate probabilitate utilizand kde

		#detectie si eliminare outlieri

		outliers_iqr_pdf = self.computeOutliersIQR(self.pdf)
		print("We identified "+str(len(outliers_iqr_pdf))+" outliers from "+str(len(X))+" points")
		'''
		print("The outliers are:")
		for outlier_id in outliers_iqr_pdf:
			print(X[outlier_id])'''
		print("======================================")

		X_aux = list()
		each_dimension_values_aux = collections.defaultdict(list)

		#refac X, x si y

		X = [X[q] for q in range(len(X)) if q not in outliers_iqr_pdf]
		y = [X[q] for q in range(len(X))]
		for dim in range(no_dims):
			each_dimension_values[dim] = [X[q][dim] for q in range(len(X))]

		#recalculez pdf, ca altfel se produc erori

		self.pdf = self.computeKDE(X, each_dimension_values) #calculez functia densitate probabilitate din nou

		'''if(self.no_dims==2):
			#coturul cu albastru este plotat doar pentru 2 dimensiuni
			f,xmin, xmax, ymin, ymax, xx, yy = self.evaluate_pdf_kde(X, each_dimension_values) #pentru afisare zone dense albastre
			plt.contourf(xx, yy, f, cmap='Blues') #pentru afisare zone dense albastre'''
			
		
		'''
		Split the dataset in density levels
		'''

		points_per_bin, bins = np.histogram(self.pdf, bins=self.no_bins)

		#plot density levels bins and create density levels partitions
		for idx_bin in range( (len(bins)-1) ):
			culoare = self.randomColorScaled()
			for idx_point in range(len(X)):
				if(self.pdf[idx_point]>=bins[idx_bin] and self.pdf[idx_point]<=bins[idx_bin+1]):
					element_to_append = list()
					for dim in range(self.no_dims):
						element_to_append.append(X[idx_point][dim])
					element_to_append.append(-1) #clusterul nearest neighbour din care face parte punctul
					element_to_append.append(self.pdf[idx_point])
					element_to_append.append(-1) #daca punctul e deja parsta nearest neighbour
					element_to_append.append(idx_point) 
					element_to_append.append(y[idx_point])
					partition_dict[idx_bin].append(element_to_append)
					#scatter doar pentru 2 sau 3 dimensiuni
					if(self.no_dims == 2):
						plt.scatter(X[idx_point][0], X[idx_point][1], color=culoare)
					elif(self.no_dims == 3):
						plt.scatter(X[idx_point][0], X[idx_point][1], X[idx_point][2], color=culoare)
		if(self.no_dims == 2 or self.no_dims == 3):
			plt.show()


		'''
		Density levels distance split
		'''
		
		finalPartitions, noise = self.splitPartitions(partition_dict) #functie care scindeaza partitiile
		
		if(self.no_dims==2):
			for k in finalPartitions:
				color = self.randomColorScaled()
				for point in finalPartitions[k]:
					plt.scatter(point[0], point[1], color=color)

			plt.show()


		joinedPartitions = collections.defaultdict(list)

		joinedPartitions = self.joinPartitions(finalPartitions, self.no_clusters)

		intermediary_centroids = []

		for joinedPartitionIndex in joinedPartitions:
			intermediary_centroids.append(self.centroid(joinedPartitions[joinedPartitionIndex]))


		self.evaluateCluster(clase_points, joinedPartitions)
		print("Evaluation")
		print("==============================")
		
		if(self.no_dims==2):
			#plt.contourf(xx, yy, f, cmap='Blues')
			#final plot
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
			l = l+1

		for cluster_id in sorted_ids:
			color = color_list[cluster_ids_sorted[cluster_id]]
			#print(color)
			for point in cluster_points[cluster_id]:
				ax.scatter(point[0], point[1], color=color)

		fig.savefig('F:\\IULIA\\GITHUB_denlac\\denlac\\results\\poze2\\denlac'+'_'+str(set_de_date)+'.png')   # save the figure to file
		plt.close(fig)

	def return_generated_colors(self):
		colors = [[0.4983913408111469, 0.9878468789867579, 0.6660097921680713], [0.9744941631787404, 0.2832566337094712, 0.9879204118216028], [0.2513270277379317, 0.2743083421568847, 0.24523147335002599], [0.9449152611869482, 0.6829811084805801, 0.23098727325934598], [0.2930994694413758, 0.4447870676048005, 0.9360225619487069], [0.7573766048982865, 0.3564335977711406, 0.5156761252908519], [0.7856267252783685, 0.8893618277470249, 0.9998901678967227], [0.454408739644873, 0.6276300415432641, 0.44436302877623274], [0.5960549019562876, 0.9169447263679981, 0.23343224756103573], [0.5043076141852516, 0.24928662375540336, 0.783126632292948], [0.9247167854639711, 0.8850738215338994, 0.5660824976182026], [0.6968162201133189, 0.5394098658486699, 0.8777137989623846], [0.24964251456446662, 0.8062739995395578, 0.7581261497155073], [0.2575944036656022, 0.7915937407896246, 0.2960661699553983], [0.6437636915214084, 0.4266693349653669, 0.23677001364815042], [0.23112259938541102, 0.32175446177894845, 0.645224195428065], [0.7243345083671118, 0.753389424009313, 0.6716029761309434], [0.9722842730592992, 0.47349469240107894, 0.4282317021959992], [0.9487569650924492, 0.6891786532046004, 0.9520338320784278], [0.5051885381513164, 0.7452481002341962, 0.953601834451638], [0.39319970873496335, 0.5150008439629207, 0.6894464075507598], [0.9907888356008789, 0.3349550392437493, 0.6631372416723879], [0.8941331011073401, 0.23083104173874827, 0.3338481968809], [0.995585861422136, 0.9539037035322647, 0.8814571710677304], [0.3229010345744149, 0.9929405485082905, 0.9199797840228496], [0.8587274228303506, 0.23960128391504704, 0.7796299268247159], [0.9755623661339603, 0.9736967761902182, 0.3368365287453637], [0.26070353957125486, 0.6611108693105839, 0.5626778400435902], [0.33209253309750436, 0.9376441530076292, 0.47506002838287176], [0.8388207042685366, 0.6295035956243679, 0.5353583425079034], [0.3222337347709434, 0.40224067198150343, 0.40979789009079776], [0.6442372806094001, 0.26292344132349454, 0.9763078755323873], [0.7668883074119105, 0.8486492161433142, 0.3841638241303332], [0.5216210516737045, 0.27506979815845595, 0.39564388714836696], [0.6036371225021209, 0.5467800941023466, 0.5990844069213549], [0.5988470728143217, 0.8689413295622888, 0.5609526743224205], [0.8935152630682563, 0.5596944902716602, 0.7784415487870969], [0.686841264479984, 0.9412597573588116, 0.849613972582678], [0.400134697318114, 0.5384071943290534, 0.24536921682148846], [0.5304620100522262, 0.6770501903569319, 0.718601456418752]]

		return colors


'''
=============================================
Denlac Algorithm
'''
if __name__ == "__main__":
	
	filename = sys.argv[1]
	no_bins = int(sys.argv[2]) #no bins
	expand_factor = float(sys.argv[3]) # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)

	#read from file

	each_dimension_values = collections.defaultdict(list)
	X = list() # dataset
	y = list() # classes for evaluatuation
	clase_points = collections.defaultdict(list)

	with open(filename) as f:
			content = f.readlines()

	content = [l.strip() for l in content]

	for l in content:
		aux = l.split(',')
		no_dims = len(aux) - 1
		for dim in range(no_dims):
			each_dimension_values[dim].append(float(aux[dim]))
		list_of_coords = list()
		for dim in range(no_dims):
			list_of_coords.append(float(aux[dim]))
		X.append(list_of_coords)
		y.append(int(aux[no_dims]))
		clase_points[int(aux[no_dims])].append(tuple(list_of_coords))
	
	no_clusters = len(set(y))
	
	denlacInstance = Denlac(no_clusters, no_bins, expand_factor)
	cluster_points = denlacInstance.fitPredict(X, y, each_dimension_values, clase_points)
	'''set_de_date = filename.split("/")[1].split(".")[0].title()
	color_list = denlacInstance.return_generated_colors()'''
	#denlacInstance.plot_clusters(cluster_points, set_de_date, color_list)