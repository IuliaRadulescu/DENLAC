from __future__ import division

import numpy as np
import sys
import collections
import matplotlib.pyplot as plt
import random
import math

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.agglomerative import agglomerative, type_link

import evaluation_measures

'''
Formatul dictionarului pentru evaluare
{ clasa_0 : { cluster_0: nr_puncte, cluster_1: nr_puncte, ... cluster_n: nr_puncte}, clasa_1: { cluster_0: nr_puncte, cluster_1: nr_puncte, ... cluster_n: nr_puncte}....}

Exemplu:

Se da un dataset cu 2 clase si 400 de puncte
- clasa 1 are 150 de puncte
- clasa 2 are 250 de puncte

dupa clusterizare avem urmatoarele:
- clusterul 1 are 140 de puncte din clasa 1 si  20 de puncte din clasa 2
- clusterul 2 are  10 de puncte din clasa 1 si 230 de puncte din clasa 2

dictionarul va arata in felul urmator:
{
clasa_1 : {cluster_1: 140, cluster_2: 10},
clasa_2 : {cluster_1:  20, cluster_2: 230}
}
'''


class EvaluateAlgorithms:

	def __init__(self, no_dims):

		self.no_dims = no_dims

	def runKMeans(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = KMeans(n_clusters=k, random_state=0).fit_predict(X)
		#print(y_pred)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runBirch(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = Birch(n_clusters=k).fit(X).predict(X)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runGaussianMixture(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = GaussianMixture(n_components=k).fit(X).predict(X)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runSpectralClustering(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()
		y_pred = SpectralClustering(n_clusters=k).fit_predict(X)
		for point_id in range(len(X)):
			cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runCURE(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()

		cure_instance = cure(data=X, number_cluster=k)
		cure_instance.process()
		clusters = cure_instance.get_clusters()
		
		for id_point in range(len(X)):
			for cluster_id in range(len(clusters)):
				point_ids_in_cluster = [int(point_id_in_cluster) for point_id_in_cluster in  clusters[cluster_id]]
				if(id_point in point_ids_in_cluster):
					cluster_points[cluster_id].append(X[id_point])

		return cluster_points

	def runCLARANS(self, k, X):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()

		clarans_instance = clarans(data=X, number_clusters=k, numlocal=5, maxneighbor=5);
		clarans_instance.process();
		clusters = clarans_instance.get_clusters();
		
		for id_point in range(len(X)):
			for cluster_id in range(len(clusters)):
				point_ids_in_cluster = [int(point_id_in_cluster) for point_id_in_cluster in  clusters[cluster_id]]
				if(id_point in point_ids_in_cluster):
					cluster_points[cluster_id].append(X[id_point])

		return cluster_points

	def runDBSCAN(self, X, mean_dist, number_of_points):
		cluster_points = {}
		y_pred = DBSCAN(min_samples=number_of_points, eps=mean_dist).fit_predict(X)

		nr_obtained_clusters = max(y_pred)+1
		for q in range(nr_obtained_clusters): #aici numarul de clustere e valoarea maxima din y_pred
			cluster_points[q] = list()

		for point_id in range(len(X)):
			#eliminam zgomotele
			if(y_pred[point_id]!=-1):
				cluster_points[y_pred[point_id]].append(X[point_id])
		#print(cluster_points)
		return cluster_points

	def runAGGLOMERATIVE(self, k, X, type_link_param):
		cluster_points = {}
		for q in range(k):
			cluster_points[q] = list()

		agglo_instance = agglomerative(data=X, number_clusters=k, link=type_link_param);
		agglo_instance.process();
		clusters = agglo_instance.get_clusters();
		for id_point in range(len(X)):
			for cluster_id in range(len(clusters)):
				point_ids_in_cluster = [int(point_id_in_cluster) for point_id_in_cluster in  clusters[cluster_id]]
				if(id_point in point_ids_in_cluster):
					cluster_points[cluster_id].append(X[id_point])

		return cluster_points


	def evaluate_cluster(self, clase_points, cluster_points, filename, nume_algoritm, nume_set_date):
		
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

		for point in point2cluster:
			evaluation_dict[point2class[point]][point2cluster[point]] += 1
				

		print('Purity: ', evaluation_measures.purity(evaluation_dict))
		print('Entropy: ', evaluation_measures.entropy(evaluation_dict)) # perfect results have entropy == 0
		print('RI ', evaluation_measures.rand_index(evaluation_dict))
		print('ARI ', evaluation_measures.adj_rand_index(evaluation_dict))

		f = open("rezultate_evaluare_"+nume_algoritm+"_"+nume_set_date+".txt", "a")
		f.write("Rezultate evaluare pentru setul de date "+str(filename)+"\n")
		f.write('Purity: '+str(evaluation_measures.purity(evaluation_dict))+"\n")
		f.write('Entropy: '+str(evaluation_measures.entropy(evaluation_dict))+"\n")
		f.write('RI: '+str(evaluation_measures.rand_index(evaluation_dict))+"\n")
		f.write('ARI: '+str(evaluation_measures.adj_rand_index(evaluation_dict))+"\n")
		f.write("\n")
		f.close()

	def random_color_scaled(self):
		b = random.randint(0, 255)
		g = random.randint(0, 255)
		r = random.randint(0, 255)
		return [round(b/255,2), round(g/255,2), round(r/255,2)]

	def DistFunc(self, x, y):

		sum_powers = 0
		for dim in range(self.no_dims):
			sum_powers = math.pow(x[dim]-y[dim], 2) + sum_powers
		return math.sqrt(sum_powers)

	def get_mean_dist(self, X):
		distances = list()
		for id_x in range(len(X)-1):
			for id_y in range(id_x+1, len(X)):
				dist = self.DistFunc(X[id_x], X[id_y])
				distances.append(dist)
		return sum(distances)/len(distances)

	def plot_clusters(self, cluster_points, algoritm, set_de_date, color_list):
		fig, ax = plt.subplots(nrows=1, ncols=1)

		#sortam cluster_points in functie de dimensiunea fiecarui cluster - pentru printare frumoasa culor
		sorted_ids = list()
		for k in sorted(cluster_points, key=lambda k: len(cluster_points[k]), reverse=True):
			sorted_ids.append(k)

		for cluster_id in range(len(sorted_ids)):
			color = color_list[sorted_ids[cluster_id]]
			#print(color)
			for point in cluster_points[cluster_id]:
				ax.scatter(point[0], point[1], color=color)

		fig.savefig('F:\\IULIA\\GITHUB_MARIACLUST\\MariaClust\\results\\poze2\\'+str(algoritm)+"_"+str(set_de_date)+'.png')   # save the figure to file
		plt.close(fig)

	def get_closest_mean(self, X, percent):
		'''
		Media distantelor celor mai apropiati k vecini pentru fiecare punct in parte
		'''

		k=int(math.ceil(percent*len(X)))
		distances = list()

		for point in X:
			deja_parsati = list()
			while(k>0):
				neigh_id = 0
				minDist = 99999
				for id_point_k in range(len(X)):
					point_k = X[id_point_k]
					if(point_k not in deja_parsati):
						dist = self.DistFunc(point, point_k)
						if(dist < minDist and dist > 0):
							minDist = dist
							neigh_id = id_point_k
				distances.append(minDist)
				neigh = X[neigh_id]
				deja_parsati.append(neigh)
				k=k-1
		distances = list(set(distances))
		return sum(distances)/len(distances)
class Colors:

	def __init__(self, N):
		self.N = N

	def get_random_color(self, pastel_factor = 0.5):
		return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

	def color_distance(self, c1,c2):
		return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

	def generate_new_color(self, existing_colors,pastel_factor = 0.5):
		max_distance = None
		best_color = None
		for i in range(0,100):
			color = self.get_random_color(pastel_factor = pastel_factor)
			if not existing_colors:
				return color
			best_distance = min([self.color_distance(color,c) for c in existing_colors])
			if not max_distance or best_distance > max_distance:
				max_distance = best_distance
				best_color = color
		return best_color

	def generate_colors(self):
		colors = []

		for i in range(0,self.N):
			colors.append(self.generate_new_color(colors, pastel_factor = 0.3))

		f = open("culori_utilizate_generare.txt", "a")
		f.write(str(colors))
		f.close()
	
		return colors

	def return_generated_colors(self):
		colors = [[0.4983913408111469, 0.9878468789867579, 0.6660097921680713], [0.9744941631787404, 0.2832566337094712, 0.9879204118216028], [0.2513270277379317, 0.2743083421568847, 0.24523147335002599], [0.9449152611869482, 0.6829811084805801, 0.23098727325934598], [0.2930994694413758, 0.4447870676048005, 0.9360225619487069], [0.7573766048982865, 0.3564335977711406, 0.5156761252908519], [0.7856267252783685, 0.8893618277470249, 0.9998901678967227], [0.454408739644873, 0.6276300415432641, 0.44436302877623274], [0.5960549019562876, 0.9169447263679981, 0.23343224756103573], [0.5043076141852516, 0.24928662375540336, 0.783126632292948], [0.9247167854639711, 0.8850738215338994, 0.5660824976182026], [0.6968162201133189, 0.5394098658486699, 0.8777137989623846], [0.24964251456446662, 0.8062739995395578, 0.7581261497155073], [0.2575944036656022, 0.7915937407896246, 0.2960661699553983], [0.6437636915214084, 0.4266693349653669, 0.23677001364815042], [0.23112259938541102, 0.32175446177894845, 0.645224195428065], [0.7243345083671118, 0.753389424009313, 0.6716029761309434], [0.9722842730592992, 0.47349469240107894, 0.4282317021959992], [0.9487569650924492, 0.6891786532046004, 0.9520338320784278], [0.5051885381513164, 0.7452481002341962, 0.953601834451638], [0.39319970873496335, 0.5150008439629207, 0.6894464075507598], [0.9907888356008789, 0.3349550392437493, 0.6631372416723879], [0.8941331011073401, 0.23083104173874827, 0.3338481968809], [0.995585861422136, 0.9539037035322647, 0.8814571710677304], [0.3229010345744149, 0.9929405485082905, 0.9199797840228496], [0.8587274228303506, 0.23960128391504704, 0.7796299268247159], [0.9755623661339603, 0.9736967761902182, 0.3368365287453637], [0.26070353957125486, 0.6611108693105839, 0.5626778400435902], [0.33209253309750436, 0.9376441530076292, 0.47506002838287176], [0.8388207042685366, 0.6295035956243679, 0.5353583425079034], [0.3222337347709434, 0.40224067198150343, 0.40979789009079776], [0.6442372806094001, 0.26292344132349454, 0.9763078755323873], [0.7668883074119105, 0.8486492161433142, 0.3841638241303332], [0.5216210516737045, 0.27506979815845595, 0.39564388714836696], [0.6036371225021209, 0.5467800941023466, 0.5990844069213549], [0.5988470728143217, 0.8689413295622888, 0.5609526743224205], [0.8935152630682563, 0.5596944902716602, 0.7784415487870969], [0.686841264479984, 0.9412597573588116, 0.849613972582678], [0.400134697318114, 0.5384071943290534, 0.24536921682148846], [0.5304620100522262, 0.6770501903569319, 0.718601456418752]]

		return colors

if __name__ == "__main__":
	
	home_path = "F:\\IULIA\\GITHUB_MARIACLUST\\MariaClust\\datasets\\"
	filenames = [home_path+"aggregation.txt", home_path+"compound.txt", home_path+"d31.txt", home_path+"flame.txt", home_path+"jain.txt", home_path+"pathbased.txt", home_path+"r15.txt", home_path+"spiral.txt"]
	dataset_names = ["Aggregation", "Compound", "D31", "Flame", "Jain", "Pathbased", "R15", "Spiral"]
	no_clusters_all = [7, 6, 31, 2, 2, 3, 15, 3]
	no_dims_all = [2, 2, 2, 2, 2, 2, 2, 2]

	#multidimensional
	'''filenames = [home_path+"dim032.txt", home_path+"dim064.txt", home_path+"dim128.txt", home_path+"dim256.txt", home_path+"dim512.txt"]
	dataset_names = ["dim032", "dim064", "dim128", "dim256", "dim512"]
	no_clusters_all = [16, 16, 16, 16, 16]
	no_dims_all = [32, 64, 128, 256, 512]'''

	color_generator = Colors(40)
	color_list = color_generator.return_generated_colors()
	#print("Culorile utilizate sunt:")
	#print(color_list)

	for nr_crt in range(len(filenames)):

		filename = filenames[nr_crt]
		no_clusters = no_clusters_all[nr_crt]
		no_dims = no_dims_all[nr_crt]

		each_dimension_values = collections.defaultdict(list)
		dataset_xy = list()
		dataset_xy_validate = list()
		clase_points = collections.defaultdict(list)

		with open(filename) as f:
				content = f.readlines()

		content = [l.strip() for l in content]

		for l in content:
			aux = l.split('\t')
			for dim in range(no_dims):
				each_dimension_values[dim].append(float(aux[dim]))
			list_of_coords = list()
			for dim in range(no_dims):
				list_of_coords.append(float(aux[dim]))
			dataset_xy.append(list_of_coords)
			dataset_xy_validate.append(int(aux[no_dims]))
			clase_points[int(aux[no_dims])].append(tuple(list_of_coords))

		evaluateAlg = EvaluateAlgorithms(no_dims)
	
		'''cluster_points = evaluateAlg.runKMeans(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "KMEANS", dataset_names[nr_crt], color_list)
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "KMEANS", dataset_names[nr_crt])
		
		cluster_points = evaluateAlg.runBirch(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "BIRCH", dataset_names[nr_crt], color_list)
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "BIRCH", dataset_names[nr_crt])

		cluster_points = evaluateAlg.runGaussianMixture(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "GAUSSIANMIXTURE", dataset_names[nr_crt], color_list)
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "GAUSSIANMIXTURE", dataset_names[nr_crt])

		cluster_points = evaluateAlg.runSpectralClustering(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "SPECTRALCLUSTERING", dataset_names[nr_crt], color_list)
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "SPECTRALCLUSTERING", dataset_names[nr_crt])

		cluster_points = evaluateAlg.runCURE(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "CURE", dataset_names[nr_crt], color_list)
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "CURE", dataset_names[nr_crt])'''
		
		cluster_points = evaluateAlg.runCLARANS(no_clusters, dataset_xy)
		evaluateAlg.plot_clusters(cluster_points, "CLARANS", dataset_names[nr_crt], color_list)
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "CLARANS", dataset_names[nr_crt])

		'''if(nr_crt == 0):
			mean_dist = evaluateAlg.get_closest_mean(dataset_xy, 0.0025)
			cluster_points = evaluateAlg.runDBSCAN(dataset_xy, mean_dist,3)
		elif(nr_crt == 1):
			mean_dist = evaluateAlg.get_closest_mean(dataset_xy, 0.005)
			cluster_points = evaluateAlg.runDBSCAN(dataset_xy, mean_dist,3)
		elif(nr_crt == 5):
			mean_dist = evaluateAlg.get_closest_mean(dataset_xy, 0.015)
			cluster_points = evaluateAlg.runDBSCAN(dataset_xy, mean_dist,3)
		elif(nr_crt == 4):
			mean_dist = evaluateAlg.get_closest_mean(dataset_xy, 0.005)
			cluster_points = evaluateAlg.runDBSCAN(dataset_xy, mean_dist,3)
		elif(nr_crt == 6):
			mean_dist = evaluateAlg.get_closest_mean(dataset_xy, 0.05)
			cluster_points = evaluateAlg.runDBSCAN(dataset_xy, mean_dist,3)
		else:
			mean_dist = evaluateAlg.get_closest_mean(dataset_xy, 0.01)
			cluster_points = evaluateAlg.runDBSCAN(dataset_xy, mean_dist,3)
		
		evaluateAlg.plot_clusters(cluster_points, "DBSCAN", dataset_names[nr_crt], color_list)'''
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "OPTICS")

		'''if(nr_crt==7 or nr_crt==3):
			cluster_points = evaluateAlg.runAGGLOMERATIVE(no_clusters, dataset_xy, type_link.SINGLE_LINK)
			
		else:
			cluster_points = evaluateAlg.runAGGLOMERATIVE(no_clusters, dataset_xy, type_link.AVERAGE_LINK)
		#evaluateAlg.evaluate_cluster(clase_points, cluster_points, filename, "HIERARCHICALAGG", dataset_names[nr_crt])
		evaluateAlg.plot_clusters(cluster_points, "HIERARCHICALAGG", dataset_names[nr_crt], color_list)'''