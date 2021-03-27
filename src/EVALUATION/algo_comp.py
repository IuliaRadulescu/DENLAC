import numpy as np
import sys
import csv
import matplotlib.pyplot as plt

from spherecluster import SphericalKMeans

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.agglomerative import agglomerative, type_link


if __name__ == "__main__":
    file = sys.argv[1]
    k = int(sys.argv[2])
    l = []
    with open(file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            l.append([float(elem) for elem in row[:-1]])

    X = np.array(l)

    # K-means chior
    y_pred = KMeans(n_clusters=k, random_state=0).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("K-Means")
    plt.show()

    # Spherical k-means
    y_pred = SphericalKMeans(n_clusters=k).fit(X).predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Sperical K_means")
    plt.show()
    

    # DBSCAN
    i = 1
    while i<=2:
        y_pred = DBSCAN(eps=i, min_samples=8).fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.title("DBSCAN i="+str(i))
        plt.show()
        i += 0.05

    # Birch
    y_pred = Birch(n_clusters=k).fit(X).predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Birch")
    plt.show()

    # Gaussian Mixture
    y_pred = GaussianMixture(n_components=k).fit(X).predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Gaussian Mixture")
    plt.show()

    # Spectral Clustering
    y_pred = SpectralClustering(n_clusters=k).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Spectral Clustering")
    plt.show()

    # CURE
    cure_instance = cure(data=X, number_cluster=k);
    cure_instance.process();
    clusters = cure_instance.get_clusters();
    visualizer = cluster_visualizer(titles=["Cure"]);
    visualizer.append_clusters(clusters, X);
    visualizer.show();

    # CLARANS
    clarans_instance = clarans(data=X, number_clusters=k, numlocal=5, maxneighbor=5);
    clarans_instance.process();
    clusters = clarans_instance.get_clusters();
    visualizer = cluster_visualizer(titles=["Clarans"]);
    visualizer.append_clusters(clusters, X);
    visualizer.show();

    # Agglomerative
    # type_link  = [SINGLE_LINK, COMPLETE_LINK, AVERAGE_LINK, CENTROID_LINK]
 
    agglo_instance = agglomerative(data=X, number_clusters=k, link=type_link.COMPLETE_LINK);
    agglo_instance.process();
    clusters = agglo_instance.get_clusters();
    visualizer = cluster_visualizer(titles=["Agglomerative"]);
    visualizer.append_clusters(clusters, X);
    visualizer.show();