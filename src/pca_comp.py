import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
import sys
import csv


def getD(x1, y1, x2, y2, x3, y3):
    return abs((y2-y1)*x3 - (x2-x1)*y3 + x2*y1-x1*y2)/math.sqrt((y2-y1)**2 + (x2-x1)**2)

def getOptimalComponents(data):
    dataStd = StandardScaler().fit_transform(data)
    #Calculating Eigenvecors and eigenvalues of Covariance matrix
    mean_vec = np.mean(dataStd, axis=0)
    cov_mat = np.cov(dataStd.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [ (np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key = lambda x: x[0], reverse= True)
    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    # Individual explained variance
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)]
    # Cumulative explained variance
    cum_var_exp = np.cumsum(var_exp)
        
    dist = {k: getD(1, var_exp[0], len(var_exp), var_exp[len(var_exp)-1], k, var_exp[k-1]) for k in range(1, len(var_exp)+1)}
    optimalComponents = max(dist, key=dist.get)
    print(optimalComponents)
    return optimalComponents

def getFeatures(data):
    dataStd = StandardScaler().fit_transform(data)
    # reduce to optimal important features
    pca = PCA(n_components=getOptimalComponents(data))
    dataPCA = pca.fit_transform(dataStd)

    return dataPCA

if __name__ == "__main__":
    fn = sys.argv[1]
    with open(fn) as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\t')
        data = [[int(elem) for elem in row] for row in readCSV]

    dataPCA = getFeatures(np.array(data))
    print(len(dataPCA[0]))
    print(dataPCA)
