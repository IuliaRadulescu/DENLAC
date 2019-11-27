# denlac

### Run parameters:
* f - filename of the dataset
* nclusters - the desired number of clusters
* nbins - usually the number of optimal density levels
* expFactor - between 0.1 and 2
* aggMethod - by default 1, 2 for centroid

Run examples: 
* python src/denlac.py -f datasets/aggregation.txt -nclusters 7 -nbins 8 -expFactor 0.8 -aggMethod 2
* python src/denlac.py -f datasets/spiral.txt -nclusters 3 -nbins 3 -expFactor 1
* python src/denlac.py -f datasets/r15.txt -nclusters 15 -nbins 3 -expFactor 0.5
* python src/denlac.py -f datasets/jain.txt -nclusters 2 -nbins 3 -expFactor 1.5
* python src/denlac.py -f datasets/pathbased.txt -nclusters 3 -nbins 3 -expFactor 0.5
* python src/denlac.py -f datasets/flame.txt -nclusters 2 -nbins 2 -expFactor 1 
* python ssrc/denlac.py -f datasets/compound.txt -nclusters 6 -nbins 3 -expFactor 1 
* python src/denlac.py -f datasets/d31.txt -nclusters 31 -nbins 7 -expFactor 0.1 -aggMethod 2
* python src/denlac.py -f datasets/irisDenlacText.txt -nclusters 3 -nbins 5 -expFactor 0.1
-------------------------------------------------------------------------------------------------

## Datasets:

### Aggregation
* Aggregation details: 
	* No. Elements: 788
	* No. Dimensions: 2
	* No. Clusters: 7
	* No. Bins: 8
	* Expand Factor: 0.8 
* article:
	* A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.

### Spiral
* Spiral details: 
	* No. Elements: 312
	* No. Dimensions: 2
	* No. Clusters: 3
	* No. Bins: 3
	* Expand Factor: 1
* article:
	* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

### R15
* R15 details: 
	* No. Elements: 600
	* No. Dimensions: 2
	* No. Clusters: 15
	* No. Bins: 8
	* Expand Factor: 0.5
* article:
	* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence, 2002. 24(9): p. 1273-1280. 

### Jain
* Jain details: 
	* No. Elements: 373
	* No. Dimensions: 2
	* No. Clusters: 2
	* No. Bins: 3
	* Expand Factor: 1
* article:
	* A. Jain and M. Law, Data clustering: A user's dilemma. Lecture Notes in Computer Science, 2005. 3776: p. 1-10. 

### Pathbased
* Pathbased details: 
	* No. Elements: 300
	* No. Dimensions: 2
	* No. Clusters: 3
	* No. Bins: 2
	* Expand Factor: 0.5
* article:
	* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

### Flame
* Flame details:
	* No. Elements: 240
	* No. Dimensions: 2
	* No. Clusters: 2
	* No. Bins: 2
	* Expand Factor: 1
* article:
	* L. Fu and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. BMC bioinformatics, 2007. 8(1): p. 3. 

### Compound
* Compound details:
	* No. Elements: 399
	* No. Dimensions: 6
	* No. Clusters: 2
	* No. Bins: 3
	* Expand Factor: 1
* article:
	* C.T. Zahn, Graph-theoretical methods for detecting and describing gestalt clusters. IEEE Transactions on Computers, 1971. 100(1): p. 68-86. 

### D31
* D31 details:
	* No. Elements: 3100
	* No. Dimensions: 2
	* No. Clusters: 31
	* No. Bins: 7
	* Expand Factor: 0.1
* article:
	* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence 2002. 24(9): p. 1273-1280.
