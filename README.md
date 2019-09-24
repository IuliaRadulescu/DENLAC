# denlac

### Run parameters for improved version (denlac_better2):
* no_clusters
* no_neigbors - the number of nearest neighbors in the k-nearest-neigbors graph

### Run parameters:
* filename of the dataset
* no_clusters
* no_bins
* expand_factor - given a center, how much a cluster can expand based on the number of neighbours
* how to compute the inter cluster dinstances:
	* 1 = centroid linkage
	* 2 = average linkage
	* 3 = single linkage
* no_dimensions

Run examples: 
* python ssrc/denlac_better.py datasets/aggregation.txt 7 8 0.8 1 2
* python src/denlac_better.py datasets/spiral.txt 3 3 1 3 2
* python src/denlac.py datasets/r15.txt 15 3 0.5 1 2
* python src/denlac_better.py datasets/jain.txt 2 3 1.5 2 2
* python src/denlac_better.py datasets/pathbased.txt 3 3 0.5 2 2
* python src/denlac_better.py datasets/flame.txt 2 2 1 3 2
* python ssrc/denlac_better.py datasets/compound.txt 6 3 1 3 2
* python src/denlac.py datasets/d31.txt 31 7 0.1 1 2
* python src/denlac_better.py datasets/irisDenlacText.txt 3 5 0.1 1 5
-------------------------------------------------------------------------------------------------

## Datasets:

### Aggregation
* Aggregation details: 
	* No. Elements: 788
	* No. Dimensions: 2
	* No. Clusters: 7
	* No. Bins: 8
	* Expand Factor: 0.8 
	* Dinstance Type: 1 (centroid)
* run:
	* python src/denlac.py datasets/aggregation.txt 7 3 1 1 2
* article:
	* A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.

### Spiral
* Spiral details: 
	* No. Elements: 312
	* No. Dimensions: 2
	* No. Clusters: 3
	* No. Bins: 3
	* Expand Factor: 1
	* Dinstance Type: 3 (single linkage)
* run:
	* python src/denlac.py datasets/spiral.txt 3 3 1 3 2
* article:
	* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

### R15
* R15 details: 
	* No. Elements: 600
	* No. Dimensions: 2
	* No. Clusters: 15
	* No. Bins: 8
	* Expand Factor: 0.5
	* Dinstance Type: 1 (centroid)
* run:
	* python src/denlac.py datasets/r15.txt 15 8 0.5 1
* article:
	* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence, 2002. 24(9): p. 1273-1280. 

### Jain
* Jain details: 
	* No. Elements: 373
	* No. Dimensions: 2
	* No. Clusters: 2
	* No. Bins: 3
	* Expand Factor: 1
	* Dinstance Type: 2 (average linkage)
* run:
	* python src/denlac.py datasets/jain.txt 2 3 1 2
* article:
	* A. Jain and M. Law, Data clustering: A user's dilemma. Lecture Notes in Computer Science, 2005. 3776: p. 1-10. 

### Pathbased
* Pathbased details: 
	* No. Elements: 300
	* No. Dimensions: 2
	* No. Clusters: 3
	* No. Bins: 2
	* Expand Factor: 0.73
	* Dinstance Type: 2 (average linkage)
* run:
	* python src/denlac.py datasets/pathbased.txt 3 2 0.73 2
* article:
	* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 

### Flame
* Flame details:
	* No. Elements: 240
	* No. Dimensions: 2
	* No. Clusters: 2
	* No. Bins: 2
	* Expand Factor: 1
	* Dinstance Type: 3 (single linkage)
* run
	* python src/denlac.py datasets/flame.txt 2 2 1 3
* article:
	* L. Fu and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. BMC bioinformatics, 2007. 8(1): p. 3. 

### Compound
* Compound details:
	* No. Elements: 399
	* No. Dimensions: 6
	* No. Clusters: 2
	* No. Bins: 3
	* Expand Factor: 1
	* Dinstance Type: 3 (single linkage)
* run:
	* python src/denlac.py datasets/compound.txt 6 3 1 3
* article:
	* C.T. Zahn, Graph-theoretical methods for detecting and describing gestalt clusters. IEEE Transactions on Computers, 1971. 100(1): p. 68-86. 

### D31
* D31 details:
	* No. Elements: 3100
	* No. Dimensions: 2
	* No. Clusters: 31
	* No. Bins: 7
	* Expand Factor: 0.1
	* Dinstance Type: 1 (centroid)	
*run:
	* python src/denlac.py datasets/d31.txt 31 7 0.1 1
* article:
	* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence 2002. 24(9): p. 1273-1280.
