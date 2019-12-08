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
* python src/denlac.py -f datasets/r15.txt -nclusters 15 -nbins 3 -expFactor 0.7 -aggMethod 2
* python src/denlac.py -f datasets/jain.txt -nclusters 2 -nbins 3 -expFactor 0.75
* python src/denlac.py -f datasets/pathbased.txt -nclusters 3 -nbins 3 -expFactor 0.35
* python src/denlac.py -f datasets/flame.txt -nclusters 2 -nbins 2 -expFactor 0.5
* python src/denlac.py -f datasets/compound.txt -nclusters 6 -nbins 3 -expFactor 1 
* python src/denlac.py -f datasets/d31.txt -nclusters 31 -nbins 7 -expFactor 0.1 -aggMethod 2
* python src/denlac.py -f datasets/irisDenlacText.txt -nclusters 3 -nbins 5 -expFactor 0.45
-------------------------------------------------------------------------------------------------

## Datasets:

### Aggregation
* Aggregation details: 
	* No. Elements: 788
	* No. Dimensions: 2
	* No. Clusters: 7
	* No. Bins: 8
	* Expand Factor: 0.8 
* Article:
	* A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.
* Evaluation results:
Purity:   0.9949238578680204
Entropy:  0.011107599359548468
RI        0.997058804558853
ARI       0.991326806021418

### Spiral
* Spiral details: 
	* No. Elements: 312
	* No. Dimensions: 2
	* No. Clusters: 3
	* No. Bins: 3
	* Expand Factor: 1
* article:
	* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 
* Evaluation results:
Purity:   1.0
Entropy:  0.0
RI        1.0
ARI       1.0

### R15
* R15 details: 
	* No. Elements: 600
	* No. Dimensions: 2
	* No. Clusters: 15
	* No. Bins: 8
	* Expand Factor: 0.5
* article:
	* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence, 2002. 24(9): p. 1273-1280. 
* Evaluation results:
Purity:   0.995
Entropy:  0.008680187258725283
RI        0.9986867000556483
ARI       0.9892138643396456

### Jain
* Jain details: 
	* No. Elements: 373
	* No. Dimensions: 2
	* No. Clusters: 2
	* No. Bins: 3
	* Expand Factor: 1
* article:
	* A. Jain and M. Law, Data clustering: A user's dilemma. Lecture Notes in Computer Science, 2005. 3776: p. 1-10. 
* Evaluation results:
Purity:   1.0
Entropy:  0.0
RI        1.0
ARI       1.0

### Pathbased
* Pathbased details: 
	* No. Elements: 300
	* No. Dimensions: 2
	* No. Clusters: 3
	* No. Bins: 2
	* Expand Factor: 0.5
* article:
	* H. Chang and D.Y. Yeung, Robust path-based spectral clustering. Pattern Recognition, 2008. 41(1): p. 191-203. 
* Evaluation details:
Purity:   0.9765886287625418
Entropy:  0.09889029183939287
RI        0.9687998024735697
ARI       0.9298035300401308

### Flame
* Flame details:
	* No. Elements: 240
	* No. Dimensions: 2
	* No. Clusters: 2
	* No. Bins: 2
	* Expand Factor: 1
* article:
	* L. Fu and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. BMC bioinformatics, 2007. 8(1): p. 3. 
* Evaluation details:
Purity:   0.9747899159663865
Entropy:  0.16695141891087922
RI        0.950643548558664
ARI       0.9005276926202117

### Compound
* Compound details:
	* No. Elements: 399
	* No. Dimensions: 6
	* No. Clusters: 2
	* No. Bins: 3
	* Expand Factor: 1
* article:
	* C.T. Zahn, Graph-theoretical methods for detecting and describing gestalt clusters. IEEE Transactions on Computers, 1971. 100(1): p. 68-86. 
* Evaluation details:
Purity:   0.8922305764411027
Entropy:  0.11445223510987003
RI        0.9667888313749197
ARI       0.9112696082383437
### D31
* D31 details:
	* No. Elements: 3100
	* No. Dimensions: 2
	* No. Clusters: 31
	* No. Bins: 7
	* Expand Factor: 0.1
* article:
	* C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence 2002. 24(9): p. 1273-1280.

## IRIS
* Evaluation results:
Purity:   1.0
Entropy:  0.0
RI        1.0
ARI       1.0
