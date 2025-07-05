---
title: 1103 Clustering - Requirements Analysis.md

---


# 1103 - Clustering (KMeans) - Requirements Analysis

<!-- Filename: 1103 - Clustering (KMeans)- Requirements Analysis.md -->

[toc]

---



<!--
This document is a requirements analysis for the Friedman Test implemented across SAS, R, and Python.

🎯 Purpose:
- Document statistical assumptions, inputs, outputs, and implementation details for each tool.
- Enable reproducibility and informed tool selection.
- Support standardization of statistical procedures across environments.

🛠️ How to Use:
- Fill in each section with details specific to the Friedman Test.
- Use consistent formatting and clarify parameter defaults, data structures, and return values.
- Leave "N/A" where a package does not support a feature.
-->

# Background

Clustering is an unsupervised learning algorithm that groups unlabeled data points into groups such that observations in the same group are as similar to each other as possible, and similarly, observations in different groups are as different to each other as possible. K-means is one method of cluster analysis that groups observations by minimizing Euclidean distances between them. 

Applications of clustering approaches in medical data analysis include disease nosology, early diagnosis of diseases, and predictions of diseases etc.



## Hypotheses
K Means clustering method is an exploratory analysis and doesn't have an explicit null and alternate hypothesis. However, by 1) minimizing variability within clusters and 2) maximizing variability between clusters, the output of more than one cluster enforces that the means in the clusters are different from each other, whereas not identifying more than one cluster implies the mean value is shared across all groups. 

## Assumptions

- All variables are continuous at the interval or ratio level. Categorical variables are converted to continuous forms.
- Rows are observations (individuals) and columns are variables. Missing values, if exist, are removed or estimated. 
- For most common clustering software, the default distance measure is the Euclidean distance. The choice of distance measure is important for clustering results. 
- Generally, the data is expected to be standardized (i.e., they have mean zero and standard deviation one) to make data comparable, as Euclidean distance measure is "scale-dependent".
- A priori specification of the number of clusters, k, should be provided.
- K Means clustering works well with clusters that are spherical- and isotropic-shaped and having similar variance and sizes.
- Initialization of cluster centroids can influence the result as well. Random initialization makes result more sensitive to outliers. 

## Test Statistic & P-value
Clustering procedure does not involve inferential statistics or p-values. We only have cluster validity indices that enable the user to choose an optimal number of clusters in the data.

Pseudo F Index (ratio of between-cluster variance to within cluster variance):  
**Pseudo F** = **(GSS/K-1)/(WSS/N-K)**
- **N** is the number of observations  
- **K** is the number of clusters  
- **GSS** is the between-group sum of squares  
- **WSS** is the within group sum of squares  

# Package Implementations

## SAS (HPCLUS)

### Function / Procedure
`PROC HPCLUS` comes from the SAS Enterprise Miner: High-Performance Procedures.

### Inputs
- **Required**: 
    - `PROC HPCLUS` statements (including but not limited to):
        - Input and output dataset options: `DATA=` (input data set)
        - Clustering Options: `MAXCLUSTERS=` (number of clusters), `DISTANCE=` (distance method for similarity measurement)
        - Data processing options: `IMPUTE=` (imputation method for numeric interval input variables)
        - `NOC=` Specifies the method for finding number of clusters
    - `INPUT` statement: specifies the names of the variables to be used in clustering
- **Optional**: statements including `ID`, `FREQ`, `SCORE` etc.

### Outputs
Tables including Model Information, WithinClusStats (statistics for numeric interval input variables within clusters), Iteration statistics, etc.

### Sample Code
```sas
proc hpclus data=inputData maxclusters=3;
   input x y;
   freq freq;
run;
```

### Limitations / Notes
- By default, random observations from the input data set are selected as initial cluster centroids. You can change the observations that are selected from the input data set by using the `SEED=` option. 
- Compared to other methods, `PROC HPCLUS`:
    - automates standardization by specifying the`STANDARDIZE=` option. 
    - automates replacing missing values with mean or mode with the statement `IMPUTE=MEAN` or `IMPUTENOM=MODE`.
    - uses parallel computing and multithreading in the `PERFORMANCE` statement.
    - provides a new technique called the aligned box criterion (ABC) for estimating the number of clusters in the data set by setting the NOC (number of clusters) statement `NOC=ABC` .
    - supports k-mode algorithm (for nominal variables) and allows specification of different disimilarity measures by setting`DISTANCENOM=` statement.

---

## SAS (FASTCLUS)

### Function / Procedure
`PROC FASTCLUS` comes from the SAS/STAT® 15.2 User's Guide.

### Inputs
- **Required**: 
    - `PROC FASTCLUS` statement: expect at least`<MAXCLUSTERS=n>` and `<RADIUS=t>`.
        - ` DATA=` 	Specifies input data set
        - ` SEED=` 	Specifies input SAS data set for selecting initial cluster seeds
        - `RADIUS=` Specifies a minimum distance by which the seeds must be separated
        - `RANDOM=`	Specifies seed to initializes pseudo-random number generator
        - `LEAST=` Optimizes an L_p criterion, where 1<=p<=infinity
    - `VAR` statement: numeric variables to be used in the cluster analysis.
- **Optional**: `BY`, `FREQ`, `ID`, `VAR`, and WEIGHT statements

### Outputs
- Cluster Summary: cluster number, frequency, weight, maximum Distance from Seed to Observation, etc.
- Statistics for all variables: total standard deviation, within-cluster standard deviation, R-Square etc.
- Pseudo F Statistic, distances Between Cluster Means, Cubic Clustering Criterion etc.

### Sample Code
```sas
proc fastclus data=x outseed=mean1 maxc=20 maxiter=0 summary;
   var x y;
run;
```

### Limitations / Notes
- Does not have a standardization option, and you should use methods such as `PROC STDIZE` to standardize input data prior to running `FASTCLUS`.
- Selects the first k complete (no missing values) observations that are RADIUS (default set to 0) apart from each other as initial cluster centroids. As such, the ordering of observations can affect initial centroids selection (`PROC FASTCLUS` is not recommended for data sets with fewer than 100 observations).
- Does not support the clustering of nominal variables compared to the latest version of `PROC HPCLUS`.
- Reports results for only a single k value to determining number of clusters k, which is generally `k=MAXCLUSTERS`. Should try different values of `MAXCLUSTERS=` & leverage pseudo F statistics or Cubic Clustering Criterion to find the best k.
-  Highly efficient but single-threaded (compared to `PROC HPCLUS`) procedure that decreases execution time by locating non-random cluster seeds.
---

## R (stats)

### Function / Procedure
`kmeans(x, centers)` from base R `stats` package.

### Inputs
- **Required**: 
    - `x` (data)
    - `centers` (the number of clusters k, or a set of initial (distinct) cluster centres; if `centers` is not a number, random initialization is used)
- **Optional**:
    - `iter.max` (the maximum number of iterations allowed)
    - `nstart` (if centers is a number, how many random sets should be chosen?)
    - `algorithm` (character: may be abbreviated. "Hartigan-Wong", "Lloyd", "Forgy", "MacQueen". Note that "Lloyd" and "Forgy" are alternative names for one algorithm)
    - `object` (an R object of class "kmeans", typically the result ob of ob <- kmeans(..).)
    - `method` (character: may be abbreviated. "centers" causes `fitted` to return cluster centers (one for each input point) and "classes" causes `fitted` to return a vector of class assignments)
    - `trace` (logical or integer number, currently only used in the default method ("Hartigan-Wong"): if positive (or true), tracing information on the progress of the algorithm is produced. Higher values may produce more tracing information)

### Outputs
kmeans returns an object of class "kmeans" with the following components:
- `cluster` (A vector of integers (from 1:k) indicating the cluster to which each point is allocated)
- `centers` (A matrix of cluster centres)
- `totss` (The total sum of squares)
- `withinss` (Vector of within-cluster sum of squares)
- `tot.withinss` (Total within-cluster sum of squares)
- `betweenss` (The between-cluster sum of squares)
- `size` (The number of points in each cluster)
- `iter` (The number of (outer) iterations)
- `ifault` (integer: indicator of a possible algorithm problem)


### Sample Code
```r
# Normalization done manually 
airbnb[, c("price", "number_of_reviews")] = scale(airbnb[, c("price", "number_of_reviews")])
# Get the two columns of interest
airbnb_2cols <- data[, c("price", "number_of_reviews")]
set.seed(123)
km.out <- kmeans(airbnb_2cols, centers = 3, nstart = 20)
km.out
```

### Limitations / Notes
- The algorithm of Hartigan and Wong (1979) is used by default.
- Manual standardization is required.
- Supports multiple random starts (specified by `nstart`). Trying a large number of`nstart` (20, 30, 50 etc.) is recommended.

---

## Python (Scikit-Learn)

### Function / Procedure
`class sklearn.cluster.KMeans` in Scikit-Learn package

- `class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')`
- `fit(X, y=None, sample_weight=None)`: Compute k-means clustering. Returns self object.
- `fit_predict(X, y=None, sample_weight=None)`: Compute cluster centers and predict cluster index for each sample. Returns labels, ndarray of shape (n_samples,) Index of the cluster each sample belongs to.
- `fit_transform(X, y=None, sample_weight=None)`: Compute clustering and transform X to cluster-distance space. Returns X_new, ndarray of shape (n_samples, n_clusters) X transformed in the new space.
- `get_params(deep=True)`: Get parameters for this estimator.
- `predict(X)`: Predict the closest cluster each sample in X belongs to. Returns labels, ndarray of shape (n_samples,) Index of the cluster each sample belongs to.
- `set_params(**params)`: Set the parameters of this estimator.
- `transform(X)`: Transform X to a cluster-distance space. In the new space, each dimension is the distance to the cluster centers.

### Inputs
- **Required**: 
    - `X` {array-like, sparse matrix} of shape (n_samples, n_features)
- **Optional**:
    - `sample_weight` array-like of shape (n_samples,), default=None
    - `n_clusters` int, default=8. The number of clusters to form as well as the number of centroids to generate. A hyperparameter. 
    - `init` {‘k-means++’, ‘random’}, Method for initialization, callable or array-like of shape (n_clusters, n_features), default=’k-means++’. A hyperparameter. 
    - `n_init` ‘auto’ or int, default=’auto’. Number of times the k-means algorithm is run with different centroid seeds.
    - `max_iter` int, default=300. Maximum number of iterations of the k-means algorithm for a single run. A hyperparameter.
    - `tol` float, default=1e-4. Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. A hyperparameter.
    - `verbose` int, default=0. Verbosity mode.
    - `random_state` int, RandomState instance or None, default=None. Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
    - `copy_x` bool, default=True. If True, the original data is not modified. If False, the original data is modified during computation and restored afterward, which may introduce small numerical differences.
    - `algorithm` {“lloyd”, “elkan”}, default=”lloyd”. K-means algorithm to use. “lloyd” is the classical EM-style algorithm. “elkan” is faster on some datasets with well-separated clusters, but uses more memory. A hyperparameter.

### Outputs
The followings are attributes of the class `class sklearn.cluster.KMeans`, which are accessible from the object returned by the function `fit(X, y=None, sample_weight=None)`:

- `cluster_centers_` ndarray of shape (n_clusters, n_features). Coordinates of cluster centers.
- `labels_` ndarray of shape (n_samples,). Labels of each point.
- `inertia_` float. Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
- `n_iter_` int. Number of iterations run.
- `n_features_in_` int. Number of features seen during fit.
- `feature_names_in_` ndarray of shape (n_features_in_,). Names of features seen during fit. Defined only when X has feature names that are all strings.

### Sample Code
```python
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_
kmeans.predict([[0, 0], [12, 3]])
kmeans.cluster_centers_
```

### Limitations / Notes
- Inertia: within-cluster sum-of-squares criterion, which the k-means algorithm aims to minimize. Inertia is not a normalized metric. Running a dimensionality reduction algorithm prior to k-means clustering is recommended
- k-means++ initialization scheme is implemented (use the `init='k-means++'` parameter). This initialization tend to lead to better results than random initialization.
- The input `X` will be converted to C ordering, which will cause a memory copy if the given data is not C-contiguous. If a sparse matrix is passed, a copy will be made if it’s not in CSR format.
---

## Python (Scipy)

### Function / Procedure
`scipy.cluster.vq.kmeans(obs, k_or_guess, iter=20, thresh=1e-05, check_finite=True, *, rng=None)`

### Inputs
[Required and optional parameters.]
- **Required**: 
    - `obs`, ndarray (each row of the M by N array is an observation vector)
    - `k_or_guess`, int or ndarray. Random initialization if passed in the number of centroids to generate. Alternatively, passing a k by N array specifies the initial k centroids.
- **Optional**: `iter` (number of iterations), `thresh` (distortion change threshold as a stopping criterion) etc.

### Outputs
- codebook, ndarray: A k by N array of k centroids
- distortion, float: mean (non-squared) Euclidean distance between the observations passed and the centroids generated.

### Sample Code
```python
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
features  = np.array([[ 1.9,2.3],
                      [ 1.5,2.5],
                      [ 0.8,0.6],
                      [ 0.4,1.8],
                      [ 0.1,0.1],
                      [ 0.2,1.8],
                      [ 2.0,0.5],
                      [ 0.3,1.5],
                      [ 1.0,1.0]])
whitened = whiten(features)
book = np.array((whitened[0],whitened[2]))
kmeans(whitened,book)
```

### Limitations / Notes
- `whiten` (to normalize a group of observations on a per feature basis) must be called prior to passing an observation matrix to kmeans.
- Check skicit learn for optimal performance library.

# Summary

SAS `HPCLUS` is optimized for large datasets with high-performance computing (parallelization, automatic standardization, and imputation) and supports both interval and nominal variables. It offers cluster number estimation via the aligned box criterion ``(NOC=ABC)``. 

`FASTCLUS` is a more lightweight SAS procedure best for moderate-sized, numeric-only data. It lacks multithreading and automatic preprocessing, but performs quickly using deterministic centroid initialization—though; requires standardization via PROC STDIZE.

In open-source environments, R’s kmeans and Scikit-Learn’s KMeans offer customizable initialization ``(nstart/init)``. Python  Scikit-Learn stands out with its object-oriented interface, multiple initialization strategies (e.g., k-means++), and integration with the Python data science ecosystem. SciPy’s kmeans is the most manual and minimalist tool.

# References

- Columbia University - K-Means Cluster Analysis: https://www.publichealth.columbia.edu/research/population-health-methods/k-means-cluster-analysis
- UC Business Analytics R Programming Guide: https://uc-r.github.io/kmeans_clustering#fn:scale
- IBM: https://www.ibm.com/think/topics/k-means-clustering; https://www.ibm.com/docs/en/spss-statistics/beta?topic=features-k-means-cluster-analysis
- TIBCO Statistica® 14.0.1: https://docs.tibco.com/pub/stat/14.0.1/doc/html/UsersGuide/user-guide/k-means-clustering.htm?TocPath=Data+Mining%7CStatistics%7CMultivariate+Exploratory+Techniques%7CCluster+Analysis+Overview%7CStatistical+Significance+Testing%7CTwo-Way+Joining%7C_____1
- Yang, W.-C.; Lai, J.-P.; Liu, Y.-H.; Lin, Y.-L.; Hou, H.-P.; Pai, P.-F. Using Medical Data and Clustering Techniques for a Smart Healthcare System. Electronics 2024, 13, 140. https://doi.org/10.3390/electronics13010140
- Scipy KMeans Clustering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html
- Sklearn KMeans Clustering: https://scikit-learn.org/stable/modules/clustering.html#k-means (user guide); https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans (syntax)
- SAS FASTCLUS: https://documentation.sas.com/doc/en/statug/15.2/statug_fastclus_syntax01.htm
- SAS HPCLUS: https://documentation.sas.com/doc/en/emhpprcref/14.2/emhpprcref_hpclus_overview.htm
- SAS Communities Library: https://communities.sas.com/t5/SAS-Communities-Library/Tip-K-means-clustering-in-SAS-comparing-PROC-FASTCLUS-and-PROC/ta-p/221369
- R KMeans: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/kmeans.html
- Sample code (R): https://www.datacamp.com/tutorial/k-means-clustering-r

