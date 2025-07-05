---
title: 1105 - PCA - Requirements Analysis

---

# 1105 - PCA - Requirements Analysis

<!-- Filename: 1105 - PCA - Requirements Analysis.md -->

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

PCA is a linear dimensionality reduction technique in which data is linearly transformed onto a new coordinate system built from a group of uncorrelated "principal components" to identify the "directions" (revealed by the PCs) capturing most of the data variance. Principal component analysis has applications in exploratory data analysis, visualization (by reducing dimensions of the data), and data preprocessing (in denoising the data). It can also be used for exploring polynomial relationships and for multivariate outlier detection. 


## Hypotheses
As proposed by a paper from Chung et al in 2014, the approach of jackstraw is a significance test for associations between feature variables and a given set, subset, or linear combination of PCs estimated from the data. If `Y = BL + E` describes how unknown parameters of interest `B` quantifies the relationship between latent variables (PCs) `L` and the observed variable (mean centered, unit-variant)`Y`, then `H0: b_i = 0` and `H1: b_i ≠ 0`, for `b_i` being the `i`th row of `B`, the matrix of unknown coefficients.

## Assumptions
- To apply PCA, it is assumed that variables have linear relationships with each other.
- PCA requires the data being standardized before applying the analysis. This is because PCA is highly sensitive to feature scaling and features with large scale may dominate the result even with low variance. PCA is sensitive to outliers. 
- PCA is not applicable to non-discrete data types (categorical data). 
- In technical implementations, a lack of missing values is assumed. 
- The underlying algorithm under PCA, eigendecomposition of covariance matrix, can make it computationally expensive confronting large datasets.

## Test Statistic & P-value
Not applicable.


# Package Implementations

## SAS (PROC PRINCOMP)

### Function / Procedure
`PROC PRINCOMP` procedure from SAS® Viya®

### Inputs
- `PROC PRINCOMP` Statement: `DATA=`, `COV` (computes the principal components from the covariance matrix), `N=` (specifies the number of principal components to be computed), `STD` (standardizes the principal component scores), `PLOTS=` (specifies options that control the details of the plots)
- Other statments including `BY`, `FREQ`, `ID`, `PARTIAL` (analyze a partial correlation or covariance matrix), `VAR`, `WEIGHT`

### Outputs
- Produced from`OUT= Data Set` or `OUTSTAT= Data Set` (also avaialble from ODS Table Names):
    - Descriptive statistics: mean and std dev per variable
    - If `COV` is specified: correlation matrix or covariance matrix; total variance
    - If `PARTIAL` is specified: partial regression results (R², RMSE, and coefficients); partial correlation/covariance matrix 
    - Eigenvalues (plus variance explained and cumulative proportion); eigenvectors (principal component loadings)
- Specified in ODS Graphics: `PaintedScorePlot`, `ScorePlot`, `ScreePlot`, `VariancePlot` etc.

### Sample Code
```sas
proc princomp data=Temperature cov plots=score(ellipse);
   var July January;
   id CityId;
run;
```

---

## SAS (PROC FACTOR)

### Function / Procedure
`PROC FACTOR` procedure from SAS® Viya®

### Inputs
- `METHOD=PRINCIPAL` (or default): selects principal components extraction.
- `N=` or `NF=`: specify number of components to extract.
- `PLOTS=SCREE` (via ODS Graphics): requests scree plot of eigenvalues.
- `COV`: uses covariance matrix instead of correlation.
- `STD`: standardizes scores to unit variance.
- `PARTIAL` statement: enables partial correlation/covariance-based scoring.

### Outputs
- Eigenvalues: includes differences, % variance explained, and cumulative variance; eigenvectors (loadings): principal component coefficients.
- Descriptive stats: mean, std dev, etc. (unless `NOPRINT`).
- With `COV`: uses covariance matrix, shows total variance and scoring coefficients; with `PARTIAL`: outputs partial correlations, regression coefficients, and R².
- ODS Graphics : scree plot and variance explained, explained variance plot, initial (unrotated) loadings, rotated loadings etc.


### Sample Code
```sas
proc princomp data=Temperature cov plots=score(ellipse);
   var July January;
   id CityId;
run;
```

### Limitations / Notes (`PROC PRINCOMP` v.s. `PROC FACTOR`)

- Both omit missing values.
- `PROC_FACTOR` scores are smaller due to unit variance assumption; rescale to match `PROC_PRINCOMP` eigenvectors.
- `PROC_PRINCOMP` supports partial correlation/covariance scores.
- `PROC_PRINCOMP` lacks rotation comapred to`PROC_FACTOR`.

---

## R (stats)

### Function / Procedure
`prcomp(x,...)` function from R's `stats` package.

### Inputs
`prcomp(x, center = TRUE, scale. = FALSE, rank. = NULL)`
- **Required**: `x` (numeric matrix or data frame)
- **Optional** (not limited to):
    - `center`: whether the variables should be centered to have mean zero.
    - `scale.`: whether the variables should be scaled to have unit variance. 
    - `rank.`: number of principal components to retain. If NULL, all components are retained.

### Outputs
`prcomp` returns a list with class "prcomp" containing the following components:
- `sdev`: the standard deviations of the principal components (i.e., the square roots of the eigenvalues of the covariance/correlation matrix).
- `rotation`: the matrix of variable loadings (i.e., a matrix whose columns contain the eigenvectors).
- `x` if `retx` (optional input) is true the value of the rotated data (the centred (and scaled if requested) data multiplied by the rotation matrix) is returned. 

The `summary()` of `prcomp` returns the standard deviation, proportion of variance, and cumulative proportion of each PC.


### Sample Code
```r
pca_result <- prcomp(data, scale. = TRUE)
summary(pca_result)

scree_data <- data.frame(
  Component = 1:length(pca_result$sdev),
  Variance = pca_result$sdev^2 / sum(pca_result$sdev^2))
```

### Limitations / Notes
- Uses SVD (not eigen) on centered and scaled data; more numerically accurate than princomp.
- `prcomp` and `princomp` differ: `prcomp` uses SVD; `princomp` uses `eigen`, so loadings may vary.
- `prcomp` computes variances with the standard divisor.

---


## Python (sklearn)

### Function / Procedure
The class `sklearn.decomposition.PCA` has functions (not limited to):
- `fit(x)`: takes in input data`x`, fits the data, and returns the PCA class instance
- `fit_transform(x)`: fit data and returns new data `x`
- `get_covariance()`: get covariance of data
- `score(x)`: log likelihood of data based on the model (probabilistic PCA)

### Inputs
- **Optional:** (all parameters are optional and have default values)
    - `n_components`: number of components to keep (default = min(number of features, number of data points))
    - `whiten`: `components_` vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances
    - `svd_solver`: specifies algorithm for computing, the default `auto` selects `covariance_eigh`, `full` (full SVD) and `randomized` (randomized SV) based on input shape; other values include `arpack` (truncated SVD)
    - `tol`, `random_state`, etc.

### Outputs
- `fit(x)` returns class instance; `fit_transform(x)` returns the transformed data 
- The class `PCA` has the following attributes:
    - `components_`: ndarray of shape (n_components, n_features). Principal axes in feature space, representing the directions of maximum variance in the data.
    - `explained_variance_`: ndarray of shape (n_components,). The amount of variance explained by each of the selected components. The variance estimation uses n_samples - 1 degrees of freedom. Equal to n_components largest eigenvalues of the covariance matrix of X.
    - `explained_variance_ratio_`: ndarray of shape (n_components,). Percentage of variance explained by each of the selected components.
    - `singular_values_`, `n_components_`, `noise_variance_` etc.


### Sample Code
```python
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
```

### Limitations / Notes
- Only supports sparse data.
- Uses `IncrementalPCA` for minibatch, `KernelPCA`/`TruncatedSVD` for advanced cases.
- `whiten=True` scales components to unit variance after centering.
- `svd_solver='randomized' offers faster, approximate computation for large datasets.

# Summary
- SAS (`PROC PRINCOMP`): perform unrotated PCA, supports partial correlation/covariance.
- SAS (`PROC FACTOR`): can perform PCA, supports rotation and richer factor analysis.
- R (`prcomp`): numerically stable, SVD-based PCA, compatible with R's `stats` package ecosystem.
- Python (`sklearn.decomposition.PCA`): has customizable solver, supports batch and incremental processing with supplementary classes, integrates with broader ML pipelines.


# References

- PCA method overview: https://www.keboola.com/blog/pca-machine-learning
- PROC PRINCOMP v.s. PROC FACTOR: https://documentation.sas.com/doc/en/pgmsascdc/v_063/statug/statug_intromult_sect002.htm
- PCA Significance test original paper: https://academic.oup.com/bioinformatics/article/31/4/545/2748186?login=false
- PCA significance test in R: https://cbml.science/post/association-test-with-principal-components/
- R prcomp (GeeksforGeeks) code sample: https://www.geeksforgeeks.org/prcomp-in-r/
- R prcomp documentation: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/prcomp\
- Comparing two R PCA packages: https://cran.r-project.org/web/packages/LearnPCA/vignettes/Vig_07_Functions_PCA.pdf
- Sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- SAS PROC PRINCOMP Procedure: https://documentation.sas.com/doc/en/pgmsascdc/v_063/statug/statug_princomp_overview.htm
- SAS PROC PRINCOMP code example: https://documentation.sas.com/doc/en/pgmsascdc/v_063/statug/statug_code_princex1.htm
- SAS PROC FACTOR Procedure: https://documentation.sas.com/doc/en/statug/15.2/statug_factor_toc.htm
- SAS PROC FACTOR PCA Example: https://documentation.sas.com/doc/en/statug/15.2/statug_factor_examples01.htm#statug_factor004176
