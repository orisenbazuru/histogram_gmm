# Histogram-based Gaussian Mixture Model 📊

This package updates the `GaussianMixture`  model implementation in `sklearn`  to support grouped input (i.e. histogram-like) data.

The input matrix `X` is of shape `(nsamples, nfeatures+1)` where the last column represents the counts of the observed unique samples.

### Example 

```python
import numpy as np
from hgmm.histogram_gmm import HistogramGaussianMixture
# each sample in X has two features and
# the last column representing the count of each sample 
X = np.array([[1, 2, 5], [1, 4, 4], [1, 0, 2], [10, 2,9], [10, 4,11], [10, 0,15]])
hgmm = HistogramGaussianMixture(n_components=2, random_state=42).fit(X)

>>> hgmm.means_
    array([[10.        ,  1.77142857],
           [ 1.        ,  2.36363636]])
>>> hgmm.predict([[0, 0, 1], [12, 3, 1]])
    array([1, 0])


```
The `HistogramGaussianMixture` has the same `API` as the `GaussianMixture` class in sklearn (see [here](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) for further info).

### Notebooks

Example for how to run `HistogramGaussianMixture` model on count (i.e. histogram-based) data is found in the `notebooks` folder (see `cfdna_data.ipynb`)