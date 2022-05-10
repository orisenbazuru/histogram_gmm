# Histogram-based Gaussian Mixture Model 📊

This package updates the `GaussianMixture`  model implementation in `sklearn`  to support grouped input (i.e. histogram-like) data.

The input matrix `X` is of shape `(nsamples, nfeatures+1)` where the last column represents the counts of the observed unique samples.

### Example I (full parameters estimation)

```python
import numpy as np
from hgmm.histogram_gmm import HistogramGaussianMixture
# each sample in X has two features and
# the last column representing the count of each sample 
X = np.array([[1, 2, 5], [1, 4, 4], [1, 0, 2], [10, 2,9], [10, 4,11], [10, 0,15]])
hgmm = HistogramGaussianMixture(n_components=2, fixed_means=False, random_state=42).fit(X)

>>> hgmm.means_
    array([[ 1.        ,  2.36363636],
           [10.        ,  1.77142857]])
>>> hgmm.covariances_
    array([[[1.00000000e-06, 3.58573139e-32],
            [3.58573139e-32, 2.04958778e+00]],

           [[1.00000000e-06, 0.00000000e+00],
            [0.00000000e+00, 2.91918467e+00]]])
>>> hgmm.predict([[0, 0, 1], [12, 3, 1]])
    array([1, 0])


```

### Example II (fixing the means of the Gaussians)

```python
import numpy as np
from hgmm.histogram_gmm import HistogramGaussianMixture
# each sample in X has two features and
# the last column representing the count of each sample 
X = np.array([[1, 2, 5], [1, 4, 4], [1, 0, 2], [10, 2,9], [10, 4,11], [10, 0,15]])
hgmm = HistogramGaussianMixture(n_components=2,
                                random_state=42,
                                means_init = np.array([[1., 2.36],[10., 1.8]]), # means of shape (n_components, n_features)
                                fixed_means=True, # to fix the means during learning (it will be equal to means_init)
                                covariance_type = 'full').fit(X)

>>> hgmm.means_
    array([[ 1.  ,  2.36],
           [10.  ,  1.8 ]])
>>> hgmm.covariances_
    array([[[1.000000e-06, 0.000000e+00],
            [0.000000e+00, 2.049601e+00]],

           [[1.000000e-06, 0.000000e+00],
            [0.000000e+00, 2.920001e+00]]])
>>> hgmm.predict([[0, 0, 1], [12, 3, 1]])
    array([1, 0])


```

The `HistogramGaussianMixture` has the same `API` as the `GaussianMixture` class in sklearn (see [here](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) for further info).


### Notebooks

Example for how to run `HistogramGaussianMixture` model on count (i.e. histogram-based) data is found in the `notebooks` folder.
- `cfdna_data.ipynb` for full parameters estimation
- `cfdna_data_with_fixedmeans.ipynb` parameters estimation while fixing the means