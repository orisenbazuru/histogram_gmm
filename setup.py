from setuptools import setup

setup(name='HistogramGMM',
      version='0.0.1',
      description='Implementation of HistogramGaussianMixture model based on sklearn GaussianMixture API.',
      url='https://github.com/orisenbazuru/histogram_gmm',
      packages=['hgmm'],
      python_requires='>=3.6.0',
      install_requires=[
            'numpy',
            'pandas>=1.1.0',
            'scipy',
            'scikit-learn',
            'matplotlib'
      ],
      zip_safe=False)