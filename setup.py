from setuptools import setup

setup(name='HistogramGMM',
      version='0.0.1',
      description='',
      url='https://github.com/orisenbazuru/histogram_gmm',
      packages=['hgmm'],
      python_requires='>=3.6.0',
      install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'scikit-learn',
            'matplotlib'
      ],
      zip_safe=False)