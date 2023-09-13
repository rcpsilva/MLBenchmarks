# MLBenchmarks

Benchmark your machine learning algorithms and pipelines with ease. Currently, regression and classification problems on tabular data. 

![GitHub](https://img.shields.io/github/license/rcpsilva/MLBenchmarks)
![GitHub last commit](https://img.shields.io/github/last-commit/rcpsilva/MLBenchmarks)
![GitHub stars](https://img.shields.io/github/stars/rcpsilva/MLBenchmarks?style=social)
![GitHub downloads](https://img.shields.io/github/downloads/rcpsilva/MLBenchmarks/latest/total)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
    - [Classification](#classification)
    - [Regression](#regression)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
pip install git+https://github.com/rcpsilva/MLBenchmarks.git@v.0.0.1-alpha#egg=MLBenchmarks
```

## Usage

See [example notebooks](https://github.com/rcpsilva/MLBenchmarks/tree/main/Examples) for more thorough code.
For simple use, see below.

### Classification

- Benchmarking a single model 

```python
from sklearn.tree import DecisionTreeClassifier
from benchmarking_methods  import load_classification_datasets, run_cross_dataset_benchmark

# Benchmark a single model
model = DecisionTreeClassifier()
datasets = load_classification_datasets()
metrics = ['accuracy','f1_weighted']

res = run_cross_dataset_benchmark(datasets, model, metrics,
                                'classification_single_model.json', cv=5)

```
- Benchmarking pipelines and models

```python

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from benchmarking_methods  import load_classification_datasets, run_cross_dataset_benchmark_models

# Benchmark pipelines
pipeline_linear_rf = Pipeline([
    ('feature_extraction', FeatureUnion([
        ('pca', PCA(n_components=5)),
        ('polynomial_features', PolynomialFeatures(degree=2)),
    ])),
    ('regressor', DecisionTreeClassifier())
])

# Add the modified pipeline to the models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Regressor": DecisionTreeClassifier(),
    "Pipeline (Linear + Decision Tree)": pipeline_linear_rf
}

metrics = ['accuracy','f1_weighted']
datasets = load_classification_datasets()

res = run_cross_dataset_benchmark_models(models, datasets, metrics, 'classification_pipeline_model.json', cv=5)

```

### Regression
- Benchmarking a single model 

```python
from sklearn.ensemble import RandomForestRegressor
from benchmarking_methods  import load_regression_datasets, run_cross_dataset_benchmark

# Benchmark a single model
model = RandomForestRegressor()
datasets = load_regression_datasets()
metrics = ['neg_mean_absolute_error','explained_variance','neg_mean_absolute_percentage_error']

res = run_cross_dataset_benchmark(datasets, model, metrics,
                                'single_model.json', cv=5)
```

- Benchmarking multiple models

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from benchmarking_methods  import load_regression_datasets, run_cross_dataset_benchmark_models

# Benchmark multiple models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest Regressor": RandomForestRegressor()
}

metrics = ['neg_mean_absolute_error','explained_variance','neg_mean_absolute_percentage_error']
datasets = load_regression_datasets()

res = run_cross_dataset_benchmark_models(models, datasets, metrics, 'multiple_model.json', cv=5)
```
- Benchmarking pipelines and models

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from benchmarking_methods  import load_regression_datasets, run_cross_dataset_benchmark_models

# Benchmark pipelines
pipeline_linear_rf = Pipeline([
    ('feature_extraction', FeatureUnion([
        ('pca', PCA(n_components=5)),
        ('polynomial_features', PolynomialFeatures(degree=2)),
    ])),
    ('regressor', RandomForestRegressor())
])

# Add the modified pipeline to the models dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Pipeline (Linear + Random Forest)": pipeline_linear_rf
}

metrics = ['neg_mean_absolute_error','explained_variance','neg_mean_absolute_percentage_error']
datasets = load_regression_datasets()

res = run_cross_dataset_benchmark_models(models, datasets, metrics, 'pipeline_model.json', cv=5)
```

## Folder Structure

Here's an overview of the folder structure of this repository:

+ MLBenchmarks
    + MLBenchmarks/
        + datasets/
            + Classification/
                + dry+bean+dataset/
                + mushroom/
                + predict+students+dropout+and+academic+success/
                + spambase/
                + wine/
            + Regression/
                + auto+mpg/
                + automobile/
                + student/
                + wine+quality/
        + Tests/
    + Examples/


## License

This project is licensed under the [GPL-3.0 license](LICENSE). You are free to use, modify, and distribute this code as per the terms of the license.
