from MLBenchmarks.benchmarking_methods import load_regression_datasets,load_classification_datasets,run_cross_dataset_benchmark,run_cross_dataset_benchmark_models
import pytest
import MLBenchmarks.regression_datasets_loaders as regression_datasets_loaders 
import MLBenchmarks.classification_datasets_loaders as classification_datasets_loaders
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

class TestMLBenchmarks:

    def test_classification_datasets(self):
        datasets = load_classification_datasets()

        # Get a list of all attributes (including methods) of the load_datasets module
        module_attributes = dir(classification_datasets_loaders)

        # Filter out only the functions (methods)
        methods = [attr for attr in module_attributes if callable(getattr(classification_datasets_loaders, attr))]
        
        for method_name in methods:
            assert method_name in datasets, f"{method_name} not found in the dataset"
            assert isinstance(datasets[method_name][0], np.ndarray), f"[data] is not a NumPy array for {method_name}"
            assert np.issubdtype(datasets[method_name][0].dtype, np.number), f"[data] does not have a numeric data type for {method_name}"
            assert isinstance(datasets[method_name][1], np.ndarray), f"[target] is not a NumPy array for {method_name}"
            assert np.issubdtype(datasets[method_name][1].dtype, np.number), f"[target] does not have a numeric data type for {method_name}"
    
    def test_load_regression_datasets(self):
        datasets = load_regression_datasets()

        # Get a list of all attributes (including methods) of the load_datasets module
        module_attributes = dir(regression_datasets_loaders)

        # Filter out only the functions (methods)
        methods = [attr for attr in module_attributes if callable(getattr(regression_datasets_loaders, attr))]
        
        for method_name in methods:
            assert method_name in datasets, f"{method_name} not found in the dataset"
            assert isinstance(datasets[method_name][0], np.ndarray), f"[data] is not a NumPy array for {method_name}"
            assert np.issubdtype(datasets[method_name][0].dtype, np.number), f"[data] does not have a numeric data type for {method_name}"
            assert isinstance(datasets[method_name][1], np.ndarray), f"[target] is not a NumPy array for {method_name}"
            assert np.issubdtype(datasets[method_name][1].dtype, np.number), f"[target] does not have a numeric data type for {method_name}"
    
    def test_classification_cross_dataset_benchmark(self):

        # Benchmark a single model
        model = DecisionTreeClassifier()
        datasets = load_classification_datasets()
        metrics = ['accuracy','f1_weighted']
        other_metrics = ['fit_time','score_time']
        memory = 'memory_usage(MB)'
        folds = 3

        res = run_cross_dataset_benchmark(datasets, model, metrics, cv=folds)
        
        for dataset in res:
            for m in metrics:
                assert 'test_' + m in res[dataset], f'Metric {m} has not been computed for dataset {dataset}'
                assert len(res[dataset]['test_' + m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} was less then {folds}'
            for m in other_metrics:
                assert m in res[dataset], f'Metric {m} has not been computed for dataset {dataset}'
                assert len(res[dataset][m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} was less then {folds}'
            
            assert memory in res[dataset], f'Metric {memory} has not been computed for dataset {dataset}'


    def test_classification_cross_dataset_benchmark_models(self):
        # Benchmark multiple models
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree Classifier": DecisionTreeClassifier()
        }

        datasets = load_classification_datasets()

        metrics = ['accuracy','f1_weighted']
        
        other_metrics = ['fit_time','score_time']
        memory = 'memory_usage(MB)'
        folds = 3

        res = run_cross_dataset_benchmark_models(models, datasets, metrics, cv=folds) 

        for model in res:
            for dataset in res[model]:
                for m in metrics:
                    assert 'test_' + m in res[model][dataset], f'Metric {m} has not been computed for dataset {dataset} using model {model}'
                    assert len(res[model][dataset]['test_' + m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} using model {model} was less then {folds}'
                for m in other_metrics:
                    assert m in res[model][dataset], f'Metric {m} has not been computed for dataset {dataset} using model {model}'
                    assert len(res[model][dataset][m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} using model {model} was less then {folds}'
            
                assert memory in res[model][dataset], f'Metric {memory} has not been computed for dataset {dataset} using model {model}' 

    def test_regression_cross_dataset_benchmark(self):

        # Benchmark a single model
        model = DecisionTreeRegressor()
        datasets = load_regression_datasets()
        metrics = ['neg_mean_absolute_error','explained_variance']
        other_metrics = ['fit_time','score_time']
        memory = 'memory_usage(MB)'
        folds = 3

        res = run_cross_dataset_benchmark(datasets, model, metrics, cv=folds)
        
        for dataset in res:
            for m in metrics:
                assert 'test_' + m in res[dataset], f'Metric {m} has not been computed for dataset {dataset}'
                assert len(res[dataset]['test_' + m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} was less then {folds}'
            for m in other_metrics:
                assert m in res[dataset], f'Metric {m} has not been computed for dataset {dataset}'
                assert len(res[dataset][m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} was less then {folds}'
            
            assert memory in res[dataset], f'Metric {memory} has not been computed for dataset {dataset}'

        
    def test_regression_cross_dataset_benchmark_models(self):
        # Benchmark multiple models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }

        datasets = load_regression_datasets()

        metrics = ['neg_mean_absolute_error','explained_variance']
        
        other_metrics = ['fit_time','score_time']
        memory = 'memory_usage(MB)'
        folds = 3

        res = run_cross_dataset_benchmark_models(models, datasets, metrics, cv=folds) 

        for model in res:
            for dataset in res[model]:
                for m in metrics:
                    assert 'test_' + m in res[model][dataset], f'Metric {m} has not been computed for dataset {dataset} using model {model}'
                    assert len(res[model][dataset]['test_' + m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} using model {model} was less then {folds}'
                for m in other_metrics:
                    assert m in res[model][dataset], f'Metric {m} has not been computed for dataset {dataset} using model {model}'
                    assert len(res[model][dataset][m]) == folds, f'Number of evaluations from metric {m} on dataset {dataset} using model {model} was less then {folds}'
            
                assert memory in res[model][dataset], f'Metric {memory} has not been computed for dataset {dataset} using model {model}'         
    

if __name__ == "__main__":
    pytest.main()


    