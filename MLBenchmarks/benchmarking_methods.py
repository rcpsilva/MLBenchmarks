import psutil
import numpy as np
import pickle
import tempfile
import os
import json
from tqdm import tqdm
from sklearn.model_selection import cross_validate
import MLBenchmarks.classification_datasets_loaders as classification_datasets_loaders
import MLBenchmarks.regression_datasets_loaders as regression_datasets_loaders

def load_specific_datasets(loader_names):

    # Get a list of all attributes (including methods) of the load_datasets module
    module_attributes = dir(regression_datasets_loaders)

    # Filter out only the functions (methods)
    methods = [attr for attr in module_attributes if callable(getattr(regression_datasets_loaders, attr))]
    datasets = {}

    # Call each method in the module
    for method_name in methods:
        if method_name in loader_names:
            method = getattr(regression_datasets_loaders, method_name)
            if callable(method):
                print(f"Running {method_name} ...")
                dataset = method()
                datasets[method_name] = (dataset['data'], dataset['target'])
                
    return datasets

def load_regression_datasets():

    # Get a list of all attributes (including methods) of the load_datasets module
    module_attributes = dir(regression_datasets_loaders)

    # Filter out only the functions (methods)
    methods = [attr for attr in module_attributes if callable(getattr(regression_datasets_loaders, attr))]
    datasets = {}

    # Call each method in the module
    for method_name in methods:
        method = getattr(regression_datasets_loaders, method_name)
        if callable(method):
            print(f"Running {method_name} ...")
            dataset = method()
            datasets[method_name] = (dataset['data'], dataset['target'])
            
    return datasets

def load_classification_datasets():

    # Get a list of all attributes (including methods) of the load_datasets module
    module_attributes = dir(classification_datasets_loaders)

    # Filter out only the functions (methods)
    methods = [attr for attr in module_attributes if callable(getattr(classification_datasets_loaders, attr))]
    datasets = {}

    # Call each method in the module
    for method_name in methods:
        method = getattr(classification_datasets_loaders, method_name)
        if callable(method):
            print(f"Running {method_name} ...")
            dataset = method()
            datasets[method_name] = (dataset['data'], dataset['target'])
            
    return datasets

def run_cross_dataset_benchmark_models(models, datasets, metric_fn, output_file = None, cv=5):
    results = {}

    for model_name, model in tqdm(models.items()):
        model_results = {}
        
        for dataset_name, (X, y) in tqdm(datasets.items()):
            scores = measure_metric_cv(model, X, y, metric_fn, cv)
            for metric_name, metric_values in scores.items():
                if isinstance(metric_values, np.ndarray):
                    scores[metric_name] = metric_values.tolist()
            model_results[dataset_name] = scores
            model.fit(X,y)
            model_results[dataset_name]['memory_usage(MB)'] = measure_memory_usage(model)

        results[model_name] = model_results
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    return results

def run_cross_dataset_benchmark(datasets, model, metric_name, output_file=None, cv=5):
    
    results = {}

    for dataset_name, (X, y) in tqdm(datasets.items()):
        scores = measure_metric_cv(model, X, y, metric_name, cv)
        results[dataset_name] = scores
        model.fit(X,y)
        results[dataset_name]['memory_usage(MB)'] = measure_memory_usage(model)

    # Convert NumPy arrays to lists
    for dataset_name, dataset_results in results.items():
        for metric_name, metric_values in dataset_results.items():
            if isinstance(metric_values, np.ndarray):
                dataset_results[metric_name] = metric_values.tolist()

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    return results

def measure_metric_cv(model, X, y, metric_name, cv=5):
    scores = cross_validate(model, X, y, cv=cv, scoring=metric_name)    
    return scores

def measure_memory_usage(model):
    # Serialize the model using pickle
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    pickle.dump(model, temp_file)
    temp_file.close()

    # Get memory usage before loading the serialized model
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    # Load the serialized model to simulate memory usage
    with open(temp_file.name, 'rb') as file:
        loaded_model = pickle.load(file)

    # Get memory usage after loading the serialized model
    memory_after = process.memory_info().rss

    # Clean up the temporary file
    os.unlink(temp_file.name)

    # Calculate the memory usage difference
    memory_usage = (memory_after - memory_before) / (1024 ** 2)  # Convert to megabytes
    return memory_usage


