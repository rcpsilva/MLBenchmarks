import pandas as pd
from sklearn import preprocessing

def load_breast_cancer_wisconsin():
    label_encoder = preprocessing.LabelEncoder()
    df = pd.read_csv('/content/wdbc.data',header=None)
    df = df.dropna()

    cat = df.select_dtypes(exclude=['number'])

    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])

    df = df.to_numpy()
    target = df[:,1]
    data = df[:,2:-1] 

    dataset = {'target':target,
               'data':data,
               'info':'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
               'date_access':'2023-10-19'}

    return dataset

def load_soybean_large():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/soybean%2Blarge/soybean-large.data',header=None,na_values='?')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])

    df = df.to_numpy()

    target = df[:,0]
    data = df[:,1:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/90/soybean+large',
            'date_access':'2023-09-20'}

    return dataset    
    

def load_spect():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/spect%2Bheart/SPECT_all.csv',header=None)
    
    df[df.columns] = df[df.columns].apply(pd.to_numeric)

    df = df.to_numpy()

    target = df[:,0]
    data = df[:,1:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/95/spect+heart',
            'date_access':'2023-09-20'}

    return dataset

def load_spectf():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/spect%2Bheart/SPECTF_all.csv',header=None)
    
    df = df.to_numpy()

    target = df[:,0]
    data = df[:,1:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/95/spect+heart',
            'date_access':'2023-09-20'}

    return dataset
    
def load_obesity_eating_habits():

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition/ObesityDataSet_raw_and_data_sinthetic.csv')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])

    df = df.to_numpy()

    target = df[:,-1]
    data = df[:,0:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition',
            'date_access':'2023-09-20'}

    return dataset

def load_wine():

    file_path = 'https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/wine/wine.data'


    df = pd.read_csv(file_path, sep=',',header=None)
    df = df.dropna()
    df = df.to_numpy()

    target = df[:,0]
    data = df[:,1:]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/109/wine',
            'date_access':'2023-09-12'}

    return dataset

def load_spambase():

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/spambase/spambase.data', sep=',',header=None)
    df = df.dropna()
    df = df.to_numpy()

    target = df[:,-1]
    data = df[:,0:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/94/spambase',
            'date_access':'2023-09-12'}

    return dataset

def load_student_dropout():

    label_encoder = preprocessing.LabelEncoder()
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/predict+students+dropout+and+academic+success/data.csv', sep=';')

    df = df.dropna()

    target = 'Target'

    df[target] = label_encoder.fit_transform(df[target])
    
    dataset = {'target': df[target].to_numpy(),
            'data': df.drop([target],axis=1).to_numpy(),
            'info':'https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success',
            'date_access':'2023-09-12'}

    return dataset

def load_dry_bean():
    
    label_encoder = preprocessing.LabelEncoder()

    df = pd.read_excel('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx')
    
    df['Class'] = label_encoder.fit_transform(df['Class'])
    
    dataset = {'target': df['Class'].to_numpy(),
            'data': df.drop(['Class'],axis=1).to_numpy(),
            'info':'https://archive.ics.uci.edu/dataset/602/dry+bean+dataset',
            'date_access':'2023-09-12'}

    return dataset

def load_mushroom():

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Classification/mushroom/agaricus-lepiota.data', sep=',', header = 0,
                     names = ['class',
                              'cap-shape',
                              'cap-surface',
                              'cap-color',
                              'bruises',
                              'odor',
                              'gill-attachment',
                              'gill-spacing',
                              'gill-size',
                              'gill-color',
                              'stalk-shape',
                              'stalk-root',
                              'stalk-surface-above-ring',
                              'stalk-surface-below-ring',
                              'stalk-color-above-ring',
                              'stalk-color-below-ring',
                              'veil-type',
                              'veil-color',
                              'ring-number',
                              'ring-type',
                              'spore-print-color',
                              'population',
                              'habitat'])
    
    label_encoder = preprocessing.LabelEncoder()
    cat = df.select_dtypes(exclude='number')

    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])


    dataset = {'target': df['class'].to_numpy(),
                'data': df.drop(['class'],axis=1).to_numpy(),
                'info':'https://archive.ics.uci.edu/dataset/73/mushroom',
                'date_access':'2023-09-12'}

    return dataset