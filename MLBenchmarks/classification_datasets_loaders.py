import pandas as pd
from sklearn import preprocessing

def load_wine():

    df = pd.read_csv('datasets/Classification/wine/wine.data', sep=',',header=None)
    df = df.dropna()
    df = df.to_numpy()

    target = df[:,0]
    data = df[:,1:]

    dataset = {'target': target,
            'data': data}

    return dataset

def load_spambase():

    df = pd.read_csv('datasets/Classification/spambase/spambase.data', sep=',',header=None)
    df = df.dropna()
    df = df.to_numpy()

    target = df[:,-1]
    data = df[:,0:-1]

    dataset = {'target': target,
            'data': data}

    return dataset



def load_student_dropout():

    label_encoder = preprocessing.LabelEncoder()
    df = pd.read_csv('datasets/Classification/predict+students+dropout+and+academic+success/data.csv', sep=';')

    df = df.dropna()

    target = 'Target'

    df[target] = label_encoder.fit_transform(df[target])
    
    dataset = {'target': df[target].to_numpy(),
            'data': df.drop([target],axis=1).to_numpy()}

    return dataset


def load_dry_bean():
    
    label_encoder = preprocessing.LabelEncoder()

    df = pd.read_excel('datasets/Classification/dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx')
    
    df['Class'] = label_encoder.fit_transform(df['Class'])
    
    dataset = {'target': df['Class'].to_numpy(),
            'data': df.drop(['Class'],axis=1).to_numpy()}

    return dataset

def load_mushroom():

    df = pd.read_csv('datasets/Classification/mushroom/agaricus-lepiota.data', sep=',', header = 0,
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
                'data': df.drop(['class'],axis=1).to_numpy()}

    return dataset