import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_spm_demagnetization_analytical():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/spm%2Bdemagnetization/Dataset_SPM_demagnetization.csv')

    df = df.dropna()
    target = df['max_OL_analytical'].to_numpy()
    
    data = df.to_numpy()[:,0:-2]

    dataset = {'target': target,
            'data': data,
            'info': 'https://www.kaggle.com/datasets/mrjacopong/spm-demagnetization-dataset',
            'date_access': '2023-07-11'}
    
    return dataset 

def load_spm_demagnetization_FEM():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/spm%2Bdemagnetization/Dataset_SPM_demagnetization.csv')

    df = df.dropna()
    target = df['max_OL'].to_numpy()
    
    data = df.to_numpy()[:,0:-2]

    dataset = {'target': target,
            'data': data,
            'info': 'https://www.kaggle.com/datasets/mrjacopong/spm-demagnetization-dataset',
            'date_access': '2023-07-11'}
    
    return dataset 


def load_facebook_post_interactions():
    
    target_idx = 18

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')

    df = df.dropna()
    
    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 


def load_facebook_post_shares():
    
    target_idx = 17

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 


def load_facebook_post_likes():
    
    target_idx = 16

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()
    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 


def load_facebook_comments():
    
    target_idx = 15

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 


def load_facebook_liked_engaged():
    
    target_idx = 14

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 


def load_facebook_reach_liked():
    
    target_idx = 13

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 


def load_facebook_impressions_liked():
    
    target_idx = 12

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 


def load_facebook_post_consumptions():
    
    target_idx = 11

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 



def load_facebook_post_consumers():
    
    target_idx = 10

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 

def load_facebook_engaged_users():
    
    target_idx = 9

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 

def load_facebook_lifetime_impressions():
    
    target_idx = 8

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 

def load_facebook_lifetime_reach():
    
    target_idx = 7

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/facebook+metrics/dataset_Facebook.csv',
                     sep=';')
    df = df.dropna()

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,target_idx]
    data = df[:,0:7]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/368/facebook+metrics',
            'date_access': '2023-09-12'}
    
    return dataset 

def load_concrete_strength():

    df = pd.read_excel('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/concrete+compressive+strength/Concrete_Data.xls')

    df = df.to_numpy()

    target = df[:,-1]
    data = df[:,1:-1]
    
    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength',
            'date_access':'2023-09-12'}
    
    return dataset 

def load_obesity_levels():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition/ObesityDataSet_raw_and_data_sinthetic.csv')

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    df = df.to_numpy()

    target = df[:,-1]
    data = df[:,:-1]

    dataset = {'target': target,
            'data': data,
            'info': 'https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition',
            'date_access': '2023-09-12'}
    
    return dataset 

def load_bike_sharing_day():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/bike+sharing+dataset/day.csv')

    df = df.drop(['instant','dteday'],axis=1)
    df = df.to_numpy(dtype=np.float32)

    target = df[:,-1]
    data = df[:,:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
            'date_access':'2023-09-12'}
    
    return dataset 

def load_bike_sharing_hour():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/bike+sharing+dataset/hour.csv')   

    df = df.drop(['instant','dteday'],axis=1)
    df = df.to_numpy(dtype=np.float32)

    target = df[:,-1]
    data = df[:,:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
            'date_access':'2023-09-12'}
    
    return dataset 

def load_real_state_valuation():
    df = pd.read_excel('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/real+estate+valuation+data+set/RealEstateValuationDataSet.xlsx')

    df = df.to_numpy()

    target = df[:,-1]
    data = df[:,1:-1]
    
    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set',
            'date_access':'2023-09-12'}
    
    return dataset 

def load_forest_fires():

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/forest+fires/forestfires.csv')

    cat = df.select_dtypes(exclude='number')
    label_encoder = preprocessing.LabelEncoder()
    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])
    

    df = df.to_numpy()
    target = df[:,-1]
    data = df[:,:-1]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/162/forest+fires',
            'date_access':'2023-09-12'}
    
    return dataset                    

def load_energy_efficiency_y1():
    df = pd.read_excel('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/energy+efficiency/ENB2012_data.xlsx')

    df = df.to_numpy()

    target = df[:,8]
    data = df[:,:8]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/242/energy+efficiency',
            'date_access':'2023-09-12'}
    
    return dataset

def load_energy_efficiency_y2():
    df = pd.read_excel('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/energy+efficiency/ENB2012_data.xlsx')

    df = df.to_numpy()

    target = df[:,9]
    data = df[:,:8]

    dataset = {'target': target,
            'data': data,
            'info':'https://archive.ics.uci.edu/dataset/242/energy+efficiency',
            'date_access':'2023-09-12'}
    
    return dataset

def load_auto_mpg():
    
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/auto+mpg/auto-mpg.data', sep=',', header = 0,
                     names = ['blank',
                              'mpg',
                              'cylinders',
                              'displacement',
                              'horsepower',
                              'weight',
                              'acceleration',
                              'model year',
                              'origin',
                              'car name'])
    
    dataset = {'target': df['mpg'].to_numpy(),
                'data': df.drop(['blank','car name','mpg'],axis=1).to_numpy(),
                'info':'https://archive.ics.uci.edu/dataset/9/auto+mpg',
                'date_access':'2023-09-12'}

    return dataset

def load_wine_quality_red():
    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/wine+quality/winequality-red.csv',sep=';')
    dataset = {'target': df['quality'].to_numpy(),
                'data': df.drop(['quality'],axis=1).to_numpy(),
                'info':'https://archive.ics.uci.edu/dataset/186/wine+quality',
                'date_access':'2023-09-12'}
    return dataset

def load_wine_quality_white():

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/wine+quality/winequality-white.csv',sep=';')
    dataset = {'target': df['quality'].to_numpy(),
                'data': df.drop(['quality'],axis=1).to_numpy(),
                'info':'https://archive.ics.uci.edu/dataset/186/wine+quality',
                'date_access':'2023-09-12'}
    return dataset

def load_student_mat():
    label_encoder = preprocessing.LabelEncoder()

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/student/studentmat.csv',sep=';') # Carrega os dados
    df = df.dropna() # Retira valores faltantes

    cat = df.select_dtypes(exclude='number')

    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])

    dataset = {'target': df['G3'].to_numpy(),
                'data': df.drop(['G1','G2','G3'],axis=1).to_numpy(),
                'info':'',
                'info':'https://archive.ics.uci.edu/dataset/320/student+performance',
                'date_access':'2023-12-09'}
    return dataset

def load_student_por():
    label_encoder = preprocessing.LabelEncoder()

    df = pd.read_csv('https://raw.githubusercontent.com/rcpsilva/MLBenchmarks/main/MLBenchmarks/datasets/Regression/student/studentpor.csv',sep=';') # Carrega os dados
    df = df.dropna() # Retira valores faltantes

    cat = df.select_dtypes(exclude='number')

    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])

    dataset = {'target': df['G3'].to_numpy(),
                'data': df.drop(['G1','G2','G3'],axis=1).to_numpy(),
                'info':'https://archive.ics.uci.edu/dataset/320/student+performance',
                'date_access':'2023-12-09'}
    
    return dataset


