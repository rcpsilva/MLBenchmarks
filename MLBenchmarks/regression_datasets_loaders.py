import pandas as pd
from sklearn import preprocessing

def load_auto_mpg():
    
    df = pd.read_csv('datasets/Regression/auto+mpg/auto-mpg.data', sep=',', header = 0,
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
                'data': df.drop(['blank','car name','mpg'],axis=1).to_numpy()}

    return dataset

def load_wine_quality_red():
    df = pd.read_csv('datasets/Regression/wine+quality/winequality-red.csv',sep=';')
    dataset = {'target': df['quality'].to_numpy(),
                'data': df.drop(['quality'],axis=1).to_numpy()}
    return dataset

def load_wine_quality_white():

    df = pd.read_csv('datasets/Regression/wine+quality/winequality-white.csv',sep=';')
    dataset = {'target': df['quality'].to_numpy(),
                'data': df.drop(['quality'],axis=1).to_numpy()}
    return dataset

def load_student_mat():
    label_encoder = preprocessing.LabelEncoder()

    df = pd.read_csv('datasets/Regression/student/studentmat.csv',sep=';') # Carrega os dados
    df = df.dropna() # Retira valores faltantes

    cat = df.select_dtypes(exclude='number')

    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])

    dataset = {'target': df['G3'].to_numpy(),
                'data': df.drop(['G1','G2','G3'],axis=1).to_numpy()}
    return dataset

def load_student_por():
    label_encoder = preprocessing.LabelEncoder()

    df = pd.read_csv('datasets/Regression/student/studentpor.csv',sep=';') # Carrega os dados
    df = df.dropna() # Retira valores faltantes

    cat = df.select_dtypes(exclude='number')

    for col in cat.columns:
        df[col] = label_encoder.fit_transform(df[col])

    dataset = {'target': df['G3'].to_numpy(),
                'data': df.drop(['G1','G2','G3'],axis=1).to_numpy()}
    return dataset


