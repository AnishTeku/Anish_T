# -- coding: utf-8 --

#Importing libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import sklearn.cluster as cluster
import scipy.optimize as opt

#cluster_tools and errors are used from class
import cluster_tools as ct
import errors as err
import importlib

#read file function to read csv files


def read_file(filepath):
    '''
    This function generates dataframe from file in the given filepath

    Parameters
    ----------
    filepath : STR
        File path or location.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame created from given filepath.

    '''
    df = pd.read_csv(filepath, skiprows=4)
    df = df.set_index('Country Name', drop=True)
    df = df.loc[:, '1960':'2021']

    return df

#transpose file to create transpose of given data


def transpose_file(df):
    '''
    This function creates transpose of given dataframe

    Parameters
    ----------
    df  : pandas.DataFrame
        DataFrame for which transpose to be found.

    Returns
    -------
    data_tr : pandas.DataFrame
        Transposed DataFrame of given DataFrame.

    '''
    df_tr = df.transpose()

    return df_tr

#correlation & scattermatrix for plotting matrix and scatter plots


def correlation_and_scattermatrix(df):
    '''
    This function plots correlation matrix and scatter plots
    of data among columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame for which analysis will be done.

    Returns
    -------
    None.

    '''
    corr = df.corr()
    print(corr)
    plt.figure(figsize=(10, 10))
    plt.matshow(corr, cmap='coolwarm')

    # xticks and yticks for corr matrix
    plt.title('CORRELATION BETWEEN YEARS & COUNTRIES OVER LIFE EXPECTANCY',
              fontsize=8)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.show()

    pd.plotting.scatter_matrix(df, figsize=(12, 12), s=5, alpha=0.8)
    plt.show()

    return
