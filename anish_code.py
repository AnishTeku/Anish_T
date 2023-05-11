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

#cluster number to calculate based on silhouette score


def cluster_number(df, df_normalised):
    '''
    This function calculates the best number of clusters based on silhouette
    score

    Parameters
    ----------
    df : pandas.DataFrame
        Actual data.
    df_normalised : pandas.DataFrame
        Normalised data.

    Returns
    -------
    INT
        Best cluster number.

    '''

    clusters = []
    scores = []
    # loop over number of clusters
    for ncluster in range(2, 10):

        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Cluster fitting
        kmeans.fit(df_normalised)
        lab = kmeans.labels_

        # Silhoutte score over number of clusters
        print(ncluster, skmet.silhouette_score(df, lab))

        clusters.append(ncluster)
        scores.append(skmet.silhouette_score(df, lab))

    clusters = np.array(clusters)
    scores = np.array(scores)

    best_ncluster = clusters[scores == np.max(scores)]
    print()
    print('best n clusters', best_ncluster[0])

    return best_ncluster[0]

#clusters and centers to plot the centers of given data


def clusters_and_centers(df, ncluster, y1, y2):
    '''
    This function will plot clusters and its centers for given data

    Parameters
    ----------
    df : pandas.DataFrame
        Data for which clusters and centers will be plotted.
    ncluster : INT
        Number of clusters.
    y1 : INT
        Column 1
    y2 : INT
        Column 2

    Returns
    -------
    df : pandas.DataFrame
        Data with cluster labels column added.
    cen : array
        Cluster Centers.

    '''
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df)

    labels = kmeans.labels_
    df['labels'] = labels
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    cen = np.array(cen)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    sc = plt.scatter(df[y1], df[y2], 10, labels, marker="o", cmap=cm)
    plt.title('CLUSTERS OF COUNTIRES OVER LIFE EXPECTANCY IN 1960 AND 2020')
    plt.scatter(xcen, ycen, 45, "k", marker="s")
    plt.legend(*sc.legend_elements(), title='clusters')
    plt.xlabel(f"Life Expectancy({y1})")
    plt.ylabel(f"Life Expectancy({y2})")

    plt.show()

    print()
    print(cen)

    return df, cen
