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

#logistic function calculation with scale factor


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 
    and growth rate g"""

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f

#forecast life expectancy to optimize data of selected country


def forecast_life_expectancy(df, country, start_year, end_year):
    '''
    This function will analyse data and optimize to forecast 
    life_expectancy of selected 
    country

    Parameters
    ----------
    df : pandas.DataFrame
        Data for which forecasting analysis is performed.
    country : STR
        Country for which forecasting is performed.
    start_year : INT
        Starting year for forecasting.
    end_year : INT
        Ending year for forecasting.

    Returns
    -------
    None.

    '''
    df = df.loc[:, country]
    df = df.dropna(axis=0)

    df_life_expectancy = pd.DataFrame()

    df_life_expectancy['Year'] = pd.DataFrame(df.index)
    df_life_expectancy['Life_expectancy'] = pd.DataFrame(df.values)
    df_life_expectancy["Year"] = pd.to_numeric(df_life_expectancy["Year"])
    importlib.reload(opt)

    param, covar = opt.curve_fit(logistic, df_life_expectancy["Year"],
                                 df_life_expectancy["Life_expectancy"],
                                 p0=(1.2e12, 0.03, 1990.0))

    sigma = np.sqrt(np.diag(covar))

    year = np.arange(start_year, end_year)
    forecast = logistic(year, *param)
    low, up = err.err_ranges(year, logistic, param, sigma)
    plt.figure()
    plt.plot(df_life_expectancy["Year"], df_life_expectancy["Life_expectancy"],
             label="Life_expectancy", color='b')
    plt.plot(year, forecast, label="forecast", color='k')
    plt.fill_between(year, low, forecast, color="red", alpha=0.7)
    plt.fill_between(year, forecast, up, color="green", alpha=0.7)
    plt.title(f'LIFE EXPECTANCY PER CAPITA FORECAST FOR {country.upper()}')
    plt.legend(loc='upper left')
    plt.xlabel("YEAR")
    plt.ylabel("LIFE EXPECTANCY in YEARS")
    plt.savefig(f'{country}.png', bbox_inches='tight', dpi=300)
    plt.show()

    life_expectancy2030 = logistic(2030, *param)/1e9

    low, up = err.err_ranges(2030, logistic, param, sigma)
    sig = np.abs(up-low)/(2.0 * 1e9)
    print()
    print(f"Life_expectancy 2030 of {country}", life_expectancy2030*1e9,
          "+/-", sig*1e9)


#Reading life_expectancy Data
df_life_expectancy = read_file("life_expectancy_years.csv")
print(df_life_expectancy.describe())

#Finding transpose of life_expectancy Data
df_life_expectancy_tr = transpose_file(df_life_expectancy)
print(df_life_expectancy_tr.head())

#Selecting years for which correlation is done for further analysis
df_le = df_life_expectancy[['1960', "1970", "1980", "1990", "2000", "2010",
                            '2020']]
print(df_le.describe())

correlation_and_scattermatrix(df_le)
year1 = "1960"
year2 = "2020"

# Extracting columns for clustering
df_ex = df_le[[year1, year2]]
df_ex = df_ex.dropna()

# Normalising data and storing minimum and maximum
df_norm, df_min, df_max = ct.scaler(df_ex)

print()
print("Number of Clusters and Scores")
ncluster = cluster_number(df_ex, df_norm)

df_norm, cen = clusters_and_centers(df_norm, ncluster, year1, year2)

#Applying backscaling to get actual cluster centers
scen = ct.backscale(cen, df_min, df_max)
print('scen\n', scen)

df_ex, scen = clusters_and_centers(df_ex, ncluster, year1, year2)

print()
print('Countries in last cluster')
print(df_ex[df_ex['labels'] == ncluster-1].index.values)

#Forecast life_expectancy for India
forecast_life_expectancy(df_life_expectancy_tr, 'India', 1960, 2031)

#Forecast life_expectancy for United States
forecast_life_expectancy(df_life_expectancy_tr, 'United States', 1960, 2031)

#Forecast life_expectancy for Pakistan
forecast_life_expectancy(df_life_expectancy_tr, 'Pakistan', 1960, 2031)
