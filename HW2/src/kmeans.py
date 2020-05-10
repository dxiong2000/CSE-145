import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def find_best_k():
    """
    Function to determine the best value of k using graphs of SSE and Silhouette.
    :return: SSE and Silhouette plots
    """
    # define k
    k_clusters = 26

    data = pd.read_csv('../data/Customer_Churn_processed.csv')
    data = data.to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    print("starting k-means training")

    # fitting multiple k-means algorithms and storing SSE and silhouette in lists
    SSE = []
    sil = []
    # training k-means
    for k in range(2, k_clusters):
        print('training k-means for k = {}'.format(k))
        kmeans = KMeans(n_jobs=-1, n_clusters=k, init='k-means++')
        kmeans.fit(data_scaled)
        SSE.append(kmeans.inertia_)
        sil.append(silhouette_score(data_scaled, kmeans.labels_, metric='euclidean'))

    print("done training")

    # converting the results into a dataframe and plotting them
    # plot SSE
    frame = pd.DataFrame({'Cluster': range(2, k_clusters), 'SSE': SSE})
    plt.figure(figsize=(12, 6))
    plt.plot(frame['Cluster'], frame['SSE'], marker='o')
    plt.xticks(range(1,k_clusters))
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum Squared Error')

    plt.show()

    # plot silhouette
    frame = pd.DataFrame({'Cluster': range(2, k_clusters), 'Silhouette': sil})
    plt.figure(figsize=(12, 6))
    plt.plot(frame['Cluster'], frame['Silhouette'], marker='o')
    plt.xticks(range(1, k_clusters))
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette')

    plt.show()


def analyze_clusters(k):
    """
    Function to analyze the clustering given by k-means
    :param k: number of clusters for k-means
    :return: prints statistical data on each cluster
    """
    data = pd.read_csv('../data/Customer_Churn_processed.csv')
    data = data.to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_jobs=-1, n_clusters=k, init='k-means++')
    kmeans.fit(data_scaled)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    clusters = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i, cluster in enumerate(labels):
        clusters[cluster].append(data[i].tolist())
    df_clusters = []
    for i in range(5):
        df_clusters.append(pd.DataFrame(clusters[i]))

    for i, df in enumerate(df_clusters):
        print('CLUSTER {}'.format(i))
        print('====================================')
        print(df.describe())
        print('====================================')
        print()


# runs k-means with k=5, and prints stats on each clustering
analyze_clusters(5)