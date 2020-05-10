import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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