'''
COMP721: Machine Learning
Project: NBA Player Outlier Detection
Author:  217008217
University of KwaZulu-Natal
School of Mathematics, Statistics and Computer Science
'''

import pandas as pd
from sklearn import preprocessing
import csv
import matplotlib.pyplot as plt
from matplotlib import cm


data = pd.read_csv("player_regular_season_career.csv")
X_train = data.iloc[:, 4:] #features
player_names = data.iloc[:, [1,2]] #name details of players

x = X_train.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

X_train = pd.DataFrame(x_scaled)
X_train.columns = ['gp', 'minutes', 'pts', 'oreb', 'dreb', 'reb', 'asts', 'stl', 'blk', 'turnover', 'pf', 'fga', 'fgm', 'fta', 'ftm', 'tpa', 'tpm']


########## DBSCAN ##########

from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps = 0.60, metric="euclidean", min_samples = 50, n_jobs = 1).fit(X_train)
outlier_df = pd.DataFrame(X_train)

colours = clusters.labels_
cmap = cm.get_cmap('Set1')
plt.scatter(X_train.iloc[:,2].values, X_train.iloc[:,1].values, c=colours, s=20, cmap=cmap, edgecolor="black")
plt.title("Player Outliers using DBSCAN")
plt.xlabel("Minutes")
plt.ylabel("Points")
plt.show()

dbscan_outliers = outlier_df[clusters.labels_==-1]
dbscan_outliers = dbscan_outliers[:].index.to_numpy()
player_names = player_names.values
print("Player Outliers using DBSCAN:")
for i in range(0, len(dbscan_outliers)):
    print(player_names[dbscan_outliers[i]])
print("DBSCAN yielded ", len(dbscan_outliers), " outliers.")    
    
    
########## One-Class Support Vector Machine ##########

from sklearn import svm
svm_model = svm.OneClassSVM(nu=.008, kernel='rbf', gamma=.001)
svm_model.fit(X_train)
y_pred = svm_model.predict(X_train)

plt.scatter(X_train.iloc[:,2].values,X_train.iloc[:,1].values,c=y_pred, cmap=cmap,s=20,edgecolor='black')
plt.xlabel("Minutes")
plt.ylabel("Points")
plt.title("Player Outliers using One-Class SVM")
plt.show()

if_outliers = []
print("\nPlayer Outliers using One-Class SVM:")
count = 0
for i in range(0, len(y_pred)):
    if y_pred[i] == -1:
        if_outliers.append(i)
        print(player_names[if_outliers[len(if_outliers) - 1]])
        count += 1
print("One-Class SVM yielded ", count, " outliers.")
        
        
        
########## Local Outlier Factor ##########

from sklearn.neighbors import LocalOutlierFactor
y_pred = LocalOutlierFactor(n_neighbors=40, contamination=.006).fit_predict(X_train)
LOF_pred = pd.Series(y_pred).replace([-1,1],[1,0])
LOF_anomalies = X_train[LOF_pred==1]
LOF_anomalies = LOF_anomalies[LOF_anomalies['pts']>=0.2]

plt.scatter(X_train.iloc[:,2],X_train.iloc[:,1],c='grey',s=20,edgecolor='black')
plt.scatter(LOF_anomalies.iloc[: ,2], LOF_anomalies.iloc[: ,1], c='red', edgecolor='black')
plt.title('Player Outliers using LOF')
plt.xlabel('Minutes')
plt.ylabel('Points')
plt.show()

lof_outliers = LOF_anomalies.index
lof_outliers = list(lof_outliers)
print("\nPlayer Outliers using LOF:")
for i in range(0, len(lof_outliers)):
    print(player_names[lof_outliers[i]])
print("LOF yielded ", len(lof_outliers), " outliers.") 


########## K-Means Clustering ##########

from sklearn.cluster import KMeans
x = X_train.values
wcss = []
for i in range(1, 18):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0).fit(x)
    wcss.append(kmeans.inertia_)
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=5)
y = kmeans.fit_predict(X_train)

plt.scatter(x[y == 0, 0], x[y == 0, 1], s=25, c='indigo', label='Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s=25, c='sienna', label='Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s=25, c='crimson', label='Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s=25, c='black', label='Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s=25, c='orange', label='Cluster 5')
plt.scatter(x[y == 5, 0], x[y == 5, 1], s=25, c='grey', label='Cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=25, c='yellow', label='centroid')
plt.legend()
plt.show()

outlier_list = []
for i in range(0, len(y)):
    if y[i] == 3:
        outlier_list.append(i)
print("Player Outliers using K-Means Clustering:")
for i in range(0, len(outlier_list)):
    print(outlier_list[i], player_names[outlier_list[i]])
print("K-Means Clustering yields ", len(outlier_list), " outliers.")