import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from clustering_file import demo_vector

data = demo_vector()
# data = np.random.rand(1000,3)
j = []
y_silhouette_score = []
inertia_score = []
calinskiharabaz_score = []
for k in range(2,11):
    clusterer = KMeans(k, random_state=1, init='k-means++')
    pred = clusterer.fit_predict(data, True)
    silhouettescore = metrics.silhouette_score(data, pred)
    print("silhouette_score for cluster '{}'".format(k))
    print(silhouettescore)
    calinskiharabazscore = metrics.calinski_harabasz_score(data,pred)
    print("calinski_harabaz_score '{}'".format(k))
    print(calinskiharabazscore)
    j.append(k)
    y_silhouette_score.append(silhouettescore)
    inertia_score.append(clusterer.inertia_)
    calinskiharabaz_score.append(calinskiharabazscore)
    print("clusterer.inertia_score for cluster '{}'".format(k))
    print(clusterer.inertia_)
# plt.figure()
# plt.plot(j,y_silhouette_score)
# plt.xlabel('kmeans-k')
# plt.ylabel('silhouette_score')
# plt.title('vectors')
# plt.show()

plt.figure()
plt.plot(j,inertia_score)
plt.xlabel('kmeans-k')
plt.ylabel('inertia_score(sum of squared)')
plt.title('matrix')
plt.show()

# plt.figure()
# plt.plot(j,calinskiharabaz_score)
# plt.xlabel('kmeans-k')
# plt.ylabel('calinski_harabaz_score')
# plt.title('vectors')
plt.show()