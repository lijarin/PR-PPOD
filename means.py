import math
import matplotlib.pyplot as plt
from clustering_file import demo_vector
import random

def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)

def k_means(dataset, k, iteration):
    #初始化簇心向量
    index = random.sample(list(range(len(dataset))), k)
    vectors = []
    for i in index:
        vectors.append(dataset[i])
    #初始化标签
    labels = []
    for i in range(len(dataset)):
        labels.append(-1)
    #根据迭代次数重复k-means聚类过程
    while(iteration > 0):
        #初始化簇
        C = []
        for i in range(k):
            C.append([])
        for labelIndex, item in enumerate(dataset):
            classIndex = -1
            minDist = 1e6
            for i, point in enumerate(vectors):
                dist = getEuclidean(item, point)
                if(dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = len(dataset[0])
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1
    return C, labels

if __name__=='__main__':
    vector = demo_vector()
    distortions = []
    C, labels = k_means(dataset=vector,k=3,iteration=100).fit(vector)
    sihouette_score(vector,)
    # vector = [[1, 2, 3], [1.5, 1.8,1.4], [5, 8,5], [8, 8,7], [1, 0.6,1.3], [9, 11,5]]
    # for i in range(1,10):
    #     C = k_means(dataset=vector,k=i,iteration=100).fit(x)
    #     C.fit(vector)
    #     distortions.append(C.inertia_)
    # plt.plot(range(1,10),distortions, marker = 'o')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Distortion')
    # plt.show()