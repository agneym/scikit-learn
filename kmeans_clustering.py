from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris.data[:,1:3]

model = KMeans(n_clusters=5, random_state=0)
model.fit(x)

print (model.predict(x))

centroids = model.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', s=170, zorder=10, c='m')
plt.scatter(x[:,0], x[:,1], c=model.labels_)
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.show()