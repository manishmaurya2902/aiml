from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[20,25],[80,85],[26,30],[86,90],[105,110],[100,105]])

model = KMeans(n_clusters=3)
model.fit(x)

print(model.labels_)

plt.scatter([x[0] for x in x],[x[1] for x in x],c=model.labels_) 
plt.xlabel("x value")
plt.ylabel("y value")
plt.title("k-means clustering")
plt.show()