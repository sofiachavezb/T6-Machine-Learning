from sklearn.cluster import KMeans
from src import SEED
from p2_a import data_pca
import matplotlib.pyplot as plt

max_clusters = 7
row_size = 3
# plot the clusters from 2 to max_clusters
fig, axs = plt.subplots((max_clusters-1)//row_size, row_size, figsize=(9, 6), dpi=400)
for i in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=i, random_state=SEED)
    kmeans.fit(data_pca)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    ax = axs[(i-2)//row_size, (i-2)%row_size]
    ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)
    ax.set_title(f'K={i}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    

fig.legend(['Data points', 'Centroids'], loc='upper right')
#fig.tight_layout()
plt.suptitle('KMeans clustering')
plt.savefig('figures/P2B_KMeans.png')
#plt.show()
plt.close()