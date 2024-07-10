from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from p2_a import data_pca
from src.config import SEED

# Parámetros para DBSCAN
eps = 0.5
min_samples = 5

# DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(data_pca)

# K-means con K=1 y K=2
kmeans_1 = KMeans(n_clusters=2, random_state=SEED)
kmeans_labels_1 = kmeans_1.fit_predict(data_pca)

kmeans_2 = KMeans(n_clusters=3, random_state=SEED)
kmeans_labels_2 = kmeans_2.fit_predict(data_pca)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=400)

axs[0].scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels, cmap='viridis', marker='o', edgecolor='k')
axs[0].set_title('DBSCAN')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')

axs[1].scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels_1, cmap='viridis', marker='o', edgecolor='k')
axs[1].scatter(kmeans_1.cluster_centers_[:, 0], kmeans_1.cluster_centers_[:, 1], c='red', s=100, alpha=0.5)
axs[1].set_title('K-means (K=2)')
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC2')

axs[2].scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels_2, cmap='viridis', marker='o', edgecolor='k')
axs[2].scatter(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1], c='red', s=100, alpha=0.5)
axs[2].set_title('K-means (K=3)')
axs[2].set_xlabel('PC1')
axs[2].set_ylabel('PC2')

plt.suptitle('DBSCAN vs K-means (K=2 y K=3)')
plt.tight_layout()
plt.savefig('figures/P2C_DBSCAN_vs_KMeans.png')
#plt.show()