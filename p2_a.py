import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from src import load_data_for_unsupervised

data = load_data_for_unsupervised(verbose=True)
scaler = StandardScaler()

data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

if __name__ == '__main__':

    # Plot main components
    plt.figure(dpi=400)
    plt.scatter(data_pca[:,0], data_pca[:,1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA')
    plt.savefig('figures/P2A_PCA.png')
    plt.show()
    plt.close()

    # Percentage of variance explained
    print(f"Percentage of variance explained by PC1: {pca.explained_variance_ratio_[0]}")
    print(f"Percentage of variance explained by PC2: {pca.explained_variance_ratio_[1]}")
    print(f"Total percentage of variance explained by PC1 and PC2: {pca.explained_variance_ratio_.sum()}")