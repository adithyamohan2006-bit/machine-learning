import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("data.csv")
X = df.drop("result", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_before = KMeans(n_clusters=2, random_state=42)
clusters_before = kmeans_before.fit_predict(X_scaled)

score_before = silhouette_score(X_scaled, clusters_before)
print("Silhouette Score (Before PCA):", score_before)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance:", pca.explained_variance_ratio_)

kmeans_after = KMeans(n_clusters=2, random_state=42)
clusters_after = kmeans_after.fit_predict(X_pca)

score_after = silhouette_score(X_pca, clusters_after)
print("Silhouette Score (After PCA):", score_after)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_before)
plt.title("Clustering Before PCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_after)
plt.title("Clustering After PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()