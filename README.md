# Clustering-with-K-Means

# Database :
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

# Code :
#kmeans clustering.py

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\katta\OneDrive\Python-ai\myenv\Mall_Customers.csv")

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to determine optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Fit KMeans with optimal K (e.g., K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = cluster_labels

# Plot clustered data
plt.figure(figsize=(8, 5))
for cluster in range(optimal_k):
    plt.scatter(
        X_scaled[cluster_labels == cluster, 0],
        X_scaled[cluster_labels == cluster, 1],
        label=f'Cluster {cluster}'
    )
plt.title('K-Means Clustering')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate with Silhouette Score
score = silhouette_score(X_scaled, cluster_labels)
print(f'Silhouette Score: {score:.3f}')

# Screenshots :
![8 1](https://github.com/user-attachments/assets/4845bf25-c248-4c15-82b9-2d849769a4f4)
![8 2](https://github.com/user-attachments/assets/dfe72c58-b34f-4365-b99b-16d141054696)

