import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

clust=Image.open("clustering.jpeg")
st.image(clust,use_column_width=True)

# Generate synthetic datasets
def generate_data(data_type, n_samples=500, n_features=2, centers=4, cluster_std=1.0, random_state=42):
    if data_type == "Blobs":
        data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    elif data_type == "Circles":
        data, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    elif data_type == "Spheres":
        data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return data

# Plot clusters
def plot_clusters(data, labels, title, dbscan_outliers=None):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot for clusters
    scatter = ax.scatter(x=data[:, 0], y=data[:, 1], c=labels, cmap='tab10')

    # Highlight DBSCAN outliers if available
    if dbscan_outliers is not None:
        ax.scatter(x=data[dbscan_outliers, 0], y=data[dbscan_outliers, 1], color='red', marker='o', s=200, label='Outliers')

    # Set plot title and labels
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    return fig  # Return the figure object

# Sidebar settings
st.sidebar.header("Clustering Settings")
clustering_algo = st.sidebar.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN"])
data_type = st.sidebar.selectbox("Select Data Type", ["Blobs", "Circles", "Spheres"])

# Common settings
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500, step=50)
n_features = st.sidebar.slider("Number of Features", 2, 5, 2)
cluster_std = st.sidebar.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0)
random_state = 42

data = generate_data(data_type, n_samples=n_samples, n_features=n_features, cluster_std=cluster_std)

if clustering_algo == "K-Means":
    st.sidebar.subheader("K-Means Settings")
    k_value = st.sidebar.slider("Number of Clusters (K)", 1, 20, 4)
    kmeans = KMeans(n_clusters=k_value, random_state=random_state)
    labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, labels)
    st.write(f"*K-Means Silhouette Score:* {silhouette_avg:.4f}")
    st.write(f"*Number of clusters (K):* {len(np.unique(labels))}")
    
    # Plot K-Means clusters
    st.subheader(f"K-Means Clustering with K={k_value}")
    fig_kmeans = plot_clusters(data, labels, f"K-Means Clustering (K={k_value})")
    st.pyplot(fig_kmeans)

elif clustering_algo == "DBSCAN":
    st.sidebar.subheader("DBSCAN Settings")
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Identify outliers (DBSCAN labels outliers as -1)
    outliers = np.where(labels == -1)[0]
    
    # Only calculate silhouette score if there are more than 1 cluster and no outliers
    if len(np.unique(labels)) > 1 and -1 not in np.unique(labels):
        silhouette_avg = silhouette_score(data, labels)
        st.write(f"*DBSCAN Silhouette Score:* {silhouette_avg:.4f}")
    else:
        st.write("*DBSCAN produced only one cluster or has outliers. Silhouette Score cannot be calculated.*")
    
    st.write(f"*Number of clusters (excluding outliers):* {len(np.unique(labels[labels != -1]))}")
    
    # Plot DBSCAN clusters with outliers
    st.subheader(f"DBSCAN Clustering with eps={eps} and min_samples={min_samples}")
    fig_dbscan = plot_clusters(data, labels, f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})", dbscan_outliers=outliers)
    st.pyplot(fig_dbscan)
