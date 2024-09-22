# Interactive Clustering Visualization using K-Means and DBSCAN in Streamlit
## Project Description:
This project provides an interactive dashboard to visualize the performance of clustering algorithms, specifically K-Means and DBSCAN, on different types of synthetic datasets. The project is developed using Streamlit for easy interaction and allows users to adjust parameters for both clustering algorithms in real-time to observe the resulting clusters and silhouette scores.

## Click here for deployment https://clsutering-k-means-dbscan-outlier-detection-5xqjmaizyr5tyr9heg.streamlit.app/

## Key Features:
- User-Friendly Interface: Developed using Streamlit for real-time parameter adjustment and visualization of clustering results.
  
## Clustering Algorithms:
- K-Means: Allows the user to choose the number of clusters (K) and view the clustering results.
- DBSCAN: Provides options for adjusting the epsilon and min_samples parameters to control the density-based clustering.
- Synthetic Datasets: Offers multiple types of data for clustering:
- Blobs: Standard dataset with Gaussian blobs.
- Circles: Non-linearly separable circular clusters.
- Spheres: Blobs in a higher-dimensional feature space.
- Dynamic Visualization: Clustering results are displayed with different colors for each cluster, and outliers (for DBSCAN) are highlighted in red.
- Silhouette Score Calculation: Displays the silhouette score for evaluating clustering performance (when applicable).
- Outlier Detection: DBSCAN's ability to detect outliers is clearly visualized with highlighted points.
- Interactive Controls: Parameters such as the number of samples, features, cluster standard deviation, and clustering algorithm settings can be adjusted via the sidebar.

## How to Run:
- Install the required libraries: pip install streamlit scikit-learn matplotlib seaborn.
- Run the Streamlit app: streamlit run app.py.
  
## Use the sidebar to:
- Select the type of clustering algorithm (K-Means or DBSCAN).
- Adjust the number of samples, features, cluster standard deviation, and clustering parameters.
- Observe the clustering results and the silhouette score in real-time.
  
## Libraries Used:
- Streamlit: For creating the interactive web app.
- Scikit-learn: For generating synthetic data and performing K-Means and DBSCAN clustering.
- Matplotlib & Seaborn: For plotting the clusters.

  ![image](https://github.com/user-attachments/assets/4fef90f9-c972-42fa-9cb3-f15f6e3955ba)

## Additional Information:
This project is an excellent tool for exploring how different clustering algorithms behave on different types of datasets, allowing users to better understand the inner workings of clustering and tune parameters effectively.
