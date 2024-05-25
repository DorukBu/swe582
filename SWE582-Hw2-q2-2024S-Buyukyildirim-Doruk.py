# NAME:       Doruk Büyükyıldırım
# STUDENT ID: 2022719033
# DATE:       25.05.2024
# SUMMARY:    SPRING 2024 SWE582 - SWE 582: Sp. Tp. Machine Learning for Data Analytic Homework #2 Question 2 Due May 26, 2024 by 23:59
#             This Python code does the following:
#             -> Plots the data using a scatter plot, assigning different colors to different labels.
#             -> Implements k-means algorithm from scratch, using the Euclidean distance as the distance metric.
#             -> Runs the k-means algorithm on the data
#             -> Plots the data points with different colors for each assigned cluster and centroids with a + marker in blue.
#             -> Loads the groundtruth labels and compares them with the cluster assignments.
#             -> Finds the wrongly labeled points and plots them with an x marker in red in a third plot.
#             * The datasets in .npy format should be placed inside the working directory
#             Acknowledgement: Generative AI is used to generate plotting style and functions
#             Required Libraries: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt

# Load the data and labels
data = np.load('./data.npy')
labels = np.load('./label.npy')

# Plot the data points with different colors for each label
cmap = 'Set2'
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap)
plt.show()

# Perform k-means clustering
def kmeans(data, k, max_iter=100):
  """
  Performs k-means clustering on the given data.

  Args:
    data: A numpy array containing the data points.
    k: The number of clusters.
    max_iter: The maximum number of iterations.

  Returns:
    A tuple containing the cluster assignments and the centroids.
  """

  # Initialize the centroids randomly
  centroids = data[np.random.choice(data.shape[0], k, replace=False)]

  for _ in range(max_iter):
    # Assign each data point to the closest centroid using Euclidean distance
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1)
    assignments = np.argmin(distances, axis=1)

    # Update the centroids to be the mean of the assigned points
    for i in range(k):
      centroids[i] = np.mean(data[assignments == i], axis=0)

  return assignments, centroids

# Run the k-means algorithm
k=np.unique(labels).shape[0]
assignments, centroids = kmeans(data, k=k)

# Plot the data points with different colors for each cluster
plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap=cmap)
plt.scatter(centroids[:, 0], centroids[:, 1],  marker='+', c='blue', s=500)
plt.show()

# Load the groundtruth labels
groundtruth_labels = labels

# Find the wrongly labeled points
wrongly_labeled_points = np.where(groundtruth_labels != assignments)[0]

# WARNING: The ground truth labels and the labels assigned by the k-means algorithm may not match.
# This can happen due to the random initialization of centroids and the nature of the k-means algorithm.
# It is important to evaluate the clustering results using appropriate metrics and not solely rely on the labels.

# Plot the data points with different colors for each cluster
plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap=cmap)

# Plot the wrongly labeled points with a different marker
plt.scatter(data[wrongly_labeled_points, 0], data[wrongly_labeled_points, 1], marker='x', c='red', s=150)

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', c='blue', s=500)
plt.show()
