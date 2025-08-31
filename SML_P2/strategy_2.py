import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import pairwise_distances_argmin_min

# Load the dataset from a .mat file
X = loadmat('./Desktop/SML_P2/AllSamples.mat')['AllSamples']


def initialize_centers(X, k):
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features))
    
    # Randomly select the first centroid
    first_centroid_idx = np.random.randint(0, n_samples)
    centroids[0] = X[first_centroid_idx]
    
    # Select remaining centroids
    for i in range(1, k):
        distances = np.zeros((n_samples,))
        for j in range(i):
            dist = np.linalg.norm(X - centroids[j], axis=1)
            distances += dist
        distances[first_centroid_idx] = -1
        next_center_idx = np.argmax(distances)
        centroids[i] = X[next_center_idx]
        distances[next_center_idx] = -1
    
    return centroids

def kmeans_custom_init(X, k, init_strategy, n_init=2):
    """ Perform K-means clustering with custom initialization. """
    best_inertia = None
    best_centroids = None
    initial_inertia = None
    
    for _ in range(n_init):
        centroids = init_strategy(X, k)
        centroids_old = np.zeros(centroids.shape)
        clusters = np.zeros(len(X))
        clusters, _ = pairwise_distances_argmin_min(X, centroids)
        initial_inertia = np.sum((X - centroids[clusters]) ** 2)  # Inertia for initial clusters
        while not np.all(centroids == centroids_old):
            centroids_old = centroids.copy()
            #clusters = pairwise_distances_argmin_min(X, centroids)[0]
            for i in range(k):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                if points: 
                    centroids[i] = np.mean(points, axis=0)
        inertia = np.sum((X - centroids[clusters]) ** 2)
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
    
    return best_inertia, best_centroids, initial_inertia

# Evaluate the objective function for different number of clusters for two initializations
k_values = range(2, 11)
objective_values1 = []
initial_objective_values1 = []
objective_values2 = []
initial_objective_values2 = []

for k in k_values:
    inertia1, _, initial_inertia1 = kmeans_custom_init(X, k, initialize_centers, n_init=2)
    inertia2, _, initial_inertia2 = kmeans_custom_init(X, k, initialize_centers, n_init=2)
    objective_values1.append(inertia1)
    initial_objective_values1.append(initial_inertia1)
    objective_values2.append(inertia2)
    initial_objective_values2.append(initial_inertia2)

# Plot the results
# first initialization
plt.figure(figsize=(8, 6))
plt.plot(k_values, objective_values1, marker='o', label='Final Objective Function')
plt.plot(k_values, initial_objective_values1, marker='o', label='Initial Objectiive Function')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Objective function value (Inertia)')
plt.title('Objective Function vs. Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()

#second initialization
plt.figure(figsize=(8, 6))
plt.plot(k_values, objective_values2, marker='o', label='Final Objective Function')
plt.plot(k_values, initial_objective_values2, marker='o', label='Initial Objectiive Function')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Objective function value (Inertia)')
plt.title('Objective Function vs. Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()
