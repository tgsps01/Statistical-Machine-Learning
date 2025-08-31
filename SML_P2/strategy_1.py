# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# load data
data = scipy.io.loadmat('./Desktop/SML_P2/AllSamples.mat')['AllSamples']

# Plot the data
plt.scatter(data[:, 0], data[:, 1], s=10, c='blue', marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of 2D Data Points')
plt.show()

# Initialize centers randomly
def initialize_centers(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# Assingn data points to clusturs
def assign_clusters(X, centers):
    distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# Update the centers
def recompute_centers(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

# Objective function
def compute_objective(X, labels, centers):
    return sum(np.linalg.norm(X[labels == i] - centers[i])**2 for i in range(len(centers)))

# k-means algorithm
def k_means(X, k, max_iters=100):
    centers = initialize_centers(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centers)
        new_centers = recompute_centers(X, labels, k)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    objective = compute_objective(X, labels, centers)
    return centers, labels, objective

# Execute the k-means algorithm and plots the results
def main():
    X = data

    # Plot for two different initializations
    objectives_1 = []
    objectives_2 = []
    ks = range(2, 11)
    for k in ks:
        # First initialization
        centers_1, labels_1, objective_1 = k_means(X, k)
        objectives_1.append(objective_1)

        # Second initialization
        centers_2, labels_2, objective_2 = k_means(X, k)
        objectives_2.append(objective_2)

    # Plotting the objective function values for both initializations
    plt.figure(figsize=(8, 6))
    plt.plot(ks, objectives_1, marker='o', label='Initialization 1')
    plt.plot(ks, objectives_2, marker='o', label='Initialization 2')
    plt.title('Objective Function vs. Number of Clusters')
    plt.xlabel('Number of Clusters k')
    plt.ylabel('Objective Function Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the main function
main()

# Plot of the clusters for K = 10
def plot_k10_clusters():
    X = data
    k = 10

    # First initialization
    centers_1, labels_1, _ = k_means(X, k)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_1, s=10, cmap='viridis', alpha=0.7)
    plt.scatter(centers_1[:, 0], centers_1[:, 1], c='red', s=100, alpha=0.9)
    plt.title(f'k-means clustering result for k={k} (Initialization 1)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

    # Second initialization
    centers_2, labels_2, _ = k_means(X, k)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_2, s=10, cmap='viridis', alpha=0.7)
    plt.scatter(centers_2[:, 0], centers_2[:, 1], c='red', s=100, alpha=0.9)
    plt.title(f'k-means clustering result for k={k} (Initialization 2)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

# Call the function to plot k=10 clusters
plot_k10_clusters()