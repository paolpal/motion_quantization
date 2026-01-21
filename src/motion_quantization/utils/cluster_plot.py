def cluster_plot(points, labels, centroids=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for k in unique_labels:
        class_members = labels == k
        plt.scatter(points[class_members, 0], points[class_members, 1], s=10, color=colors(k), label=f'Cluster {k}' if k != -1 else 'Noise')

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

    plt.title('Cluster Plot')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid()
    plt.show()