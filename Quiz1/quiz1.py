import argparse
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import open3d as o3d
import polyscope as ps

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_ply_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a point cloud from a .ply file.
    Returns points and binary ground truth labels derived from color (1=ground, 0=other).
    """
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points found in {path}")

    labels = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape == points.shape:
             # Ground color reference (Brown)
             ground_color_ref = np.array([0.6, 0.4, 0.1])
             # Check for exact equality
             is_ground = np.all(np.isclose(colors, ground_color_ref, atol=1e-2), axis=1)
             labels = is_ground.astype(int)

    return points, labels


# --------------------------------------------------------------------------
# K-Means
# --------------------------------------------------------------------------

def kmeans_clustering(points, k):

    print("Perfoming K-Means clustering, K = " + str(k))\
    
    scaled_points = StandardScaler().fit_transform(points)

    cluster_labels = KMeans(n_clusters=k, init='k-means++').fit_predict(scaled_points)

    return cluster_labels

# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------

def return_PCA(points, cluster_labels, k):

    centers = []
    pc1 = []
    pc2 = []
    pc3 = []
    features = []

    for i in range(k):

        cluster_points = points[cluster_labels ==i]

        center = cluster_points.mean(axis=0)
        centers.append(center)

        pca = PCA(n_components=3).fit(cluster_points)
        axis_std = np.sqrt(np.maximum(pca.explained_variance_, 0.0))
        pc1.append(pca.components_[0] * axis_std[0])
        pc2.append(pca.components_[1] * axis_std[1])
        pc3.append(pca.components_[2] * axis_std[2])

        variances = pca.explained_variance_ratio_

        if i == 0 or i == 199:
            print(variances[0])
            print(variances[1])
            print(variances[2])
            print(pca.components_[0])
            print(pca.components_[1])
            print(pca.components_[2])

        height = center[2]

        feature_vector = [
            variances[0],
            variances[1],
            variances[2],
            height
        ]

        features.append(feature_vector)

    return(
        np.array(centers),
        np.array(pc1),
        np.array(pc2),
        np.array(pc3),
        np.array(features)
    )

# --------------------------------------------------------------------------
# Cluster ground-truth generation
# --------------------------------------------------------------------------

def cluster_ground_truth(cluster_labels, point_gt_labels, k, threshold = 0.5):
    
    cluster_gt = []
    for i in range(k):
        idx = np.where(cluster_labels == i)[0]
        ground_fraction = float(point_gt_labels[idx].mean())
        if ground_fraction >= threshold:
            cluster_gt.append(1)
        else:
            cluster_gt.append(0)

    return cluster_gt


# --------------------------------------------------------------------------
# SVM (train on clusters, predict points)
# --------------------------------------------------------------------------

def train_svm_and_predict_points(features, cluster_labels, cluster_gt, k, point_gt_labels):

    # Normalize features so SVM isn't dominated by the "largest-scale" feature.
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    svm = SVC(kernel="rbf").fit(X, cluster_gt)

    cluster_pred = svm.predict(X)

    cluster_accuracy = np.mean(cluster_pred == cluster_gt)
    print("SVM training accuracy on clusters:", cluster_accuracy)

    point_pred = np.zeros(len(cluster_labels), dtype=int)
    for i in range(k):
        point_pred[cluster_labels == i] = int(cluster_pred[i])

    point_accuracy = np.mean(point_pred == point_gt_labels)
    print("SVM training accuracy on points:", point_accuracy)

    return point_pred


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(points, labels, k, cluster_labels, pca_centers, pc1, pc2, pc3, features, svm_predictions = None):
    """
    Set up Polyscope serialization.
    """
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    # 1. Register main point cloud
    cloud = ps.register_point_cloud("Processed Point Cloud", points, radius=0.0015)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", points[:, 2], enabled=True)

    # 2. Add ground truth labels
    cloud.add_scalar_quantity("Ground truth data", labels, enabled=False)
    
    # 3. Add colors for clusters

    cloud.add_scalar_quantity("K-Means Clusters", cluster_labels, enabled = True)

    # 4. Register PCA visualization (Cluster Centers and Principal Components)
    # you can register pointclouds: https://polyscope.run/py/structures/point_cloud/basics/
    # you can then add vector quantities: https://polyscope.run/py/structures/point_cloud/vector_quantities/
    pca_cloud = ps.register_point_cloud("Cluster PCA Centers", pca_centers, radius=0.0006)
    pca_cloud.add_vector_quantity("PC1 (Major)", pc1, enabled=True, color=(1, 0, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC2 (Minor)", pc2, enabled=True, color=(0, 1, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC3 (Normal)", pc3, enabled=True, color=(0, 0, 1), vectortype="ambient")

    # 5. Add SVM Predictions
    if svm_predictions is not None:
        cloud.add_scalar_quantity("SVM prediction", svm_predictions, enabled=False)


    ps.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation pipeline (K-Means -> PCA -> SVM)")
    parser.add_argument("path", nargs="?", default="airport_downsample.ply")
    parser.add_argument("-k", "--clusters", type=int, default=150, help="Number of k-means clusters (default: 6)")
    args = parser.parse_args()

    # 1. Load Data
    ## returns points and labels
    points, point_gt_labels = load_ply_point_cloud(args.path)
    print(f"Loaded {len(points)} points from {args.path}")

    # 2. Clustering
    cluster_labels = kmeans_clustering(points, args.clusters)

    # once the clustering is solved, you might consider saving the cluster in a .npy file
    # np.save("cluster_labels.npy", cluster_labels)
    # cluster_labels = np.load("cluster_labels.npy")

    # 3. Feature Extraction (PCA)

    pca_centers, pc1, pc2, pc3, features = return_PCA(points, cluster_labels, args.clusters)

    # 4. Ground Truth Generation (for training SVM)
    # We only have GT at the point level (from .ply colors). Convert it to a
    # cluster label so we can train one label per cluster feature-vector.
    cluster_gt = cluster_ground_truth(cluster_labels, point_gt_labels, args.clusters, threshold=0.5)

    # 5. SVM Classification
    svm_point_predictions = train_svm_and_predict_points(features, cluster_labels, cluster_gt, args.clusters, point_gt_labels)

    # 6. Visualization
    visualize(points, point_gt_labels, args.clusters, cluster_labels, pca_centers, pc1, pc2, pc3, features, svm_point_predictions)



if __name__ == "__main__":
    main()