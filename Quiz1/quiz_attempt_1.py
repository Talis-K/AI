import argparse
from typing import Optional, Tuple

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

    pcd = o3d.io.read_point_cloud(path)

    points = np.asarray(pcd.points)

    if points.size == 0:
        raise ValueError(f"No points found in {path}")

    labels = None

    if pcd.has_colors():

        colors = np.asarray(pcd.colors)

        ground_color_ref = np.array([0.6, 0.4, 0.1])

        is_ground = np.all(np.isclose(colors, ground_color_ref, atol=1e-2), axis=1)

        labels = is_ground.astype(int)

    return points, labels


# ---------------------------------------------------------------------------
# KMEANS CLUSTERING
# ---------------------------------------------------------------------------
def perform_kmeans(points, k):

    print("Running K-Means clustering...")

    scaler = StandardScaler()

    scaled_points = scaler.fit_transform(points)

    kmeans = KMeans(n_clusters=k, n_init=10)

    cluster_labels = kmeans.fit_predict(scaled_points)

    return cluster_labels


# ---------------------------------------------------------------------------
# PCA FEATURE EXTRACTION
# ---------------------------------------------------------------------------
def compute_cluster_pca(points, cluster_labels, k):

    centers = []
    pc1 = []
    pc2 = []
    pc3 = []

    features = []

    for i in range(k):

        cluster_points = points[cluster_labels == i]

        if len(cluster_points) < 3:
            continue

        center = cluster_points.mean(axis=0)

        pca = PCA(n_components=3)
        pca.fit(cluster_points)

        centers.append(center)

        pc1.append(pca.components_[0])
        pc2.append(pca.components_[1])
        pc3.append(pca.components_[2])

        variances = pca.explained_variance_ratio_

        height = center[2]

        feature_vector = [
            variances[0],
            variances[1],
            variances[2],
            height
        ]

        features.append(feature_vector)

    return (
        np.array(centers),
        np.array(pc1),
        np.array(pc2),
        np.array(pc3),
        np.array(features)
    )


# ---------------------------------------------------------------------------
# SVM TRAINING
# ---------------------------------------------------------------------------
def train_svm(features, cluster_labels, point_gt_labels, points, k):

    cluster_gt = []

    for i in range(k):

        indices = np.where(cluster_labels == i)[0]

        if len(indices) == 0:
            cluster_gt.append(0)
            continue

        gt = point_gt_labels[indices]

        label = 1 if np.mean(gt) > 0.5 else 0

        cluster_gt.append(label)

    cluster_gt = np.array(cluster_gt)

    scaler = StandardScaler()

    X = scaler.fit_transform(features)

    svm = SVC(kernel="rbf")

    # 10-fold cross validation
    cv_folds = min(10, len(cluster_gt))
    scores = cross_val_score(svm, X, cluster_gt, cv=cv_folds)

    print("10-fold cross validation accuracy:")
    print(scores)
    print("Mean accuracy:", scores.mean())

    svm.fit(X, cluster_gt)

    cluster_predictions = svm.predict(X)

    point_predictions = np.zeros(len(points))

    for i in range(k):

        point_predictions[cluster_labels == i] = cluster_predictions[i]

    return point_predictions


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------
def visualize(points, gt_labels, cluster_labels, pca_centers, pc1, pc2, pc3, svm_predictions):

    print("Visualizing results in Polyscope...")

    ps.init()

    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    cloud = ps.register_point_cloud(
        "Processed Point Cloud",
        points,
        radius=0.0015
    )

    cloud.set_point_render_mode("quad")

    cloud.add_scalar_quantity(
        "Elevation",
        points[:, 2],
        enabled=True
    )

    if gt_labels is not None:

        cloud.add_scalar_quantity(
            "Ground Truth",
            gt_labels,
            enabled=False
        )

    cloud.add_scalar_quantity(
        "KMeans Clusters",
        cluster_labels,
        enabled=True
    )

    cloud.add_scalar_quantity(
        "SVM Prediction",
        svm_predictions,
        enabled=False
    )

    pca_cloud = ps.register_point_cloud(
        "Cluster Centers",
        pca_centers,
        radius=0.003
    )

    pca_cloud.add_vector_quantity(
        "PC1 (Major)",
        pc1,
        enabled=True
    )

    pca_cloud.add_vector_quantity(
        "PC2 (Minor)",
        pc2,
        enabled=False
    )

    pca_cloud.add_vector_quantity(
        "PC3 (Normal)",
        pc3,
        enabled=False
    )

    ps.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Point Cloud Segmentation pipeline (K-Means -> PCA -> SVM)"
    )

    parser.add_argument(
        "path",
        nargs="?",
        default="airport_downsample.ply"
    )

    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=6,
        help="Number of k-means clusters"
    )

    args = parser.parse_args()

    # 1 Load Data
    points, point_gt_labels = load_ply_point_cloud(args.path)

    print(f"Loaded {len(points)} points from {args.path}")

    # 2 KMeans
    cluster_labels = perform_kmeans(points, args.clusters)

    # 3 PCA
    pca_centers, pc1, pc2, pc3, features = compute_cluster_pca(
        points,
        cluster_labels,
        args.clusters
    )

    # 4 SVM
    svm_predictions = train_svm(
        features,
        cluster_labels,
        point_gt_labels,
        points,
        args.clusters
    )

    # 5 Visualization
    visualize(
        points,
        point_gt_labels,
        cluster_labels,
        pca_centers,
        pc1,
        pc2,
        pc3,
        svm_predictions
    )


if __name__ == "__main__":
    main()