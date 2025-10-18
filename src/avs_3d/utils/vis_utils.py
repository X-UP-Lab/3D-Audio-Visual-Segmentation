import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_clusters_3d(target_object_grid_points, large_cluster_masks, output_dir='visuals'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        labels = sorted(large_cluster_masks.keys())
        colors = plt.cm.Spectral(np.linspace(0, 1, len(labels)))
        for k, col in zip(labels, colors):
            class_member = large_cluster_masks[k]
            xyz = target_object_grid_points[class_member] - np.mean(target_object_grid_points, axis=0)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], label=f'Cluster {k}', s=5)
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        plt.savefig(os.path.join(output_dir, 'clusters_3d.png'), dpi=300)
        plt.close(fig)
    except Exception:
        pass


def save_gt_vs_pred_comparison(preds, gts, output_dir, max_images=20):
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    num_images = min(len(preds), max_images)
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 4 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(num_images):
        axes[i, 0].imshow(preds[i], cmap='gray')
        axes[i, 0].set_title('Prediction')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(gts[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gt_vs_pred.png'), dpi=300)
    plt.close(fig)