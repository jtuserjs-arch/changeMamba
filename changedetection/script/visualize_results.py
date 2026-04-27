import os
import matplotlib.pyplot as plt
import imageio
import numpy as np

def visualize_example(image_path, pred_path, uncertainty_path, gate_weights_paths, save_path):
    """
    生成单个样本的可视化组合图：原图、变化检测、不确定性、门控权重（三个通道）
    """
    img = imageio.imread(image_path)
    pred = imageio.imread(pred_path)
    uncertainty = imageio.imread(uncertainty_path) / 255.0
    gate_seq = imageio.imread(gate_weights_paths['seq']) / 255.0
    gate_cross = imageio.imread(gate_weights_paths['cross']) / 255.0
    gate_par = imageio.imread(gate_weights_paths['par']) / 255.0

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(img)
    axes[0,0].set_title('Input Image')
    axes[0,0].axis('off')
    axes[0,1].imshow(pred, cmap='gray')
    axes[0,1].set_title('Change Detection')
    axes[0,1].axis('off')
    axes[0,2].imshow(uncertainty, cmap='hot', vmin=0, vmax=1)
    axes[0,2].set_title('Uncertainty')
    axes[0,2].axis('off')
    axes[1,0].imshow(gate_seq, cmap='hot')
    axes[1,0].set_title('Gate: Sequential')
    axes[1,0].axis('off')
    axes[1,1].imshow(gate_cross, cmap='hot')
    axes[1,1].set_title('Gate: Cross')
    axes[1,1].axis('off')
    axes[1,2].imshow(gate_par, cmap='hot')
    axes[1,2].set_title('Gate: Parallel')
    axes[1,2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":

    pass