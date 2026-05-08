import os
import argparse
import random

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt


def find_dir(root, candidates):
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Cannot find any of {candidates} under {root}")


def read_rgb(path):
    img = imageio.imread(path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] > 3:
        img = img[..., :3]

    return img


def read_gray(path):
    img = imageio.imread(path)

    if img.ndim == 3:
        img = img[..., 0]

    return img


def normalize_gray(x):
    x = x.astype(np.float32)

    if x.max() > 1.0:
        x = x / 255.0

    return np.clip(x, 0.0, 1.0)


def overlay_heatmap(image, heat, alpha=0.45, cmap_name="jet"):
    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    heat = normalize_gray(heat)
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(heat)[..., :3]

    out = (1.0 - alpha) * image + alpha * heat_rgb
    return np.clip(out, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--result_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 兼容不同 SYSU-CD 文件夹命名
    t1_dir = find_dir(args.data_root, ["A", "T1", "t1", "im1", "before", "image1"])
    t2_dir = find_dir(args.data_root, ["B", "T2", "t2", "im2", "after", "image2"])
    label_dir = find_dir(args.data_root, ["label", "Label", "labels", "GT", "gt"])

    change_dir = os.path.join(args.result_root, "change_map")
    uncertainty_dir = os.path.join(args.result_root, "uncertainty")
    gate_dir = os.path.join(args.result_root, "gate_weights")

    if not os.path.isdir(change_dir):
        raise FileNotFoundError(change_dir)

    names = sorted([
        f for f in os.listdir(change_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    if len(names) == 0:
        raise RuntimeError(f"No prediction maps found in {change_dir}")

    if len(names) > args.num_samples:
        names = random.sample(names, args.num_samples)

    for name in names:
        base = os.path.splitext(name)[0]

        t1_path = os.path.join(t1_dir, name)
        t2_path = os.path.join(t2_dir, name)
        gt_path = os.path.join(label_dir, name)
        pred_path = os.path.join(change_dir, name)

        if not os.path.exists(t1_path):
            print(f"Skip {name}: T1 not found")
            continue

        t1 = read_rgb(t1_path)
        t2 = read_rgb(t2_path)
        gt = read_gray(gt_path)
        pred = read_gray(pred_path)

        panels = []
        titles = []

        panels.extend([t1, t2, gt, pred])
        titles.extend(["T1", "T2", "GT", "Prediction"])

        # uncertainty
        unc_path = os.path.join(uncertainty_dir, name)
        if os.path.exists(unc_path):
            unc = read_gray(unc_path)
            panels.append(unc)
            titles.append("Uncertainty")

            panels.append(overlay_heatmap(t2, unc))
            titles.append("Uncertainty Overlay")

        # gate weights
        gate_names = ["seq", "cross", "par"]
        for gate_name in gate_names:
            gate_path = os.path.join(gate_dir, f"{base}_{gate_name}.png")
            if os.path.exists(gate_path):
                gate = read_gray(gate_path)
                panels.append(gate)
                titles.append(f"Gate-{gate_name}")

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        if n == 1:
            axes = [axes]

        for ax, img, title in zip(axes, panels, titles):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()

        save_path = os.path.join(args.save_dir, f"{base}_vis.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()