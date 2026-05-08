import os
import argparse

import imageio.v2 as imageio
import numpy as np


def find_dir(root, candidates):
    for name in candidates:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Cannot find any of {candidates} under {root}")


def read_gray(path):
    img = imageio.imread(path)

    if img.ndim == 3:
        img = img[..., 0]

    return img


def read_prob(path):
    img = read_gray(path).astype(np.float32)

    if img.max() > 1.0:
        if img.max() > 255:
            img = img / 65535.0
        else:
            img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def binary_label(x):
    return (x > 127).astype(np.uint8)


def compute_ece(prob_change, label, n_bins=15):
    pred = (prob_change > 0.5).astype(np.uint8)
    confidence = np.maximum(prob_change, 1.0 - prob_change)
    correct = (pred == label).astype(np.float32)

    ece = 0.0
    total = confidence.size

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        count = mask.sum()

        if count == 0:
            continue

        acc = correct[mask].mean()
        conf = confidence[mask].mean()

        ece += (count / total) * abs(acc - conf)

    return ece


def compute_f1(pred, label):
    pred = pred.astype(bool)
    label = label.astype(bool)

    tp = np.logical_and(pred, label).sum()
    fp = np.logical_and(pred, ~label).sum()
    fn = np.logical_and(~pred, label).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1


def compute_risk_coverage(prob_change, label, uncertainty=None):
    pred = (prob_change > 0.5).astype(np.uint8)

    if uncertainty is None:
        confidence = np.maximum(prob_change, 1.0 - prob_change)
        score = 1.0 - confidence
    else:
        score = uncertainty.astype(np.float32)
        if score.max() > 1.0:
            score = score / 255.0

    score = score.reshape(-1)
    pred = pred.reshape(-1)
    label = label.reshape(-1)

    order = np.argsort(score)  # low uncertainty first

    coverages = [0.5, 0.7, 0.8, 0.9, 1.0]
    f1_at_cov = {}
    risk_at_cov = {}

    n = len(order)

    for cov in coverages:
        k = max(1, int(n * cov))
        idx = order[:k]

        pred_k = pred[idx]
        label_k = label[idx]

        f1 = compute_f1(pred_k, label_k)
        acc = (pred_k == label_k).mean()
        risk = 1.0 - acc

        f1_at_cov[cov] = f1
        risk_at_cov[cov] = risk

    aurc = np.mean(list(risk_at_cov.values()))

    return f1_at_cov, risk_at_cov, aurc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--result_root", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=15)

    args = parser.parse_args()

    label_dir = find_dir(args.data_root, ["label", "Label", "labels", "GT", "gt"])

    prob_dir = os.path.join(args.result_root, "prob_change")
    uncertainty_dir = os.path.join(args.result_root, "uncertainty")

    if not os.path.isdir(prob_dir):
        raise FileNotFoundError(f"prob_change dir not found: {prob_dir}")

    has_uncertainty = os.path.isdir(uncertainty_dir)

    names = sorted([
        f for f in os.listdir(prob_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    ece_list = []
    f1_cov_sum = {0.5: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}
    risk_cov_sum = {0.5: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}
    aurc_list = []

    for name in names:
        prob = read_prob(os.path.join(prob_dir, name))

        label_path = os.path.join(label_dir, name)
        label = binary_label(read_gray(label_path))

        uncertainty = None
        if has_uncertainty:
            unc_path = os.path.join(uncertainty_dir, name)
            if os.path.exists(unc_path):
                uncertainty = read_prob(unc_path)

        ece = compute_ece(prob, label, n_bins=args.n_bins)
        ece_list.append(ece)

        f1_at_cov, risk_at_cov, aurc = compute_risk_coverage(
            prob,
            label,
            uncertainty=uncertainty,
        )

        for cov in f1_cov_sum:
            f1_cov_sum[cov].append(f1_at_cov[cov])
            risk_cov_sum[cov].append(risk_at_cov[cov])

        aurc_list.append(aurc)

    print("=" * 60)
    print(f"Result root: {args.result_root}")
    print(f"Images: {len(names)}")
    print(f"ECE ↓: {np.mean(ece_list):.6f}")
    print(f"AURC ↓: {np.mean(aurc_list):.6f}")

    print("\nF1 at Coverage ↑")
    for cov in sorted(f1_cov_sum.keys()):
        print(f"  F1@{int(cov * 100)}%: {np.mean(f1_cov_sum[cov]):.6f}")

    print("\nRisk at Coverage ↓")
    for cov in sorted(risk_cov_sum.keys()):
        print(f"  Risk@{int(cov * 100)}%: {np.mean(risk_cov_sum[cov]):.6f}")


if __name__ == "__main__":
    main()