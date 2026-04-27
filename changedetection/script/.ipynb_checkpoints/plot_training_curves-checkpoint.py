import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_metrics(history_file, save_dir="plots", dpi=300):
    with open(history_file, "r") as f:
        data = json.load(f)

    # 提取验证集指标
    iters = []
    f1 = []
    iou = []
    kappa = []
    for record in data:
        if record["split"] == "Validation":
            iters.append(record["iter"])
            f1.append(record["f1"])
            iou.append(record["iou"])
            kappa.append(record["kappa"])

    os.makedirs(save_dir, exist_ok=True)

    # F1 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(iters, f1, 'b-o', label='F1')
    plt.xlabel("Iteration")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 over Training")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "f1_curve.png"), dpi=dpi)
    plt.close()

    # IoU 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(iters, iou, 'r-s', label='IoU')
    plt.xlabel("Iteration")
    plt.ylabel("IoU")
    plt.title("Validation IoU over Training")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "iou_curve.png"), dpi=dpi)
    plt.close()

    # 多指标合图
    plt.figure(figsize=(12, 6))
    plt.plot(iters, f1, 'b-o', label='F1')
    plt.plot(iters, iou, 'r-s', label='IoU')
    plt.plot(iters, kappa, 'g-^', label='Kappa')
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "all_metrics.png"), dpi=dpi)
    plt.close()

    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True, help="Path to metrics_history.json")
    parser.add_argument("--save_dir", type=str, default="./plots", help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()
    plot_metrics(args.history, args.save_dir, args.dpi)