#!/usr/bin/env python3
"""根据 record.csv 绘制 MAE-Loss、MIoU-Loss 随 epoch 变化的折线图。"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def plot_record(csv_path: str, out_dir: str = None, dpi: int = 150, task: str = 'SOD'):
    """读取 record.csv，绘制 MAE-Loss、MIoU-Loss 对比折线图。"""
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        print(f"文件不存在: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        print("CSV 缺少 epoch 列")
        sys.exit(1)

    out_dir = out_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    epochs = df["epoch"].values
    loss = df["loss"].values if "loss" in df.columns else None
    mae = df[f"{task}-mae"].values if f"{task}-mae" in df.columns else None
    miou = df[f"{task}-miou"].values if f"{task}-miou" in df.columns else None

    # 1. MAE-Loss 对比
    if mae is not None and loss is not None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(epochs, loss, color="tab:blue", label="Loss", linewidth=1.5)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("MAE", color="tab:orange")
        ax2.plot(epochs, mae, color="tab:orange", label="MAE", linewidth=1.5)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.title("MAE vs Loss over Epoch")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{task}_mae_loss.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"已保存: {out_path}")

    # 2. MIoU-Loss 对比
    if miou is not None and loss is not None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(epochs, loss, color="tab:blue", label="Loss", linewidth=1.5)
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("MIoU", color="tab:green")
        ax2.plot(epochs, miou, color="tab:green", label="MIoU", linewidth=1.5)
        ax2.tick_params(axis="y", labelcolor="tab:green")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.title("MIoU vs Loss over Epoch")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{task}_miou_loss.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制 record.csv 的 MAE-Loss、MIoU-Loss 折线图")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="record.csv 路径，或 log 目录（将自动查找 record/record.csv）",
    )
    parser.add_argument("-o", "--out_dir", default=None, help="输出目录，默认与 csv 同目录")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片 DPI")
    args = parser.parse_args()

    if args.csv_path is None:
        parser.print_help()
        sys.exit(1)

    path = args.csv_path
    if os.path.isdir(path):
        path = os.path.join(path, "record", "record.csv")
    if not os.path.isfile(path):
        print(f"未找到: {path}")
        sys.exit(1)

    for task in ['SOD', 'COD','USOD']:
        plot_record(path, out_dir=args.out_dir, dpi=args.dpi, task=task)


if __name__ == "__main__":
    main()
