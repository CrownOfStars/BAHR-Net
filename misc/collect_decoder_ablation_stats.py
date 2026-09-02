#!/usr/bin/env python3
"""汇总 decoder 消融实验 log，输出 CSV 到 misc/ 目录。

扫描 log/dec_abl_*_{ISOD,COD}/record/record.csv，提取验证集 best/final 指标及训练配置。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "log"
DEFAULT_OUT_DIR = PROJECT_ROOT / "misc"

ABLATION_ORDER = [
    "full",
    "wo_current",
    "wo_skip",
    "wo_fusion",
    "only_current",
    "only_skip",
    "only_fusion",
]

LOG_DIR_RE = re.compile(r"^dec_abl_(?P<abl>.+)_(?P<task>ISOD|COD)$")
LOG_TEST_RE = re.compile(
    r"#TEST#:Epoch:(?P<epoch>\d+) MAE:(?P<mae>[0-9.eE+-]+) "
    r"bestEpoch:(?P<best_epoch>\d+) bestMAE:(?P<best_mae>[0-9.eE+-]+)"
)


def _abl_sort_key(name: str) -> tuple[int, str]:
    try:
        return (ABLATION_ORDER.index(name), name)
    except ValueError:
        return (len(ABLATION_ORDER), name)


def discover_runs(log_root: Path) -> list[Path]:
    runs = []
    for p in sorted(log_root.iterdir()):
        if not p.is_dir():
            continue
        if LOG_DIR_RE.match(p.name):
            runs.append(p)
    return runs


def parse_log_best(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    last = None
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_TEST_RE.search(line)
            if m:
                last = m.groupdict()
    if last is None:
        return None
    return {
        "log_best_mae": float(last["best_mae"]),
        "log_best_mae_epoch": int(last["best_epoch"]),
        "log_last_val_mae": float(last["mae"]),
        "log_last_val_epoch": int(last["epoch"]),
    }


def load_args(log_dir: Path) -> dict:
    args_path = log_dir / "args.json"
    if not args_path.is_file():
        return {}
    with open(args_path, encoding="utf-8") as f:
        return json.load(f)


def summarize_run(log_dir: Path) -> dict | None:
    m = LOG_DIR_RE.match(log_dir.name)
    if not m:
        return None

    record_csv = log_dir / "record" / "record.csv"
    if not record_csv.is_file():
        print(f"[skip] no record.csv: {log_dir}")
        return None

    df = pd.read_csv(record_csv)
    if df.empty or "epoch" not in df.columns:
        print(f"[skip] empty record: {record_csv}")
        return None

    args = load_args(log_dir)
    log_info = parse_log_best(log_dir / "log.txt") or {}

    decoder_ablation = args.get("decoder_ablation") or m.group("abl")
    task = args.get("task") or m.group("task")

    row = {
        "tag": log_dir.name,
        "log_dir": str(log_dir.relative_to(PROJECT_ROOT)),
        "task": task,
        "decoder_ablation": decoder_ablation,
        "backbone": args.get("backbone", ""),
        "pretrain_size": args.get("pretrain_size", ""),
        "finetune_size": args.get("finetune_size", ""),
        "warmup_epoch": args.get("warmup_epoch", ""),
        "max_epoch": args.get("max_epoch", ""),
        "pretrain_batch": args.get("pretrain_batch", ""),
        "finetune_batch": args.get("finetune_batch", ""),
        "loss_preset": args.get("loss", ""),
        "num_record_epochs": int(len(df)),
    }

    if "loss" in df.columns:
        last = df.iloc[-1]
        row["final_train_loss"] = float(last["loss"])
        row["final_val_mae"] = float(last["mae"]) if "mae" in df.columns else None
        row["final_val_miou"] = float(last["miou"]) if "miou" in df.columns else None
        row["final_epoch"] = int(last["epoch"])

    if "mae" in df.columns:
        best_mae_idx = df["mae"].astype(float).idxmin()
        row["best_val_mae"] = float(df.loc[best_mae_idx, "mae"])
        row["best_val_mae_epoch"] = int(df.loc[best_mae_idx, "epoch"])

    if "miou" in df.columns:
        best_miou_idx = df["miou"].astype(float).idxmax()
        row["best_val_miou"] = float(df.loc[best_miou_idx, "miou"])
        row["best_val_miou_epoch"] = int(df.loc[best_miou_idx, "epoch"])

    row.update(log_info)
    return row


def load_all_epochs(log_dir: Path) -> pd.DataFrame | None:
    record_csv = log_dir / "record" / "record.csv"
    if not record_csv.is_file():
        return None
    df = pd.read_csv(record_csv)
    m = LOG_DIR_RE.match(log_dir.name)
    if m and "decoder_ablation" not in df.columns:
        df["decoder_ablation"] = m.group("abl")
    if m and "task" not in df.columns:
        df["task"] = m.group("task")
    df["tag"] = log_dir.name
    df["log_dir"] = str(log_dir.relative_to(PROJECT_ROOT))
    return df


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_abl_order"] = df["decoder_ablation"].map(lambda x: _abl_sort_key(str(x))[0])
    df = df.sort_values(["task", "_abl_order", "decoder_ablation"]).drop(columns=["_abl_order"])
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="汇总 decoder 消融实验指标到 misc/*.csv")
    parser.add_argument(
        "--log_root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="实验 log 根目录，默认 ./log",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="CSV 输出目录，默认 ./misc",
    )
    args = parser.parse_args()

    log_root = args.log_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(log_root)
    if not runs:
        print(f"未找到 dec_abl_* 实验目录: {log_root}")
        return

    summary_rows = []
    epoch_frames = []
    for run_dir in runs:
        row = summarize_run(run_dir)
        if row:
            summary_rows.append(row)
        ep = load_all_epochs(run_dir)
        if ep is not None:
            epoch_frames.append(ep)

    summary = sort_summary(pd.DataFrame(summary_rows))

    summary_path = out_dir / "decoder_ablation_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"已写入: {summary_path} ({len(summary)} runs)")

    for task in ("ISOD", "COD"):
        sub = summary[summary["task"] == task].copy()
        if sub.empty:
            continue
        path = out_dir / f"decoder_ablation_{task}.csv"
        sub.to_csv(path, index=False, float_format="%.6f")
        print(f"已写入: {path} ({len(sub)} runs)")

    if epoch_frames:
        epochs = pd.concat(epoch_frames, ignore_index=True)
        epochs = epochs.sort_values(["task", "decoder_ablation", "epoch"])
        epochs_path = out_dir / "decoder_ablation_all_epochs.csv"
        epochs.to_csv(epochs_path, index=False, float_format="%.6f")
        print(f"已写入: {epochs_path} ({len(epochs)} rows)")

    # 宽表：便于论文表格（行=消融，列=指标）
    for task in ("ISOD", "COD"):
        sub = summary[summary["task"] == task]
        if sub.empty:
            continue
        pivot = sub[
            [
                "decoder_ablation",
                "best_val_mae",
                "best_val_mae_epoch",
                "best_val_miou",
                "best_val_miou_epoch",
                "final_val_mae",
                "final_val_miou",
            ]
        ].copy()
        pivot_path = out_dir / f"decoder_ablation_{task}_pivot.csv"
        pivot.to_csv(pivot_path, index=False, float_format="%.6f")
        print(f"已写入: {pivot_path}")


if __name__ == "__main__":
    main()
