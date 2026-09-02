#!/usr/bin/env python3
"""汇总 alpha1/alpha2 grid search 实验结果 -> misc/grid_search_alpha_summary.csv"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "log"
DEFAULT_OUT = PROJECT_ROOT / "misc" / "grid_search_alpha_summary.csv"

TAG_RE_A2 = re.compile(r"^gs_a1_(?P<a1>.+)_a2_(?P<a2>.+)_(?P<task>ISOD|COD)$")
TAG_RE_BETA = re.compile(r"^gs_a1_(?P<a1>.+)_b_(?P<beta>.+)_(?P<task>ISOD|COD)$")


def _unfmt(s: str) -> float:
    return float(s.replace("p", "."))


def _parse_tag(tag: str) -> dict:
    m = TAG_RE_BETA.match(tag)
    if m:
        return {
            "alpha1": _unfmt(m.group("a1")),
            "beta": _unfmt(m.group("beta")),
            "task": m.group("task"),
        }
    m = TAG_RE_A2.match(tag)
    if m:
        return {
            "alpha1": _unfmt(m.group("a1")),
            "alpha2": _unfmt(m.group("a2")),
            "task": m.group("task"),
        }
    return {}


def summarize_log_dir(log_dir: Path) -> dict | None:
    record_csv = log_dir / "record" / "record.csv"
    if not record_csv.is_file():
        return None

    df = pd.read_csv(record_csv)
    args = {}
    args_path = log_dir / "args.json"
    if args_path.is_file():
        with open(args_path, encoding="utf-8") as f:
            args = json.load(f)

    row = {
        "tag": log_dir.name,
        "log_dir": str(log_dir.relative_to(PROJECT_ROOT)),
        "alpha1": args.get("alpha1"),
        "alpha2": args.get("alpha2"),
        "beta": args.get("beta"),
        "task": args.get("task"),
        "loss": args.get("loss"),
    }
    parsed = _parse_tag(log_dir.name)
    for k, v in parsed.items():
        if row.get(k) is None:
            row[k] = v

    if "mae" in df.columns:
        best_idx = df["mae"].astype(float).idxmin()
        row["best_val_mae"] = float(df.loc[best_idx, "mae"])
        row["best_val_mae_epoch"] = int(df.loc[best_idx, "epoch"])
    if "miou" in df.columns:
        best_iou = df["miou"].astype(float).idxmax()
        row["best_val_miou"] = float(df.loc[best_iou, "miou"])
        row["best_val_miou_epoch"] = int(df.loc[best_iou, "epoch"])
    if not df.empty:
        last = df.iloc[-1]
        row["final_val_mae"] = float(last["mae"]) if "mae" in df.columns else None
        row["final_val_miou"] = float(last["miou"]) if "miou" in df.columns else None

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    for p in sorted(args.log_root.iterdir()):
        if p.is_dir() and p.name.startswith("gs_a1_"):
            row = summarize_log_dir(p)
            if row:
                rows.append(row)

    if not rows:
        print(f"未找到 gs_a1_* 目录: {args.log_root}")
        return

    sort_cols = [c for c in ("task", "alpha1", "alpha2", "beta") if c in pd.DataFrame(rows).columns]
    df = pd.DataFrame(rows).sort_values(sort_cols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"已写入: {args.out} ({len(df)} runs)")


if __name__ == "__main__":
    main()
