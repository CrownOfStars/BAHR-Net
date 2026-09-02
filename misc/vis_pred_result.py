import os

import numpy as np
import panel as pn
from PIL import Image

pn.extension()

# ----- 路径配置（可按需修改）-----
POLY_TEST_ROOT = "/home/data1/ShiqiangShu/best/HRMNetv3_1/dataset/Crack/test"
LOG_SAVE_ROOT = (
    "/home/data1/ShiqiangShu/best/HRMNetv3/log/2026-05-10-02:27:48-GSformer-segswin-base+segswin-small/save"
)
DEFAULT_DATASET = "CFD"


def build_paths(dataset_name: str):
    base_dir = os.path.join(POLY_TEST_ROOT, dataset_name)
    pred_dir = os.path.join(LOG_SAVE_ROOT, dataset_name)
    rgb_dir = os.path.join(base_dir, "RGB")
    gt_dir = os.path.join(base_dir, "GT")
    return base_dir, pred_dir, rgb_dir, gt_dir


def list_dataset_names():
    if not os.path.isdir(POLY_TEST_ROOT):
        return []
    return sorted(
        d
        for d in os.listdir(POLY_TEST_ROOT)
        if os.path.isdir(os.path.join(POLY_TEST_ROOT, d))
    )


def resolve_path(dir_path: str, img_name: str):
    p = os.path.join(dir_path, img_name)
    if os.path.exists(p):
        return p
    alt = (
        img_name.replace(".jpg", ".png")
        if img_name.endswith(".jpg")
        else img_name.replace(".png", ".jpg")
    )
    p2 = os.path.join(dir_path, alt)
    return p2 if os.path.exists(p2) else None


def _pil_to_chw_float(img: Image.Image) -> np.ndarray:
    """RGB, float32 [0,1], shape (3, H, W)"""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def mae_pil_gt_pred(pil_gt: Image.Image, pil_pred: Image.Image) -> float:
    """GT 尺度为准，预测对齐尺寸后计算 RGB 逐像素 MAE。"""
    if pil_pred.size != pil_gt.size:
        pil_pred = pil_pred.resize(pil_gt.size, Image.Resampling.BILINEAR)
    g = _pil_to_chw_float(pil_gt)
    p = _pil_to_chw_float(pil_pred)
    return float(np.mean(np.abs(g - p)))


def mae_for_pair(gt_path: str, pred_path: str) -> float:
    pil_gt = Image.open(gt_path)
    pil_pred = Image.open(pred_path)
    return mae_pil_gt_pred(pil_gt, pil_pred)


def sorted_image_names_by_mae(dataset_name: str):
    """按 GT 与 Pred 的 MAE 从低到高排序的图片名列表。"""
    _, pred_dir, rgb_dir, gt_dir = build_paths(dataset_name)
    if not os.path.isdir(rgb_dir):
        return []

    names = [
        f
        for f in os.listdir(rgb_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    scored = []
    for name in names:
        gt_path = resolve_path(gt_dir, name)
        pred_path = resolve_path(pred_dir, name)
        if not gt_path or not pred_path:
            continue
        try:
            m = mae_for_pair(gt_path, pred_path)
            scored.append((m, name))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0])
    return [n for _, n in scored]


datasets = list_dataset_names()
if DEFAULT_DATASET in datasets:
    _init_ds = DEFAULT_DATASET
elif datasets:
    _init_ds = datasets[0]
else:
    _init_ds = None

dataset_select = pn.widgets.Select(
    name="dataset_name",
    options=datasets if datasets else ["(无数据集目录)"],
    value=_init_ds if _init_ds else datasets[0] if datasets else "(无数据集目录)",
)

_init_names = (
    sorted_image_names_by_mae(dataset_select.value)
    if _init_ds and datasets
    else []
)
image_select = pn.widgets.Select(
    name="image (MAE 从低到高)",
    options=_init_names if _init_names else ["(无可用图片)"],
    value=_init_names[0] if _init_names else "(无可用图片)",
)


def on_dataset_change(event):
    if event.new == "(无数据集目录)":
        image_select.options = ["(无可用图片)"]
        image_select.value = "(无可用图片)"
        return
    opts = sorted_image_names_by_mae(event.new)
    image_select.options = opts if opts else ["(无可用图片)"]
    image_select.value = opts[0] if opts else "(无可用图片)"


dataset_select.param.watch(on_dataset_change, "value")


@pn.depends(dataset_select.param.value, image_select.param.value)
def show_image(dataset_name, img_name):
    if not dataset_name or dataset_name == "(无数据集目录)":
        return pn.pane.Markdown("请检查 `POLY_TEST_ROOT` 下是否有子数据集目录。")
    if not img_name or img_name.startswith("("):
        return pn.pane.Markdown("当前数据集下没有可对比的 RGB/GT/Pred 图片。")

    _, pred_dir, rgb_dir, gt_dir = build_paths(dataset_name)
    rgb_path = resolve_path(rgb_dir, img_name)
    gt_path = resolve_path(gt_dir, img_name)
    pred_path = resolve_path(pred_dir, img_name)

    if not rgb_path or not gt_path or not pred_path:
        return pn.pane.Markdown("找不到 RGB / GT / Pred 中某一文件路径。")

    pil_rgb = Image.open(rgb_path)
    pil_gt = Image.open(gt_path)
    pil_pred = Image.open(pred_path)
    mae = mae_pil_gt_pred(pil_gt, pil_pred)

    return pn.Column(
        pn.pane.Markdown(f"**MAE (GT vs Pred):** `{mae:.6f}`"),
        pn.Row(
            pn.Column("### RGB", pn.pane.Image(pil_rgb, width=400)),
            pn.Column("### GT", pn.pane.Image(pil_gt, width=400)),
            pn.Column("### Prediction", pn.pane.Image(pil_pred, width=400)),
        ),
    )


interactive_dashboard = pn.Column(
    "### 可视化（按 MAE 排序浏览）",
    pn.Row(dataset_select, image_select),
    show_image,
)

interactive_dashboard.show()
