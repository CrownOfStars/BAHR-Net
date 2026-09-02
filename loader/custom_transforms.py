import random
from PIL import Image,ImageEnhance
import numpy as np
import cv2
import torch.nn.functional as F
import math

def random_flip(*images):
    out = list(images)
    if random.randint(0, 1):
        out = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in out]
    if random.randint(0, 1):
        out = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in out]
    return tuple(out)


def random_crop(*images):
    image_width, image_height = images[0].size[0], images[0].size[1]
    border_width, border_height = image_width * 0.1, image_height * 0.1
    crop_win_width = np.random.randint(int(image_width - border_width), image_width + 1)
    crop_win_height = np.random.randint(int(image_height - border_height), image_height + 1)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return tuple(img.crop(random_region) for img in images)



def random_rotation(*images):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        return tuple(img.rotate(random_angle, Image.BICUBIC) for img in images)
    return images


def color_enhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def suppress_foreground_contrast(img_rgb, gt_mask):
    """
    将前景颜色混合为背景均值，实现颜色弱化
    :param img_rgb: RGB图像 (PIL)
    :param gt_mask: 二值GT图像 (PIL)，前景为1，背景为0
    :param alpha: 融合权重，越大越接近背景
    :return: PIL.Image
    """
    alpha = random.uniform(0.3,0.7)
    img_np = np.array(img_rgb).astype(np.float32)
    mask_np = np.array(gt_mask.convert("L"))
    mask_bin = (mask_np > 127).astype(np.uint8)

    # 计算背景颜色均值
    bg_pixels = img_np[mask_bin == 0]
    if len(bg_pixels) == 0:  # 防止全是前景
        return img_rgb
    bg_mean = bg_pixels.mean(axis=0)  # [R, G, B]

    # 融合前景像素
    img_np[mask_bin == 1] = (1 - alpha) * img_np[mask_bin == 1] + alpha * bg_mean

    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

def suppress_contrast_with_overlay(img_rgb: Image.Image, gt_mask: Image.Image):
    """
    给前景和背景同时加一个灰色偏移掩膜
    :param img_rgb: RGB图像 (PIL)
    :param gt_mask: GT图像 (PIL)
    :param overlay_color: 掩膜颜色（推荐浅灰）
    :param alpha: 掩膜权重
    :return: PIL.Image
    """
    alpha = random.uniform(0.3,0.7)
    img_np = np.array(img_rgb).astype(np.float32)
    mask_np = np.array(gt_mask.convert("L"))
    mask_bin = (mask_np > 127).astype(np.uint8)

    # 计算背景颜色均值
    bg_pixels = img_np[mask_bin == 0]
    if len(bg_pixels) == 0:  # 防止全是前景
        return img_rgb
    bg_mean = bg_pixels.mean(axis=0)  # [R, G, B]
    overlay = np.full_like(img_np, bg_mean, dtype=np.float32)

    # 全图混合
    img_np = (1 - alpha) * img_np + alpha * overlay

    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


def pair_color_enhance(image1, image2):
    bright_intensity = random.randint(5, 15) / 10.0
    image1 = ImageEnhance.Brightness(image1).enhance(bright_intensity)
    image2 = ImageEnhance.Brightness(image2).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image1 = ImageEnhance.Contrast(image1).enhance(contrast_intensity)
    image2 = ImageEnhance.Contrast(image2).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image1 = ImageEnhance.Color(image1).enhance(color_intensity)
    image2 = ImageEnhance.Color(image2).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image1 = ImageEnhance.Sharpness(image1).enhance(sharp_intensity)
    image2 = ImageEnhance.Sharpness(image2).enhance(sharp_intensity)
    return image1, image2

def image_suffix(f):
    return f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')

