import cv2
import albumentations as A


transform = A.Compose([
    A.Blur(blur_limit=5, p=0.3),
    A.MedianBlur(blur_limit=5, p=0.3)
])

transform = A.Compose([
    A.RandomGamma(gamma_limit=(80, 120), p=1.0)
])

transform = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5)

A.Sharpen(alpha=(0.0, 1.0), lightness=(0.0, 2.0), p=1.0)


A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=200, val_shift_limit=0, p=1.0)

A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.5, 0.5), p=1.0)

A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=0, p=1.0)

A.Rotate(limit=15, interpolation=1, p=0.2)

import albumentations as A
from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip

transform = A.Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5)
], additional_targets={
    'image1': 'image',
    'image2': 'image',
    # 可继续添加更多图像
})
