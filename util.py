# /usr/bin/env python
"""
Author: zhiying
Date: 24-4-5 下午20:28
Description: 数据增强工具
"""
from torchvision import transforms
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

PILToTensor = transforms.PILToTensor()
ToPILImage = transforms.ToPILImage()
ToTensor = transforms.ToTensor()

transforms_train = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation((-180, 180)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ]
)


transforms_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]
    )

transforms_trainV2 = albumentations.Compose(
    [
        albumentations.Resize(320, 320),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Rotate(limit=180, p=0.7),
        albumentations.RandomBrightnessContrast(),
        albumentations.ShiftScaleRotate(
            shift_limit=0.25, scale_limit=0.1, rotate_limit=0
        ),
        albumentations.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            max_pixel_value=255.0, always_apply=True
        ),
        ToTensorV2(p=1.0),
    ]
)


transforms_testV2 = albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ]
    )