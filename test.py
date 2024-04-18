# /usr/bin/env python
"""
Author: zhiying
Date: 24-4-5 下午20:28
Description: CatsCls training testing
"""
from torchvision import transforms
from util import transforms_train
from util import transforms_trainV2
from Dataset import Cats_Dataset


import numpy as np
from PIL import Image
import time
from tqdm import *

img_path = './imgs/cat_12_train/YGyx4qCdOb7j8tzBuNfoFHLi6gU0SE3T.jpg'
img = Image.open(img_path)
# print(f'img.shape={img.shape}\n')
img2 = transforms_train(img)
print(f'img.shape={np.array(img2).shape}\n')
# PILToTensor = transforms.PILToTensor()
# ToPILImage = transforms.ToPILImage()
# print(PILToTensor(img2).shape)
# ToPILImage(PILToTensor(img2)).show()

# img3 = transforms_trainV2(image=np.array(img))
# print(img3)
# for i in tqdm(range(1000)):
#     time.sleep(.01)
ls = [i for i in range(100)]
testset = Cats_Dataset(ls, transforms_train)
print(testset)
for id, data in enumerate(testset):
    print(data[0].shape)

