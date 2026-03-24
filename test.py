import sys
sys.path.append("C:/Users/user/Desktop/code/XFeat-LightGlue/accelerated_features")
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# LightGlue
from lightglue import LightGlue
from lightglue.utils import load_image, rbd

# XFeat
from modules.xfeat import XFeat

# ------------------------
# 1. 讀取影像
# ------------------------
img0 = cv2.imread(r"C:\Users\user\Desktop\code\XFeat-LightGlue\image\map.jpg")   # 大圖
img1 = cv2.imread(r"C:\Users\user\Desktop\code\XFeat-LightGlue\image\uav_1.jpg")   # 小圖

img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# ------------------------
# 2. XFeat 特徵點
# ------------------------
xfeat = XFeat().eval().cuda()

def extract(img):
    img = torch.from_numpy(img / 255.).float()[None, None].cuda()
    feats = xfeat(img)
    return feats

feat0 = extract(img0_gray)
feat1 = extract(img1_gray)

# ------------------------
# 3. LightGlue 匹配
# ------------------------
matcher = LightGlue(features="xfeat").eval().cuda()

matches = matcher({
    "image0": feat0,
    "image1": feat1
})

matches = rbd(matches)

kpts0 = matches["keypoints0"].cpu().numpy()
kpts1 = matches["keypoints1"].cpu().numpy()

# ------------------------
# 4. Homography
# ------------------------
H, mask = cv2.findHomography(kpts1, kpts0, cv2.RANSAC)

# ------------------------
# 5. 找 UAV 中心對應位置
# ------------------------
h, w = img1.shape[:2]
center = np.array([[[w/2, h/2]]], dtype=np.float32)

mapped = cv2.perspectiveTransform(center, H)
x, y = mapped[0][0]

print(f"定位結果 (pixel): x={x:.2f}, y={y:.2f}")

# ------------------------
# 6. 視覺化
# ------------------------
img_vis = img0.copy()
cv2.circle(img_vis, (int(x), int(y)), 20, (0,0,255), -1)

plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.title("UAV Location on Map")
plt.show()