import os
import cv2
import numpy as np
import pandas as pd


####################################################
def find_left_right(x, y, component_mask):
    # 找出該區域內所有點的 x, y 座標，是相對座標
    ys, xs = component_mask.nonzero()
    xs_full = xs + x
    ys_full = ys + y

    # 找最左與最右的 x
    left_idx = xs_full.argmin()
    right_idx = xs_full.argmax()

    left_point = (xs_full[left_idx], ys_full[left_idx])
    right_point = (xs_full[right_idx], ys_full[right_idx])
    return left_point, right_point
####################################################

radius, GREEN, RED, thickness = 6, [0, 255, 0], [0, 0, 255], 2
color_list = [
    ((255, 0, 0), "blue"),
    ((0, 255, 255), "yellow"),
    ((255, 255, 0), "cyan"),
    ((255, 0, 255), "magenta"),
    ((0, 165, 255), "orange"),
    ((128, 0, 128), "purple"),
    ((42, 42, 165), "brown")
]

version = 0
img_dir = f"./outputs_{version}/marginal_ridge"
angle_list = []

for name in os.listdir(img_dir):
    img_path = img_dir + "/" + name
    img = cv2.imread(img_path)

    temp_color = []
    for color, color_name in color_list:
        if color in img:
            img_mask = np.all(img == color, axis=2).astype(np.uint8)  # 找出紅色區域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_mask, connectivity=8)
            sorted_indices = stats[1:, cv2.CC_STAT_LEFT].argsort() + 1 # 從 1 開始，就部會把背景 0 算進去

            for i in range(1, len(sorted_indices)):          
                ridx1 = sorted_indices[i-1]       # 前一個物件
                ridx2 = sorted_indices[i]         # 現在的物件
                x1, y1, w1, h1, area1 = stats[ridx1]
                x2, y2, w2, h2, area2 = stats[ridx2]

                component_mask1 = (labels[y1:y1+h1, x1:x1+w1] == ridx1)
                component_mask2 = (labels[y2:y2+h2, x2:x2+w2] == ridx2)
                left_1, right_1 = find_left_right(x1, y1, component_mask1)
                left_2, right_2 = find_left_right(x2, y2, component_mask2)

                # 計算向量
                x1, y1 = left_1
                x2, y2 = right_1
                x3, y3 = left_2
                x4, y4 = right_2
                v1 = np.array([x2 - x1, y2 - y1])  # 向量 1
                v2 = np.array([x4 - x3, y4 - y3])

                # print("v1: ", v1, "v2: ", v2)
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 避免超出範圍
                angle_deg = np.degrees(angle_rad)
                # print(cos_theta)
                if not np.isnan(angle_deg) and color_name not in temp_color:
                    temp_color.append(color_name)
                    angle_list.append({
                        "filename": name,
                        "color": color_name,
                        "angle": angle_deg
                    })
    print("save: ", name)

df = pd.DataFrame(angle_list)
df.to_excel(f"./outputs_{version}/ridge_angles.xlsx", index=False)
