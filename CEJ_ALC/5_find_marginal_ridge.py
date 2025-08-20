import os
import cv2
import numpy as np

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
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 165, 255),    # orange
    (128, 0, 128),    # purple
    (42, 42, 165)     # brown
]

version = 0
# PA_dir = f"./PA_dataset/tooth_dataset/images"
PA_dir = r"C:\Users\howar\Desktop\IoT Medical\marginal ridge\PA_dataset\10_imgs"
PA_output_dir = f"./outputs_{version}/marginal_ridge_pa"
img_dir = f"./outputs_{version}/CEJ_ALC_level_2"
img_output_dir = f"./outputs_{version}/marginal_ridge"
os.makedirs(img_output_dir, exist_ok=True)
os.makedirs(PA_output_dir, exist_ok=True)

for name in os.listdir(img_dir):
    img_path = img_dir + "/" + name
    pa_path = PA_dir + "/" + name
    img = cv2.imread(img_path)
    pa = cv2.imread(pa_path)

    green_mask = np.all(img == GREEN, axis=2).astype(np.uint8)  # 找出綠色區域
    red_mask = np.all(img == RED, axis=2).astype(np.uint8)  # 找出紅色區域

    g_num_labels, g_labels, g_stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
    g_sorted_indices = g_stats[1:, cv2.CC_STAT_LEFT].argsort() + 1 # 從 1 開始，就部會把背景 0 算進去
    r_num_labels, r_labels, r_stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    r_sorted_indices = r_stats[1:, cv2.CC_STAT_LEFT].argsort() + 1 # 從 1 開始，就部會把背景 0 算進去
    # print("red sorted indices: ", r_sorted_indices)

    ####################################################
    #### ALC ####
    # 從 1 開始，跳過背景
    for i in range(1, len(r_sorted_indices)):          
        ridx1 = r_sorted_indices[i-1]       # 前一個物件
        ridx2 = r_sorted_indices[i]         # 現在的物件
        x1, y1, w1, h1, area1 = r_stats[ridx1]
        x2, y2, w2, h2, area2 = r_stats[ridx2]

        component_mask1 = (r_labels[y1:y1+h1, x1:x1+w1] == ridx1)
        component_mask2 = (r_labels[y2:y2+h2, x2:x2+w2] == ridx2)
        left_alc1, right_alc1 = find_left_right(x1, y1, component_mask1)
        left_alc2, right_alc2 = find_left_right(x2, y2, component_mask2)

        cur_color = color_list[i]
        cv2.line(img, right_alc1, left_alc2, cur_color, thickness)
        cv2.line(pa, right_alc1, left_alc2, cur_color, thickness)

    #### CEJ ####
    for i in range(1, len(g_sorted_indices)):
        ridx1 = g_sorted_indices[i-1]       # 前一個物件
        ridx2 = g_sorted_indices[i]         # 現在的物件
        x1, y1, w1, h1, area1 = g_stats[ridx1]
        x2, y2, w2, h2, area2 = g_stats[ridx2]

        component_mask1 = (g_labels[y1:y1+h1, x1:x1+w1] == ridx1)
        component_mask2 = (g_labels[y2:y2+h2, x2:x2+w2] == ridx2)
        left_cej1, right_cej1 = find_left_right(x1, y1, component_mask1)
        left_cej2, right_cej2 = find_left_right(x2, y2, component_mask2)

        cur_color = color_list[i]
        cv2.line(img, right_cej1, left_cej2, cur_color, thickness)
        cv2.line(pa, right_cej1, left_cej2, cur_color, thickness)

    img_out_path = img_output_dir + "/" + name
    pa_out_path = PA_output_dir + "/" + name
    cv2.imwrite(img_out_path, img)
    cv2.imwrite(pa_out_path, pa)
    print("save: ", img_out_path)


# cv2.CC_STAT_LEFT：左上角 x
# cv2.CC_STAT_TOP：左上角 y
# cv2.CC_STAT_WIDTH：寬度
# cv2.CC_STAT_HEIGHT：高度
# cv2.CC_STAT_AREA：面積（像素數）
