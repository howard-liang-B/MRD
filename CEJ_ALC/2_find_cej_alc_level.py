import cv2
import os
import numpy as np

version = 0
tooth_dir = f"./pred_mask_{version}/tooth_pred_mask"
crown_dir = f"./pred_mask_{version}/crown_pred_mask"
bone_dir = f"./pred_mask_{version}/bone_pred_mask"

# img_dir = f"./PA_dataset/tooth_dataset/images"
img_dir = r"C:\Users\howar\Desktop\IoT Medical\marginal ridge\PA_dataset\10_imgs"
output_dir = f"./outputs_{version}/CEJ_ALC_level_1"
os.makedirs(output_dir, exist_ok=True)

GREEN = [0, 255, 0] 
RED = [0, 0, 255]

for img_name in os.listdir(tooth_dir):
    print(f"Start processing {img_name}")
    img_path = img_dir + "/" + img_name     # 設置影像路徑
    out_path = output_dir + "/" + img_name
    t_path = tooth_dir + "/" + img_name 
    c_path = crown_dir + "/" + img_name
    b_path = bone_dir + "/" + img_name

    o_img = cv2.imread(img_path)
    t_img = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE) # 讀影像
    c_img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
    b_img = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)

    h, w = t_img.shape

    t_pix_value = np.unique(t_img)                  # 找出這張影像有哪些 pixel value
    t_pix_value = t_pix_value[t_pix_value != 0]     # 把背景黑色 pixel value = 0 的刪掉
    c_pix_value = np.unique(c_img)
    c_pix_value = c_pix_value[c_pix_value != 0]
    b_pix_value = np.unique(b_img)
    b_pix_value = b_pix_value[b_pix_value != 0]    
    # print(f"tooth pixel value {t_pix_value}")


    for t_pix in t_pix_value:
        # 找出重疊面積最大的一對遮罩(tooth 和 crown)
        max_iou = 0
        miou_t_mask = None
        miou_c_mask = None
        miou_b_mask = b_img == b_pix_value[0]
        for c_pix in c_pix_value:
            t_mask = t_img == t_pix
            c_mask = c_img == c_pix
            overlap = np.logical_and(t_mask, c_mask)
            iou = np.sum(overlap)
            if iou > max_iou:
                max_iou = iou
                miou_t_mask = t_mask
                miou_c_mask = c_mask
        if miou_t_mask is None or miou_c_mask is None:
            continue

        kernel_a = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 11))
        kernel_b = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 17))
        eroded_t = cv2.erode(miou_t_mask.astype(np.uint8) * 255, kernel_a)
        eroded_t = eroded_t > 0
        dilated_c = cv2.dilate(miou_c_mask.astype(np.uint8) * 255, kernel_a)
        dilated_c = dilated_c > 0
        dilated_b = cv2.dilate(miou_b_mask.astype(np.uint8) * 255, kernel_b)
        dilated_b = dilated_b > 0
        
        

        # 定位 CEJ level、ALC level
        kernel_size = 5
        half_k_size = kernel_size // 2
        for i in range(2, w - 2):
            for j in range(2, h - 2):
                t_kernel = eroded_t[j-2:j+2, i-2:i+2]
                c_kernel = dilated_c[j-2:j+2, i-2:i+2]
                b_kernel = dilated_b[j-2:j+2, i-2:i+2]
                if True in t_kernel and True in c_kernel and False in c_kernel:
                    o_img[j, i] = GREEN 
                if True in t_kernel and True in b_kernel and False in b_kernel:
                    o_img[j, i] = RED

        # 把過小的 CEJ level 刪除，如果夠大的話，把最左右兩邊標記起來儲存
        green_mask = np.all(o_img == [0, 255, 0], axis=2).astype(np.uint8)  # 找出綠色區域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)

        min_area = 80  # 設定要保留的最小面積

        # 建立新的遮罩，只保留大面積的綠色區域
        filtered_green_mask = np.zeros_like(green_mask)
        for i in range(1, num_labels):  # 跳過背景
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_green_mask[labels == i] = 1

        # 建構過濾後的新圖：原圖拷貝後，把小面積綠區塊設為黑色
        res_img = o_img.copy()
        res_img[np.logical_and(green_mask == 1, filtered_green_mask == 0)] = (0, 0, 0)

    cv2.imwrite(out_path, res_img)
    print(f"save cej alc points image at {out_path}.\n")