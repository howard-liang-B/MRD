import os
import json
import cv2
import numpy as np
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug.augmenters as iaa
from tqdm import tqdm


class_name = "bone"
tooth_color = (0, 0, 128)
crown_color = (0, 128, 0) # BGR
bone_color = (128, 0, 0)
color = bone_color

# 路徑設定
BASE_DIR = r"C:\Users\howar\Desktop\IoT Medical\marginal ridge\PA_dataset\class_name_dataset".replace("class_name", class_name)
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")
OUT_IMG_DIR = os.path.join(BASE_DIR, "images_aug")
OUT_MASK_DIR = os.path.join(BASE_DIR, "masks_aug")
ANN_PATH = os.path.join(BASE_DIR, f"{class_name}_annotations.json")
OUT_ANN_PATH = os.path.join(BASE_DIR, f"{class_name}_annotations_aug.json")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# 讀取 COCO JSON
with open(ANN_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

augmenter = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Fliplr(1.0)),                          # 水平翻轉
    iaa.Sometimes(0.2, iaa.Flipud(1.0)),                          # 垂直翻轉（小機率）
    iaa.Sometimes(0.9, iaa.Affine(rotate=(-30, 30))),  # 旋轉 + 縮放
    iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),                 # 亮度變化
    iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 10))),  # 高斯雜訊
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.5))),          # 模糊
    iaa.Sometimes(0.5, iaa.ContrastNormalization((0.75, 1.5))),    # 對比度
    iaa.Sometimes(0.6, iaa.ElasticTransformation(alpha=1.0, sigma=0.25)),  # 彈性形變
    iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1))), # 銳化
    iaa.Invert(1.0, per_channel=True),         # 顏色反轉（低機率）
])

next_image_id = max(img["id"] for img in coco["images"]) + 1
next_ann_id = max(ann["id"] for ann in coco["annotations"]) + 1

ann_by_image = {}
for ann in coco["annotations"]:
    ann_by_image.setdefault(ann["image_id"], []).append(ann)

new_images = []
new_annotations = []

for image in tqdm(coco["images"], desc="Augmenting"):
    fname = image["file_name"].replace(".jpg", ".png")
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)
    mask_path = os.path.join(MASK_DIR, fname).replace(".jpg", ".png")
    mask = cv2.imread(mask_path)
    cv2.imwrite(img_path.replace("images", "images_aug"), img)     # 輸入影像複製到aug目錄
    cv2.imwrite(img_path.replace("images", "masks_aug"), mask)     # 輸入影像複製到aug目錄
    if img is None:
        continue

    anns = ann_by_image.get(image["id"], [])
    polygons = []
    valid_anns = []

    for ann in anns:
        if "segmentation" in ann and len(ann["segmentation"]) > 0:
            flat = ann["segmentation"][0]
            if len(flat) < 6:
                continue  # 略過無效的三角形以下的區塊
            pts = np.array(flat, dtype=np.float32).reshape((-1, 2))
            poly = Polygon(pts)
            polygons.append(poly)
            valid_anns.append(ann)


    if len(polygons) == 0:
        continue

    for i in range(1, 6):
        polys_on_img = PolygonsOnImage(polygons, shape=img.shape)
        img_aug, polys_aug = augmenter(image=img, polygons=polys_on_img)

        
        new_fname = fname.replace(".png", f"a{i}.png")
        out_img_path = os.path.join(OUT_IMG_DIR, new_fname)                 # augmentation 過的影像
        cv2.imwrite(out_img_path, img_aug)

        new_image = {
            "id": next_image_id,
            "width": image["width"],
            "height": image["height"],
            "file_name": new_fname,
            "license": image.get("license", 0),
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        new_images.append(new_image)

        # 初始化 mask (單通道黑底)
        mask = np.zeros((image["height"], image["width"], 3), dtype=np.uint8)

        for poly_aug, old_ann in zip(polys_aug.polygons, valid_anns):
            if not poly_aug.is_valid or len(poly_aug.coords) < 3:
                continue

            coords = poly_aug.coords.astype(np.int32)
            cv2.fillPoly(mask, [coords], color=color)

            x, y, w, h = cv2.boundingRect(coords)
            new_ann = {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": old_ann["category_id"],
                "segmentation": [coords.flatten().tolist()],
                "area": cv2.contourArea(coords.astype('float32')),
                "bbox": [x, y, w, h],
                "iscrowd": 0,
                "attributes": old_ann.get("attributes", {})
            }
            new_annotations.append(new_ann)
            next_ann_id += 1

        # 儲存 mask
        mask_fname = new_fname.replace(".jpg", ".png").replace(".jpeg", ".png").replace(".JPG", ".png")
        cv2.imwrite(os.path.join(OUT_MASK_DIR, mask_fname), mask)

        next_image_id += 1

# 更新 COCO 標註
coco["images"].extend(new_images)
coco["annotations"].extend(new_annotations)

with open(OUT_ANN_PATH, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

print("✅ 增強圖片與 masks 已儲存，COCO 標註已更新。")
