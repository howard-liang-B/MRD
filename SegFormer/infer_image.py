from transformers import (
    SegformerFeatureExtractor, 
    SegformerForSemanticSegmentation
)
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import (
    draw_segmentation_map, 
    image_overlay,
    predict
)

import argparse
import cv2
import os
import glob

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from metrics import get_roc_auc

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='input/inference_data/images'
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou'
)
args = parser.parse_args()

overlay_out_dir = 'outputs/inference_results_image' # 儲存 mask 疊加原圖的照片
masks_out_dir = 'outputs/mask_results(320 x 320)' # 儲存純 mask
os.makedirs(overlay_out_dir, exist_ok=True)
os.makedirs(masks_out_dir, exist_ok=True)

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

###### 設置變數 ######
counter = 0
total_auc = 0
auc_df = pd.DataFrame(columns=['Image Name', 'auc'])
########################

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_name in image_paths:
    print("image_name: ", image_name)
    image = cv2.imread(image_name)
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 讀取遮罩
    val_dir = args.input
    print("val_dir: ", val_dir)
    img_mask_path = os.path.join(val_dir.replace("images", "masks"), os.path.basename(image_name).replace(".jpg", ".png"))
    print("img_mask_path: ", img_mask_path)
    image_mask = Image.open(img_mask_path)
    image_mask = image.resize((512, 512))
    image_mask = image_mask.convert('L')
    image_mask = np.array(image_mask.point(lambda x: 0 if x < 128 else 255, '1')) # change 

    # Get labels.
    pred_labels = predict(model, extractor, image, args.device)
    print(f'predict labels: {pred_labels}')

    # Get segmentation map.
    seg_map = draw_segmentation_map(pred_labels.cpu(), LABEL_COLORS_LIST)
    overlay_outputs, mask_outputs = image_overlay(image, seg_map) # change 

    ####### 儲存每個 roc、auc #######
    roc_dir = "outputs/roc_auc"
    os.makedirs(roc_dir, exist_ok=True)
    roc_save_path = os.path.join(roc_dir, image_name)

#     fpr, tpr, auc = get_roc_auc(image_mask, outputs)
#     plt.plot(fpr, tpr, 'r', label = "AUC = %0.2f" % auc)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend(loc='best')
#     plt.savefig(roc_save_path)
#     plt.clf()
#     #################################
#     counter += 1
#     total_auc += auc
#     #################################
    
#     # Save path.
#     image_name = image_path.split(os.path.sep)[-1]
#     save_overlay_path = os.path.join(overlay_out_dir, image_name)
#     save_mask_path = os.path.join(masks_out_dir, image_name.replace(".jpg", ".png"))
    
#     cv2.imwrite(save_overlay_path, overlay_outputs)
#     cv2.imwrite(save_mask_path, mask_outputs)
#     print("## save: ", image_name)


# ################ 儲存 Excel ################
# excel_file_path = "outputs/roc_auc/results.xlsx"
# average_auc = auc_df['AUC'].mean()
# average_auc = round(average_auc, 4)

# # 將平均 AUC 添加到 DataFrame 中
# average_row = pd.DataFrame({'Image Name': ['Average DSC'], 'AUC': [average_auc]})
# auc_df = pd.concat([auc_df, average_row], ignore_index=True)
# print(f'\n-- average auc: {average_auc} --')

# auc_df.to_excel(excel_file_path, index=False)

# print("\n== inference_image complete ==")
# ############################################