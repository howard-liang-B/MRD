import torch
import argparse
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model

from metrics import get_roc_auc

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input dir')
parser.add_argument(
    '--model',
    default='outputs/best_model_iou.pth',
    help='path to the model checkpoint'
)
args = parser.parse_args()

out_overlay_dir = os.path.join('outputs', 'inference_results', "overlay")
out_masks_dir = os.path.join('outputs', 'inference_results', "masks")
os.makedirs(out_overlay_dir, exist_ok=True)
os.makedirs(out_masks_dir, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

###### 設置變數 ######
counter = 0
total_auc = 0
auc_df = pd.DataFrame(columns=['Image Name', 'auc'])
image_siz = (256, 256)
#####################

all_image_names = os.listdir(args.input)
for i, image_name in enumerate(all_image_names):
    print(f"\nImage {i+1}, image_path: {image_name}")

    # Read the image.
    val_dir = args.input
    roc_dir = "outputs/roc_auc"
    os.makedirs(roc_dir, exist_ok=True)

    img_path = os.path.join(val_dir, image_name)
    img_mask_path = os.path.join(val_dir.replace("images", "masks"), image_name.replace(".jpg", ".png"))
    roc_save_path = os.path.join(roc_dir, image_name)
    
    image = Image.open(img_path)
    image = image.resize(image_siz)

    image_mask = Image.open(img_mask_path)
    print(f'# {np.unique(image_mask)}')
    image_mask = image.resize(image_siz)
    image_mask = image_mask.convert('L')
    image_mask = np.array(image_mask.point(lambda x: 0 if x < 128 else 255, '1')) # change 

    # Do forward pass and get the output dictionary.
    outputs = get_segment_labels(image, model, device)
    outputs = outputs['out']

    ####### 儲存每個 roc、auc #######
    fpr, tpr, auc = get_roc_auc(image_mask, outputs)
    plt.plot(fpr, tpr, 'r', label = "AUC = %0.2f" % auc)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')
    plt.savefig(roc_save_path)
    plt.clf()
    #################################
    counter += 1
    total_auc += auc
    #################################


    ####### 儲存每個 roc、auc #######
    auc_df = pd.concat([auc_df, pd.DataFrame({'Image Name': [image_name], 'AUC': [auc]})], ignore_index=True)

    # 獲得分割圖
    segmented_image = draw_segmentation_map(outputs)
    final_image = image_overlay(image, segmented_image)

    out_masks_path = os.path.join(out_masks_dir, image_name.replace(".jpg", ".png"))
    out_overlay_path = os.path.join(out_overlay_dir, image_name)
    cv2.imwrite(out_masks_path, segmented_image)
    cv2.imwrite(out_overlay_path, final_image)
    print(f'### save {out_masks_path}, {out_overlay_path}')

excel_file_path = "outputs/roc_auc/results.xlsx"
average_auc = auc_df['AUC'].mean()
average_auc = round(average_auc, 4)

# 將平均 AUC 添加到 DataFrame 中
average_row = pd.DataFrame({'Image Name': ['Average DSC'], 'AUC': [average_auc]})
auc_df = pd.concat([auc_df, average_row], ignore_index=True)
print(f'--> average auc: {average_auc}')

auc_df.to_excel(excel_file_path, index=False)

print("\n== inference_image complete ==")