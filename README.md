# Auxiliary Evaluation of Marginal Ridge Discrepancy in Periodontal Disease Using Deep Learning on Periapical Radiographs

<img src="resource/mrd.png" width="500" height="300">

## Abstract
Marginal Ridge Discrepancy (MRD) is an important early indicator of periodontal disease, often resulting from tooth inclination or alveolar bone loss, leading to uneven interproximal ridge height. Although periapical radiographs commonly observe bone and root structures, image overlap and angle variation often hinder accurate clinical interpretation. This study proposes a deep learning-based system integrating image segmentation and angular evaluation to assist dentists in objectively classifying MRD severity and improving diagnostic efficiency. 

## Methods
We adopted a Mask R-CNN model with ResNet-101 as the backbone, incorporating warm-up and learning rate scheduling strategies to ensure stable convergence. Moreover, Mask R-CNN localized the cemento-enamel junction and alveolar crest by overlapping the mask image. We also introduced a novel angular measurement method to quantify the MRD between adjacent ridges and categorize periodontal disease severity. 

## Results
ResNet-101 achieved the best segmentation performance among tested backbones with 98.11% pixel-wise accuracy. Recall scores reached 97.60% for teeth, 97.24% for crowns, and 97.53% for bone structures. The MRD classification model achieved 93.41% accuracy with a mean angular error of only 0.85°, demonstrating strong clinical reliability. Conclusions: The proposed method effectively addresses challenges in evaluating ridge loss on periapical radiographs. Providing accurate and objective assessment enhances early periodontal diagnosis, reduces clinical workload, and supports improved medical quality and treatment planning.

## Directory Structure
* : Training Model
* : CEJ & ALC location algorithm
* : Dataset

## Citation
```
@inproceedings{
      title={Auxiliary Evaluation of Marginal Ridge Discrepancy in Periodontal Disease Using Deep Learning on Periapical Radiographs},
      author={Yuan-Jin Lin, Yi-Cheng Mao, Tai-Jung Lin, Chin-Hao Liang, Shih-Lun Chen, Tsung-Yi Chen, Chiung-An Chen, Kuo-Chen Li, Wei-Chen Tu and Patricia Angela R. Abu 9},
      year={2025},
      version={1.0},
      doi={[doi]}
}
```
