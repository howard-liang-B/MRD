import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large


# # backbone --> ResNet50
# def prepare_model(num_classes=2):
#     model = deeplabv3_resnet50(weights='DEFAULT')
#     model.classifier[4] = nn.Conv2d(256, num_classes, 1)
#     model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
#     return model


# # backbone --> ResNet101
# def prepare_model(num_classes=2):
#     model = deeplabv3_resnet101(weights='DEFAULT')
#     model.classifier[4] = nn.Conv2d(256, num_classes, 1)
#     model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
#     return model

# backbone --> deeplabv3_mobilenet_v3_large
def prepare_model(num_classes=2):
    model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(10, num_classes, 1)
    return model


if __name__ == '__main__':
    model = prepare_model()
    print(model)