import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from stable_baselines3 import PPO

import torch.nn as nn
#if __name__ == '__main__':

model = PPO.load("./sample_train/best_model_1.zip") # replace with the saved torch model

network_model = model.policy

torch.save(network_model, "ppo.pth")
model = torch.load("ppo.pth")
features_extractor = model.features_extractor

n_flatten = features_extractor.cnn(torch.as_tensor(model.observation_space.sample()[None]).float()).shape[1]
features_extractor.linear = nn.Sequential(nn.Linear(809472, 512), nn.ReLU())
# from torchsummary import summary
# summary(features_extractor, input_size=(model.observation_space.shape))

# layer = network_model.features_extractor
#layer = model.features_extractor
# features_extractor = {k[len('features_extractor.'):]: v for k, v in model.items() if k.startswith('features_extractor.cnn.0.weight')}
# features_extractor = {k: v for k, v in model.items() if k.startswith('features_extractor.cnn.0.weight')}

# Define the sub-model for features_extractor
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
target_layers = [features_extractor]
'''
image_path = "./mario_datasets/train/right,B/2de60cfa-eb34-11ee-8ad3-e0c264fc39e6.jpg"

rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) '''

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
# targets = [ClassifierOutputTarget(281)]
targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
cam = LayerCAM(model=model, target_layers=target_layers)
'''
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
cam.batch_size = 32
grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)

grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gb_model = GuidedBackpropReLUModel(model=model, device=device)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    #os.makedirs(args.output_dir, exist_ok=True)

    output_id = "sample_2"

    cam_output_path = os.path.join("C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_saliency", f"{output_id}_scorecam.jpg")
    gb_output_path = os.path.join("C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_saliency", f"{output_id}_scorecam_gb.jpg")
    cam_gb_output_path = os.path.join("C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_saliency", f"{output_id}_scorecam_cam_gb.jpg")

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb) '''