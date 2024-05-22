import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import time 
'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args
'''

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    '''
    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }
    '''

    from torchvision.models import resnet50, ResNet50_Weights

    #model = models.resnet50(pretrained=True).to(torch.device('cpu')).eval()
    # new weights with accuracy 80.858%
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(torch.device('cpu')).eval()  
   
    # from torchsummary import summary
    # summary(model, (3, 224, 224))

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
    

    target_layers = [model.layer4]

    directory = os.path.join('../', 'image_datasets/')
    #directory = os.path.join('../', 'datasets_for_efficientnet/')
    filenames = os.listdir(directory)
    '''filenames = ["ruler.png",
                "scorpion.png", "screwdriver.png", "shark.png", "snail.png", "snake.png", 
                "stingray.png", "tarantula.png", "tench.png", "tick.png", "turtle.png"] '''
    # for f in filenames:
        #rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    f = "scorpion.png"
    rgb_img = cv2.imread(os.path.join(directory, f))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    #cam_algorithm = methods[args.method]
    cam_algorithm = LayerCAM
    with cam_algorithm(model=model,
                    target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        '''
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth) '''
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, device='cpu')
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    #output_dir = os.path.join('../', 'datasets_for_efficientnet_outputs/')
    output_dir = os.path.join('../', 'image_datasets_outputs/')
    os.makedirs(output_dir, exist_ok=True)

    object_name = f.split('.')[0]
    cam_output_path = os.path.join(output_dir, f'{object_name}_layercam.jpg')

    cv2.imwrite(cam_output_path, cam_image)
