import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import os 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
import time

import torch

'''#setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# demonstrates the flow of the game (10000 frames stacked)
done = True
#Loop through each frame in the game
for step in range(100000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    #show the game on screen
    env.render()
# close the game
env.close() '''

# preprocess environment
from gym.wrappers import GrayScaleObservation    # allows AI to analyse the velocity of character
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

import gym
from gym.spaces import Box

import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#env.observation_space = Box(0, 255, (96, 96, 3), np.uint8)

# env = GrayScaleObservation(env, keep_dim=True)
#env = DummyVecEnv([lambda: env])    # wrap inside the Dummy Environment
#env = VecFrameStack(env, 4, channels_order='last')  # stack the frames

'''
state = env.reset()

state, reward, done, info = env.step([5])

# shows the 4 frames stacked together
plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()
'''
'''
class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True
 
# set up model saving callback
# CHECKPOINT_DIR = './train/'
CHECKPOINT_DIR = './sample_train/'
# LOG_DIR = './logs/'
LOG_DIR = './sample_logs/'
checkpoint_callback = TrainAndLoggingCallback(check_freq=1, save_path=CHECKPOINT_DIR)

# create AI model
# CnnPolicy is interchangable with MlpPolicy
# learning rate --- should not be too fast (ineffective learning), or too slow (could take up years)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=512)

# train the AI model, this is where the model starts to learn
model.learn(total_timesteps=1, callback=checkpoint_callback)
'''
'''
# test it out
#PPO_model = PPO.load('../train/best_model_1600000.zip')

# demonstrates the agent behaviour

state = env.reset()
#print(SIMPLE_MOVEMENT[PPO_model.predict(state)[0][0]])   # initial state that the model predicts
# loop through the game

while True:
    action, _state = PPO_model.predict(state)
    state, reward, done, info = env.step(action)
    
    action_file = open("./epoch_policies/actions.txt", "a")
    predicted_action = SIMPLE_MOVEMENT[PPO_model.predict(state)[0][0]]
    # output the predicted actions into a text file
    #time.sleep(0.3) # slows down the frame
    print(f"{predicted_action}\n", file=action_file)
    action_file.close()
    
    env.render() 
'''
# transfer PPO model to existing resnet50 model for transfer learning + fine tuning
# import train_resnet50 as tl50
'''
# transfer learning
tl_resnet50 = tl50.train_model(tl50.resnet50_model, tl50.criterion, tl50.optimizer, tl50.step_lr_scheduler, num_epochs=100)
torch.save(tl_resnet50, "../tl_resnet50.pth")
# fine-tuning
model_conv = tl50.train_model(tl50.model_conv, tl50.criterion, tl50.optimizer, tl50.step_lr_scheduler, num_epochs=100)

# save the trained model
torch.save(model_conv, "../fttl_resnet50.pth")
'''

# output saliency maps
import mario_cam

import argparse
import os
import cv2
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#image_path = "../mario_datasets/train/right,B/535584af-eb34-11ee-8248-e0c264fc39e6.jpg"
image_path = "../mario_screenshot/koopa.png"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

cam = LayerCAM(model=mario_cam.model, target_layers=mario_cam.target_layers)

cam.batch_size = 32
grayscale_cam = cam(input_tensor=input_tensor,
                        targets=mario_cam.targets,
                            aug_smooth=True,
                            eigen_smooth=True)

grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gb_model = GuidedBackpropReLUModel(model=mario_cam.model, device=device)
gb = gb_model(input_tensor, target_category=None)

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)
gb = deprocess_image(gb)

output_id = "tl_6"

cam_output_path = os.path.join("C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_saliency", f"{output_id}_scorecam.jpg")
gb_output_path = os.path.join("C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_saliency", f"{output_id}_scorecam_gb.jpg")
cam_gb_output_path = os.path.join("C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_saliency", f"{output_id}_scorecam_cam_gb.jpg")

cv2.imwrite(cam_output_path, cam_image)
# cv2.imwrite(gb_output_path, gb)
# cv2.imwrite(cam_gb_output_path, cam_gb) 
