import json
from PIL import Image

import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

import os
import cv2 

efficientnet_dict = {"agama":[], "anemone":[], "centipede":[], "chameleon":[], "cock":[], 
                     "crab": [], "dog":[], "eagle":[], "flamingo":[], "frog":[], "gecko":[],
                    "goldfish":[], "goose":[], "hornbill":[], "iguana":[], "jellyfish":[], 
                    "kite":[], "koala":[], "macaw":[], "magpie":[], "notebook":[], "orca":[], 
                    "ostrich":[], "peacock":[], "penguin":[], "quail":[], "quill":[], "racecar":[],
                    "radio":[], "ruler":[], "scorpion":[], "screwdriver":[], "shark":[], "snail":[], 
                    "snake":[], "stingray":[], "tarantula":[], "tench":[], "tick":[], "turtle":[]}

model_name = 'efficientnet-b7'
image_size = EfficientNet.get_image_size(model_name)
'''
directory = os.path.join('../', 'cropped_images/')
filenames = os.listdir(directory)

not_detected = 0

# img = Image.open(os.path.join("../cropped_images/agama_ablationcam_0.jpg")).convert('RGB')
# # Preprocess image
# tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
#                         transforms.ToTensor(),
#                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(img).unsqueeze(0)

# # Load class names
# labels_map = json.load(open('labels_map.txt'))
# labels_map = [labels_map[str(i)] for i in range(1000)]

# # Classify with EfficientNet
# model = EfficientNet.from_pretrained(model_name)
# model.eval()
# with torch.no_grad():
#     logits = model(img)
# preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

# print('-----')

# for idx in preds:
#     label = labels_map[idx]
#     prob = torch.softmax(logits, dim=1)[0, idx].item()
#     print('{:<75} ({:.2f}%)'.format(label, prob*100))

for f in filenames:
    if "score" in f:
        img = Image.open(os.path.join(directory, f)).convert('RGB')
        #img = Image.open(os.path.join(directory, filenames[0])).convert('RGB')

        # Preprocess image
        tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = tfms(img).unsqueeze(0)

        # Load class names
        labels_map = json.load(open('labels_map.txt'))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        # Classify with EfficientNet
        model = EfficientNet.from_pretrained(model_name)
        model.eval()
        with torch.no_grad():
            logits = model(img)
        preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

        print('-----')
        
        #for idx in preds:
        #    label = labels_map[idx]
        #    prob = torch.softmax(logits, dim=1)[0, idx].item()
        #    print('{:<75} ({:.2f}%)'.format(label, prob*100))
        

        file_name = f.split('.')[0]  # separate the name of f
        object_name = file_name.split('_')[0]   # get the name of the object

        label = labels_map[preds[0]]
        prob = torch.softmax(logits, dim=1)[0, preds[0]].item()
        #print('{:<75} ({:.2f}%)'.format(label, prob*100))

        #if label not in efficientnet_dict[object_name]:
        efficientnet_dict[object_name].append(label)

# check for number of unpredicted images
for prediction in efficientnet_dict.items():
    if len(prediction[1]) == 0:
        not_detected += 1

print(efficientnet_dict)
print("Number of segments not recognised: ", not_detected)
'''
# -------------------------------------------------------------------------------------
# train efficientNet model
# from roboflow import Roboflow
# rf = Roboflow(api_key="plG3ZxsU4xy6wv6BIAIK")
# project = rf.workspace("cits4010research").project("efficientnet-training")
# version = project.version(1)
# dataset = version.download("multiclass")

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# EfficientNet implementation
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
'''
train_data_dir = '../mario_datasets/train'
val_data_dir = '../mario_datasets/val'

# Data generators
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2, 
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224), # (600, 600)
    batch_size = 16,  #32
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size = (224, 224),  # (600, 600)
    batch_size = 16,    # 32
    class_mode = 'categorical'
)

base_model = EfficientNetB7(weights=None, include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
# patience parameter stops the training process after no improvements for 5 consecutive epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
# factor --> reduces learning rate by 20%
# patience --> number of epochs with no improvement after which learning rate will be reduced
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# training
history = model.fit(
    train_generator,
    epochs = 200,
    validation_data = val_generator,
    callbacks=[early_stopping, reduce_lr]
)

model.save('../efficientNetb7_trained.h5')

plt.figure(10, 5)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
'''
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/efficientNetb7_trained.h5')
# test the trained model

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0   # normalize the image
    return img_array

img_path = "C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_datasets/big_pipe_img.png"
# img_path = "C:/Users/USER/Documents/UWA/CITS4010 - AI_Research/AI_Research/mario_datasets/test/question_block/question_block_30.png"
img_array = preprocess_image(img_path)

predictions = model.predict(img_array)

class_names = ['goomba', 'jump', 'koopa', 'mushroom', 'pipe', 'question_block', 'run']

predicted_class = class_names[np.argmax(predictions)]
print(f'Predicted class: {predicted_class}')
'''

# evaluate the saved model
'''
test_data_dir = '../mario_datasets/test'
test_datagen = ImageDataGenerator(rescale=0.25)
test_generator = test_datagen.flow_from_directory(
    test_data_dir, 
    target_size=(224, 224),
    batch_size = 16,
    class_mode = 'categorical',
    shuffle=False
)

# num_test_samples = test_generator.samples
# test_loss, test_accuracy = model.evaluate(test_generator, steps=num_test_samples // test_generator.batch_size)
