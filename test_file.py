# %env CUDA_VISIBLE_DEVICES=1

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from random import shuffle

import pandas as pd

import pickle

from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,ExtraTreesClassifier
from sklearn import svm,metrics,preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

#from scipy.misc import imread, imresize

# %matplotlib inline
ALL_DATA_DIR = '/home/HDD6TB/datasets/emotions/'
INPUT_SIZE = (224, 224)
#INPUT_SIZE = (299, 299)
AFFECT_DATA_DIR=ALL_DATA_DIR+'AffectNet/'
AFFECT_TRAIN_DATA_DIR = AFFECT_DATA_DIR+'full_res/train'
AFFECT_VAL_DATA_DIR = AFFECT_DATA_DIR+'full_res/val'
AFFECT_SEVEN_TRAIN_DATA_DIR = AFFECT_DATA_DIR+'full_res/seven_emotions/train'
AFFECT_SEVEN_VAL_DATA_DIR = AFFECT_DATA_DIR+'full_res/seven_emotions/val'

AFFECT_IMG_TRAIN_DATA_DIR = AFFECT_DATA_DIR+str(INPUT_SIZE[0])+'/train'
AFFECT_IMG_VAL_DATA_DIR = AFFECT_DATA_DIR+str(INPUT_SIZE[0])+'/val'
AFFECT_IMG_SEVEN_TRAIN_DATA_DIR = AFFECT_DATA_DIR+str(INPUT_SIZE[0])+'/seven_emotions/train'
AFFECT_IMG_SEVEN_VAL_DATA_DIR = AFFECT_DATA_DIR+str(INPUT_SIZE[0])+'/seven_emotions/val'
AFFECT_TRAIN_ORIG_DATA_DIR = AFFECT_DATA_DIR+'orig/train'
AFFECT_VAL_ORIG_DATA_DIR = AFFECT_DATA_DIR+'orig/val'

IMG_AFFECT_DATA_DIR = AFFECT_DATA_DIR+'Manually_Annotated_Images/'
AFFECT_TRAIN_FILE=AFFECT_DATA_DIR+'training.csv'
AFFECT_TRAIN_FILTERED_FILE=AFFECT_DATA_DIR+'training_filtered.csv'
AFFECT_VAL_FILE=AFFECT_DATA_DIR+'validation.csv'
AFFECT_VAL_FILTERED_FILE=AFFECT_DATA_DIR+'validation_filtered.csv'

AFFECT_TRAIN_ALIGNED_DATA_DIR = AFFECT_DATA_DIR+'full_res_aligned/train'# 8 emotions
AFFECT_VAL_ALIGNED_DATA_DIR = AFFECT_DATA_DIR+'full_res_aligned/val'
AFFECT_TRAIN_SEVEN_ALIGNED_DATA_DIR = AFFECT_DATA_DIR+'full_res_aligned/seven_emotions/train'# 7 emotions
AFFECT_VAL_SEVEN_ALIGNED_DATA_DIR = AFFECT_DATA_DIR+'full_res_aligned/seven_emotions/val'
import csv
def save_csv(filename,outfile, dir_to_save):
    affect_df = pd.read_csv(filename)
    affect_vals=[d for i,d in affect_df.iterrows()]
    with open(os.path.join(AFFECT_DATA_DIR,outfile), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filepath','emotion','valence', 'arousal'])
        writer.writeheader()
        for d in affect_vals:
            #print(d.subDirectory_filePath,d.face_x,d.face_y, d.face_width, d.face_height, d.expression)
            if d.expression>=len(emotion_labels) or d.face_width<0:
                continue
            input_path=os.path.join(IMG_AFFECT_DATA_DIR,d.subDirectory_filePath)
            dst_file_path=os.path.join(emotion_labels[d.expression],os.path.basename(d.subDirectory_filePath))
            #print(input_path,dst_file_path)
            if os.path.exists(os.path.join(dir_to_save,dst_file_path)):
                #writer.writerow({'filepath':dst_file_path[len(AFFECT_DATA_DIR):],'emotion':emotion_labels[d.expression],'valence':d.valence, 'arousal':d.arousal})
                writer.writerow({'filepath':dst_file_path,'emotion':emotion_labels[d.expression],'valence':d.valence, 'arousal':d.arousal})

if False:
    save_csv(AFFECT_VAL_FILE,AFFECT_VAL_FILTERED_FILE,AFFECT_IMG_VAL_DATA_DIR)
    save_csv(AFFECT_TRAIN_FILE,AFFECT_TRAIN_FILTERED_FILE,AFFECT_IMG_TRAIN_DATA_DIR)

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm


print(f"Torch: {torch.__version__}")

# Training settings
batch_size = 64 #48# 32# 32 #16 #8 #
epochs = 40
lr = 3e-5
gamma = 0.7
seed = 42
device = 'cuda'
use_cuda = torch.cuda.is_available()
print(use_cuda)

#net_description='affectnet_'+net_description
train_dir,test_dir=AFFECT_TRAIN_DATA_DIR,AFFECT_VAL_DATA_DIR
#train_dir,test_dir=AFFECT_IMG_TRAIN_DATA_DIR,AFFECT_IMG_VAL_DATA_DIR
#train_dir,test_dir=AFFECT_SEVEN_TRAIN_DATA_DIR,AFFECT_SEVEN_VAL_DATA_DIR
#train_dir,test_dir=AFFECT_IMG_SEVEN_TRAIN_DATA_DIR,AFFECT_IMG_SEVEN_VAL_DATA_DIR
#train_dir,test_dir=AFFECT_TRAIN_ORIG_DATA_DIR,AFFECT_VAL_ORIG_DATA_DIR

#train_dir,test_dir=AFFECT_TRAIN_ALIGNED_DATA_DIR,AFFECT_VAL_ALIGNED_DATA_DIR
#train_dir,test_dir=AFFECT_TRAIN_SEVEN_ALIGNED_DATA_DIR,AFFECT_VAL_SEVEN_ALIGNED_DATA_DIR

print(train_dir,test_dir)
USE_ENET2=True #False #
IMG_SIZE=260 if USE_ENET2 else 224 # 300 # 80 #
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)
print(test_transforms)

#adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
# FER only model
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs) 

print(len(train_dataset), len(test_dataset))
# 287651 4000
(unique, counts) = np.unique(train_dataset.targets, return_counts=True)
cw=1/counts
cw/=cw.min()
class_weights = {i:cwi for i,cwi in zip(unique,cw)}
print(counts, class_weights.values())
# [ 24882   3750   3803   6378 134415  74874  25459  14090] dict_values([5.402097902097902, 35.844, 35.34446489613463, 21.07478833490122, 1.0, 1.7952159628175335, 5.279665344279037, 9.539744499645138])
num_classes=len(train_dataset.classes)
print(num_classes)
# 8
# loss function
weights = torch.FloatTensor(list(class_weights.values())).cuda()
if False:
    criterion = nn.CrossEntropyLoss(weight=weights)
    #criterion = nn.CrossEntropyLoss()
else:
    def label_smooth(target, n_classes: int, label_smoothing=0.1):
        # convert to one-hot
        batch_size = target.size(0)
        target = torch.unsqueeze(target, 1)
        soft_target = torch.zeros((batch_size, n_classes), device=target.device)
        soft_target.scatter_(1, target, 1)
        # label smoothing
        soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
        return soft_target

    def cross_entropy_loss_with_soft_target(pred, soft_target):
        #logsoftmax = nn.LogSoftmax(dim=-1)
        return torch.mean(torch.sum(- weights*soft_target * torch.nn.functional.log_softmax(pred, -1), 1))

    def cross_entropy_with_label_smoothing(pred, target):
        soft_target = label_smooth(target, pred.size(1)) #num_classes) #
        return cross_entropy_loss_with_soft_target(pred, soft_target)

    criterion=cross_entropy_with_label_smoothing
from robust_optimization import RobustOptimizer
import copy
def train(model,n_epochs=epochs, learningrate=lr, robust=False):
    # optimizer
    if robust:
        optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=learningrate)
        #print(optimizer)
    else:
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    best_acc=0
    best_model=None
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            if robust:
                #optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
  
                # second forward-backward pass
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = (output.argmax(dim=1) == label).float().sum()
            epoch_accuracy += acc
            epoch_loss += loss
        epoch_accuracy /= len(train_dataset)
        epoch_loss /= len(train_dataset)
        
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss
        epoch_val_accuracy /= len(test_dataset)
        epoch_val_loss /= len(test_dataset)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        if best_acc<epoch_val_accuracy:
            best_acc=epoch_val_accuracy
            best_model=copy.deepcopy(model.state_dict())
        #scheduler.step()
    
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Best acc:{best_acc}")
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss
        epoch_val_accuracy /= len(test_dataset)
        epoch_val_loss /= len(test_dataset)
        print(
            f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    else:
        print(f"No best model Best acc:{best_acc}")
# Finetune CNN
from torchvision.models import resnet101,mobilenet_v2
import timm
model=timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
model.classifier=torch.nn.Identity()
model.load_state_dict(torch.load('../../models/pretrained_faces/state_vggface2_enet0_new.pt')) #_new
# <All keys matched successfully>
model.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=num_classes)) #1792 #1280 #1536
#model.head.fc=nn.Linear(in_features=3072, out_features=num_classes)
#model.head=nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))
model=model.to(device)
print(model)
# EfficientNet(
#   (conv_stem): Conv2dSame(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
#   (bn1): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   (act1): SiLU(inplace=True)
#   (blocks): Sequential(
#     (0): Sequential(
#       (0): DepthwiseSeparableConv(
#         (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
#         (bn1): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pw): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn2): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): Identity()
#       )
#     )
#     (1): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(96, 96, kernel_size=(3, 3), stride=(2, 2), groups=96, bias=False)
#         (bn2): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
#         (bn2): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (2): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(144, 144, kernel_size=(5, 5), stride=(2, 2), groups=144, bias=False)
#         (bn2): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
#         (bn2): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (3): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(240, 240, kernel_size=(3, 3), stride=(2, 2), groups=240, bias=False)
#         (bn2): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
#         (bn2): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): InvertedResidual(
#         (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
#         (bn2): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (4): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
#         (bn2): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
#         (bn2): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): InvertedResidual(
#         (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
#         (bn2): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (5): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(672, 672, kernel_size=(5, 5), stride=(2, 2), groups=672, bias=False)
#         (bn2): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (3): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (6): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#   )
#   (conv_head): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (bn2): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   (act2): SiLU(inplace=True)
#   (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=True)
#   (classifier): Sequential(
#     (0): Linear(in_features=1280, out_features=8, bias=True)
#   )
# )
set_parameter_requires_grad(model, requires_grad=False)
set_parameter_requires_grad(model.classifier, requires_grad=True)
train(model,3,0.001,robust=True)
#Best acc:0.48875007033348083
#7: Best acc:0.558712363243103
# RobustOptimizer (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     eps: 1e-08
#     lr: 0.001
#     rho: 0.05
#     weight_decay: 0
# )
 

set_parameter_requires_grad(model, requires_grad=True)
train(model,6,1e-4,robust=True)
# RobustOptimizer (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     eps: 1e-08
#     lr: 0.0001
#     rho: 0.05
#     weight_decay: 0
# )


if USE_ENET2:
    if False: # 7 emotions
        PATH='../../models/affectnet_emotions/enet_b2_7.pt'
        model_name='enet2_7_pt'
    else:
        #PATH='../../models/affectnet_emotions/enet_b2_8.pt'
        PATH='../../models/affectnet_emotions/enet_b2_8_best.pt'
        model_name='enet2_8_pt'
else:
    if False: # 7 emotions from AFFECT_IMG_SEVEN_TRAIN_DATA_DIR and AFFECT_IMG_SEVEN_VAL_DATA_DIR
        PATH='../../models/affectnet_emotions/enet_b0_7.pt'
        model_name='enet0_7_pt'
    else:
        PATH='../../models/affectnet_emotions/enet_b0_8_best_vgaf.pt'
        #PATH='../../models/affectnet_emotions/enet_b0_8_best_afew.pt'
        model_name='enet0_8_pt'
print(PATH)
# Save
torch.save(model, PATH)
# Load
print(PATH)
model = torch.load(PATH)
model=model.eval()

class_to_idx=train_dataset.class_to_idx
print(class_to_idx)
idx_to_class={idx:cls for cls,idx in class_to_idx.items()}
print(idx_to_class)
{'Anger': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4, 'Neutral': 5, 'Sadness': 6, 'Surprise': 7}
{0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
print(test_dir)
y_val,y_scores_val=[],[]
model.eval()
for class_name in tqdm(os.listdir(test_dir)):
    if class_name in class_to_idx:
        class_dir=os.path.join(test_dir,class_name)
        y=class_to_idx[class_name]
        for img_name in os.listdir(class_dir):
            filepath=os.path.join(class_dir,img_name)
            img = Image.open(filepath)
            img_tensor = test_transforms(img)
            img_tensor.unsqueeze_(0)
            scores = model(img_tensor.to(device))
            scores=scores[0].data.cpu().numpy()
            #print(scores.shape)
            y_scores_val.append(scores)
            y_val.append(y)

y_scores_val=np.array(y_scores_val)
y_val=np.array(y_val)
print(y_scores_val.shape,y_val.shape)
# /home/HDD6TB/datasets/emotions/AffectNet/full_res/val
#   0%|          | 0/8 [00:00<?, ?it/s]
# (4000, 8) (4000,)
y_pred=np.argmax(y_scores_val,axis=1)
acc=100.0*(y_val==y_pred).sum()/len(y_val)
print(acc)

y_train=np.array(train_dataset.targets)

for i in range(y_scores_val.shape[1]):
    _val_acc=(y_pred[y_val==i]==i).sum()/(y_val==i).sum()
    print('%s %d/%d acc: %f' %(idx_to_class[i],(y_train==i).sum(),(y_val==i).sum(),100*_val_acc))

#-Contempt
сontempt_idx=class_to_idx['Contempt']
y_scores_val_filtered=y_scores_val[:, [i!=сontempt_idx for i in idx_to_class]]
print(y_scores_val_filtered.shape)
y_pred_filtered=np.argmax(y_scores_val_filtered,axis=1)
other_indices=y_val!=сontempt_idx
y_val_new=np.array([y if y<сontempt_idx else y-1 for y in y_val if y!=сontempt_idx])
acc=100.0*np.mean(y_val_new==y_pred_filtered[other_indices])
print(acc)


labels=list(class_to_idx.keys())
print(labels)
IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
def plt_conf_matrix(y_true,y_pred,labels):
    print(y_pred.shape,y_true.shape, (y_pred==y_true).mean())

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_confusion_matrix(IC, y_pred,y_true,display_labels=labels,cmap=plt.cm.Blues,ax=ax,colorbar=False) #,normalize='true'
    plt.tight_layout()
    plt.show()
plt_conf_matrix(y_val,y_pred,labels)


# Evaluate single-task models
test_dir = AFFECT_DATA_DIR+'full_res/val'
class_to_idx={'Anger': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4, 'Neutral': 5, 'Sadness': 6, 'Surprise': 7}
idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

for model_name in ['enet_b2_8_best.pt','enet_b2_8.pt','enet_b0_8_best_vgaf.pt','enet_b0_8_best_afew.pt']:
    print(model_name)
    model = torch.load('../../models/affectnet_emotions/'+model_name,map_location=device)
    model=model.eval()
    img_size=260 if 'enet_b2' in model_name else 224
    test_transforms = transforms.Compose(
        [
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ]
    )
    y_val,y_scores_val=[],[]
    imgs=[]
    for class_name in tqdm(os.listdir(test_dir)):
        if class_name in class_to_idx:
            class_dir=os.path.join(test_dir,class_name)
            y=class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                filepath=os.path.join(class_dir,img_name)
                img = Image.open(filepath)
                img_tensor = test_transforms(img)
                imgs.append(img_tensor)
                y_val.append(y)
                if len(imgs)>=32:
                    scores = model(torch.stack(imgs, dim=0).to(device))
                    scores=scores.data.cpu().numpy()
                    #print(scores.shape)
                    if len(y_scores_val)==0:
                        y_scores_val=scores
                    else:
                        y_scores_val=np.concatenate((y_scores_val,scores),axis=0)

                    imgs=[]

    if len(imgs)>0:
        scores = model(torch.stack(imgs, dim=0).to(device))
        scores=scores.data.cpu().numpy()
        #print(scores.shape)
        if len(y_scores_val)==0:
            y_scores_val=scores
        else:
            y_scores_val=np.concatenate((y_scores_val,scores),axis=0)

        imgs=[]
    y_val=np.array(y_val)

    y_pred=np.argmax(y_scores_val,axis=1)
    acc=100.0*(y_val==y_pred).sum()/len(y_val)
    print(model_name,'Accuracy (8 classes):',acc, 'total samples:',len(y_val))

    #-Contempt
    сontempt_idx=class_to_idx['Contempt']
    y_scores_val_filtered=y_scores_val[:, [i!=сontempt_idx for i in idx_to_class]]
    y_pred_filtered=np.argmax(y_scores_val_filtered,axis=1)
    other_indices=y_val!=сontempt_idx
    y_val_new=np.array([y if y<сontempt_idx else y-1 for y in y_val if y!=сontempt_idx])
    acc=100.0*np.mean(y_val_new==y_pred_filtered[other_indices])
    print('Accuracy (7 classes):',acc, 'total samples:',len(y_val_new))
    print()



    test_dir = AFFECT_DATA_DIR+'full_res/seven_emotions/val'
class_to_idx={'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 6}
idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

for model_name in ['enet_b2_7.pt','enet_b0_7.pt']:
    print(model_name)
    model = torch.load('../../models/affectnet_emotions/'+model_name,map_location=device)
    model=model.eval()
    img_size=260 if 'enet_b2' in model_name else 224
    test_transforms = transforms.Compose(
        [
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ]
    )
    y_val,y_scores_val=[],[]
    imgs=[]
    for class_name in tqdm(os.listdir(test_dir)):
        if class_name in class_to_idx:
            class_dir=os.path.join(test_dir,class_name)
            y=class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                filepath=os.path.join(class_dir,img_name)
                img = Image.open(filepath)
                img_tensor = test_transforms(img)
                imgs.append(img_tensor)
                y_val.append(y)
                if len(imgs)>=32:
                    scores = model(torch.stack(imgs, dim=0).to(device))
                    scores=scores.data.cpu().numpy()
                    #print(scores.shape)
                    if len(y_scores_val)==0:
                        y_scores_val=scores
                    else:
                        y_scores_val=np.concatenate((y_scores_val,scores),axis=0)

                    imgs=[]

    if len(imgs)>0:
        scores = model(torch.stack(imgs, dim=0).to(device))
        scores=scores.data.cpu().numpy()
        #print(scores.shape)
        if len(y_scores_val)==0:
            y_scores_val=scores
        else:
            y_scores_val=np.concatenate((y_scores_val,scores),axis=0)

        imgs=[]
    y_val=np.array(y_val)

    y_pred=np.argmax(y_scores_val,axis=1)
    acc=100.0*(y_val==y_pred).sum()/len(y_val)
    print(model_name,'Accuracy (7 classes):',acc, 'total samples:',len(y_val))


    # Multi-task: FER+Valence-Arousal
affectnet_expr2emotion={0:'Neutral',1:'Happiness', 2:'Sadness', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
#idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
class_to_idx={cls:idx for idx,cls in idx_to_class.items()}
def ConcordanceCorCoeff(prediction, ground_truth):
    mean_gt = torch.mean (ground_truth, 0)
    mean_pred = torch.mean (prediction, 0)
    var_gt = torch.var (ground_truth, 0)
    var_pred = torch.var (prediction, 0)
    v_pred = prediction - mean_pred
    v_gt = ground_truth - mean_gt
    cor = torch.sum (v_pred * v_gt) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)))
    sd_gt = torch.std(ground_truth)
    sd_pred = torch.std(prediction)
    numerator=2*cor*sd_gt*sd_pred
    denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
    ccc = numerator/denominator
    return ccc

def ConcordanceCorCoeffLoss(prediction, ground_truth):
    return (1-ConcordanceCorCoeff(prediction, ground_truth))/2
# Train
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file, root, transform):
        df=pd.read_csv(csv_file)
        df=df[df['emotion'].isin(class_to_idx.keys())]
        self.paths = list(df.filepath)#[:4000]
        self.targets = np.array([class_to_idx[cls] for cls in df.emotion])#[:4000]
        self.valence_arousal = df[['valence','arousal']].to_numpy()#[:4000]
        self.transform = transform
        self.root=root

    def __len__(self): return len(self.paths)

    def __getitem__(self,idx):
        #dealing with the image
        img = Image.open(os.path.join(self.root,self.paths[idx])).convert('RGB')
        img = self.transform(img)

        #dealing with the labels
        emotion_label = self.targets[idx]
        valence=torch.tensor(float(self.valence_arousal[idx,0]), dtype=torch.float32)
        arousal=torch.tensor(float(self.valence_arousal[idx,1]), dtype=torch.float32)
        
        return img.data, (emotion_label, valence, arousal)

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

batch_size=48
train_dataset = MultiTaskDataset(csv_file=AFFECT_TRAIN_FILTERED_FILE, root=train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_dataset = MultiTaskDataset(csv_file=AFFECT_VAL_FILTERED_FILE,root=test_dir, transform=test_transforms)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs) 

print(len(train_dataset), len(test_dataset))
# 287651 4000
(unique, counts) = np.unique(train_dataset.targets, return_counts=True)
cw=1/counts
cw/=cw.min()
class_weights = {i:cwi for i,cwi in zip(unique,cw)}
print(counts, class_weights.values())

num_classes=len(class_to_idx)
print(num_classes)
# [ 24882   3750   3803   6378 134416  74874  25459  14090] dict_values([5.4021380917932635, 35.84426666666667, 35.344727846437024, 21.07494512386328, 1.0, 1.7952293185885622, 5.2797046231195255, 9.539815471965934])
# 8
import timm
model=timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
model.classifier=torch.nn.Identity()
model.load_state_dict(torch.load('../../models/pretrained_faces/state_vggface2_enet0_new.pt')) #_new
model.classifier=nn.Linear(in_features=1280, out_features=num_classes+2) #1792 #1280 #1536 #1408
model=model.to(device)
print(model)
# EfficientNet(
#   (conv_stem): Conv2dSame(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
#   (bn1): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   (act1): SiLU(inplace=True)
#   (blocks): Sequential(
#     (0): Sequential(
#       (0): DepthwiseSeparableConv(
#         (conv_dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
#         (bn1): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pw): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn2): BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): Identity()
#       )
#     )
#     (1): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(96, 96, kernel_size=(3, 3), stride=(2, 2), groups=96, bias=False)
#         (bn2): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
#         (bn2): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (2): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(144, 144, kernel_size=(5, 5), stride=(2, 2), groups=144, bias=False)
#         (bn2): BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
#         (bn2): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (3): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(240, 240, kernel_size=(3, 3), stride=(2, 2), groups=240, bias=False)
#         (bn2): BatchNorm2d(240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
#         (bn2): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): InvertedResidual(
#         (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
#         (bn2): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (4): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
#         (bn2): BatchNorm2d(480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
#         (bn2): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): InvertedResidual(
#         (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
#         (bn2): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (5): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2dSame(672, 672, kernel_size=(5, 5), stride=(2, 2), groups=672, bias=False)
#         (bn2): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (2): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (3): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (6): Sequential(
#       (0): InvertedResidual(
#         (conv_pw): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): SiLU(inplace=True)
#         (conv_dw): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
#         (bn2): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#         (act2): SiLU(inplace=True)
#         (se): SqueezeExcite(
#           (conv_reduce): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
#           (act1): SiLU(inplace=True)
#           (conv_expand): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (conv_pwl): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#   )
#   (conv_head): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
#   (bn2): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   (act2): SiLU(inplace=True)
#   (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=True)
#   (classifier): Linear(in_features=1280, out_features=10, bias=True)
# )
class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()
        #self.task_num = 3
        #self.log_vars = nn.Parameter(torch.zeros((self.task_num)))
        weights = torch.FloatTensor(list(class_weights.values())).to(device)
        self.loss_emotions = nn.CrossEntropyLoss(weight=weights)
        self.loss_valence=ConcordanceCorCoeffLoss #nn.MSELoss()
        self.loss_arousal=ConcordanceCorCoeffLoss #nn.MSELoss()

    def forward(self, preds, target):
        loss_emotions=self.loss_emotions(preds[:,:num_classes],target[0])
        loss_valence=self.loss_valence(preds[:,num_classes],target[1])
        loss_arousal=self.loss_arousal(preds[:,num_classes+1],target[2])
        return loss_emotions+(loss_valence+loss_arousal)*1
my_criterion=MultiTaskLossWrapper()
from robust_optimization import RobustOptimizer
import copy
def train(model,n_epochs=epochs, learningrate=lr, robust=False):
    # optimizer
    if robust:
        optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=learningrate)
        #print(optimizer)
    else:
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    best_acc=0
    best_model=None
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_mse_valence=epoch_mse_arousal=0
        model.train()
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = [l.to(device) for l in label]
            
            output = model(data)
            loss = my_criterion(output, label)

            if robust:
                #optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
  
                # second forward-backward pass
                output = model(data)
                loss = my_criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = (output[:,:num_classes].argmax(dim=1) == label[0]).float().sum()
            epoch_accuracy += acc
            
            mse_valense = ((output[:,num_classes] - label[1])**2).float().sum()
            epoch_mse_valence += mse_valense
            mse_arousal = ((output[:,num_classes+1] - label[2])**2).float().sum()
            epoch_mse_arousal += mse_arousal
            
            epoch_loss += loss

        epoch_accuracy /= len(train_dataset)
        mse_valense /= len(train_dataset)
        mse_arousal /= len(train_dataset)
        epoch_loss /= len(train_dataset)

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            epoch_val_mse_valence=epoch_val_mse_arousal=0
            for data, label in test_loader:
                data = data.to(device)
                label = [l.to(device) for l in label]
                
                val_output = model(data)
                val_loss = my_criterion(val_output, label)

                acc = (val_output[:,:num_classes].argmax(dim=1) == label[0]).float().sum()
                epoch_val_accuracy += acc

                mse_valense = ((val_output[:,num_classes] - label[1])**2).float().sum()
                epoch_val_mse_valence += mse_valense
                mse_arousal = ((val_output[:,num_classes+1] - label[2])**2).float().sum()
                epoch_val_mse_arousal += mse_arousal
                epoch_val_loss += val_loss
        
        epoch_val_accuracy /= len(test_dataset)
        epoch_val_mse_valence /= len(test_dataset)
        epoch_val_mse_arousal /= len(test_dataset)
        epoch_val_loss /= len(test_dataset)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - mse (valence): {epoch_mse_valence:.4f} - mse (arousal): {epoch_mse_arousal:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - val_mse (valence): {epoch_val_mse_valence:.4f} - val_mse (arousal): {epoch_val_mse_arousal:.4f}\n"
        )
        if best_acc<epoch_val_accuracy:
            best_acc=epoch_val_accuracy
            best_model=copy.deepcopy(model.state_dict())
        #scheduler.step()
    
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Best acc:{best_acc}")
    else:
        print(f"No best model Best acc:{best_acc}")
set_parameter_requires_grad(model, requires_grad=False)
set_parameter_requires_grad(model.classifier, requires_grad=True)
train(model,3,0.001,robust=True)
set_parameter_requires_grad(model, requires_grad=True)
train(model,6,1e-4,robust=True)
PATH='../../models/affectnet_emotions/enet_b0_8_va_mtl.pt'
model_name='enet0_8_mtl_pt'
print(PATH)
if False:
    torch.save(model, PATH)
else:
    model = torch.load(PATH)
# ../../models/affectnet_emotions/enet_b0_8_va_mtl.pt
model=model.to(device)
model.eval()
with torch.no_grad():
    epoch_val_accuracy = 0
    epoch_val_mse_valence=epoch_val_mse_arousal=0
    for data, label in test_loader:
        data = data.to(device)
        label = [l.to(device) for l in label]

        val_output = model(data)

        acc = (val_output[:,:num_classes].argmax(dim=1) == label[0]).float().sum()
        epoch_val_accuracy += acc

        mse_valense = ((val_output[:,num_classes] - label[1])**2).float().sum()
        epoch_val_mse_valence += mse_valense
        mse_arousal = ((val_output[:,num_classes+1] - label[2])**2).float().sum()
        epoch_val_mse_arousal += mse_arousal
epoch_val_accuracy /= len(test_dataset)
epoch_val_mse_valence /= len(test_dataset)
epoch_val_mse_arousal /= len(test_dataset)
print(
    f"val_acc: {epoch_val_accuracy:.4f} - val_mse (valence): {epoch_val_mse_valence:.4f} - val_mse (arousal): {epoch_val_mse_arousal:.4f}\n"
)
# val_acc: 0.6193 - val_mse (valence): 0.1880 - val_mse (arousal): 0.1496

y_val,y_scores_val=[],[]
model.eval()
for class_name in tqdm(os.listdir(test_dir)):
    if class_name in class_to_idx:
        class_dir=os.path.join(test_dir,class_name)
        y=class_to_idx[class_name]
        for img_name in os.listdir(class_dir):
            filepath=os.path.join(class_dir,img_name)
            img = Image.open(filepath)
            img_tensor = test_transforms(img)
            img_tensor.unsqueeze_(0)
            scores = model(img_tensor.to(device))
            scores=scores[0].data.cpu().numpy()
            #print(scores.shape)
            y_scores_val.append(scores[:-2])
            y_val.append(y)
#   0%|          | 0/8 [00:00<?, ?it/s]
y_scores_val=np.array(y_scores_val)
y_val=np.array(y_val)
print(y_scores_val.shape,y_val.shape)

y_pred=np.argmax(y_scores_val,axis=1)
acc=100.0*(y_val==y_pred).sum()/len(y_val)
print(acc)

y_train=np.array(train_dataset.targets)
for i in range(y_scores_val.shape[1]):
    _val_acc=(y_pred[y_val==i]==i).sum()/(y_val==i).sum()
    print('%s %d/%d acc: %f' %(idx_to_class[i],(y_train==i).sum(),(y_val==i).sum(),100*_val_acc))

#-Contempt
сontempt_idx=class_to_idx['Contempt']
y_scores_val_filtered=y_scores_val[:, [i!=сontempt_idx for i in idx_to_class]]
print(y_scores_val_filtered.shape)
y_pred=np.argmax(y_scores_val_filtered,axis=1)
other_indices=y_val!=сontempt_idx
y_val_new=np.array([y if y<сontempt_idx else y-1 for y in y_val if y!=сontempt_idx])
acc=100.0*np.mean(y_val_new==y_pred[other_indices])
print(acc)
# (4000, 8) (4000,)
# 61.925
# Anger 24882/500 acc: 64.000000
# Contempt 3750/500 acc: 57.800000
# Disgust 3803/500 acc: 53.800000
# Fear 6378/500 acc: 56.200000
# Happiness 134416/500 acc: 81.000000
# Neutral 74874/500 acc: 55.800000
# Sadness 25459/500 acc: 64.600000
# Surprise 14090/500 acc: 62.200000
# (4000, 7)
# 64.97142857142858
affect_df = pd.read_csv(AFFECT_VAL_FILE)
X_VA_val=[]
VA_preds=[]
for i,d in tqdm(affect_df.iterrows()):
    if d.expression not in affectnet_expr2emotion:
        continue
    X_VA_val.append([d.valence, d.arousal])
    img = Image.open(os.path.join(test_dir,affectnet_expr2emotion[d.expression],os.path.basename(d.subDirectory_filePath))).convert('RGB')
    img_tensor = test_transforms(img)
    img_tensor.unsqueeze_(0)
    scores = model(img_tensor.to(device))
    scores=scores[0].data.cpu().numpy()
    VA_preds.append([scores[-2],scores[-1]])
X_VA_val=np.array(X_VA_val)
VA_preds=np.array(VA_preds)
print(X_VA_val.shape,VA_preds.shape)
# 0it [00:00, ?it/s]
# (4000, 2) (4000, 2)
print('Valence,Arousal')
mse=((X_VA_val-VA_preds)**2).mean(axis=0)
print('MSE',mse)
print('RMSE',np.sqrt(mse))
print('MAE',abs((X_VA_val-VA_preds)).mean(axis=0))
print('CCC',ConcordanceCorCoeff(torch.from_numpy(X_VA_val[:,0]),torch.from_numpy(VA_preds[:,0])).numpy(),
      ConcordanceCorCoeff(torch.from_numpy(X_VA_val[:,1]),torch.from_numpy(VA_preds[:,1])).numpy())
# Valence,Arousal
# MSE [0.18803859 0.14956716]
# RMSE [0.43363417 0.38673914]
# MAE [0.31258092 0.30441801]
# CCC 0.5936892154066524 0.5486192135795945
# Evaluate multi-task models
import mobilefacenet
test_dir = AFFECT_DATA_DIR+'full_res/val'
class_to_idx={'Anger': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4, 'Neutral': 5, 'Sadness': 6, 'Surprise': 7}
idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
affect_df = pd.read_csv(AFFECT_VAL_FILE)
for model_name in ['mbf_mtl.pt','enet_b0_8_va_mtl.pt','mobilevit_mtl.pt']:
    print(model_name)
    model = torch.load('../../models/affectnet_emotions/'+model_name,map_location=device)
    model=model.eval()
    if 'mbf_' in model_name: 
        test_transforms = transforms.Compose(
        [
            transforms.Resize((112,112)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
        ]
    )
    else:
        test_transforms = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            ]
        )

    y_val,y_scores_val=[],[]
    VA_val,VA_preds=[],[]
    imgs=[]
    
    for i,d in affect_df.iterrows():
        if d.expression not in affectnet_expr2emotion:
            continue
        class_name=affectnet_expr2emotion[d.expression]
        img = Image.open(os.path.join(test_dir,class_name,os.path.basename(d.subDirectory_filePath))).convert('RGB')
        img_tensor = test_transforms(img)
        y_val.append(class_to_idx[class_name])
        VA_val.append([d.valence, d.arousal])
        imgs.append(img_tensor)
        if len(imgs)>=32:
            scores = model(torch.stack(imgs, dim=0).to(device))
            scores=scores.data.cpu().numpy()
            #print(scores.shape)
            if len(y_scores_val)==0:
                y_scores_val=scores
            else:
                y_scores_val=np.concatenate((y_scores_val,scores),axis=0)

            imgs=[]

    if len(imgs)>0:
        scores = model(torch.stack(imgs, dim=0).to(device))
        scores=scores.data.cpu().numpy()
        #print(scores.shape)
        if len(y_scores_val)==0:
            y_scores_val=scores
        else:
            y_scores_val=np.concatenate((y_scores_val,scores),axis=0)

        imgs=[]
    
    y_val=np.array(y_val)
    VA_val=np.array(VA_val)
    VA_preds=y_scores_val[:,-2:]
    y_scores_val=y_scores_val[:,:-2]
    #print(y_scores_val.shape,VA_val.shape,VA_preds.shape)

    y_pred=np.argmax(y_scores_val,axis=1)
    acc=100.0*(y_val==y_pred).sum()/len(y_val)
    print(model_name,'Accuracy (8 classes):',acc, 'total samples:',len(y_val))

    #-Contempt
    сontempt_idx=class_to_idx['Contempt']
    y_scores_val_filtered=y_scores_val[:, [i!=сontempt_idx for i in idx_to_class]]
    y_pred_filtered=np.argmax(y_scores_val_filtered,axis=1)
    other_indices=y_val!=сontempt_idx
    y_val_new=np.array([y if y<сontempt_idx else y-1 for y in y_val if y!=сontempt_idx])
    acc=100.0*np.mean(y_val_new==y_pred_filtered[other_indices])
    print('Accuracy (7 classes):',acc, 'total samples:',len(y_val_new))
    
    mse=((VA_val-VA_preds)**2).mean(axis=0)
    print('Valence,Arousal','MSE',mse,'RMSE',np.sqrt(mse),'MAE',abs((VA_val-VA_preds)).mean(axis=0),
          'CCC',ConcordanceCorCoeff(torch.from_numpy(VA_val[:,0]),torch.from_numpy(VA_preds[:,0])).numpy(),
          ConcordanceCorCoeff(torch.from_numpy(VA_val[:,1]),torch.from_numpy(VA_preds[:,1])).numpy())
    print()
# mbf_mtl.pt
# (4000, 8) (4000, 2) (4000, 2)
# mbf_mtl.pt Accuracy (8 classes): 62.325 total samples: 4000
# Accuracy (7 classes): 65.17142857142856 total samples: 3500
# Valence,Arousal MSE [0.20012764 0.14959607] RMSE [0.44735628 0.38677651] MAE [0.32998172 0.30524319] CCC 0.5773172049436921 0.5468350979153239

# enet_b0_8_va_mtl.pt
# (4000, 8) (4000, 2) (4000, 2)
# enet_b0_8_va_mtl.pt Accuracy (8 classes): 61.925 total samples: 4000
# Accuracy (7 classes): 64.97142857142858 total samples: 3500
# Valence,Arousal MSE [0.18803859 0.14956715] RMSE [0.43363417 0.38673913] MAE [0.31258092 0.30441801] CCC 0.5936892133929977 0.5486192228642244

# mobilevit_mtl.pt
# (4000, 8) (4000, 2) (4000, 2)
# mobilevit_mtl.pt Accuracy (8 classes): 62.5 total samples: 4000
# Accuracy (7 classes): 66.45714285714286 total samples: 3500
# Valence,Arousal MSE [0.17911225 0.13746791] RMSE [0.42321655 0.37076665] MAE [0.30257006 0.28658601] CCC 0.5991092380989274 0.5649780970197389

# Example usage
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
set_session(sess)
from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)
import matplotlib.pyplot as plt
fpath='../test_images/20180720_174416.jpg'
frame_bgr=cv2.imread(fpath)
plt.figure(figsize=(5, 5))
frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
bounding_boxes, points = imgProcessing.detect_faces(frame)
points = points.T
for bbox,p in zip(bounding_boxes, points):
    box = bbox.astype(np.int)
    x1,y1,x2,y2=box[0:4]    
    face_img=frame[y1:y2,x1:x2,:]
    
    img_tensor = test_transforms(Image.fromarray(face_img))
    img_tensor.unsqueeze_(0)
    scores = model(img_tensor.to(device))
    scores=scores[0].data.cpu().numpy()
    plt.figure(figsize=(3, 3))
    plt.imshow(face_img)
    if False:
        plt.title(idx_to_class[np.argmax(scores)])
    else:
        plt.title(f"{idx_to_class[np.argmax(scores[:8])]} V:{scores[8]:.2f} A:{scores[9]:.2f}")





# path to weights
# path_eff_b0_7=/c/Users/ilias/Python/Thesis-Project/models/weights/enet_b0_7.pt
# path_eff_b2_7=/c/Users/ilias/Python/Thesis-Project/models/weights/enet_b2_7.pt


