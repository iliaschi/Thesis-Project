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

train_dir,test_dir=AFFECT_TRAIN_DATA_DIR,AFFECT_VAL_DATA_DIR


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

model=model.to(device)


set_parameter_requires_grad(model, requires_grad=False)
set_parameter_requires_grad(model.classifier, requires_grad=True)
train(model,3,0.001,robust=True)

 

set_parameter_requires_grad(model, requires_grad=True)



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