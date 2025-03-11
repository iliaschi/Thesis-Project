"""
finetune_synthetic.py

A simpler, more efficient script for:
1. Loading a pretrained EfficientNet model
2. Creating DataLoaders for synthetic 'train' and 'test'
3. Fine-tuning the model on the synthetic dataset
4. Evaluating performance
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import timm  # for create_model
from tqdm import tqdm

#######################
# 1) Configuration
#######################
# Paths to your synthetic data
BASE_SYNTH_DIR = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic"
TRAIN_DIR = os.path.join(BASE_SYNTH_DIR, "synth_train")  # e.g. .../angry_men_proc, angry_women_proc, ...
TEST_DIR  = os.path.join(BASE_SYNTH_DIR, "synth_test")

# A path to your existing pretrained EfficientNet weights (as a state_dict).
PRETRAINED_WEIGHTS = r"PATH\TO\pretrained_faces\my_efficientnet_b0_state_dict.pth"
SAVE_FINETUNED_MODEL = r"PATH\TO\save\my_efficientnet_b0_finetuned.pt"

# Basic training hyperparameters
IMG_SIZE       = 224
NUM_CLASSES    = 8
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
EPOCHS         = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

#######################
# 2) Data Transforms
#######################
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#######################
# 3) Dataset & DataLoader
#######################
def get_data_loaders(train_dir, test_dir, batch_size):
    """
    Create DataLoaders for train and test sets from folder structures.
    train_dir / test_dir each have subfolders of emotion_category
    """
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset  = datasets.ImageFolder(root=test_dir,  transform=test_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Train Set: {len(train_dataset)} images across {len(train_dataset.classes)} classes")
    print(f"Test  Set: {len(test_dataset)} images across {len(test_dataset.classes)} classes")

    return train_loader, test_loader, train_dataset.classes

#######################
# 4) Model Setup
#######################
def create_efficientnet_b0(num_classes):
    """
    Create an EfficientNet-B0 architecture from timm, 
    then load a state_dict if available, 
    and replace the final layer with a new classifier for 'num_classes' outputs.
    """
    # 1) Create baseline B0, no pretrained weights
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    
    # 2) Load your pretrained feature extractor weights (state_dict)
    state_dict = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)  # If final layer mismatch, do strict=False

    # 3) Replace the final layer with a new classifier
    #    Some variants have 'model.classifier = nn.Linear(...)'
    #    We'll use nn.Sequential for future expandability
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=num_classes)
    )
    return model

#######################
# 5) Training Function
#######################
def train_and_evaluate(model, train_loader, test_loader, device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Finetune the model on the synthetic train_loader, evaluate on test_loader.
    Returns the best model state_dict found during training.
    """
    # Basic cross-entropy
    criterion = nn.CrossEntropyLoss()
    
    # Use Adam for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        #################
        #   TRAINING
        #################
        model.train()
        running_loss = 0.0
        running_correct = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Train"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = 100.0 * running_correct / len(train_loader.dataset)

        #################
        #  VALIDATION
        #################
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} - Val"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / len(test_loader.dataset)
        val_acc  = 100.0 * val_correct / len(test_loader.dataset)

        print(f"Epoch [{epoch}/{epochs}] -- "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Track best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    return best_state, best_acc


import random
from torch.utils.data import Subset

def subset_dataset(full_dataset, fraction=0.01):
    """
    Return a smaller random subset of the given dataset.

    Parameters
    ----------
    full_dataset : torch.utils.data.Dataset
        An existing dataset, e.g. from torchvision.datasets.ImageFolder.
    fraction : float
        Fraction of the dataset to keep. For example, 0.1 => keep 10% of images.

    Returns
    -------
    Subset
        A PyTorch Subset object containing fraction * len(full_dataset) samples.
    """
    total_len = len(full_dataset)
    subset_len = int(total_len * fraction)
    indices = list(range(total_len))
    random.shuffle(indices)
    subset_indices = indices[:subset_len]
    return Subset(full_dataset, subset_indices)

import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

def evaluate_and_save_metrics(
    model, data_loader, device, class_names,
    output_dir="results", csv_name="evaluation_metrics.csv", do_gradcam=False
):
    """
    Run inference on the data_loader, compute confusion matrix, save metrics to a CSV.
    Optionally, placeholders for Grad-CAM or interpretability can be added.

    Parameters
    ----------
    model : torch.nn.Module
        The trained/fine-tuned model.
    data_loader : torch.utils.data.DataLoader
        Data loader for the evaluation dataset.
    device : str
        "cpu" or "cuda"
    class_names : list
        List of class names, e.g. ["Anger","Contempt","Disgust",...].
    output_dir : str
        Directory path to save metrics/results.
    csv_name : str
        Name of the CSV file to store confusion matrix and per-class stats.
    do_gradcam : bool
        If True, you might add code for Grad-CAM heatmaps or interpretability here.

    Returns
    -------
    None
        But writes out confusion matrix and metrics into a CSV, plus an optional text summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    # Per-class accuracy
    class_acc = []
    for i in range(len(class_names)):
        mask = (all_labels == i)
        correct = (all_preds[mask] == i).sum()
        total_i = mask.sum()
        accuracy_i = 100.0 * correct / total_i if total_i > 0 else 0
        class_acc.append(accuracy_i)

    # Prepare a DataFrame with metrics
    df = pd.DataFrame(cm, columns=class_names, index=class_names)
    df.index.name = 'True'
    df.columns.name = 'Predicted'
    
    # Save confusion matrix
    cm_csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(cm_csv_path)
    
    # Save per-class accuracy as well
    acc_data = {
        'class': class_names,
        'accuracy(%)': class_acc,
        'samples': [cm[i, :].sum() for i in range(len(class_names))]
    }
    df_acc = pd.DataFrame(acc_data)
    acc_csv_path = os.path.join(output_dir, "per_class_accuracy.csv")
    df_acc.to_csv(acc_csv_path, index=False)

    # Print or log summary
    overall_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print("Per-class results saved to:", acc_csv_path)

    if do_gradcam:
        # Placeholder: Grad-CAM requires hooking your model’s last conv layer,
        # generating heatmaps for each image, etc.
        # This is too large to detail here, but you can see e.g. pytorch-grad-cam library.
        pass


#######################
# 6) Main Script
#######################
def main_synthetic_demo():
    """
    Demonstration of:
    1) Creating data loaders from synthetic train/test
    2) Optionally subset the training data to a fraction
    3) Fine-tuning an EfficientNet
    4) Evaluating the results, saving confusion matrix & per-class stats to CSV
    """
    # 1. Data
    from torchvision import datasets
    from torch.utils.data import DataLoader
    
    # get the full train dataset
    full_train_ds = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    
    # Let's say we want only 10% to see if it runs quickly
    train_fraction = 0.01
    small_train_ds = subset_dataset(full_train_ds, fraction=train_fraction)
    train_loader = DataLoader(small_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_ds = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    class_names = list(full_train_ds.classes)
    print("Training classes:", class_names)
    
    # 2. Model
    model = create_efficientnet_b0(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # 3. Fine-tuning
    best_state, best_acc = train_and_evaluate(
        model, train_loader, test_loader,
        device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE
    )
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Best validation accuracy: {best_acc:.2f}%")
        # Save final model if desired
        torch.save(model, SAVE_FINETUNED_MODEL)
        print(f"[INFO] Finetuned model saved at: {SAVE_FINETUNED_MODEL}")
    else:
        print("[WARNING] No improvement found, skipping save.")

    # 4. Evaluate + Save Metrics (Confusion Matrix, Per-class stats)
    evaluate_and_save_metrics(
        model, test_loader, DEVICE, class_names,
        output_dir="results_synth_demo", 
        csv_name="confusion_matrix.csv",
        do_gradcam=False  # or True if you expand with Grad-CAM
    )
    print("[INFO] Evaluation metrics saved in 'results_synth_demo' folder.")

if __name__ == "__main__":
    main_synthetic_demo()

