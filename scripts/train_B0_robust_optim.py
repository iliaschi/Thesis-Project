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
import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import timm  # for create_model
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib as plt

import random
import collections

import time

from datetime import datetime


# ------------------------------
# 1) Configuration
# ------------------------------

# Basic training hyperparameters
IMG_SIZE       = 224
NUM_CLASSES    = 8
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
EPOCHS         = 30 # 10 # 40 as the paper
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

dropout_rate = 0.2 # only for vggface2 as a base for only synth training
EARLY_STOP_PATIENCE = 20 # 5
TRAIN_FRACTION  = 1 # Fraction of the training dataset to use, e.g. 0.1 => 10%

# Paths to your synthetic data
# Should have the splits inside 
# BASE_SYNTH_DIR = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic" # old folder without splits
BASE_SYNTH_DIR = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\train_splits" # with splits

# training data splits
train_splits = os.path.join(BASE_SYNTH_DIR, "train_splits")

VAL_DIR = "val_real"

# A path to your existing pretrained EfficientNet weights (as a state_dict).
# C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_base_vggface2_state_dict.pth # base state dict
PRETRAINED_WEIGHTS = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_8_best_afew_state_dict.pth" # real trained state dict
# PRETRAINED_WEIGHTS = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_base_vggface2_state_dict.pth" # base face recognition state dict
SAVE_FINETUNED_MODEL = r"C:\Users\ilias\Python\Thesis-Project\models\weights\my_EffNet_b0_finetuned_test_cuda_synth_data_real_val.pt"



# ------------------------------
# 2) DATA TRANSFORMS
# ------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ------------------------------
# 3) SUBSET DATASET
# ------------------------------
def subset_dataset(full_dataset, fraction=1.0):
    """
    Return a random subset of 'fraction' * len(full_dataset).
    If fraction >= 1.0, return the entire dataset.
    """
    if fraction >= 1.0:
        return full_dataset
    
    total_len = len(full_dataset)
    subset_len = int(total_len * fraction)
    indices = list(range(total_len))
    random.shuffle(indices)
    chosen_indices = indices[:subset_len]
    return Subset(full_dataset, chosen_indices)



# ------------------------------
# 4) Model Creation
# ------------------------------
def create_efficientnet_b0(num_classes, weights_path=None, map_location="cpu"):
    """
    Unified approach that sets up the classifier layer,
    then attempts to load either a state dict or a full model object
    if 'weights_path' is given.
    """
    
        # Create baseline model
    # model = timm.create_model('tf_efficientnet_b0', pretrained=False, drop_rate=dropout_rate) # Dropout for synthetic data only
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    
    # # Step 2: define the final classifier EXACTLY as in your testing code
    model.classifier = nn.Sequential(nn.Linear(1280, num_classes))

    
    # Step 3: if we have a weights_path, check if it’s a state dict or full model
    if weights_path is not None and len(weights_path) > 0:
        try:
            checkpoint = torch.load(weights_path, map_location=map_location)
            print(f"[INFO] Loaded checkpoint from {weights_path}")
            
            if isinstance(checkpoint, collections.OrderedDict):
                # It's already a pure state dict
                # model.load_state_dict(checkpoint, strict=False) # False for synthetic data only
                model.load_state_dict(checkpoint, strict=True)
                print("[INFO] Loaded pure state_dict directly.")
            else:
                # Possibly a full model or something else
                # e.g. checkpoint might have .state_dict()
                if hasattr(checkpoint, 'state_dict'):
                    # model.load_state_dict(checkpoint.state_dict(), strict=False) # False for synthetic data only
                    model.load_state_dict(checkpoint.state_dict(), strict=True)
                    print("[INFO] Extracted state_dict from entire model object.")
                else:
                    print("[WARNING] checkpoint is neither an OrderedDict nor a full model with state_dict. Skipping load.")
        except Exception as e:
            print(f"[ERROR] Could not load {weights_path}: {str(e)}")
    else:
        print("[WARNING] No weights_path provided or file does not exist. Starting from scratch.")
    
    return model


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

def cross_entropy_loss_with_soft_target(pred, soft_target, weights=None):
    # pred shape: (B, n_classes)
    # soft_target: (B, n_classes)
    if weights is None:
        # standard label-smoothing CE
        return torch.mean(torch.sum(- soft_target * torch.nn.functional.log_softmax(pred, dim=-1), dim=1))
    else:
        # Weighted version
        # we expect "weights" to be shape (n_classes,)
        # We multiply log-softmax by (weights * soft_target) per class
        # Expand weights => (1, n_classes)
        weights = weights.view(1, -1)
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        # scale by weights * soft_target
        # sum over classes => average over batch
        weighted = - weights * soft_target * log_probs
        return torch.mean(torch.sum(weighted, dim=1))

def cross_entropy_with_label_smoothing(pred, target, n_classes, label_smoothing=0.1, weights=None):
    soft_target = label_smooth(target, n_classes, label_smoothing=label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target, weights=weights)



##############################
# TRAIN FUNCTION
##############################
from robust_optimization import RobustOptimizer
def train_model(
    model,
    train_loader,
    val_loader=None,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    save_path=SAVE_FINETUNED_MODEL,
    early_stopping_patience=EARLY_STOP_PATIENCE
):
    """
    Training loop that uses a robust optimizer (two-step approach) for each batch.
    Replaces standard Adam with e.g. SAM or robust_optimization.RobustOptimizer.
    
    We still do:
      - CrossEntropyLoss
      - LR scheduling (optional)
      - Early stopping
      - Metric logging
    """
    # Make sure you import or define robust_optimization elsewhere:
    # from robust_optimization import RobustOptimizer

    # For normal cross-entropy
    criterion = nn.CrossEntropyLoss()

    # Initialize robust optimizer (similar to SAM approach)
    #  e.g. RobustOptimizer(params, base_optimizer=optim.Adam, lr=lr)
    optimizer = RobustOptimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        optim.Adam,
        lr=lr
    )

    # If you still want a scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer.base_optimizer, mode='max', factor=0.1, patience=3)
    
    # Container for epoch metrics
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'time_taken': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ---- TRAIN PHASE ----
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = len(train_loader.dataset)

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # First backward + step
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward + backward pass (SAM approach)
            outputs2 = model(images)
            loss2 = criterion(outputs2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            # Track training loss
            total_loss += loss.item() * images.size(0)

            # Compute training accuracy on this batch
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_loss = total_loss / total_samples
        train_acc = 100.0 * correct / total_samples

        # ---- VALIDATION PHASE ----
        val_loss = 0.0
        val_acc = 0.0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total   = len(val_loader.dataset)
            val_loss_sum= 0.0

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    vloss   = criterion(outputs, labels).item() * images.size(0)
                    val_loss_sum += vloss
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()

            val_loss = val_loss_sum / val_total
            val_acc  = 100.0 * val_correct / val_total

            # Step the scheduler with val_acc if you like
            scheduler.step(val_acc)

            # Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model so far
                best_model_path = save_path.replace('.pt', '_best.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best model => Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"[INFO] Val Acc did not improve => patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"[INFO] Early stopping triggered at epoch {epoch}")
                    break

        # ---- LOG EPOCH METRICS ----
        elapsed = time.time() - epoch_start
        current_lr = optimizer.base_optimizer.param_groups[0]['lr']  # note: robust uses .base_optimizer
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss if val_loader else float('nan'))
        metrics['val_acc'].append(val_acc if val_loader else float('nan'))
        metrics['learning_rate'].append(current_lr)
        metrics['time_taken'].append(elapsed)

        print(f"Epoch {epoch}/{epochs} - Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loader:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        # Save metrics to CSV each epoch
        csv_path = os.path.dirname(save_path) + '/training_metrics.csv'
        pd.DataFrame(metrics).to_csv(csv_path, index=False)
        print(f"[INFO] Metrics saved => {csv_path}")

    # ---- END OF TRAINING ----
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Final model => {save_path}")
    print(f"[INFO] Best Val Acc => {best_val_acc:.2f}%")

    return model, metrics



# ------------------------------
# 6) Directory Creation
# ------------------------------
def create_results_directory(base_dir=None):
    """
    Create a timestamped results directory with proper structure
    """
    if base_dir is None:
        base_dir = r"C:\Users\ilias\Python\Thesis-Project\results"
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"training_experiment_2_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    dirs = {
        "root": results_dir,
        "models": os.path.join(results_dir, "models"),
        "metrics": os.path.join(results_dir, "metrics"),
        "plots": os.path.join(results_dir, "plots"),
        "timestamp": timestamp
    }
    os.makedirs(dirs["models"], exist_ok=True)
    os.makedirs(dirs["metrics"], exist_ok=True)
    os.makedirs(dirs["plots"], exist_ok=True)
    print(f" [INFO] Created results directory: {results_dir}")
    print(f" [INFO] Created models directory: {dirs['models']}")
    
    return dirs


# ------------------------------
# 7) run_all_splits
# ------------------------------
def run_all_splits(split_list):
    """
    Run training on multiple data splits and save results
    
    Parameters:
    -----------
    split_list : list
        List of split names to process
    """
    # Create results directory
    results_dir = create_results_directory()
    
    # Save experiment configuration
    config = {
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "PRETRAINED_WEIGHTS": PRETRAINED_WEIGHTS,
        "TRAIN_FRACTION": TRAIN_FRACTION,
        "DEVICE": DEVICE,
        "splits": split_list,
        "NUM_CLASSES": NUM_CLASSES
    }
    
    config_path = os.path.join(results_dir["root"], "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Store results to compare later
    all_results = {}
    
    for split_name in split_list:
        # I want this outside the loop
        # Create a unique save path for this split's model
        print(f" training directory is {train_dir}")
        
        train_dir = os.path.join(BASE_SYNTH_DIR, split_name) # to change
        # old
        # train_dir = os.path.join(r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\train_splits", split_name)
        
        # Create a unique save path for this split's model
        save_model_path = os.path.join(results_dir["models"], f"enet_b0_{split_name}_finetuned.pt")
        
        print(f"\n{'='*50}")
        print(f"[INFO] Training on split: {split_name}")
        print(f"[INFO] Using train folder: {train_dir}")
        print(f"{'='*50}\n")
        
        # 1) Load dataset
        try:
            dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
            print(f"[INFO] Found classes: {dataset.classes}")
            print(f"[INFO] Full dataset size: {len(dataset)}")
        except Exception as e:
            print(f"[ERROR] Could not load dataset from {train_dir}: {str(e)}")
            continue
        
        # 2) Subset if needed
        partial_dataset = subset_dataset(dataset, fraction=TRAIN_FRACTION)
        print(f"[INFO] Using {len(partial_dataset)} images for training ({TRAIN_FRACTION*100:.1f}% of available data)")
        
        train_loader = DataLoader(partial_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        # 3) Create model
        model = create_efficientnet_b0(
            num_classes=NUM_CLASSES,
            weights_path=PRETRAINED_WEIGHTS,
            map_location="cpu"
        )
        model.to(DEVICE)
        
        # 4) Optional: Load a separate val set
        try:
            val_dataset = datasets.ImageFolder(root=os.path.join(BASE_SYNTH_DIR, "val_real"), transform=train_transforms)
            print(f"[INFO] Validation dataset size: {len(val_dataset)}")
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        except Exception as e:
            print(f"[WARNING] Could not load validation dataset: {str(e)}")
            print("[INFO] Training without validation")
            val_loader = None
        
        # 5) Train
        try:
            _, metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=DEVICE,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                save_path=save_model_path
            )
            
            # Store metrics for this split
            metrics_file = os.path.join(results_dir["metrics"], f"{split_name}_metrics.csv")
            pd.DataFrame(metrics).to_csv(metrics_file, index=False)
            
            # Extract key metrics for summary
            best_val_acc = max(metrics['val_acc']) if val_loader else 0
            best_val_epoch = metrics['epoch'][np.argmax(metrics['val_acc'])] if val_loader else 0
            
            all_results[split_name] = {
                'best_val_acc': best_val_acc,
                'best_val_epoch': best_val_epoch,
                'final_val_acc': metrics['val_acc'][-1] if val_loader else 0,
                'best_train_acc': max(metrics['train_acc']),
                'final_train_acc': metrics['train_acc'][-1],
                'avg_epoch_time': np.mean(metrics['time_taken'])
            }
            
            # Generate learning curves plot
            try:
                plt.figure(figsize=(12, 5))
                
                # Plot training and validation accuracy
                plt.subplot(1, 2, 1)
                plt.plot(metrics['epoch'], metrics['train_acc'], 'b-', label='Training Accuracy')
                if val_loader:
                    plt.plot(metrics['epoch'], metrics['val_acc'], 'r-', label='Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title(f'Accuracy Curves - {split_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot training and validation loss
                plt.subplot(1, 2, 2)
                plt.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Training Loss')
                if val_loader:
                    plt.plot(metrics['epoch'], metrics['val_loss'], 'r-', label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Loss Curves - {split_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir["plots"], f"{split_name}_learning_curves.png"), dpi=300)
                plt.close()
            except Exception as e:
                print(f"[WARNING] Could not generate learning curves plot: {str(e)}")
            
        except Exception as e:
            print(f"[ERROR] Training failed for {split_name}: {str(e)}")
            all_results[split_name] = {'error': str(e)}
        
        print(f"[INFO] Finished training on {split_name}.\n")
    
    # Save summary of all results
    try:
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        summary_path = os.path.join(results_dir["metrics"], "splits_comparison.csv")
        results_df.to_csv(summary_path)
        print(f"[INFO] Comparison of all splits saved to {summary_path}")
        
        # Generate summary bar plot for validation accuracy
        if 'best_val_acc' in results_df.columns:
            plt.figure(figsize=(12, 6))
            results_df['best_val_acc'].plot(kind='bar', color='skyblue')
            plt.title('Best Validation Accuracy by Split')
            plt.xlabel('Split')
            plt.ylabel('Validation Accuracy (%)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir["plots"], "validation_accuracy_comparison.png"), dpi=300)
            plt.close()
    except Exception as e:
        print(f"[WARNING] Could not save comparison results: {str(e)}")
    
    print(f"\n[INFO] All training complete. Results saved to {results_dir['root']}")
    return results_dir, all_results



def main():

    # List of splits to process  
    
    # full names of training splits
    # split_list = ["100M_0W", "100M_25W", "100M_50W", "100M_75W", "100M_100W", "100W_0M", "100W_25M", "100W_50M", "100W_75M"]
    
    # full dataset for overview
    split_list = ["100M_100W"]

    run_all_splits(split_list)

if __name__ == "__main__":
    main()
