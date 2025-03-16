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

#######################
# 1) Configuration
#######################
# Paths to your synthetic data
BASE_SYNTH_DIR = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic"
TRAIN_DIR = r"data/synthetic/train_splits/100M_100W" #os.path.join(BASE_SYNTH_DIR, "100M_100W")  # e.g. .../angry_men_proc, angry_women_proc, ...
TEST_DIR  = os.path.join(BASE_SYNTH_DIR, "100M_100W")
VAL_DIR = "val_real"

# A path to your existing pretrained EfficientNet weights (as a state_dict).
PRETRAINED_WEIGHTS = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_8_best_afew_state_dict.pth"
SAVE_FINETUNED_MODEL = r"C:\Users\ilias\Python\Thesis-Project\models\weights\my_efficientnet_b0_finetuned_test_cuda_full_real.pt"

# Basic training hyperparameters
IMG_SIZE       = 224
NUM_CLASSES    = 8
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
EPOCHS         = 10
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

dropout_rate = 0.2


# Fraction of the training dataset to use, e.g. 0.1 => 10%
TRAIN_FRACTION  = 1


##############################
# DATA TRANSFORMS
##############################
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


##############################
# SUBSET DATASET
##############################
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



def create_efficientnet_b0(num_classes, weights_path=None, map_location="cpu"):
    """
    Unified approach that sets up the classifier layer,
    then attempts to load either a state dict or a full model object
    if 'weights_path' is given.
    """
    
        # Create baseline model
    model = timm.create_model('tf_efficientnet_b0', pretrained=False, drop_rate=dropout_rate)
    
    # # Replace classifier with dropout + linear layer
    # model.classifier = nn.Sequential(
    #     nn.Dropout(dropout_rate),
    #     nn.Linear(1280, num_classes)
    # )

    # model.classifier = nn.Sequential(
    # nn.Dropout(p=0.4),
    # nn.Linear(1280, num_classes)
    # )

    # # Step 1: create baseline model
    # model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    
    # # Step 2: define the final classifier EXACTLY as in your testing code
    model.classifier = nn.Sequential(nn.Linear(1280, num_classes))

    
    # Step 3: if we have a weights_path, check if it’s a state dict or full model
    if weights_path is not None and len(weights_path) > 0:
        try:
            checkpoint = torch.load(weights_path, map_location=map_location)
            print(f"[INFO] Loaded checkpoint from {weights_path}")
            
            if isinstance(checkpoint, collections.OrderedDict):
                # It's already a pure state dict
                model.load_state_dict(checkpoint, strict=True)
                print("[INFO] Loaded pure state_dict directly.")
            else:
                # Possibly a full model or something else
                # e.g. checkpoint might have .state_dict()
                if hasattr(checkpoint, 'state_dict'):
                    model.load_state_dict(checkpoint.state_dict(), strict=True)
                    print("[INFO] Extracted state_dict from entire model object.")
                else:
                    print("[WARNING] checkpoint is neither an OrderedDict nor a full model with state_dict. Skipping load.")
        except Exception as e:
            print(f"[ERROR] Could not load {weights_path}: {str(e)}")
    else:
        print("[WARNING] No weights_path provided or file does not exist. Starting from scratch.")
    
    return model


##############################
# TRAIN FUNCTION
##############################
# def train_model(model, train_loader, device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE):
#     """
#     Minimal training loop (no validation), just to demonstrate partial training.
#     """
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(1, epochs+1):
#         model.train()
#         total_loss = 0.0
#         correct = 0
#         total_samples = len(train_loader.dataset)
        
#         for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item() * images.size(0)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
        
#         epoch_loss = total_loss / total_samples
#         epoch_acc = 100.0 * correct / total_samples
#         print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

#     print("[INFO] Finished training (no validation used here).")

#     # Save just the state_dict (recommended)
#     torch.save(model.state_dict(), SAVE_FINETUNED_MODEL)
#     print(f "[INFO] Saved fine-tuned model state_dict to '{SAVE_FINETUNED_MODEL}' ")



#### Train function 2
# def train_model(model, train_loader, val_loader=None, device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE, save_path=SAVE_FINETUNED_MODEL):
def train_model(model, train_loader, val_loader=None, device=DEVICE, epochs=EPOCHS, 
                lr=LEARNING_RATE, save_path=SAVE_FINETUNED_MODEL, 
                early_stopping_patience=3):
    """
    Training loop with metrics tracking and CSV logging.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader, optional
        Validation data loader (if None, no validation is performed)
    device : str
        Device to train on ('cuda' or 'cpu')
    epochs : int
        Number of epochs to train for
    lr : float
        Learning rate
    save_path : str
        Path to save the trained model state_dict
    """
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    # Initialize dictionary to store metrics
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
    
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = len(train_loader.dataset)
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
        
        train_loss = total_loss / total_samples
        train_acc = 100.0 * correct / total_samples
        
        # Validation phase (if val_loader provided)
        val_loss = 0.0
        val_acc = 0.0
        

        if val_loader:
            model.eval()
            val_correct = 0
            val_total = len(val_loader.dataset)
            val_loss_sum = 0.0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    
                    val_loss_batch = criterion(outputs, labels).item() * images.size(0)
                    val_loss_sum += val_loss_batch
                    
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
            
            val_loss = val_loss_sum / val_total
            val_acc = 100.0 * val_correct / val_total
            
            # Update scheduler based on validation accuracy
            scheduler.step(val_acc)
            
            # Save best model
            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     torch.save(model.state_dict(), save_path.replace('.pt', '_best.pt'))
            #     print(f"[INFO] New best model saved with validation accuracy: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), save_path.replace('.pt', '_best.pt'))
                print(f"[INFO] New best model saved with validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"[INFO] Validation accuracy did not improve. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"[INFO] Early stopping triggered after {epoch} epochs")
                    break         
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Record metrics
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss if val_loader else float('nan'))
        metrics['val_acc'].append(val_acc if val_loader else float('nan'))
        metrics['learning_rate'].append(current_lr)
        metrics['time_taken'].append(epoch_time)
        
        # Print metrics
        print(f"Epoch {epoch}/{epochs} - Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loader:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save metrics to CSV after each epoch
        csv_path = os.path.dirname(save_path) + '/training_metrics.csv'
        pd.DataFrame(metrics).to_csv(csv_path, index=False)
        print(f"[INFO] Metrics saved to '{csv_path}'")
    
    print("[INFO] Training complete.")
    
    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Final model state_dict saved to '{save_path}'")
    
    return model, metrics


def create_results_directory(base_dir=None):
    """
    Create a timestamped results directory with proper structure
    """
    if base_dir is None:
        base_dir = r"C:\Users\ilias\Python\Thesis-Project\results"
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"training_experiment_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    models_dir = os.path.join(results_dir, "models")
    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    return {
        "root": results_dir,
        "models": models_dir,
        "metrics": metrics_dir,
        "plots": plots_dir,
        "timestamp": timestamp
    }

def perform_cross_validation(dataset, num_folds=5, batch_size=BATCH_SIZE, 
                            epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE,
                            model_path_prefix=SAVE_FINETUNED_MODEL):
    """
    Perform k-fold cross-validation on the dataset.
    
    Parameters:
    -----------
    dataset : torchvision.datasets.ImageFolder
        The dataset to use for cross-validation
    num_folds : int
        Number of folds for cross-validation
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs per fold
    lr : float
        Learning rate
    device : str
        Device to use for training
    model_path_prefix : str
        Prefix for saving models (will append _fold{i})
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    # Setup k-fold cross-validation
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    all_metrics = []
    
    # Track dataset indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Start cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'='*40}\nFold {fold+1}/{num_folds}\n{'='*40}")
        
        # Create train and validation data loaders
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=train_subsampler,
            num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_subsampler,
            num_workers=4
        )
        
        print(f"[INFO] Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Create a fresh model for each fold
        model = create_efficientnet_b0(
            num_classes=NUM_CLASSES,
            weights_path=PRETRAINED_WEIGHTS,
            map_location=device
        )
        model.to(device)
        
        # Train with validation
        fold_save_path = model_path_prefix.replace('.pt', f'_fold{fold+1}.pt')
        _, fold_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            save_path=fold_save_path
        )
        
        # Store fold metrics with fold number
        fold_metrics['fold'] = [fold+1] * len(fold_metrics['epoch'])
        all_metrics.append(pd.DataFrame(fold_metrics))
    
    # Combine all fold metrics
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Save combined metrics
    cv_metrics_path = os.path.dirname(model_path_prefix) + '/cv_metrics.csv'
    all_metrics_df.to_csv(cv_metrics_path, index=False)
    print(f"[INFO] Cross-validation metrics saved to '{cv_metrics_path}'")
    
    # Calculate and print average metrics across folds
    avg_metrics = all_metrics_df.groupby('fold')[['val_acc', 'val_loss']].max().mean()
    print("\nCross-Validation Results:")
    print(f"Average Best Validation Accuracy: {avg_metrics['val_acc']:.2f}%")
    print(f"Average Best Validation Loss: {avg_metrics['val_loss']:.4f}")
    
    return all_metrics_df

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
    
    with open(os.path.join(results_dir["root"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Store results to compare later
    all_results = {}
    
    for split_name in split_list:
        train_dir = os.path.join(r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\train_splits", split_name)
        
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
    split_list = ["100M_0W", "100M_25W", "100M_50W", "100M_75W", "100M_100W", "100W_0M", "100W_25M", "100W_50M", "100W_75M"]
    run_all_splits(split_list)

if __name__ == "__main__":
    main()


# ##############################
# # MAIN
# ##############################
# def main():
#     print(f"Using device: {DEVICE}")
    
#     # Load entire dataset
#     full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
#     print("[INFO] Found classes in train folder:", full_train_dataset.classes)
#     print("[INFO] Full training set size:", len(full_train_dataset))
    
#     # Subset
#     partial_train_dataset = subset_dataset(full_train_dataset, fraction=TRAIN_FRACTION)
#     print("[INFO] Subset training set size:", len(partial_train_dataset))
    
#     # DataLoader
#     train_loader = torch.utils.data.DataLoader(partial_train_dataset,
#                                               batch_size=BATCH_SIZE,
#                                               shuffle=True,
#                                               num_workers=4)
    
#     # Create the model with the same approach used in testing
#     model = create_efficientnet_b0(num_classes=NUM_CLASSES, weights_path=PRETRAINED_WEIGHTS, map_location="cpu")
#     model.to(DEVICE)
    
#     # Choose whether to use cross-validation or standard validation
#     use_cross_validation = False  # Set to True to use cross-validation

#     if use_cross_validation:
#         # Use k-fold cross-validation on the training set
#         cv_metrics = perform_cross_validation(
#             dataset=partial_train_dataset,
#             num_folds=5,  # 5-fold cross-validation
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             lr=LEARNING_RATE,
#             device=DEVICE
#         )
#         print("[INFO] Cross-validation completed.")
#     else:
#         # Standard training with separate validation set
#         val_dataset = datasets.ImageFolder(root=os.path.join(BASE_SYNTH_DIR, "val_real"), transform=train_transforms)
#         print("[INFO] Found classes in validation folder:", val_dataset.classes)
#         print("[INFO] Validation set size:", len(val_dataset))
        
#         val_loader = torch.utils.data.DataLoader(val_dataset,
#                                                 batch_size=BATCH_SIZE,
#                                                 shuffle=False,
#                                                 num_workers=4)
        
#         model, metrics = train_model(
#             model=model, 
#             train_loader=train_loader, 
#             val_loader=val_loader,
#             device=DEVICE, 
#             epochs=EPOCHS, 
#             lr=LEARNING_RATE,
#             save_path=SAVE_FINETUNED_MODEL
#         )
        
#         print("[INFO] Standard training with validation completed.")

# if __name__ == "__main__":
#     main()

# no cross val no val dataset
##############################
# MAIN
##############################
# def main():
#     print(f"Using device: {DEVICE}")
    
#     # Load entire dataset
#     full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
#     print("[INFO] Found classes in train folder:", full_train_dataset.classes)
#     print("[INFO] Full training set size:", len(full_train_dataset))
    
#     # Subset
#     partial_train_dataset = subset_dataset(full_train_dataset, fraction=TRAIN_FRACTION)
#     print("[INFO] Subset training set size:", len(partial_train_dataset))
    
#     # DataLoader
#     train_loader = torch.utils.data.DataLoader(partial_train_dataset,
#                                                batch_size=BATCH_SIZE,
#                                                shuffle=True,
#                                                num_workers=4)
    

#     # 2) Create the model with the same approach used in testing
#     model = create_efficientnet_b0(num_classes = NUM_CLASSES, weights_path = PRETRAINED_WEIGHTS, map_location="cpu")

#     model.to(DEVICE)
    
#     # Simple training
#     train_model(model, train_loader, device=DEVICE, epochs=EPOCHS, lr=LEARNING_RATE)

#     print("[INFO] Done. No validation or saving metrics in this script.")

# if __name__ == "__main__":
#     main()