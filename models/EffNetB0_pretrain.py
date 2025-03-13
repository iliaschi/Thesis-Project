# model_utils.py
import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import csv
from datetime import datetime

import os
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import time
import json
import re
from datetime import datetime
import sys
import collections
import torch.nn.functional as F



def load_pretrained_EfficientNet_B0_NS(weights_path, num_classes, device):
    """
    Load EfficientNet-B0 model with pretrained weights
    
    Parameters:
    -----------
    weights_path : str
        Path to pretrained model weights
    num_classes : int
        Number of emotion classes
    device : str
        Device to load model on ('cpu' or 'cuda')
    
    Returns:
    --------
    torch.nn.Module
        Loaded model
    """
    try:
        # Initialize model with Sequential classifier to match old model structure
        model = timm.create_model('tf_efficientnet_b0', pretrained=False)
        model.classifier = nn.Sequential(nn.Linear(1280, num_classes))
        
        # Load state dict
        checkpoint = torch.load(weights_path, map_location=device)
        print(f"----- Successfully loaded complete model from {weights_path} -----")
        
        # Check if the loaded file is already a state dict (OrderedDict)
        if isinstance(checkpoint, collections.OrderedDict):
            # It's already a state dict, so use it directly
            model.load_state_dict(checkpoint)
            print('----- Loaded state dict directly -----')
        else:
            # It's a full model, extract state dict
            model.load_state_dict(checkpoint.state_dict())
            print('----- Extracted state dict from full model -----')
            
        print(f"----- Successfully loaded weights into model -----")
        
        # Set to evaluation mode and move to device
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"----- Error loading model: {str(e)} -----")
        raise


def get_preprocessing_transform(img_size=224):
    """
    Get image preprocessing pipeline for EfficientNet-B0
    
    Parameters:
    -----------
    img_size : int
        Image size for resizing (224 for B0, 260 for B2)
    
    Returns:
    --------
    torchvision.transforms.Compose
        Transform pipeline for preprocessing images
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def plot_confusion_matrix_single(y_true, y_pred, class_names, out_path):
    """
    Plots and saves a confusion matrix for the given predictions,
    focusing on the distribution among all classes for this single folder.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_reliability_diagram(y_true, raw_logits, out_path, n_bins=10):
    # raw_logits shape: (N, num_classes)
    # y_true shape: (N,) with class indices
    # 1) convert to probabilities via softmax
    logits_tensor = torch.from_numpy(raw_logits)
    prob_tensor = F.softmax(logits_tensor, dim=1)
    prob_array = prob_tensor.numpy()  # shape (N, num_classes)

    # 2) for each sample, pick the probability assigned to its ground-truth class
    correct_probs = []
    correct_or_not = []
    for i in range(len(y_true)):
        gt_class = y_true[i]
        p = prob_array[i, gt_class]  # prob assigned to the actual class
        correct_probs.append(p)
        correct_or_not.append(1 if gt_class == np.argmax(prob_array[i]) else 0)

    # 3) calibration_curve
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(correct_or_not, correct_probs, n_bins=n_bins)
    
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1],[0,1], '--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def compute_classification_report(y_true, y_pred, class_names):
    # Force classification_report to handle all classes from 0..7, 
    # ignoring any zero-division issues.
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),   # e.g. [0,1,2,3,4,5,6,7]
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    return report_dict


def annotate_false_positives_in_report(report_dict, y_val, y_pred, class_names):
    """
    For each class, compute the number of false positives (model predicted class i, but y_val != i).
    Insert that integer into the 'report_dict[class_name]["false positives"]' field.
    """
    y_val = np.array(y_val)
    y_pred = np.array(y_pred)
    
    # For each class 'i' in [0..(len(class_names)-1)]
    for i, class_name in enumerate(class_names):
        # false positives => model predicted 'i' but ground truth != i
        fp_count = np.sum((y_pred == i) & (y_val != i))
        
        # Make sure the class is in the report (it usually is if there's support>0)
        # But if scikit-learn skipped it due to no support, we might create it
        if class_name not in report_dict:
            report_dict[class_name] = {}
        
        # Set the field
        report_dict[class_name]["false positives"] = int(fp_count)
    
    return report_dict


def evaluate_single_emotion_folder(model, folder_path, class_to_idx, output_dir, device, batch_size=32, img_size=224):
    """
    Evaluate model on a single emotion folder
    
    Parameters:
    -----------
    model : torch.nn.Module
        Loaded emotion recognition model
    folder_path : str
        Path to folder containing images of a single emotion
    class_to_idx : dict
        Mapping of class names to indices
    output_dir : str
        Directory to save evaluation results
    device : str
        Device to run inference on
    batch_size : int
        Batch size for processing
    img_size : int
        Image size for preprocessing
        
    Returns:
    --------
    tuple
        (result_dir, accuracy, predictions, true_labels, scores)
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.basename(folder_path)
    result_dir = os.path.join(output_dir, f"evaluation_{folder_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    #     # 1) create custom folder name
    # parent_folder = os.path.basename(os.path.dirname(folder_path))  # e.g. 'test'
    # timestamp = datetime.now().strftime("%Y%m%d")
    # folder_name = os.path.basename(folder_path)
    # custom_name = f"{parent_folder}_results_{timestamp}_evaluation"
    # result_dir = os.path.join(output_dir, custom_name)
    # os.makedirs(result_dir, exist_ok=True)
    
    # Extract true emotion from folder name
    # Expects folder name like 'angry_6' or 'happy_2'
    emotion_match = re.match(r'([a-zA-Z]+)_?\d*', folder_name)
    true_emotion = emotion_match.group(1) if emotion_match else folder_name
    
    # Map to standardized emotion name if needed
    emotion_mapping = {
        'angry': 'Angry',
        'anger': 'Angry',
        'contempt': 'Contempt',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happiness',
        'happiness': 'Happiness',
        'neutral': 'Neutral',
        'sad': 'Sadness',
        'sadness': 'Sadness',
        'surprise': 'Surprise',
        'surprised': 'Surprise'
    }
    
    true_emotion = emotion_mapping.get(true_emotion.lower(), true_emotion)
    
    # Check if the emotion is in our class_to_idx
    if true_emotion not in class_to_idx:
        print(f"Warning: Folder emotion '{true_emotion}' not found in class_to_idx. Available emotions: {list(class_to_idx.keys())}")
        closest_match = None
        for cls in class_to_idx.keys():
            if cls.lower() in true_emotion.lower() or true_emotion.lower() in cls.lower():
                closest_match = cls
                break
        
        if closest_match:
            print(f"Using '{closest_match}' as the true emotion.")
            true_emotion = closest_match
        else:
            print("Could not determine the true emotion. Results may be incorrect.")
    
    # Get transform
    transform = get_preprocessing_transform(img_size)
    
    print(f"Evaluating folder: {folder_path}")
    print(f"True emotion: {true_emotion}")
    start_time = time.time()
    
    # Check if the folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: Folder {folder_path} does not exist or is not a directory")
        return None, 0, None, None, None
    
    # Get true class index
    if true_emotion in class_to_idx:
        true_class_idx = class_to_idx[true_emotion]
    else:
        print(f"Warning: True emotion {true_emotion} not found in class_to_idx. Using -1.")
        true_class_idx = -1
    
    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return None, 0, None, None, None
    
    print(f"Found {len(image_files)} images")
    
    # Initialize arrays to store results
    y_val = []
    y_scores_val = []
    imgs = []
    file_paths = []
    
    # Create inverse mapping from index to class
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Process all images
    for img_name in tqdm(image_files, desc=f"Processing {true_emotion} images"):
        filepath = os.path.join(folder_path, img_name)
        
        try:
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform(img)
            
            imgs.append(img_tensor)
            y_val.append(true_class_idx)
            file_paths.append(filepath)
            
            # Process in batches
            if len(imgs) >= batch_size:
                # Stack tensors into a batch
                batch = torch.stack(imgs, dim=0).to(device)
                
                # Get model predictions
                with torch.no_grad():
                    scores = model(batch)
                    scores = scores.data.cpu().numpy()
                
                # Store scores
                if len(y_scores_val) == 0:
                    y_scores_val = scores
                else:
                    y_scores_val = np.concatenate((y_scores_val, scores), axis=0)
                
                # Clear the batch
                imgs = []
                
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    # Process any remaining images
    if len(imgs) > 0:
        batch = torch.stack(imgs, dim=0).to(device)
        
        with torch.no_grad():
            scores = model(batch)
            scores = scores.data.cpu().numpy()
        
        if len(y_scores_val) == 0:
            y_scores_val = scores
        else:
            y_scores_val = np.concatenate((y_scores_val, scores), axis=0)
    
    # Convert to numpy arrays
    y_val = np.array(y_val)
    
    # Calculate predictions
    y_pred = np.argmax(y_scores_val, axis=1)
    
    # Calculate accuracy
    correct = (y_val == y_pred)
    accuracy = 100.0 * correct.sum() / len(y_val)
    

    ##### Extra Metrics
        # After computing y_pred, y_val
    # (1) Classification report
    class_names = list(idx_to_class.values())
    report_dict = compute_classification_report(y_val, y_pred, class_names)
    report_dict = annotate_false_positives_in_report(report_dict, y_val, y_pred, class_names) # false positives
    with open(os.path.join(result_dir, "classification_report.json"), 'w') as f:
        json.dump(report_dict, f, indent=4)
    
    # (2) Single confusion matrix
    cm_path = os.path.join(result_dir, "single_confusion_matrix.png")
    plot_confusion_matrix_single(y_val, y_pred, class_names, cm_path)
    
    # (3) Reliability diagram example
    # Build correct_or_not, correct_probs
    correct_or_not = (y_pred == y_val).astype(int)
    correct_probs  = [y_scores_val[i, y_val[i]] for i in range(len(y_val))]
    rel_path = os.path.join(result_dir, "reliability_diagram.png")
    plot_reliability_diagram(y_val, y_scores_val, rel_path, n_bins=10)

    print(f"Accuracy for {true_emotion}: {accuracy:.2f}%, total samples: {len(y_val)}")
    
    # Create detailed results
    results = []
    for i in range(len(y_val)):
        result = {
            'file': os.path.basename(file_paths[i]),
            'path': file_paths[i],
            'true_emotion': true_emotion,
            'true_idx': int(y_val[i]),
            'predicted_emotion': idx_to_class[y_pred[i]],
            'predicted_idx': int(y_pred[i]),
            'confidence': float(y_scores_val[i, y_pred[i]]),
            'correct': bool(correct[i])
        }
        
        # Add probabilities for all classes
        for cls_idx, cls_name in idx_to_class.items():
            prob_key = f'prob_{cls_name}'
            if cls_idx < y_scores_val.shape[1]:
                result[prob_key] = float(y_scores_val[i, cls_idx])
        
        results.append(result)

    # False Positives Addition
    # --- Step 4: figure out false positives
    false_positives = []
    for row in results:
        if not row['correct']:
            false_positives.append({
                'file': row['file'],
                'predicted_emotion': row['predicted_emotion'],
                'true_emotion': row['true_emotion'],
                # etc.
            })

    # # --- Step 5: classification report
    # class_names = list(idx_to_class.values())
    # report_dict = compute_classification_report(y_val, y_pred, class_names)

    # # --- Step 6: attach false positives
    # report_dict['false_positives'] = false_positives

    # # --- Step 7: write out
    # cr_json_path = os.path.join(result_dir, 'classification_report.json')
    # with open(cr_json_path, 'w') as f:
    #     json.dump(report_dict, f, indent=4)
    # # if false_positives:
    # #     fp_df = pd.DataFrame(false_positives)
    # #     fp_csv = os.path.join(result_dir, 'false_positives.csv')
    # #     fp_df.to_csv(fp_csv, index=False)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(result_dir, f"results_{true_emotion}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to: {csv_path}")
    
    # Count predictions for each class
    prediction_counts = {}
    for idx, emotion in idx_to_class.items():
        count = np.sum(y_pred == idx)
        percentage = 100.0 * count / len(y_pred)
        prediction_counts[emotion] = {'count': int(count), 'percentage': float(percentage)}
        print(f"  Predicted as {emotion}: {count} ({percentage:.1f}%)")
    
    # Save raw scores and labels for further analysis
    np.save(os.path.join(result_dir, "scores.npy"), y_scores_val)
    np.save(os.path.join(result_dir, "true_labels.npy"), y_val)
    np.save(os.path.join(result_dir, "pred_labels.npy"), y_pred)
    
    # Calculate confidence statistics
    mean_confidence = float(np.mean([result['confidence'] for result in results]))
    mean_correct_conf = float(np.mean([result['confidence'] for result in results if result['correct']]) 
                             if any(result['correct'] for result in results) else 0)
    mean_incorrect_conf = float(np.mean([result['confidence'] for result in results if not result['correct']])
                              if any(not result['correct'] for result in results) else 0)
    
    # Save metadata
    processing_time = time.time() - start_time
    metadata = {
        'timestamp': timestamp,
        'folder': folder_path,
        'true_emotion': true_emotion,
        'num_classes': len(class_to_idx),
        'total_images': len(y_val),
        'accuracy': float(accuracy),
        'processing_time': processing_time,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'prediction_counts': prediction_counts,
        'mean_confidence': mean_confidence,
        'mean_correct_confidence': mean_correct_conf,
        'mean_incorrect_confidence': mean_incorrect_conf
    }
    
    with open(os.path.join(result_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return result_dir, accuracy, y_pred, y_val, y_scores_val

def main():
    # Configuration
    # weights_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_8_best_afew.pt"
    #new_state_dict_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_8_best_afew_state_dict.pth"
    weights_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_8_best_afew_state_dict.pth"


    # For 8 classes (AffectNet)
    class_to_idx_8 = {
        'Angry': 0, 
        'Contempt': 1, 
        'Disgust': 2, 
        'Fear': 3, 
        'Happiness': 4, 
        'Neutral': 5, 
        'Sadness': 6, 
        'Surprise': 7
    }   # Map to standardized emotion name if needed


    # Choose which emotion set to use based on model
    class_to_idx = class_to_idx_8  # Change to class_to_idx_8 for 8-class models
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Determine model type from filename and set appropriate img_size
    img_size = 224  # 224 for B0, 260 for B2
    if 'b2' in weights_path.lower():
        img_size = 260
    
    # Load model
    model = load_pretrained_EfficientNet_B0_NS(
        weights_path=weights_path,
        num_classes=len(class_to_idx),
        device=device
    )

    base_output_dir = r"C:\Users\ilias\Python\Thesis-Project\results\Results_2.0"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_folder_name = f"RAFDB_results_{timestamp}"

    output_dir = os.path.join(base_output_dir, custom_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    #C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test\happy_4
    folder_path = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test"
    test_root = r"C:\Users\ilias\Python\Thesis-Project\data\real\RAF_DB\DATASET\test"

    
    # Define emotion labels explicitly - choose the appropriate one

    for folder_name in os.listdir(test_root):
        folder_path = os.path.join(test_root, folder_name)
        if os.path.isdir(folder_path):
            result_dir, accuracy, _, _, _ = evaluate_single_emotion_folder(
                model=model,
                folder_path=folder_path,
                class_to_idx=class_to_idx,
                output_dir=output_dir,
                device=device,
                batch_size=32,
                img_size=img_size
            )
            print(f"Done with {folder_name}, results in: {result_dir}")


    
    print(f"Results saved to: {result_dir}")

if __name__ == "__main__":
    main()



