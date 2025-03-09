# facial_emotion_evaluation.py
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from datetime import datetime
import csv
import platform

# -----------------------------------------------------------------------------
# 1. Model Definition: EfficientNet-B0
# -----------------------------------------------------------------------------

def create_efficientnet_model(weights_path=None, num_classes=7, device='cpu'):
    """
    Create EfficientNet-B0 model for Facial Emotion Recognition
    
    Parameters:
    -----------
    weights_path : str
        Path to pretrained weights file
    num_classes : int
        Number of emotion classes (default: 7)
    device : str
        Device to load the model on ('cpu' or 'cuda')
    
    Returns:
    --------
    torch.nn.Module
        Loaded EfficientNet model
    
    Notes:
    ------
    EfficientNet-B0 has ~5.3M parameters, making it much smaller than VGG or ResNet
    while maintaining competitive accuracy. The architecture uses mobile inverted
    bottleneck blocks with squeeze-and-excitation optimization.
    
    Key specifications for B0:
    - Resolution: 224x224
    - Depth multiplier: 1.0
    - Width multiplier: 1.0
    - MBConv blocks: 16
    - Feature dimensions: 1280
    """
    # Initialize model
    model = timm.create_model('tf_efficientnet_b0', pretrained=True)
    
    # Replace classifier with custom emotion classifier
    model.classifier = nn.Linear(1280, num_classes)
    
    # Load custom weights if provided
    if weights_path and os.path.exists(weights_path):
        try:
            # Load weights with compatibility handling
            weights = torch.load(weights_path, map_location=device)
            
            # Handle different weight formats
            if isinstance(weights, dict) and 'state_dict' in weights:
                model.load_state_dict(weights['state_dict'])
            elif isinstance(weights, dict):
                model.load_state_dict(weights)
            else:
                # If it's the entire model
                model = weights
                
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
    
    # Set to evaluation mode and move to device
    model = model.to(device)
    model.eval()
    
    return model

# -----------------------------------------------------------------------------
# 2. Image Preprocessing for Model Input
# -----------------------------------------------------------------------------

def get_preprocessing_transform():
    """
    Get image preprocessing pipeline for EfficientNet-B0
    
    Returns:
    --------
    torchvision.transforms.Compose
        Transform pipeline for preprocessing images
    
    Notes:
    ------
    EfficientNet requires specific preprocessing:
    - Resize to 224x224 pixels
    - Convert to tensor (0-1 range)
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet-B0 input size
        transforms.ToTensor(),          # Convert to tensor (0-1 range)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225]   # ImageNet std
        )
    ])

def preprocess_image(image_path, transform=None):
    """
    Preprocess a single image for model input
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    transform : torchvision.transforms.Compose
        Transform pipeline (if None, will use default)
    
    Returns:
    --------
    torch.Tensor
        Preprocessed image tensor ready for model input
    """
    if transform is None:
        transform = get_preprocessing_transform()
        
    # Load and convert image to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

# -----------------------------------------------------------------------------
# 3. Model Evaluation Functions
# -----------------------------------------------------------------------------

def predict_emotion(model, image_tensor, device='cpu'):
    """
    Predict emotion for a preprocessed image tensor
    
    Parameters:
    -----------
    model : torch.nn.Module
        Loaded emotion recognition model
    image_tensor : torch.Tensor
        Preprocessed image tensor
    device : str
        Device to run inference on
    
    Returns:
    --------
    dict
        Prediction details including emotion label, confidence, and all probabilities
    """
    # Emotion mapping
    emotions = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad', 
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # Move tensor to device
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    # Return prediction details
    return {
        'emotion': emotions[predicted_class],
        'class_idx': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy()
    }

def evaluate_dataset(model, data_dir, true_emotion=None, device='cpu', limit=None):
    """
    Evaluate model on a dataset of images
    
    Parameters:
    -----------
    model : torch.nn.Module
        Loaded emotion recognition model
    data_dir : str
        Directory containing images to evaluate
    true_emotion : str or None
        If provided, all images are assumed to have this emotion label
    device : str
        Device to run inference on
    limit : int or None
        Maximum number of images to evaluate
    
    Returns:
    --------
    list
        List of dictionaries with prediction results
    """
    results = []
    transform = get_preprocessing_transform()
    
    # Mapping between folder names and emotion labels
    emotion_mapping = {
        'angry': 'Angry', 
        'disgust': 'Disgust', 
        'fear': 'Fear', 
        'happy': 'Happy',
        'sad': 'Sad', 
        'surprise': 'Surprise', 'surprised': 'Surprise',
        'neutral': 'Neutral'
    }
    
    # Check if we're evaluating a single emotion directory
    if true_emotion:
        if os.path.isdir(data_dir):
            # List image files
            image_files = [f for f in os.listdir(data_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.jfif'))]
            
            # Apply limit if needed
            if limit and len(image_files) > limit:
                image_files = image_files[:limit]
            
            print(f"Evaluating {len(image_files)} images with true emotion: {true_emotion}")
            
            # Process each image
            for filename in image_files:
                image_path = os.path.join(data_dir, filename)
                try:
                    # Preprocess and predict
                    image_tensor = preprocess_image(image_path, transform)
                    prediction = predict_emotion(model, image_tensor, device)
                    
                    # Store result
                    results.append({
                        'file': filename,
                        'path': image_path,
                        'true_emotion': true_emotion,
                        'predicted_emotion': prediction['emotion'],
                        'confidence': prediction['confidence'],
                        'class_idx': prediction['class_idx'],
                        'probabilities': prediction['probabilities'],
                        'correct': true_emotion == prediction['emotion']
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    
    # If we're evaluating a directory with emotion subdirectories
    else:
        for emotion_dir in os.listdir(data_dir):
            # Check if it's a directory
            dir_path = os.path.join(data_dir, emotion_dir)
            if not os.path.isdir(dir_path):
                continue
            
            # Map directory name to emotion
            if emotion_dir.lower() in emotion_mapping:
                true_emotion = emotion_mapping[emotion_dir.lower()]
            else:
                # Try to extract emotion from directory name
                for key in emotion_mapping:
                    if key in emotion_dir.lower():
                        true_emotion = emotion_mapping[key]
                        break
                else:
                    print(f"Could not determine emotion for directory: {emotion_dir}")
                    continue
            
            # List image files
            image_files = [f for f in os.listdir(dir_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.jfif'))]
            
            print(f"Processing {len(image_files)} images in {emotion_dir} (label: {true_emotion})")
            
            # Process each image
            for filename in image_files:
                image_path = os.path.join(dir_path, filename)
                try:
                    # Preprocess and predict
                    image_tensor = preprocess_image(image_path, transform)
                    prediction = predict_emotion(model, image_tensor, device)
                    
                    # Store result
                    results.append({
                        'file': filename,
                        'path': image_path,
                        'true_emotion': true_emotion,
                        'predicted_emotion': prediction['emotion'],
                        'confidence': prediction['confidence'],
                        'class_idx': prediction['class_idx'],
                        'probabilities': prediction['probabilities'],
                        'correct': true_emotion == prediction['emotion']
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    
    return results

def evaluate_from_csv(csv_file, model, image_root_dir=None, device='cpu'):
    """
    Evaluate model based on image paths and true labels from a CSV file
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with image paths and true labels
    model : torch.nn.Module
        Loaded emotion recognition model
    image_root_dir : str
        Root directory to prefix to relative image paths in CSV
    device : str
        Device to run inference on
    
    Returns:
    --------
    list
        List of dictionaries with prediction results
    """
    results = []
    transform = get_preprocessing_transform()
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Check required columns
    required_cols = ['file', 'true_emotion']
    if not all(col in df.columns for col in required_cols):
        print(f"CSV must contain columns: {required_cols}")
        return results
    
    print(f"Evaluating {len(df)} images from CSV: {csv_file}")
    
    # Process each row
    for i, row in df.iterrows():
        try:
            # Get image path
            if image_root_dir:
                image_path = os.path.join(image_root_dir, row['file'])
            else:
                image_path = row['file']
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Preprocess and predict
            image_tensor = preprocess_image(image_path, transform)
            prediction = predict_emotion(model, image_tensor, device)
            
            # Store result
            results.append({
                'file': row['file'],
                'path': image_path,
                'true_emotion': row['true_emotion'],
                'predicted_emotion': prediction['emotion'],
                'confidence': prediction['confidence'],
                'class_idx': prediction['class_idx'],
                'probabilities': prediction['probabilities'],
                'correct': row['true_emotion'] == prediction['emotion']
            })
        except Exception as e:
            print(f"Error processing {row['file']}: {str(e)}")
    
    return results

# -----------------------------------------------------------------------------
# 4. Results Analysis and Metrics
# -----------------------------------------------------------------------------

def calculate_metrics(results):
    """
    Calculate performance metrics from prediction results
    
    Parameters:
    -----------
    results : list
        List of prediction result dictionaries
    
    Returns:
    --------
    dict
        Dictionary of calculated metrics
    """
    if not results:
        return {}
    
    # Extract true and predicted labels
    y_true = [r['true_emotion'] for r in results]
    y_pred = [r['predicted_emotion'] for r in results]
    
    # Get unique emotion labels
    emotions = sorted(list(set(y_true + y_pred)))
    
    # Calculate overall accuracy
    accuracy = sum(r['correct'] for r in results) / len(results)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=emotions)
    
    # Calculate per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Group results by true emotion
    by_emotion = {}
    for emotion in emotions:
        emotion_results = [r for r in results if r['true_emotion'] == emotion]
        if emotion_results:
            emotion_accuracy = sum(r['correct'] for r in emotion_results) / len(emotion_results)
        else:
            emotion_accuracy = 0
        
        # Get confidence statistics
        confidences = [r['confidence'] for r in emotion_results]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            correct_confidences = [r['confidence'] for r in emotion_results if r['correct']]
            incorrect_confidences = [r['confidence'] for r in emotion_results if not r['correct']]
            
            if correct_confidences:
                avg_correct_conf = sum(correct_confidences) / len(correct_confidences)
            else:
                avg_correct_conf = 0
                
            if incorrect_confidences:
                avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences)
            else:
                avg_incorrect_conf = 0
        else:
            avg_confidence = avg_correct_conf = avg_incorrect_conf = 0
        
        by_emotion[emotion] = {
            'count': len(emotion_results),
            'accuracy': emotion_accuracy,
            'avg_confidence': avg_confidence,
            'avg_correct_conf': avg_correct_conf,
            'avg_incorrect_conf': avg_incorrect_conf
        }
    
    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'emotions': emotions,
        'by_emotion': by_emotion,
        'report': report,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def generate_confusion_matrix_plot(metrics, output_path=None):
    """
    Create and save a confusion matrix visualization
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary with confusion matrix and emotion labels
    output_path : str
        Path to save the confusion matrix image
    
    Returns:
    --------
    matplotlib.figure.Figure
        Confusion matrix figure
    """
    if 'confusion_matrix' not in metrics or 'emotions' not in metrics:
        return None
    
    cm = metrics['confusion_matrix']
    emotions = metrics['emotions']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
    
    # Plot confusion matrix
    sns.heatmap(
        cm_norm,           # Normalized for colors
        annot=cm,          # Show raw counts
        fmt='d',           # Integer format for counts
        cmap='Blues',      # Color map
        xticklabels=emotions,
        yticklabels=emotions
    )
    
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def generate_confidence_distribution_plot(results, output_path=None):
    """
    Create and save a plot of confidence distributions for correct vs incorrect predictions
    
    Parameters:
    -----------
    results : list
        List of prediction result dictionaries
    output_path : str
        Path to save the plot image
    
    Returns:
    --------
    matplotlib.figure.Figure
        Confidence distribution figure
    """
    # Separate confidences for correct and incorrect predictions
    correct_conf = [r['confidence'] for r in results if r['correct']]
    incorrect_conf = [r['confidence'] for r in results if not r['correct']]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(
        correct_conf, 
        bins=20, 
        alpha=0.7, 
        label=f'Correct Predictions ({len(correct_conf)})',
        color='green'
    )
    plt.hist(
        incorrect_conf, 
        bins=20, 
        alpha=0.7, 
        label=f'Incorrect Predictions ({len(incorrect_conf)})',
        color='red'
    )
    
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def generate_per_class_metrics_plot(metrics, output_path=None):
    """
    Create and save a plot of performance metrics for each emotion class
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary with per-emotion statistics
    output_path : str
        Path to save the plot image
    
    Returns:
    --------
    matplotlib.figure.Figure
        Per-class metrics figure
    """
    if 'report' not in metrics or 'emotions' not in metrics:
        return None
    
    emotions = metrics['emotions']
    
    # Extract class metrics
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for emotion in emotions:
        if emotion in metrics['report']:
            precisions.append(metrics['report'][emotion]['precision'])
            recalls.append(metrics['report'][emotion]['recall'])
            f1_scores.append(metrics['report'][emotion]['f1-score'])
            supports.append(metrics['report'][emotion]['support'])
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            supports.append(0)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot precision, recall, F1 on first subplot
    x = np.arange(len(emotions))
    width = 0.25
    
    ax1.bar(x - width, precisions, width, label='Precision', color='#4285F4')
    ax1.bar(x, recalls, width, label='Recall', color='#EA4335')
    ax1.bar(x + width, f1_scores, width, label='F1-Score', color='#FBBC05')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(emotions)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, and F1-Score by Emotion')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot support (class distribution) on second subplot
    ax2.bar(x, supports, color='#34A853')
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotions)
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Class Distribution')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

# -----------------------------------------------------------------------------
# 5. Results Saving and Reporting
# -----------------------------------------------------------------------------

def save_evaluation_results(results, metrics, base_output_path):
    """
    Save evaluation results and metrics to files
    
    Parameters:
    -----------
    results : list
        List of prediction result dictionaries
    metrics : dict
        Metrics dictionary
    base_output_path : str
        Base path for output files (without extension)
    
    Returns:
    --------
    dict
        Dictionary with paths to saved files
    """
    os.makedirs(os.path.dirname(base_output_path), exist_ok=True)
    
    # Save detailed results CSV
    csv_path = f"{base_output_path}.csv"
    save_results_csv(results, metrics, csv_path)
    
    # Save metrics report
    report_path = f"{base_output_path}_report.txt"
    save_metrics_report(results, metrics, report_path)
    
    # Save confusion matrix
    cm_path = f"{base_output_path}_confusion_matrix.png"
    generate_confusion_matrix_plot(metrics, cm_path)
    
    # Save confidence distribution
    conf_path = f"{base_output_path}_confidence_distribution.png"
    generate_confidence_distribution_plot(results, conf_path)
    
    # Save per-class metrics
    metrics_path = f"{base_output_path}_class_metrics.png"
    generate_per_class_metrics_plot(metrics, metrics_path)
    
    return {
        'csv': csv_path,
        'report': report_path,
        'confusion_matrix': cm_path,
        'confidence_distribution': conf_path,
        'class_metrics': metrics_path
    }

def save_results_csv(results, metrics, output_path):
    """
    Save detailed results to CSV file
    
    Parameters:
    -----------
    results : list
        List of prediction result dictionaries
    metrics : dict
        Metrics dictionary
    output_path : str
        Path to save the CSV file
    """
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Select and reorder columns
    columns = [
        'file', 'true_emotion', 'predicted_emotion', 
        'confidence', 'correct', 'class_idx'
    ]
    df = df[columns]
    
    # Add header with summary stats
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Evaluation Results Summary'])
        writer.writerow([f'Accuracy: {metrics["accuracy"]:.4f}'])
        writer.writerow([f'F1 Score (Macro): {metrics["f1_macro"]:.4f}'])
        writer.writerow([f'F1 Score (Weighted): {metrics["f1_weighted"]:.4f}'])
        writer.writerow([f'Total Images: {len(results)}'])
        writer.writerow([]) # Empty row
    
    # Append dataframe
    df.to_csv(output_path, mode='a', index=False)
    
    print(f"Results saved to CSV: {output_path}")

def save_metrics_report(results, metrics, output_path):
    """
    Save comprehensive metrics report to text file
    
    Parameters:
    -----------
    results : list
        List of prediction result dictionaries
    metrics : dict
        Metrics dictionary
    output_path : str
        Path to save the report file
    """
    # Get system info
    system_info = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': 'GPU' if torch.cuda.is_available() else 'CPU'
    }
    
    # Open file for writing
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FACIAL EMOTION RECOGNITION EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # System information
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Date and Time: {system_info['date']}\n")
        f.write(f"Platform: {system_info['platform']}\n")
        f.write(f"Python Version: {system_info['python']}\n")
        f.write(f"Device: {system_info['device']}\n")
        f.write(f"Total Images Processed: {len(results)}\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score (Macro): {metrics['f1_macro']:.4f}\n")
        f.write(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Emotion':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Accuracy':<10}\n")
        
        for emotion in metrics['emotions']:
            if emotion in metrics['report']:
                cls = metrics['report'][emotion]
                per_class = metrics['by_emotion'][emotion]
                f.write(f"{emotion:<10} {cls['precision']:.4f}{'':<6} {cls['recall']:.4f}{'':<6} {cls['f1-score']:.4f}{'':<6} {cls['support']:<10} {per_class['accuracy']:.4f}\n")
        f.write("\n")
        
        # Confusion matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")
        cm = metrics['confusion_matrix']
        emotions = metrics['emotions']
        
        # Write header
        f.write(f"{'':>10}")
        for emotion in emotions:
            f.write(f"{emotion[:7]:>8} ")
        f.write("\n")
        
        # Write data rows
        for i, emotion in enumerate(emotions):
            f.write(f"{emotion:<10}")
            for j in range(len(emotions)):
                f.write(f"{cm[i, j]:>8} ")
            f.write("\n")
        f.write("\n")
        
        # Confidence analysis
        f.write("CONFIDENCE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Emotion':<10} {'Avg Conf':<10} {'Correct Conf':<15} {'Incorrect Conf':<15}\n")
        
        for emotion in metrics['emotions']:
            if emotion in metrics['by_emotion']:
                conf = metrics['by_emotion'][emotion]
                f.write(f"{emotion:<10} {conf['avg_confidence']:.4f}{'':<6} {conf['avg_correct_conf']:.4f}{'':<11} {conf['avg_incorrect_conf']:.4f}\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Detailed report saved to: {output_path}")

# -----------------------------------------------------------------------------
# 6. Main Execution Function
# -----------------------------------------------------------------------------

def main():
    """Main execution function for facial emotion evaluation"""
    # Configuration
    weights_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_7.pt"
    data_dir = r"C:\Users\ilias\Python\Thesis-Project\data\real\FER2013\test"
    true_emotion = None  # Set to a specific emotion if evaluating a single-emotion folder
    
    # Results directory
    results_dir = r"C:\Users\ilias\Python\Thesis-Project\results"
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base output path
    base_folder = os.path.basename(data_dir)
    base_output_path = os.path.join(results_dir, f"fer_evaluation_{base_folder}_{timestamp}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = create_efficientnet_model(weights_path, device=device)
    
    # Evaluate dataset
    start_time = datetime.now()
    results = evaluate_dataset(model, data_dir, true_emotion, device)
    evaluation_time = (datetime.now() - start_time).total_seconds()
    
    if results:
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Save results and generate reports
        output_files = save_evaluation_results(results, metrics, base_output_path)
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total images: {len(results)}")
        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 score (weighted): {metrics['f1_weighted']:.4f}")
        print(f"Processing time: {evaluation_time:.2f} seconds")
        print(f"Results saved to: {os.path.dirname(base_output_path)}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()