import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import warnings
import os
import random
warnings.filterwarnings('ignore')

import os

results_dir = r'C:\Users\ilias\Python\Thesis-Project\results'
os.makedirs(results_dir, exist_ok=True)

def create_emotion_model(device='cpu'):
    model = timm.create_model('tf_efficientnet_b0', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(1280, 7)
    model = model.to(device)
    model.eval()
    return model

def predict_emotion(image_path, model, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    emotions = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
            
        return emotions[predicted_class], confidence
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

def process_folder_limited(folder_path, model, limit=1000, device='cpu'):
    results = []
    all_images = []
    
    # Collect all valid images first
    for emotion_folder in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion_folder)
        if os.path.isdir(emotion_path):
            for image_file in os.listdir(emotion_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    all_images.append((emotion_folder, os.path.join(emotion_path, image_file), image_file))
    
    # Randomly sample 'limit' number of images
    selected_images = random.sample(all_images, min(limit, len(all_images)))
    print(f"Processing {len(selected_images)} images...")
    
    # Process selected images
    for emotion_folder, image_path, image_file in selected_images:
        predicted_emotion, confidence = predict_emotion(image_path, model, device)
        
        if predicted_emotion is not None:
            results.append({
                'file': image_file,
                'true_emotion': emotion_folder,
                'predicted_emotion': predicted_emotion,
                'confidence': confidence
            })
            print(f"{image_file}: True: {emotion_folder}, Predicted: {predicted_emotion} ({confidence:.2f})")
    
    return results

# def save_results(results, output_file):
#     with open(output_file, 'w') as f:
#         f.write("File,True Emotion,Predicted Emotion,Confidence\n")
#         for r in results:
#             f.write(f"{r['file']},{r['true_emotion']},{r['predicted_emotion']},{r['confidence']:.2f}\n")

def save_results(results, output_file):
    # Calculate accuracy
    total = len(results)
    correct = sum(1 for r in results if r['true_emotion'].lower() == r['predicted_emotion'].lower())
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    with open(output_file, 'w') as f:
        # Write overall accuracy header
        f.write(f"Overall Accuracy,{accuracy:.2f}%,{correct}/{total}\n\n")
        
        # Write detailed results header
        f.write("File,True Emotion,Predicted Emotion,Confidence\n")
        
        # Write individual results
        for r in results:
            f.write(f"{r['file']},{r['true_emotion']},{r['predicted_emotion']},{r['confidence']:.2f}\n")
    
    # Optional: Print accuracy to console as well
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy  # Return accuracy in case you want to use it later

def main():
    # Paths
    test_folder = r"C:\Users\ilias\Python\Thesis-Project\data\real\FER2013\test"
    results_file = r"C:\Users\ilias\Python\Thesis-Project\results\fer2013_results_100.csv"
    
    # Create model
    device = 'cpu'
    model = create_emotion_model(device)
    
    # Process limited number of images
    print("Starting emotion prediction on test folder...")
    results = process_folder_limited(test_folder, model, limit=1000, device=device)
    
    # Save results
    save_results(results, results_file)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    total = len(results)
    correct = sum(1 for r in results if r['true_emotion'].lower() == r['predicted_emotion'].lower())
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()