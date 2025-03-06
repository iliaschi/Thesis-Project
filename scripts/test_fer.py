import torch
from PIL import Image
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from models.model import load_emotion_model
#from utils.data_processing import get_transform, EMOTIONS

# import torch
# import torch.nn as nn
# import timm
# from torchvision import transforms
# from PIL import Image
# import os

# # path to weights
# path_eff_b0_7 = r'C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_7.pt'
# path_eff_b2_7 = r'C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b2_7.pt'


# # Constants and Mappings
# EMOTIONS = {
#     0: 'Angry',
#     1: 'Disgust',
#     2: 'Fear',
#     3: 'Happy',
#     4: 'Sad',
#     5: 'Surprise',
#     6: 'Neutral'
# }

# def load_emotion_model(weights_path, device='cuda'):
#     """Loads EfficientNet-B0 model with emotion recognition weights"""
#     model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
#     model.classifier = nn.Linear(1280, 7)  # 7 emotions for FER2013
    
#     try:
#         model.load_state_dict(torch.load(weights_path))
#         print(f"Loaded weights from {weights_path}")
#     except Exception as e:
#         print(f"Error loading weights: {e}")
#         return None
    
#     model = model.to(device)
#     model.eval()
#     return model

# def get_transform():
#     """Returns transform for preprocessing images"""
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

# def predict_emotion(image_path, model, transform):
#     """Predict emotion for a single image"""
#     try:
#         image = Image.open(image_path).convert('RGB')
#         image_tensor = transform(image).unsqueeze(0)
        
#         with torch.no_grad():
#             output = model(image_tensor.cuda())
#             prob = torch.softmax(output, dim=1)
#             pred = torch.argmax(prob, dim=1).item()
        
#         return EMOTIONS[pred], prob[0][pred].item()
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None, None

# def main():
#     # Paths - Update these!
#     weights_path = path_eff_b0_7 # "path/to/enet_b0_7.pt"
#     test_image_path = "path/to/test/image.jpg"
    
#     # Device setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Model setup
#     model = load_emotion_model(weights_path, device)
#     if model is None:
#         return
    
#     # Transform setup
#     transform = get_transform()
    
#     # Test prediction
#     emotion, confidence = predict_emotion(test_image_path, model, transform)
#     if emotion:
#         print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")

# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def create_emotion_model(device='cpu'):
    """Create model without loading weights initially"""
    model = timm.create_model('tf_efficientnet_b0', pretrained=True)  # Changed from tf_efficientnet_b0_ns
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier = nn.Linear(1280, 7)
    model = model.to(device)
    model.eval()
    return model

def predict_emotion(image_path, model, device='cpu'):
    """Predict emotion for a single image"""
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
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
    
    # Process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        
    return emotions[predicted_class], confidence

def main():
    # Paths
    test_image = r"C:\Users\ilias\Python\Thesis-Project\data\real\FER2013\test\angry\PrivateTest_10131363.jpg"

    # Create model
    device = 'cpu'
    model = create_emotion_model(device)
    
    # Test prediction
    emotion, confidence = predict_emotion(test_image, model, device)
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()