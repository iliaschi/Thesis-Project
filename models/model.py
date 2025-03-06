import torch
import torch.nn as nn
import timm

def load_emotion_model(weights_path, device='cuda'):
    """
    Loads EfficientNet-B0 model with emotion recognition weights
    Args:
        weights_path: Path to the .pt weights file
        device: 'cuda' or 'cpu'
    """
    model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
    model.classifier = nn.Linear(1280, 7)  # 7 emotions for FER2013
    
    try:
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    return model