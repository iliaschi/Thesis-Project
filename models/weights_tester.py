"""
use_new_model.py

This script runs in the NEW environment. It loads just the saved state_dict and
rebuilds the model architecture using the current timm version, avoiding old references.
"""

import torch
import torch.nn as nn
import timm

def load_effnet_b0_state_dict(state_dict_path, num_classes, device='cpu'):
    """
    Load a state_dict for EfficientNet-B0 in the new environment.
    
    Parameters:
    -----------
    state_dict_path : str
        Path to the .pth file containing only the model's state_dict.
    num_classes : int
        Number of classes for the new classification head.
    device : str
        'cpu' or 'cuda' device to load the model onto.
    
    Returns:
    --------
    torch.nn.Module
        An EfficientNet-B0 model with the loaded state_dict.
    """
    # 1. Create a baseline B0 architecture with no pretrained weights
    model = timm.create_model('tf_efficientnet_b0', pretrained=False)
    
    # 2. Adjust the classifier layer to match the desired number of classes
    #    Typically, tf_efficientnet_b0 has 1280 features going into the classifier
    # model.classifier = nn.Linear(in_features=1280, out_features=num_classes)
    model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=num_classes)) #1792 #1280 #1536
    ### VERY important as this is the classifier used in the trained model, usefull to have Sequential
    # load it correctly and also to be able to implement dropout and other layers in the future !!! 
    
    # 3. Load the stored state_dict
    state_dict = torch.load(state_dict_path, map_location=device)
    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict, strict=True)
    
    # 4. Move to device and set evaluation mode
    model.to(device)
    model.eval()
    
    print(f"Loaded state_dict from {state_dict_path} into a new EfficientNet-B0 model.")
    return model

def main():
    # Example usage:
    # This file is presumably in a new environment with a possibly updated timm.
    
    # Path to the re-saved state dict from the old environment
    new_state_dict_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_8_best_afew_state_dict.pth"


    # Number of classes for your classification task
    num_classes = 8
    
    # Device config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the model
    model = load_effnet_b0_state_dict(new_state_dict_path, num_classes, device)
    
    # Now you can run inference or fine-tune
    # e.g., do a quick test with a random input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("Test output shape:", output.shape)  # should be [1, 8]

if __name__ == "__main__":
    main()
