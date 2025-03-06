# Create a new file called check_compatibility.py

import timm
import torch
import sys

def check_model_compatibility():
    print("====== Model and Library Compatibility Check ======")
    
    # 1. Version Check
    print(f"\nLibrary Versions:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Timm: {timm.__version__}")
    
    # 2. Model Availability Check
    try:
        print("\nChecking EfficientNet model availability...")
        model_name = 'tf_efficientnet_b0_ns'
        model = timm.create_model(model_name, pretrained=False)
        print(f"✓ Model '{model_name}' is available in current timm version")
        print(f"✓ Model feature dimension: {model.num_features}")
    except Exception as e:
        print(f"✗ Error creating model: {str(e)}")
    
    # 3. Weight Loading Test
    try:
        print("\nAttempting to load weights...")
        weights_path = r'C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_7.pt'
        state_dict = torch.load(weights_path, weights_only=True)
        print("✓ Weights file is readable")
        
        # Check state dict keys
        print("\nState dict structure:")
        print(f"Number of layers: {len(state_dict.keys())}")
        print("Sample layer names:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {key}")
            
    except FileNotFoundError:
        print("✗ Weights file not found")
    except Exception as e:
        print(f"✗ Error loading weights: {str(e)}")
    
    # 4. Model Architecture Check
    print("\nModel Architecture:")
    print(f"Model structure:\n{model}")

if __name__ == "__main__":
    check_model_compatibility()
    