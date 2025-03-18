"""
export_old_model.py

This script runs in the OLD environment (where the model was originally trained).
It loads the entire model object (which references older timm versions, etc.)
and then saves only the state_dict() so that a newer environment can load it safely.
"""

import torch

# def export_model_to_state_dict(old_model_path, state_dict_path):
#     """
#     Load the entire old model object and re-save just its state_dict.
    
#     Parameters:
#     -----------
#     old_model_path : str
#         Path to the .pt file containing the fully saved model.
#     state_dict_path : str
#         Path to the output file (will contain only the state dict).
#     """
#     # Load the entire model (includes architecture references)
#     model = torch.load(old_model_path, map_location='cpu')
#     print(f"Successfully loaded full model from {old_model_path}.")
    
#     # Extract the model's state_dict
#     model_sd = model.state_dict()
    
#     # Save just the state_dict
#     torch.save(model_sd, state_dict_path)
#     print(f"Successfully exported state_dict to {state_dict_path}.")


def export_model_to_state_dict(old_model_path, state_dict_path):
    obj = torch.load(old_model_path, map_location='cpu')
    print(f"Loaded object of type: {type(obj)} from {old_model_path}")

    if hasattr(obj, 'state_dict'):
        # It's likely a full model
        state_dict = obj.state_dict()
        torch.save(state_dict, state_dict_path)
        print("[INFO] Extracted state_dict from full model.")
    else:
        # Probably it's already a dict or something else
        # If it's an OrderedDict or a plain dict, just save it again
        torch.save(obj, state_dict_path)
        print("[INFO] The file was already a state dict. Re-saved directly.")


if __name__ == "__main__":
    # Example usage:
    old_model_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\state_vggface2_enet0_new.pt"
    state_dict_path = r"C:\Users\ilias\Python\Thesis-Project\models\weights\enet_b0_base_vggface2_state_dict.pth"
    
    export_model_to_state_dict(old_model_path, state_dict_path)