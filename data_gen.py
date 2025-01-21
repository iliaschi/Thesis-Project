access_token = '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'

# Here we will generate the data for the thesis

### 1
# We will have a folder checker that will chech if the folder of the data exists
# If it does not exist, it will create the folder

### 2
# Then we will as for an input from the user of Yes and No

### 3
# Then the number of data to be generated


# import os
# import logging
# import argparse

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# def validate_folder_name(folder_name):
#     """
#     Validate the folder name for invalid characters or empty input.
#     """
#     if not folder_name or any(char in folder_name for char in r'<>:"/\|?*'):
#         print("Invalid folder name. Please try again.")
#         return False
#     return True

# def list_folder_contents(folder_name):
#     """
#     List the contents of a folder if it exists.
#     """
#     contents = os.listdir(folder_name)
#     if contents:
#         print(f"Contents of '{folder_name}':")
#         for item in contents:
#             print(f"  - {item}")
#     else:
#         print(f"The folder '{folder_name}' is empty.")

# def create_folder(folder_name):
#     """
#     Create a folder and log the action.
#     """
#     try:
#         os.makedirs(folder_name)
#         print(f"The folder '{folder_name}' has been created.")
#         logging.info(f"Folder '{folder_name}' created successfully.")
#     except PermissionError:
#         print("Error: Insufficient permissions to create the folder.")
#         logging.error(f"Failed to create folder '{folder_name}': Permission denied.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         logging.error(f"Failed to create folder '{folder_name}': {e}")

# def main():
#     """
#     Main function to manage folder creation and checking.
#     """
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Folder management script.")
#     parser.add_argument("--folder", type=str, help="Name of the folder.")
#     parser.add_argument("--create", action="store_true", help="Create the folder if it doesn't exist.")
#     args = parser.parse_args()

#     # If no folder is provided via arguments, ask the user
#     folder_name = args.folder
#     if not folder_name:
#         folder_name = input("Enter the name of the folder to save data (default: 'data'): ").strip() or "data"

#     # Validate folder name
#     if not validate_folder_name(folder_name):
#         return

#     # Check if folder exists
#     if os.path.exists(folder_name):
#         print(f"The folder '{folder_name}' already exists.")
#         logging.info(f"Folder '{folder_name}' already exists.")
#         list_folder_contents(folder_name)
#     else:
#         print(f"The folder '{folder_name}' does not exist.")
#         if args.create:
#             create_folder(folder_name)
#         else:
#             create = input("Do you want to create it? (yes/no): ").strip().lower()
#             if create in ['yes', 'y']:
#                 create_folder(folder_name)
#             else:
#                 print(f"The folder '{folder_name}' was not created. User declined.")
#                 logging.info(f"User declined to create folder '{folder_name}'.")


# # Now we will create the part where we extract the data from the 
# # Ai synthetic image data generator 



# if __name__ == "__main__":
#     main()


### First way to stable dif generator


# from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
# from diffusers import StableDiffusion3Pipeline
# import torch



# model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# model_nf4 = SD3Transformer2DModel.from_pretrained(
#     model_id,
#     subfolder="transformer",
#     quantization_config=nf4_config,
#     torch_dtype=torch.bfloat16,
#     token=access_token
# )

# t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

# pipeline = StableDiffusion3Pipeline.from_pretrained(
#     model_id, 
#     transformer=model_nf4,
#     text_encoder_3=t5_nf4,
#     torch_dtype=torch.bfloat16
# )
# pipeline.enable_model_cpu_offload()

# prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape. It features the distinctive, bulky body shape of a hippo. However, instead of the usual grey skin, the creature's body resembles a golden-brown, crispy waffle fresh off the griddle. The skin is textured with the familiar grid pattern of a waffle, each square filled with a glistening sheen of syrup. The environment combines the natural habitat of a hippo with elements of a breakfast table setting, a river of warm, melted butter, with oversized utensils or plates peeking out from the lush, pancake-like foliage in the background, a towering pepper mill standing in for a tree.  As the sun rises in this fantastical world, it casts a warm, buttery glow over the scene. The creature, content in its butter river, lets out a yawn. Nearby, a flock of birds take flight"

# image = pipeline(
#     prompt=prompt,
#     num_inference_steps=4,
#     guidance_scale=0.0,
#     max_sequence_length=512,
# ).images[0]
# image.save("whimsical.png")



###################################### installations

# pip install transformers
# pip install accelerate



### Here the simplified and newer way to test 
import json
import torch
from pathlib import Path
import logging
from datetime import datetime
from diffusers import StableDiffusion3Pipeline
import time
import os

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model and generation parameters defined in the script
MODEL_SETTINGS = {
    "model_id": "stabilityai/stable-diffusion-3.5-large-turbo",
    "torch_dtype": torch.bfloat16,
    "use_safetensors": True
}

GENERATION_PARAMS = {
    "guidance_scale": 0.0,
    "num_inference_steps": 4,
    "max_sequence_length": 512
}

# Default settings
DEFAULT_PROMPT = "A serene landscape with mountains and a lake at sunset"
OUTPUT_DIR = "generated_images"
BASE_FILENAME = "sd_generated_image"

def load_api_key(config_path="config/api_key.json"):
    """Load the API key from the configuration file and set it in the environment."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            api_key = config.get("huggingface_key")
            if not api_key:
                raise KeyError("API key not found in configuration file")
            
            # Set the API key in the environment
            os.environ["HUGGING_FACE_HUB_TOKEN"] = api_key
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key configuration file not found at {config_path}. "
            "Please create a config/api_key.json file with your Hugging Face API key."
        )

def generate_image(prompt=DEFAULT_PROMPT, custom_params=None):
    """
    Generate an image using Stable Diffusion model.
    
    Args:
        prompt (str): The text prompt for image generation (defaults to DEFAULT_PROMPT)
        custom_params (dict, optional): Override default parameters
    
    Returns:
        Path: Path to the generated image
    """
    start_time = time.time()
    
    # Ensure we have the API key loaded
    api_key = load_api_key()
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the model with authentication
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_SETTINGS["model_id"],
        torch_dtype=MODEL_SETTINGS["torch_dtype"],
        use_safetensors=MODEL_SETTINGS["use_safetensors"],
        token=api_key  # Pass the API key directly to from_pretrained
    )
    
    pipe.enable_model_cpu_offload()
    
    params = GENERATION_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    
    try:
        # Generate the image
        image = pipe(prompt, **params).images[0]
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"{BASE_FILENAME}_{timestamp}.png"
        
        # Save the image
        image.save(output_file)
        
        generation_time = time.time() - start_time
        logger.info(f"Image created: {output_file.name}")
        logger.info(f"Generation time: {generation_time:.2f} seconds")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion model")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                      help=f"Text prompt for image generation (default: {DEFAULT_PROMPT})")
    parser.add_argument("--steps", type=int, help="Override number of inference steps")
    
    args = parser.parse_args()
    
    custom_params = {}
    if args.steps:
        custom_params['num_inference_steps'] = args.steps
    
    try:
        output_path = generate_image(args.prompt, custom_params)
        print(f"Successfully generated image: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")



