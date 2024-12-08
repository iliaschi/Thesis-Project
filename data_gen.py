# Here we will generate the data for the thesis

### 1
# We will have a folder checker that will chech if the folder of the data exists
# If it does not exist, it will create the folder

### 2
# Then we will as for an input from the user of Yes and No

### 3
# Then the number of data to be generated


import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def validate_folder_name(folder_name):
    """
    Validate the folder name for invalid characters or empty input.
    """
    if not folder_name or any(char in folder_name for char in r'<>:"/\|?*'):
        print("Invalid folder name. Please try again.")
        return False
    return True

def list_folder_contents(folder_name):
    """
    List the contents of a folder if it exists.
    """
    contents = os.listdir(folder_name)
    if contents:
        print(f"Contents of '{folder_name}':")
        for item in contents:
            print(f"  - {item}")
    else:
        print(f"The folder '{folder_name}' is empty.")

def create_folder(folder_name):
    """
    Create a folder and log the action.
    """
    try:
        os.makedirs(folder_name)
        print(f"The folder '{folder_name}' has been created.")
        logging.info(f"Folder '{folder_name}' created successfully.")
    except PermissionError:
        print("Error: Insufficient permissions to create the folder.")
        logging.error(f"Failed to create folder '{folder_name}': Permission denied.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"Failed to create folder '{folder_name}': {e}")

def main():
    """
    Main function to manage folder creation and checking.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Folder management script.")
    parser.add_argument("--folder", type=str, help="Name of the folder.")
    parser.add_argument("--create", action="store_true", help="Create the folder if it doesn't exist.")
    args = parser.parse_args()

    # If no folder is provided via arguments, ask the user
    folder_name = args.folder
    if not folder_name:
        folder_name = input("Enter the name of the folder to save data (default: 'data'): ").strip() or "data"

    # Validate folder name
    if not validate_folder_name(folder_name):
        return

    # Check if folder exists
    if os.path.exists(folder_name):
        print(f"The folder '{folder_name}' already exists.")
        logging.info(f"Folder '{folder_name}' already exists.")
        list_folder_contents(folder_name)
    else:
        print(f"The folder '{folder_name}' does not exist.")
        if args.create:
            create_folder(folder_name)
        else:
            create = input("Do you want to create it? (yes/no): ").strip().lower()
            if create in ['yes', 'y']:
                create_folder(folder_name)
            else:
                print(f"The folder '{folder_name}' was not created. User declined.")
                logging.info(f"User declined to create folder '{folder_name}'.")


# Now we will create the part where we extract the data from the 
# Ai synthetic image data generator 



if __name__ == "__main__":
    main()


# pip install accelerate

import torch
from diffusers import FluxPipeline
print('ok1')

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "A cat holding a sign that says hello world"
# image = pipe(
#     prompt,
#     guidance_scale=0.0,
#     num_inference_steps=4,
#     max_sequence_length=256,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save("flux-schnell.png")


# import torch
# from diffusers import StableDiffusion3Pipeline

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
# pipe = pipe.to("cuda")

# image = pipe(
#     "A capybara holding a sign that reads Hello World",
#     num_inference_steps=28,
#     guidance_scale=3.5,
# ).images[0]
# image.save("capybara.png")


from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

access_token = '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'

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    token=access_token
)

t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    text_encoder_3=t5_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape. It features the distinctive, bulky body shape of a hippo. However, instead of the usual grey skin, the creature's body resembles a golden-brown, crispy waffle fresh off the griddle. The skin is textured with the familiar grid pattern of a waffle, each square filled with a glistening sheen of syrup. The environment combines the natural habitat of a hippo with elements of a breakfast table setting, a river of warm, melted butter, with oversized utensils or plates peeking out from the lush, pancake-like foliage in the background, a towering pepper mill standing in for a tree.  As the sun rises in this fantastical world, it casts a warm, buttery glow over the scene. The creature, content in its butter river, lets out a yawn. Nearby, a flock of birds take flight"

image = pipeline(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    max_sequence_length=512,
).images[0]
image.save("whimsical.png")
