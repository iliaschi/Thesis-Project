# import requests
# import json
# from pathlib import Path
# from datetime import datetime

# # The model will generate 1024x1024 images by default
# API_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
# DEFAULT_PROMPT = "Image of a man who looks happy"

# def load_api_key(config_path="config/api_key.json"):
#     """Load the Stability AI API key from configuration file."""
#     with open(config_path, 'r') as f:
#         return json.load(f)["stability_key"]

# def generate_image(prompt=DEFAULT_PROMPT):
#     """
#     Generate a single image using SD 3.5 Turbo model.
#     Each generation costs 4 credits.
#     """
#     # Load API key and prepare the output directory
#     api_key = load_api_key()
#     output_dir = Path("generated_images_remote")
#     output_dir.mkdir(exist_ok=True)
    
#     # Make the API request
#     response = requests.post(
#         API_ENDPOINT,
#         headers={
#             "Authorization": f"Bearer {api_key}",
#             "Accept": "image/*"
#         },
#         files={"none": ''},
#         data={
#             "prompt": prompt,
#             "model": "sd3.5-large-turbo"
#         }
#     )
    
#     if response.status_code == 200:
#         # Save the image with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_file = output_dir / f"image_{timestamp}.png"
        
#         with open(output_file, 'wb') as f:
#             f.write(response.content)
        
#         print(f"Image created: {output_file}")
#         return output_file
#     else:
#         print(f"Error: {response.json()}")
#         return None

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Generate an image using SD 3.5 Turbo")
#     parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
#                       help="Text prompt for image generation")
    
#     args = parser.parse_args()
#     generate_image(args.prompt)




# Configuration
API_KEY = 'sk-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'  # Replace with your actual API key

import requests
import json
from pathlib import Path
from datetime import datetime


API_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
DEFAULT_PROMPT = "Image of a man who looks happy"

def generate_image(prompt=DEFAULT_PROMPT):
    """
    Generate a single image using SD 3.5 Turbo model.
    Each generation costs 4 credits.
    """
    # Prepare the output directory
    output_dir = Path("generated_images_remote")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare headers and data
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "image/*"
    }
    
    # Prepare form data
    form_data = {
        "prompt": (None, prompt),
        "model": (None, "sd3.5-large-turbo")
    }
    
    print("Making request with prompt:", prompt)
    
    try:
        # Make the API request using multipart/form-data
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            files=form_data  # Using files parameter for multipart/form-data
        )
        
        # Print response status and content for debugging
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"image_{timestamp}.png"
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"Image created: {output_file}")
            return output_file
        else:
            print(f"Error Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate an image using SD 3.5 Turbo")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                      help="Text prompt for image generation")
    
    args = parser.parse_args()
    generate_image(args.prompt)