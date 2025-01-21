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