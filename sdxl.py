from diffusers import AutoPipelineForText2Image
import torch
import os
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Robot in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt).images[0]

if not os.path.exists('./images'):
    os.makedirs('./images')
image.save(f'./images/{prompt}.png')