import torch
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
# image_name = 'An image of a walking robot in a jungle'
image_name = "Robot in a jungle, cold color palette, muted colors, detailed, 8k"
image = load_image(f'./images/{image_name}.png')
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=16, generator=generator, motion_bucket_id=500, noise_aug_strength=0.1).frames[0]

if not os.path.exists('./svd'):
    os.makedirs('./svd')
export_to_video(frames, f'./svd/{image_name}.mp4', fps=7)