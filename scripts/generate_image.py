import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from pathlib import Path
import gradio as gr
from PIL import Image

# Define model path
model_path = str(Path(__file__).resolve().parent.parent / "models/stable_diffusion")

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load text-to-image pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_path, torch_dtype=torch_dtype
).to(device)
pipe.enable_xformers_memory_efficient_attention()  # Optimized memory usage

# Load inpainting pipeline for editing existing images
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    model_path, torch_dtype=torch_dtype
).to(device)
pipe_inpaint.enable_xformers_memory_efficient_attention()


# Function to generate a single image
def generate_image(prompt, num_steps=50, guidance_scale=7.5):
    print(f"🎨 Generating image for: {prompt}")
    image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]

    # Save output
    output_path = str(Path(__file__).resolve().parent / "generated_output.png")
    image.save(output_path)
    print(f"✅ Image saved at {output_path}")
    return output_path


# Function to generate multiple images in a batch
def generate_images(prompts, num_steps=50, guidance_scale=7.5):
    output_dir = Path(__file__).resolve().parent / "generated_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for i, prompt in enumerate(prompts):
        print(f"🎨 Generating image {i+1}/{len(prompts)}: {prompt}")
        image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]

        output_path = output_dir / f"generated_output_{i+1}.png"
        image.save(output_path)
        images.append(str(output_path))

    return images


# Function to inpaint (edit) an image
def inpaint_image(prompt, image_path, mask_path, num_steps=50, guidance_scale=7.5):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Mask should be grayscale (white areas get changed)

    print(f"🖌 Editing image with prompt: {prompt}")
    edited_image = pipe_inpaint(
        prompt=prompt, image=image, mask_image=mask, num_inference_steps=num_steps, guidance_scale=guidance_scale
    ).images[0]

    output_path = str(Path(__file__).resolve().parent / "edited_output.png")
    edited_image.save(output_path)
    print(f"✅ Edited image saved at {output_path}")
    return output_path


# Web UI using Gradio
def generate_image_gradio(prompt):
    return pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]


iface = gr.Interface(fn=generate_image_gradio, inputs="text", outputs="image")
if __name__ == "__main__":
    print("🚀 Starting Gradio Web UI...")
    iface.launch()
