import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Define model path
base_model_path = Path(__file__).resolve().parent.parent / "models/stable_diffusion"

# Check if snapshots exist
snapshots_path = base_model_path / "models--runwayml--stable-diffusion-v1-5" / "snapshots"
snapshot_dirs = list(snapshots_path.glob("*"))  # Get all snapshot folders

# Use latest snapshot if available, else default model path
model_path = str(snapshot_dirs[0]) if snapshot_dirs else str(base_model_path)

# Load model from local directory
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32  # Use float16 only for CUDA
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)

# Function to generate images
def generate_image(prompt):
    print(f"🎨 Generating image for: {prompt}")

    # Generate image with improved parameters
    image = pipe(prompt, num_inference_steps=100, guidance_scale=10).images[0]

    # Define output path
    output_dir = Path(__file__).resolve().parent.parent / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    image_path = output_dir / "generated_output.png"

    # Save the image
    image.save(image_path)
    print(f"✅ Image saved at {image_path}")

    return str(image_path)

# Run test case if executed as script
if __name__ == "__main__":
    test_prompt = "A futuristic classroom with AI-powered robots teaching students, ultra-realistic, highly detailed, 4K, cinematic lighting."
    generate_image(test_prompt)
