from diffusers import StableDiffusionPipeline
import torch

# Set model save directory
model_path = "../models/stable_diffusion"

print("⏳ Downloading Stable Diffusion v1.5... (This may take time)")

for attempt in range(3):  # Retry up to 3 times
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            resume_download=True,  # Resume interrupted downloads
            timeout=120  # Increase timeout
        )
        pipeline.save_pretrained(model_path)
        print(f"✅ Model downloaded and saved in {model_path}")
        break
    except Exception as e:
        print(f"⚠️ Download failed. Retrying {attempt + 1}/3... Error: {e}")

