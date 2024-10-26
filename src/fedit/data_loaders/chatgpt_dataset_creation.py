import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image
import random
from tqdm import tqdm
import openai


def load_images(image_dir):
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images = []
    image_paths = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(supported_formats):
            path = os.path.join(image_dir, filename)
            img = Image.open(path).convert('RGB')
            images.append(img)
            image_paths.append(path)
    return images, image_paths


def generate_captions(images, device='cuda'):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    captions = []
    for img in tqdm(images, desc="Generating Captions"):
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions


def generate_prompts(caption, num_prompts=20):
    # Define a set of modifiers to enhance prompts
    modifiers = [
        "in a vibrant and colorful style",
        "with a dark and moody atmosphere",
        "in a minimalist design",
        "with a futuristic look",
        "in a surreal and abstract manner",
        "with high detail and realism",
        "in a watercolor painting style",
        "as a digital illustration",
        "with enhanced lighting and shadows",
        "in a vintage retro style",
        "with a cinematic feel",
        "in a cartoonish manner",
        "with a focus on texture",
        "in a high-definition quality",
        "with dynamic composition",
        "in a hyper-realistic style",
        "with bold and vivid colors",
        "in an impressionist painting style",
        "with a dreamy and ethereal quality",
        "in a sketch-style drawing"
    ]
    
    # Generate prompts by combining the caption with different modifiers
    prompts = []
    for modifier in modifiers[:num_prompts]:
        prompt = f"{caption}, {modifier}"
        prompts.append(prompt)
    return prompts


openai.api_key = 'your_openai_api_key'  # Replace with your OpenAI API key

def generate_prompts_gpt(caption, num_prompts=20):
    prompts = []
    for _ in range(num_prompts):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Generate a creative and diverse prompt for editing an image based on the following caption:\n\nCaption: {caption}\n\nPrompt:",
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )
        prompt = response.choices[0].text.strip()
        prompts.append(prompt)
    return prompts


def initialize_stable_diffusion(model_name="runwayml/stable-diffusion-v1-5", device='cuda'):
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    )
    
    # Use the UniPC scheduler for faster inference (optional)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to(device)
    return pipe

def edit_image_with_prompt(pipe, original_image, prompt, num_inference_steps=50, guidance_scale=7.5, device='cuda'):
    """
    Edits an image based on the provided prompt using Stable Diffusion.

    Args:
        pipe: The Stable Diffusion pipeline.
        original_image: PIL.Image object of the original image.
        prompt: Text prompt to guide image editing.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Controls how much the image follows the prompt.
        device: 'cuda' or 'cpu'.

    Returns:
        Edited PIL.Image object.
    """
    # Convert the original image to a tensor
    # Stable Diffusion expects images as PIL Images
    edited_image = pipe(prompt=prompt, init_image=original_image, strength=0.8, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return edited_image

# Note: The exact parameters for `init_image` and related arguments may vary based on the Stable Diffusion model version.
# Ensure that the pipeline supports image-to-image generation.


def main():
    # Configuration
    image_directory = 'path_to_your_images_directory'  # Replace with your image directory path
    output_directory = 'edited_images'
    os.makedirs(output_directory, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Step 1: Load Images
    images, image_paths = load_images(image_directory)
    print(f"Loaded {len(images)} images.")
    
    # Step 2: Generate Captions using BLIP
    captions = generate_captions(images, device=device)
    print("Generated Captions:")
    for idx, (path, caption) in enumerate(zip(image_paths, captions)):
        print(f"{idx+1}. {os.path.basename(path)}: {caption}")
    
    # Step 3: Initialize Stable Diffusion Model
    sd_model = initialize_stable_diffusion(device=device)
    print("Initialized Stable Diffusion model.")
    
    # Step 4: Generate Prompts and Edit Images
    for idx, (img, caption, path) in enumerate(zip(images, captions, image_paths)):
        print(f"\nProcessing Image {idx+1}: {os.path.basename(path)}")
        
        # Generate 20 prompts based on the caption
        prompts = generate_prompts(caption, num_prompts=20)
        
        # For each prompt, edit the image and save the result
        for prompt_idx, prompt in enumerate(prompts):
            print(f"  Generating edited image with Prompt {prompt_idx+1}: {prompt}")
            
            # Edit the image
            edited_img = edit_image_with_prompt(sd_model, img, prompt, device=device)
            
            # Define the output path
            image_name = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(output_directory, f"{image_name}_edited_{prompt_idx+1}.png")
            
            # Save the edited image
            edited_img.save(output_path)
    
    print("\nAll images have been processed and saved.")

if __name__ == "__main__":
    main()

