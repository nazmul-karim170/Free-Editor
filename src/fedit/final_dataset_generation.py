import argparse
import json
import sys
from pathlib import Path

import os 
import k_diffusion
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

# sys.path.append("./")
# sys.path.append("./stable_diffusion")

# from ldm.modules.attention import CrossAttention
# from ldm.util import instantiate_from_config
from data_loaders.metrics.clip_similarity import ClipSimilarity
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
# from create_training_dataset import create_training_dataset
from data_loaders import dataset_dict
import config_data_generation
import openai


################################################################################
# Modified K-diffusion Euler ancestral sampler with prompt-to-prompt.
# https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py

# print(data_loaders.dataset_dict)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def initialize_stable_diffusion(model_name="stable_diffusion/stable-diffusion-3-medium", device='cuda'):
    
    ## Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    )
    
    ## Use the UniPC Scheduler for Faster Inference (optional)
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
    ## Convert the original image to a tensor
    ## Stable Diffusion expects images as PIL Images
    edited_image = pipe(prompt=prompt, init_image=original_image, strength=0.8, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    
    return edited_image


### GPT4-based Editing Prompt Generation
openai.api_key = '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'  # Replace with your OpenAI API key

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

def torch_to_pil(image: torch.Tensor) -> Image.Image:
    image = 255.0 * rearrange(image.cpu().numpy(), "c h w -> h w c")
    image = Image.fromarray(image.astype(np.uint8))
    return image


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--out_dir",
    #     type=str,
    #     required=True,
    #     help="Path to output dataset directory.",
    # )
    # parser.add_argument(
    #     "--prompts_file",
    #     type=str,
    #     required=True,
    #     help="Path to prompts .jsonl file.",
    # )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
        help="Path to stable diffusion checkpoint.",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="stable_diffusion/models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt",
        help="Path to vae checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of sampling steps.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to generate per prompt (before CLIP filtering).",
    )
    parser.add_argument(
        "--max-out-samples",
        type=int,
        default=2,
        help="Max number of output samples to save per prompt (after CLIP filtering).",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=1,
        help="Number of total partitions.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        help="Partition index.",
    )
    parser.add_argument(
        "--min-p2p",
        type=float,
        default=0.1,
        help="Min prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--max-p2p",
        type=float,
        default=0.9,
        help="Max prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--min-cfg",
        type=float,
        default=7.5,
        help="Min classifier free guidance scale.",
    )
    parser.add_argument(
        "--max-cfg",
        type=float,
        default=15,
        help="Max classifier free guidance scale.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.5,
        help="CLIP threshold for text-image similarity of each image.",
    )
    parser.add_argument(
        "--clip-dir-threshold",
        type=float,
        default=0.2,
        help="Directional CLIP threshold for similarity of change between pairs of text and pairs of images.",
    )
    parser.add_argument(
        "--clip-img-threshold",
        type=float,
        default=0.7,
        help="CLIP threshold for image-image similarity.",
    )
    opt = parser.parse_args()

    global_seed = torch.randint(1 << 32, ()).item()
    print(f"Global seed: {global_seed}")
    seed_everything(global_seed)

    clip_similarity = ClipSimilarity().cuda()

    # out_dir = Path(opt.out_dir)
    # out_dir.mkdir(exist_ok=True, parents=True)

    ## BLIP Captioning Model ##
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")

    ## Stable Diffusion Model 
    SD_pipe = initialize_stable_diffusion()

    ## List All the Dataset Classes and Go through all of them one by one 
    parser = config_data_generation.config_parser()
    args = parser.parse_args()
    train_datasets = ['nerf_synth_generation','nerf_synthetic', 'llff', 'spaces', 'ibrnet_collected', 'realestate', 'google_scanned', 'deepvoxels']
    mode = 'train'

            ### Generate the Editing Prompts (Update these) ###
    painting_styles = ["Baroque","Realism","Impressionism","Op Art","Fauvism","Tonalism","Ashcan School","Rococo","Symbolism","Outsider Art"]
    painters_style  = ["Leonardo da Vinci", "Vincent van Gogh", "Sam Francis","Max Ernst", "Henri Matisse", "Eva Hesse", "Carl Andre", "Cy Twombly"]
    color           = ["pink", "red", "orange", "white","purple","green","blue", "silver", "gold", 'bronze']
    # remove_parts  = []
    # add parts     = []
    
    total_edits = 5
    number_of_edits = 20 
    
    # Each dataset is different and we may need to go through them manually
    for dataset in train_datasets:
        
        dataset = 'nerf_synthetic'  ## For Testing 

        ## Metadata Save Directory
        dataset_dir = os.path.join("../../../data", dataset)
        if dataset == 'nerf_synthetic' or "nerf_synth_generation":
            train_scenes = ["chair", "drums", "lego", "hotdog", "materials", "mic", "ship"]

        for scene in train_scenes:
            
            ## Scene Directory
            scene_dir  = os.path.join(dataset_dir, scene)

            ## Get the Dataloader (We have to go through scene by scene)
            train_dataset = dataset_dict[dataset](
                args,
                mode,
                scenes=scene,
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                worker_init_fn=lambda _: np.random.seed(),
                num_workers=args.workers,
                shuffle=True,               ## We want the selected images to be diverse 
            )

            # starting_view_files = []
            # target_view_files = []
            # nearest_ids_list  = []
            # CLIP_text_image_sim = []
            # CLIP_image_sim  = []

            ## We will apply multiple edits to each scene
            for index_ed in range(number_of_edits):

                ### Go through only 15 images, not all images of a scene
                num_starting_view = 15
                for b_id, batch in enumerate(train_loader):

                    caption_rgb = batch["caption_rgb"]
                    inputs = processor(caption_rgb, return_tensors="pt").to("cuda")                       ## Put the Image into GPU
                    out    = model.generate(**inputs)                                                     ## Get the Prediction
                    generated_cap = processor.batch_decode(out, skip_special_tokens=True)[0].strip()      ## Generate the Caption

                    
                    if b_id == 0:   ## We generate 1 prompt per scene

                                        ## GPT-4 based prompt generation ##
                        editing_prompt_SD = generate_prompts_gpt(generated_cap, num_prompts=1)[0]
                        editing_prompt = editing_prompt_SD

                                        ## Manual Editing ##
                        # edit_choice = np.random.choice([1,2,3], 1)
                        # if edit_choice==1:
                        #     id_edit = np.random.choice(len(painters_style),1, replace = False)
                        #     edit = painters_style[id_edit]
                        #     editing_prompt = edit + "painting of " + generated_cap 
                        # elif edit_choice==2:
                        #     id_edit = np.random.choice(len(color),1)
                        #     edit = color[id_edit]
                        #     editing_prompt = generated_cap + "in" + edit + "color"
                        # else:
                        #     id_edit = np.random.choice(len(painting_styles),1)
                        #     edit = painting_styles[id_edit]
                        #     editing_prompt = generated_cap + "in" + edit + "style" 

                        # editing_prompt_SD = editing_prompt +  + ', relaistic look with accuracte details'
                
                    ## Get the starting and target views 
                    starting_view = Image.fromarray((255*batch["starting_view"]).astype(np.uint8))
                    target_view   = Image.fromarray((255*batch["traget_rgb"]).astype(np.uint8))
                    nearest_pose_ids  = batch["nearest_pose_ids"]
                    save_dir_target   = scene_dir + "_edited"
                    # save_dir_starting = os.path.dirname(batch["starting_rgb_path"]) + "_edited"
                    os.makedirs(save_dir_target, exist_ok=True)
                    # os.makedirs(save_dir_starting, exist_ok=True)
                    
                    ## Edit the Target and Starting Views 
                    results = {}
                    
                    ## For Each Prompt, generate "n_samples" edited starting and edited view  
                    with tqdm(total=opt.n_samples, desc="Samples") as progress_bar:

                        while len(results) < opt.n_samples:
                            seed = torch.randint(1 << 32, ()).item()
                            if seed in results:
                                continue
                            torch.manual_seed(seed)

                        cfg_scale        = opt.min_cfg + torch.rand(()).item() * (opt.max_cfg - opt.min_cfg)
                        starting_view_ed = edit_image_with_prompt(SD_pipe, starting_view, editing_prompt_SD, num_inference_steps = 30, guidance_scale = cfg_scale)
                        target_view_ed   = edit_image_with_prompt(SD_pipe, target_view, editing_prompt_SD, num_inference_steps = 30, guidance_scale = cfg_scale)

                        clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = clip_similarity(
                            target_view_ed, starting_view_ed, [editing_prompt], [editing_prompt]
                        )

                        results[seed] = dict(
                            image_0=torch_to_pil(starting_view_ed),
                            image_1=torch_to_pil(target_view_ed),
                            cfg_scale=cfg_scale,
                            clip_sim_0=clip_sim_0[0].item(),
                            clip_sim_1=clip_sim_1[0].item(),
                            clip_sim_dir=clip_sim_dir[0].item(),
                            clip_sim_image=clip_sim_image[0].item(),
                        )

                        progress_bar.update()

                    ## CLIP filter to get best samples for each prompt.
                    metadata = [
                        (np.mean(result["clip_sim_0"], result["clip_sim_1"]), seed)
                        for seed, result in results.items()
                        if result["clip_sim_image"] >= opt.clip_img_threshold   ## Fix this 
                        and result["clip_sim_dir"] >= opt.clip_dir_threshold
                        and result["clip_sim_0"] >= opt.clip_threshold
                        and result["clip_sim_1"] >= opt.clip_threshold
                    ]
                    metadata.sort(reverse=True)
                    print("metadata:", metadata)

                    ## Save only the best pair 
                    for _, seed in metadata[: opt.max_out_samples]:
                        result = results[seed]
                        image_0 = result.pop("image_0")
                        image_1 = result.pop("image_1")
                        image_0.save(save_dir_target.joinpath(f"s_view_{index_ed}_{b_id}_{seed}.jpg"), quality=100, subsampling=0)
                        image_1.save(save_dir_target.joinpath(f"t_view{index_ed}_{b_id}_{seed}.jpg"), quality=100, subsampling=0)
                        
                        ## If we want to save all edited pair info of a scene
                        # starting_view_files.append(save_dir_starting.joinpath(f"s_view_{index_ed}_{b_id}_{seed}.jpg"))
                        # target_view_files.append(save_dir_target.joinpath(f"t_view_{index_ed}_{b_id}_{seed}.jpg"))
                        # nearest_ids_list.append(nearest_pose_ids)
                        # CLIP_text_image_sim.append(np.mean(clip_sim_0[0].item(),clip_sim_1[0].item()),)
                        # CLIP_image_sim.append(clip_sim_image[0].item())    

                        ## Saving one edited pair info at a time
                        starting_view_file  = save_dir_target.joinpath(f"s_view_{index_ed}_{b_id}_{seed}.jpg")
                        target_view_file    = save_dir_target.joinpath(f"t_view_{index_ed}_{b_id}_{seed}.jpg")
                        CLIP_text_image_sim = np.mean(clip_sim_0[0].item(),clip_sim_1[0].item())

                    ## If 15 images have been edited 
                    if b_id == num_starting_view:
                        break
            
                    ## Save the metadata of each edited pair
                    Scene_info = dict(
                        # dataset_name = dataset, 
                        scene_name = scene,
                        # editing_prompt = editing_prompt,
                        starting_view_file=starting_view_file,
                        target_view_file=target_view_file,
                        nearest_pose_ids = nearest_pose_ids,              ## We need this to select the non-edited images from each scene
                        render_pose = batch["render_pose"],
                        camera_matrices  = batch["camera_matrices"],   
                        # total_num_images = batch["num_images_in_scene"],   
                        # depth_range = batch["depth_range"],
                        # cfg_scale=cfg_scale,
                        clip_text_to_img_similarity=CLIP_text_image_sim,  ## while training, we will prioritize edited samples based on this 
                        # clip_image_similarity=clip_sim_image[0].item(),
                    )

                    ## Save the metadata
                    with open(scene_dir.joinpath(f"{scene}_edited_metadata.jsonl"), "a") as fp:
                        fp.write(f"{json.dumps(dict(seed=seed, **Scene_info))}\n")

    print("Done.")


if __name__ == "__main__":
    main()
