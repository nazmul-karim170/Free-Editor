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
import gc
import torchvision.transforms as transforms
from pathlib import Path
import pickle
import glob

# sys.path.append("./")
# sys.path.append("./stable_diffusion")

# from ldm.modules.attention import CrossAttention
# from ldm.util import instantiate_from_config
from data_loaders_gen.metrics.clip_similarity import ClipSimilarity
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusion3Pipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, UniPCMultistepScheduler
from data_loaders_gen import dataset_dict         # from create_training_dataset import create_training_dataset
import config_data_generation

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



            ### Edit with Stable Diffusion if you have enough GPU resources ##
            ##################################################################
def initialize_stable_diffusion(model_name="stable_diffusion/stable-diffusion-3-medium", device='cuda'):
    
    ## Load the Stable Diffusion pipeline  
    # pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)         ## It does not support text-to-image editing yet
    pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", custom_pipeline="stable_diffusion_mega", torch_dtype=torch.float16, variant="fp16")

    ## Use the UniPC Scheduler for Faster Inference (optional)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe

def edit_image_with_prompt(pipe, original_image, prompt, num_inference_steps=30, guidance_scale=7.5, device='cuda'):
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
    edited_image = pipe.img2img(prompt=prompt, image=original_image, strength=0.8, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    
    return edited_image


    ## If there are not enough GPU resources, Use IP2P model ##
    ###########################################################
def IP2P_intitialization(model_id = "timbrooks/instruct-pix2pix", device = 'cuda'):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()  ## For Faster Calucations
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe

def Edit_with_IP2P(pipe, image, prompt, num_inference_steps=50, guidance_scale=7.5, image_guidance_scale=1.5):
    # `image` is an RGB PIL.Image
    print("Started Editing....")
    images = pipe(prompt = prompt, image=image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale).images
    print("Editing Complete")
    return images[0]


def torch_to_pil(image: torch.Tensor) -> Image.Image:
    image = 255.0 * rearrange(image.cpu().numpy(), "c h w -> h w c")
    image = Image.fromarray(image.astype(np.uint8))
    return image

def pil_to_torch(image):
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    return tensor_image.unsqueeze(0).to("cuda")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,       ## Ideally, it should be 10-15
        help="Number of samples to generate per prompt (before CLIP filtering).",
    )
    parser.add_argument(
        "--num-edits",
        type=int,
        default=1,       ## ideally, it should be 15-20
        help="Number of DDIM sampling steps.",
    )
    parser.add_argument(
        "--num-startview",
        type=int,
        default=1,       ## Ideally, it should be 20-25
        help="Number of samples to generate per prompt (before CLIP filtering).",
    )
    parser.add_argument(
        "--max-out-samples",
        type=int,
        default=1,
        help="Max number of output samples to save per prompt (after CLIP filtering).",
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
        default=6,
        help="Min classifier free guidance scale.",
    )
    parser.add_argument(
        "--max-cfg",
        type=float,
        default=10,
        help="Max classifier free guidance scale.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.1,
        help="CLIP threshold for text-image similarity of each image.",
    )
    parser.add_argument(
        "--clip-dir-threshold",
        type=float,
        default=0.1,
        help="Directional CLIP threshold for similarity of change between pairs of text and pairs of images.",
    )
    parser.add_argument(
        "--clip-img-threshold",
        type=float,
        default=0.4,
        help="CLIP threshold for image-image similarity.",
    )
    opt = parser.parse_args()

    global_seed = torch.randint(1 << 32, ()).item()
    print(f"Global seed: {global_seed}")
    seed_everything(global_seed)

    ## For Metrics 
    clip_similarity = ClipSimilarity().cuda()

                ## Initialize Models###
                #######################
    ## Caption Model
    model_name_caption = "Salesforce/blip-image-captioning-base" 
    processor = BlipProcessor.from_pretrained(model_name_caption)
    processor.image_processor.do_rescale = False                                          ## Ensure rescaling is disabled if input images are already normalized
    cap_model = BlipForConditionalGeneration.from_pretrained(model_name_caption).to("cuda:0")

    # ## Load the Stable Diffusion Model
    SD_pipe = initialize_stable_diffusion(device="cuda:1")

    # ## Load the IP2P model (if enough GPU resources are not available)
    # IP2P_pipe = IP2P_intitialization(device="cuda:1")

    ## List All the Dataset Classes and Go through all of them one by one 
    parser = config_data_generation.config_parser()
    args = parser.parse_args()
    train_datasets = ['nerf_synthetic', 'real_iconic_noface',  'shiny', 'spaces', 'ibrnet_collected', 'realestate10k', 'google_scanned_objects', 'deepvoxels']
         
            ### Sample Editing Styles ###
    painting_styles = ["Baroque",  "sci-fi style", "Realism","Impressionism","Op Art","Fauvism","Tonalism","Ashcan School","Rococo","Symbolism","Outsider Art"]
    painters_style  = ["Leonardo da Vinci", "Vincent van Gogh", "Sam Francis","Max Ernst", "Henri Matisse", "Eva Hesse", "Carl Andre", "Cy Twombly", "watercolor painting"]
    color           = ["pink", "Cartoonish", "Watercolor", "black-and-white color", "mix of rainbow", "red", "icy", "marble", "orange", " glossy metallic", "white","purple","green","blue", "silver", "gold", 'bronze']

    ## Total number of edits per view 
    number_of_edits   = opt.num_edits 

    ### Go through only 15 views, not all images of a scene
    num_starting_view = opt.num_startview    

    ## Each dataset is different and we may need to go through them manually
    for dataset in train_datasets:

        ## Dataset Directory
        relative_data_path = "../../../data"        ## Change it according to your directories
        dataset_dir = os.path.join(relative_data_path, dataset)
        print("We are Processing the Dataset: ", dataset)
        
        if dataset == 'nerf_synthetic':
            train_scenes = ["chair", "drums", "lego", "hotdog", "materials", "mic", "ship"]
        elif dataset == 'realestate10k':
            dataset_dir = os.path.join(dataset_dir, 'frames')
            train_scenes = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
        elif dataset == 'ibrnet_collected':
            folder_path_1 = os.path.join(dataset_dir, 'ibrnet_collected_1')
            train_scenes  = os.listdir(folder_path_1)
            train_scenes_1 =  [os.path.join(folder_path_1, name) for name in train_scenes if os.path.isdir(os.path.join(folder_path_1, name))]  
            folder_path_2 = os.path.join(dataset_dir, 'ibrnet_collected_2')
            train_scenes  = os.listdir(folder_path_2)
            train_scenes_2 =  [os.path.join(folder_path_2, name) for name in train_scenes if os.path.isdir(os.path.join(folder_path_1, name))]  
            train_scenes = train_scenes_1 + train_scenes_2
        elif dataset == 'deepvoxels':
            modes = ['train', 'test', 'validation']
            train_scenes = []
            modes_list = []
            for mode in modes:
                folder_path  = os.path.join(dataset_dir, mode)
                train_scenes_1 =  [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]  
                train_scenes.extend(train_scenes_1)   
                modes_list.extend([mode]*len(train_scenes_1))    

        elif dataset == 'spaces':
            dataset_dir  =  os.path.join(dataset_dir, 'data/800')
            train_scenes =  [os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]  

        elif dataset in ["real_iconic_noface","google_scanned_objects","shiny"]:
            train_scenes = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]            


        ## Go Scene-by-Scene
        for i, scene in enumerate(train_scenes):            
            print("Processing the Scene: ", scene)

            if i > 0:
                break 

            ## Edited Data Saving Directory (Relative)
            if dataset in ['ibrnet_collected', 'spaces', 'deepvoxels']:
                scene_name = scene.split("/")[-1]
                scene_edited_name = scene_name + "_edited"
                scene_edited_path = Path(scene_edited_name)
                scene_dir = Path(scene)                     ## Meta data saving directory
                save_dir_target = Path(scene + "_edited")   ## Edited Images Save directory
            else:
                scene_edited_name = scene + "_edited"
                scene_edited_path = Path(scene_edited_name)
                scene_dir_in      = os.path.join(dataset_dir, scene)
                scene_dir = Path(scene_dir_in)              ## Meta data saving directory
                save_dir_target = Path(scene_dir_in + "_edited")

            os.makedirs(save_dir_target, exist_ok=True)

            ## Get the Dataloader (We have to go through scene by scene)
            if dataset == 'deepvoxels':
                mode = modes_list[i]
            else:
                mode = 'train' 
                
            train_dataset = dataset_dict[dataset](args, mode, scenes=scene)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=1,
                worker_init_fn=lambda _: np.random.seed(),
                num_workers=args.workers,
                shuffle=True,               ## We want the selected images to be diverse 
            )

            ## We will apply multiple edits to each scene
            for index_ed in range(number_of_edits):

                for b_id, batch in enumerate(train_loader):

                    ## BLIP Captioning Model
                    with torch.no_grad():
                        caption_rgb = batch["caption_rgb"]
                        inputs = processor(caption_rgb, return_tensors="pt").to("cuda:0")                       ## Put the Image into GPU
                        out    = cap_model.generate(**inputs)                                                   ## Get the Prediction
                        # generated_cap = processor.batch_decode(out, skip_special_tokens=True)[0].strip()      ## Generate the Caption
                        generated_cap = processor.decode(out[0], skip_special_tokens=True)                      ## Generate the Caption
                        print("generated caption:", generated_cap)
    
                    ## We formulate one editing prompt per scene
                    if b_id == 0:   

                        #               ## GPT-4 based prompt generation ##
                        # editing_prompt_SD = generate_prompts_gpt(generated_cap, num_prompts=1)[0]
                        # editing_prompt = editing_prompt_SD

                                        ## Manual Prompt Formulation ##
                        edit_choice = np.random.choice([1,2,3], 1)
                        if edit_choice==1:
                            id_edit = int(np.random.choice(len(painters_style),1, replace = False))
                            edit = painters_style[id_edit]
                            editing_prompt = edit + " painting of " + generated_cap 
                        elif edit_choice==2:
                            id_edit = int(np.random.choice(len(color),1))
                            edit = color[id_edit]
                            editing_prompt = generated_cap + " in " + edit + " color"
                        else:
                            id_edit = int(np.random.choice(len(painting_styles),1))
                            edit = painting_styles[id_edit]
                            editing_prompt = generated_cap + " in " + edit + " style" 

                        editing_prompt_SD = editing_prompt + ', with a focus on accurate color and shape, photorealistic details'
                    
                    print("Generated Prompt:", editing_prompt)
                
                    ## Get the starting and target views 
                    starting_view = Image.fromarray(255*batch["starting_view"].numpy().squeeze().astype(np.uint8))
                    target_view   = Image.fromarray(255*batch["traget_rgb"].numpy().squeeze().astype(np.uint8))
                    nearest_pose_ids  = batch["nearest_pose_ids"]

                    width, height = starting_view.size
                    print(f"Width: {width}, Height: {height}")

                    ## Edit the Target and Starting Views 
                    results = {}
                    
                    ## For Each Prompt, generate "n_samples" edited starting and edited view  
                    with tqdm(total=opt.n_samples, desc="Samples") as progress_bar:

                        while len(results) < opt.n_samples:
                            seed = torch.randint(1 << 32, ()).item()
                            if seed in results:
                                continue
                            torch.manual_seed(seed)

                            cfg_scale  = opt.min_cfg + torch.rand(()).item() * (opt.max_cfg - opt.min_cfg)
                            
                            ### If we use stable diffusion
                            with torch.no_grad():
                                starting_view_ed = edit_image_with_prompt(SD_pipe, starting_view, editing_prompt_SD, num_inference_steps = opt.steps, guidance_scale = cfg_scale)
                                target_view_ed   = edit_image_with_prompt(SD_pipe, target_view, editing_prompt_SD, num_inference_steps = opt.steps, guidance_scale = cfg_scale)

                            # ## If we use IP2P pipeline
                            # with torch.no_grad(): 
                            #     starting_view_ed = Edit_with_IP2P(IP2P_pipe, starting_view, editing_prompt_SD, num_inference_steps=30, guidance_scale=cfg_scale, image_guidance_scale=1.5)
                            #     target_view_ed   = Edit_with_IP2P(IP2P_pipe, target_view, editing_prompt_SD, num_inference_steps=30, guidance_scale=cfg_scale, image_guidance_scale=1.5)

                            clip_sim_0, clip_sim_1, clip_sim_dir_0, clip_sim_dir_1, clip_sim_image = clip_similarity(
                                pil_to_torch(target_view_ed), pil_to_torch(starting_view_ed), [editing_prompt], [editing_prompt],  pil_to_torch(target_view), pil_to_torch(starting_view),[generated_cap]
                            )
                            # print(clip_sim_0[0].cpu().item(), clip_sim_1, clip_sim_dir_0, clip_sim_dir_1, clip_sim_image)

                            results[seed] = dict(
                                image_0=starting_view_ed,
                                image_1=target_view_ed,
                                cfg_scale=cfg_scale,
                                clip_sim_0=clip_sim_0[0].cpu().item(),
                                clip_sim_1=clip_sim_1[0].cpu().item(),
                                clip_sim_dir=np.mean([clip_sim_dir_0[0].cpu().item(), clip_sim_dir_1[0].cpu().item()]),
                                clip_sim_image=clip_sim_image[0].item(),
                            )

                            progress_bar.update()

                    ## CLIP-Score based filtering to get best samples for each prompt.
                    metadata = [
                        (np.mean([result["clip_sim_0"], result["clip_sim_1"]]), seed)
                        for seed, result in results.items()
                        if result["clip_sim_image"] >= opt.clip_img_threshold    
                        and result["clip_sim_dir"] >= opt.clip_dir_threshold
                        and result["clip_sim_0"] >= opt.clip_threshold
                        and result["clip_sim_1"] >= opt.clip_threshold
                    ]
                    metadata.sort(reverse=True)
                    print("metadata:", metadata)

                    if metadata:
                        ## Save only the best pair 
                        for _, seed in metadata[: opt.max_out_samples]:
                            result  = results[seed]
                            image_0 = result.pop("image_0")
                            image_1 = result.pop("image_1")
                            image_0.save(save_dir_target.joinpath(f"s_view_{index_ed}_{b_id}_{seed}.jpg"), quality=100, subsampling=0)
                            image_1.save(save_dir_target.joinpath(f"t_view{index_ed}_{b_id}_{seed}.jpg"), quality=100, subsampling=0)

                            ## Saving one edited pair info at a time
                            starting_view_file  = os.path.join(scene_edited_path,f"s_view_{index_ed}_{b_id}_{seed}.jpg")
                            target_view_file    = os.path.join(scene_edited_path,f"t_view_{index_ed}_{b_id}_{seed}.jpg")
                            CLIP_text_image_sim = np.mean([clip_sim_0[0].cpu().item(),clip_sim_1[0].cpu().item()])
                            print("Edited Images Saved  ................", starting_view_file)

                
                        ## Save the metadata of each edited pair
                        Scene_info = dict(
                            dataset_name = dataset, 
                            scene_name = scene,
                            depth_range = batch["depth_range"],
                            starting_view_file = starting_view_file,
                            target_view_file = target_view_file,
                            nearest_pose_ids = nearest_pose_ids,              ## We need this to select the non-edited images from each scene
                            render_pose = batch["render_pose"],
                            target_camera_matrices  = batch["target_camera_matrices"],   
                            starting_camera_matrices  = batch["starting_camera_matrices"],   
                            clip_text_to_img_similarity=CLIP_text_image_sim,  ## while training, we will prioritize edited samples based on this 
                        )

                        ## Save the metadata using pickle
                        with open(scene_dir.joinpath(f"{scene_edited_name}_metadata.pkl"), "ab") as fp:
                            pickle.dump(dict(seed=seed, **Scene_info), fp)

                    ## If "num_starting_view" images have been edited 
                    if b_id == num_starting_view:
                        break

                # # Clear cache after using SD model (FOR GPU Resource Limitations)
                # del SD_pipe
                # gc.collect()             #Forces the garbage collector to free up memory immediately
                # torch.cuda.empty_cache() 

    print("Done.")


if __name__ == "__main__":
    main()
