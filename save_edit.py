from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
from utils import save_image, concat_images_in_square_grid, TaskVector
import argparse
import torch.nn.functional as F
import open_clip

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', type=str, default="stabilityai/stable-diffusion-2", help='pretrained model')
    parser.add_argument('--model_finetuned', type=str, default="", help='finetuned model')
    parser.add_argument('--num_images', type=int, default=30, help='number of images')
    parser.add_argument('--output_dir', type=str, default="diffusers_ckpt/output", help='output directory')
    parser.add_argument('--tv_edit_alpha', type=float, default=0.5, help='amount of edit to task vector layer')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_pretrained = StableDiffusionPipeline.from_pretrained(args.model_pretrained, torch_dtype=torch.float16, safety_checker=None)
    pipe_pretrained.to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    
    pipe_pretrained = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None)
    pipe_finetuned = StableDiffusionPipeline.from_pretrained("task_vector_exp/van_gogh_sd_1.4_empty_finetune=[unet]", torch_dtype=torch.float16, safety_checker=None)
    pipe_pretrained.to("cuda")
    pipe_finetuned.to("cuda")
    
    #edit process
    unet_pretrained = pipe_pretrained.unet
    unet_finetuned = pipe_finetuned.unet

    #save model unet
    torch.save(unet_pretrained, "unet_pretrained.pt")
    torch.save(unet_finetuned, "unet_finetuned.pt")

    task_vector_unet = TaskVector(pretrained_checkpoint="unet_pretrained.pt", 
                            finetuned_checkpoint="unet_finetuned.pt")
    
    unet_edited = task_vector_unet.apply_to("unet_pretrained.pt", scaling_coef=args.tv_edit_alpha)
        
    pipe_pretrained.unet = unet_edited

    pipe_pretrained.save_pretrained(args.output_dir)
    
    os.remove("unet_pretrained.pt")
    os.remove("unet_finetuned.pt")
    
    