import os
import re
import sys
import cv2
import torch
import random
import argparse
import numpy as np
import transformers 
from model.geopixel import GeoPixelForCausalLM

def rgb_color_text(text, r, g, b): # for printing
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def parse_args(args):
    parser = argparse.ArgumentParser(description="Chat with GeoPixel")
    parser.add_argument("--version", default="MBZUAI/GeoPixel-7B")            # their final trained weights -> we also use it for fine-tuning in `finetune.py`
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)  # where to save the final infered images
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)

    os.makedirs(args.vis_save_path, exist_ok=True)

    print(f'initialing tokenizer from: {args.version}')
    tokenizer = transformers.AutoTokenizer.from_pretrained( # GeoPixel-7B uses IXC-2.5 internally as LLM: tokenizer will be loaded from there, including the special tokens: <p></p>[SEG]
        args.version,
        cache_dir=None,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    ) 
    tokenizer.pad_token = tokenizer.unk_token # we already talked about it in train.py
    seg_token_idx, bop_token_idx, eop_token_idx = [
        tokenizer(token, add_special_tokens=False).input_ids[0] for token in ['[SEG]','<p>', '</p>'] # we load these since we need to load the model for inferecne (LINE 53)
    ]
   
    kwargs = {"torch_dtype": torch.bfloat16}    
    geo_model_args = {
        "vision_pretrained": 'facebook/sam2-hiera-large',
        "seg_token_idx" : seg_token_idx, # segmentation token index
        "bop_token_idx" : bop_token_idx, # begining of phrase token index
        "eop_token_idx" : eop_token_idx  # end of phrase token index
    }
    
    # Load model 
    print(f'Load model from: {args.version}')
    model = GeoPixelForCausalLM.from_pretrained(
        args.version, 
        low_cpu_mem_usage=True, 
        **kwargs,
        **geo_model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer
    
    model = model.bfloat16().cuda().eval()

    while True: # talk to the user for infinite number of times
        
        query = input("Please input your query: ")
        #query = f"Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases." 
        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image = [image_path]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            response, pred_masks = model.evaluate(tokenizer, query, images = image, max_new_tokens = 300) # our guess for TEXT_PART, having max_seq_len=374 seems to be correct
            #the model outputs two things: 1) Output Text that might contain <p>|</p>|[SEG] tokens. 2) Binary maks corresponding to [SEG] tokens.
        
        if pred_masks and '[SEG]' in response: # if there is no [SEG], then you are doing a simple VQA and you show output text-only to the user : even if there is <p>|</p>.
            pred_masks = pred_masks[0]  # In model.evaluate() we have surely loaded `E`(Nx256) and applied P_t and D to get all masks.
            pred_masks = pred_masks.detach().cpu().numpy()
            pred_masks = pred_masks > 0 # only consider the active part of masks (pixels > 1.0): NxHxWx1
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            save_img = image_np.copy()
            pattern = r'<p>(.*?)</p>\s*\[SEG\]'
            matched_text = re.findall(pattern, response)
            phrases = [text.strip() for text in matched_text]

            for i in range(pred_masks.shape[0]): #Iterate over each of those N masks
                mask = pred_masks[i] #HxWx1
                
                color = [random.randint(0, 255) for _ in range(3)]
                if matched_text:
                    phrases[i] = rgb_color_text(phrases[i], color[0], color[1], color[2])
                mask_rgb = np.stack([mask, mask, mask], axis=-1) #HxWx3
                color_mask = np.array(color, dtype=np.uint8) * mask_rgb

                save_img = np.where(mask_rgb, 
                        (save_img * 0.45 + color_mask * 0.65).astype(np.uint8), #alpha blending: Maybe it is better to replace it with (save_img * 0.0 + color_mask * 1.0)
                        save_img)
            if matched_text:    
                split_desc = response.split('[SEG]')
                cleaned_segments = [re.sub(r'<p>(.*?)</p>', '', part).strip() for part in split_desc]
                reconstructed_desc = ""
                for i, part in enumerate(cleaned_segments):
                    reconstructed_desc += part + ' '
                    if i < len(phrases):
                        reconstructed_desc += phrases[i] + ' '    
                print(reconstructed_desc)
            else:
                print(response.replace("\n", "").replace("  ", " "))
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            save_path = "{}/{}_masked.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0]
                )
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))
        else:
            print(response.replace("\n", "").replace("  ", " "))

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
