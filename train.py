# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import json
import sys
import os
import random
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
print("=======>STARTING....", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
import transformers
from accelerate.utils import DistributedType
from data_mix import Mix_dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from model.geopixel import GeoPixelForCausalLM

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

                                    ########IXC-2.5 is a VLM that has LLM and ViT inside (we use both). The ViT part is our I(.) vision encoder#########
                                                         #build_mlp.py for more information -> def build_vision_tower():#
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='') #IXC-2.5
    # GeoPixelModel arguments
    vision_pretrained: Optional[str] = field(default='facebook/sam2-hiera-large') # It seems that they use this as: I() and I_g() and D() in the paper.
    train_mask_decoder: bool = True xtract features for each channel.
The extracted features are then aggregated and reduced in
size using bilinear interpolation via the AnyRes block
    out_dim : int = 256
    ce_loss_weight : float = 1.0
    dice_loss_weight : float = 0.5
    bce_loss_weight : float = 2.0
    is_pretrained: bool = False

@dataclass
class DataArguments:
    data_path: str = field(
        default='data.txt', metadata={'help': 'Path to the training data.'})
    given_num: bool = False
    batch_size: int = 4   # in the paper effective bs=2x10 (batch_size x gradient_accumulation_steps)
    resolution: int = 560 # Bs=560
    hd_num: int = 18      # P = 9 in the paper (k1*k2 < P)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_length: int = field(
        default=8192, # I guess this is max_len of TEXT_PART. Because for the IMAGE_PART I computed it and it is a fixed length of 16010. 
                      # They set this to 16384 in train.sh. 16384-16010=374: maybe TEXT_MAX_LENGTH is 374. Might make sense.
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = True
    auto_resume: bool = False
    resume_dir: Optional[str] = field(default=None)
    start_epoch : int = field(default=0)
    label_names: List[str] = field(default_factory=lambda: ['samples'])

@dataclass
class LoraArguments:
    lora_r: int = 8  # DxK => Dxr * rxK    :   where r << D and r << K    
    lora_alpha: int = 16 # Delta_W = alpha * A * B. In the paper they say alpha=8
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'attention.wqkv', #QKV
        'attention.wo',   #O
        'feed_forward.w1',#MLP
        'feed_forward.w2',#MLP
        'feed_forward.w3',#MLP
    ])
    lora_weight_path: str = ''
    lora_bias: str = 'none'

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance['samples'] for instance in instances]
        text_input, data_type = tuple(
            [instance[key] for instance in instances]
            for key in ('text_input', 'data_type'))
        if 'image' not in instances[0]:
            text_input = [instance['text_input'][0] for instance in instances]
        batch = dict(
            text_input=text_input,
            data_type=data_type,
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image'] = images
            if 'masks' in instances[0]:
                batch['image_g'] = [instance['image_g'] for instance in instances]
                batch['ori_hw'] = [instance['ori_hw'] for instance in instances]
                batch['masks'] = [instance['masks'] for instance in instances]  
        return dict(samples=batch)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print('Loading data...')
    if data_args.data_path.endswith('json'):
        train_json = json.load(open(data_args.data_path))
    elif data_args.data_path.endswith('txt'): # we use data.txt and hence we enter here
        train_json = {}
        with open(data_args.data_path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.split(' ')
            with open(line[0]) as f:
                temp = json.load(f)
            if data_args.given_num:
                assert len(line) == 2
                num = int(float(line[1]) * 1000) #so <data.txt num> loads 'num k' (1000*k) data
                if len(temp) > num: # if the len(actual_data) is less than 1000k, sample 1000k randomly
                    temp = random.sample(temp, num)
                else:               # else it means that we already have len(actual_data) and for the rest we again get a repetitive data from actual_data
                    ex_temp = []
                    for i in range(num - len(temp)):
                        ex_temp.append(random.choice(temp))
                    temp.extend(ex_temp)
            else: # if we dont get num, and instead we get a ratio float number, we load ratio part of len(actual_data). Now if ratio > 1.0, do same as above
                if len(line) == 2:
                    ratio = float(line[1])
                    new_len = int(len(temp) * ratio)
                    if ratio < 1:
                        temp = random.sample(temp, new_len)
                    elif ratio > 1:
                        ex_temp = []
                        for i in range(new_len - len(temp)):
                            ex_temp.append(random.choice(temp))
                        temp.extend(ex_temp)
            rank0_print(f'Load {len(temp)} samples from {line}')
            train_json[line[0]] = temp # the final selected data is in train_json


    train_dataset = Mix_dataset( #w.r.t. train_json, load a mixed version of all datasets whose names are in train_json.keys()
        train_json,
        data_args.batch_size, #4
        resolution=data_args.resolution, #560
        hd_num=data_args.hd_num, #18
        local_rank=local_rank)
    print(str(len(train_dataset)) + ' samples are loaded')
    eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset, # This is important
        eval_dataset=eval_dataset,   # None
        data_collator=data_collator, # __call__ magic method
    )

class CustomTrainer(Trainer): # this is only for logging. It receives all losses and split it to different subparts: {ce + mask_bce + mask_dice + mask}
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        logs = {}
        if hasattr(outputs, 'ce_loss'):
            logs['ce_loss'] = outputs.ce_loss.detach().cpu().item()
        if hasattr(outputs, 'mask_bce_loss'):
            logs['mask_bce_loss'] = outputs.mask_bce_loss.detach().cpu().item()
        if hasattr(outputs, 'mask_dice_loss'):
            logs['mask_dice_loss'] = outputs.mask_dice_loss.detach().cpu().item()
        if hasattr(outputs, 'mask_loss'):
            logs['mask_loss'] = outputs.mask_loss.detach().cpu().item()

        self.log(logs) # this part is important

        return (loss, outputs) if return_outputs else loss # so basically we dont do anything we get and return the same thing
    
def train():
    global local_rank

    parser = transformers.HfArgumentParser( # only for argument parsing and nothing else, presented by Hugging Face
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    
    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED # GPU-related to speed up

    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained( # in RoPE we dont add PE to embeddings and instead rotate the embedding (still not learnable):   NO {x_i+PE_i}    YES {x_i.R^i_theta} 
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length #8192

    geo_model_args = {
        "vision_pretrained": model_args.vision_pretrained,   #SAM2
        "train_mask_decoder": model_args.train_mask_decoder, #True
        "out_dim": model_args.out_dim,                       #256 -> the output after P_t has dim=256 to be fed into Pixel_Decoder
        "ce_loss_weight": model_args.ce_loss_weight,         #1.0
        "dice_loss_weight": model_args.dice_loss_weight,     #0.5
        "bce_loss_weight": model_args.bce_loss_weight,       #2.0
    }

    # initializing tokenizer
    rank0_print(f'initialing tokenizer from: {model_args.model_name_or_path}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side='right',   # we do right padding
        use_fast=False,
        trust_remote_code=True, # The library downloads and executes Python code from the model repository on Hugging Face Hub.
    )
    tokenizer.pad_token = tokenizer.unk_token # for padding we use UNKNOWN token and not ZERO or PADD. {just naming}
    special_tokens = ['[SEG]','<p>', '</p>'] 
    '''
    Known categories is placed between <p> and </p>. Right after this known cetegory, we need to have [SEG] token for that.
    For an image and for each [SEG] token inside that we have ground truth masks in a .json file (NOW YOU UNDERSTAND BCE LOSS). 
    For approximating this gt mask, we use last embedding vector of [SEG] token (E) and feed it to P_t and D.
    Example:
        There is a <p> tennis court </p> [SEG] next to the <p> footbal court </p> [SEG].
    '''
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    seg_token_idx,bop_token_idx, eop_token_idx = [
        tokenizer(token, add_special_tokens=False).input_ids[0] for token in special_tokens
    ]

    geo_model_args.update({
        "seg_token_idx" : seg_token_idx, # segmentation token index        ===================> [SEG]
        "bop_token_idx" : bop_token_idx, # begining of phrase token index  ===================> <p>
        "eop_token_idx" : eop_token_idx  # end of phrase token index       ===================> </p>
    })

    torch_dtype = torch.float32
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.half

    # Load model and tokenizer
    rank0_print(f'Load model from: {model_args.model_name_or_path}')
    model = GeoPixelForCausalLM.from_pretrained(
        model_args.model_name_or_path,   #InternLM-XComposer-2.5 {IXC-2.5}: 7B parameters as our LLM
        config=config,
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        **geo_model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tokenizer = tokenizer

    if not model_args.is_pretrained:
        rank0_print(f'Initializing Vision Modules: {model_args.vision_pretrained}')
        model.model.initialize_geopixel_modules(model.config)

    if training_args.fix_vit:
        model.vit.requires_grad_(False) # freeze the ViT encoder
    else:
        model.vit.requires_grad_(True)
        model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity()

    if training_args.fix_sampler:
        model.vision_proj.requires_grad_(False)# it freezes P_v
    else:
        model.vision_proj.requires_grad_(True) # we do this bro in the paper! P_v is learnable

    if training_args.use_lora:

        for name, param in model.model.named_parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_args.lora_r, #8
            lora_alpha=lora_args.lora_alpha, #8 in the paper, 16 here
            target_modules=lora_args.lora_target_modules, #QKV-O + MLP + MLP + MLP
            lora_dropout=lora_args.lora_dropout, #0.05
            bias=lora_args.lora_bias, #none
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, lora_config) #uses PEFT library to apply LoRA on QKVO-MLP-MLP-MLP
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # make some modules trainable
    trainable_modules = ["output", "tok_embeddings", "sam_mask_decoder", "text_hidden_fcs"]
    for name, param in model.named_parameters():
        if any([ module in name for module in trainable_modules]):
            param.requires_grad = True
    
    model.resize_token_embeddings(len(tokenizer))
    if training_args.use_lora:
        model.print_trainable_parameters()
    model.to(torch.bfloat16)
    
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Data type: {param.dtype} | Trainable: {param.requires_grad} | Size: {param.size()}")

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args)
    transformers.processing_utils.logging.enable_progress_bar()
    
    # Start trainer
    trainer = CustomTrainer( # from Hugging Face 
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module, 
    )
    trainer.train()
    trainer.save_state() 

    global_step = trainer.state.global_step
    last_checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-last")
    os.makedirs(last_checkpoint_dir, exist_ok=True)

    trainer.model_wrapped.save_checkpoint(last_checkpoint_dir)
    trainer.save_model(last_checkpoint_dir)
    
    rank0_print(f"Final checkpoint saved at step {global_step} in 'checkpoint-last/'")

if __name__ == '__main__':
    train()
