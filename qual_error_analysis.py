import os
import sys
import logging
import json
import argparse
import tarfile
import io
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import webdataset as wds
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import math
import types

import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, get_model_config,get_input_dtype
from open_clip_train.main_distill_memory import (
    auto_layers_string,
    mp_parallelize_all,
    build_memory_args_automatically,
    load_weights_with_memory_layers,
)
from open_clip_train.data import get_data 
from open_clip.memory import ProductKeyArgs,HashingMemory
from open_clip_train.train_distill import evaluate,unwrap_model
from open_clip_train.precision import get_autocast
from open_clip.transformer import Attention

# from open_clip_train.distributed import init_distributed_device  # if you ever want multi‐GPU

def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--checkpoint",   required=True,
    #                help="path to epoch_X.pt")
    p.add_argument("--model",        default="ViT-B-16")
    p.add_argument("--pretrained",   default="openai")
    p.add_argument("--val-data",     default=None,
                   help="webdataset shards, e.g. '/…/val/000000.tar' or '*.tar'")
    p.add_argument("--val-num-samples", type=int, default=None)
    p.add_argument("--dataset-name", default=None,
                   help="sharegpt4v or mscoco")
    p.add_argument("--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )

    p.add_argument("--train-data",      default=None,
                   help="(unused here) placeholder so get_data() won’t crash")
    p.add_argument("--train-num-samples", type=int, default=None,
                   help="(unused) so get_data() can inspect train size if needed")
    p.add_argument("--imagenet-val",type=str,default=None,
                   help="Path to imagenet val set for conducting zero shot evaluation.")
    p.add_argument("--imagenet-v2",type=str,default=None,
                   help="Path to imagenet v2 for conducting zero shot evaluation.")
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--workers",      type=int, default=2)
    p.add_argument("--precision",    choices=["fp16","amp_bf16","fp32"], default="amp_bf16")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--K",            type=int,   default=5,
                   help="top-K considered a “hit”")
    p.add_argument("--top-n",        type=int,   default=10,
                   help="how many worst failures to plot")
    p.add_argument("--min-rank", type=int, default=1, 
                   help="Minimum rank of failures to save (1-based).")
    p.add_argument("--max-rank", type=int, default=None,
                   help="Maximum rank of failures to save (1-based).")
    p.add_argument("--teacher-ckpt", default=None,
                   help="If provided, load these weights into the teacher (else use HF pretrained)")
    p.add_argument("--student-ckpt", required=True,
                   help="Distilled student checkpoint (resume flag in your slurm)")
    p.add_argument("--student-epoch", type=int, default=0,
                   help="Epoch of the student checkpoint, used for evaluation purposes.")
    p.add_argument("--use-memory", action='store_true', default=False, 
                   help="Enabling memory layers inside Vision encoder of Student Model.")
    p.add_argument("--mem-n-keys", type=int, default=None,
                   help="Number of keys (mem_n_keys) for the memory config. Overrides default.")
    p.add_argument("--mem-v-dim", type=int, default=None,
                   help="Value dimension (mem_v_dim) for the memory config. Overrides default.")
    p.add_argument("--out-jsonl",    default="failures.jsonl")
    p.add_argument("--output-dir", type=str, default="./qual_error_analysis/",
        help="Where to store qual results. Use None to avoid storing logs.",
    )
    p.add_argument(
        "--eval-mode",
        choices=['webdataset', 'retrieval-framework'],
        default='webdataset',
        help="Choose the evaluation pipeline. 'webdataset' for .tar analysis, "
             "'retrieval-framework' for COCO/Flickr-style evaluation."
    )

    p.add_argument("--dict-root-dir",type=str,default=None,help="Path to the preprocessed dictionaries to filter the dataset.")
    p.add_argument("--coco-data-root-dir", type=str, default='', help="Root directory to the COCO dataset.")
    p.add_argument("--flickr-data-root-dir", type=str, default='', help="Root directory to the flickr datasets (but we simply use the root of the whole dataset).")
    p.add_argument("--sharegpt4v-retrieval-dir", type=str, default='', help="Root directory to the share4v dataset.")
    p.add_argument("--dci-retrieval-dir", type=str, default='', help="Root directory to the train dci dataset.")
    p.add_argument("--iiw-retrieval-dir", type=str, default='', help="Root directory to the image in words dataset.")
    p.add_argument("--docci-retrieval-dir", type=str, default='', help="Root directory to fine-grained docci retrieval.")
    p.add_argument("--urban-1k-retrieval-dir", type=str, default='', help="Root directory to fine-grained urban-1k retrieval.")
    p.add_argument("--retrieval-coco", action="store_true", default=False, help="Enable COCO retrieval task.")
    p.add_argument("--retrieval-dci", action="store_true", default=False, help="Enable DCI retrieval task.")
    p.add_argument("--retrieval-iiw", action="store_true", default=False, help="Enable IIW retrieval task.")
    p.add_argument("--retrieval-sharegpt4v-1k", action="store_true", default=False, help="Enable ShareGPT4V retrieval (1k size).")
    p.add_argument("--retrieval-sharegpt4v-10k", action="store_true", default=False, help="Enable ShareGPT4V retrieval (10k size).")
    p.add_argument("--retrieval-flickr", action="store_true", default=False, help="Enable Flickr retrieval task.")
    p.add_argument("--retrieval-urban-1k", action="store_true", default=False, help="Enable Urban-1k retrieval task.")
    p.add_argument("--retrieval-docci", action="store_true", default=False, help="Enable DOCCI retrieval task.")
    p.add_argument("--use_finegrained_iiw",default=True,action="store_true",
        help="If set to true, under the condition that we enable iiw, we further use the fine-grained iiw mode.")
    p.add_argument("--inference-with-flair", action='store_true', default=False, help="If set, use the FLAIR library for inference. This is only relevant if --baseline is set to 'flair'.")
    p.add_argument("--flickr-val-or-test",type=str,default='val', choices=['val', 'testing'],
        help="Which dataset to be used for inference, default choices are val or test.")
    p.add_argument(
        "--visualize-attention",
        action="store_true",
        help="Enable attention map visualization for specific examples found in the 'gap' analysis."
    )
    p.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="How many of the top 'gap' examples to generate visualizations for."
    )
    p.add_argument(
        "--attention-dir",
        type=str,
        default="attention_maps",
        help="Subdirectory within the output directory to save attention map plots."
    )


    return p.parse_args()


def init_teacher(args, device):
    teacher, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model, 
        args.pretrained,
        device=device, 
        precision=args.precision,
        output_dict=True,
        memory_args=None, # memory_args=None for teacher
    )
    if args.teacher_ckpt:
        sd = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(sd)

    teacher.to(device).eval()
    return teacher, preprocess_train, preprocess_val


def init_student(args, device):

    # 0) Load the checkpoint blob (train‐style or bare)
    assert args.student_ckpt, "Must pass --student-ckpt"
    ckpt_blob = torch.load(args.student_ckpt, map_location="cpu")
    sd = ckpt_blob.get("state_dict", ckpt_blob)

    # 1) Strip any leading "module." prefixes so we can compare apples→apples
    sd = { (k[len("module."): ] if k.startswith("module.") else k):v
           for k,v in sd.items() }

    # 2) Auto-detect whether this checkpoint truly contains memory layers
    checkpoint_has_mem = any("mlp.values" in k for k in sd.keys())
    use_mem = args.use_memory or checkpoint_has_mem
    if checkpoint_has_mem and not args.use_memory:
        logging.warning("Checkpoint has memory-layer keys → enabling memory_args")

    # 3) Build memory_args *based on use_mem*, *not* just args.use_memory
    memory_args = None
    if use_mem:
        cfg = get_model_config(args.model)
        num_layers = cfg["vision_cfg"]["layers"]
        memory_args = build_memory_args_automatically(num_layers)

        if args.mem_n_keys is not None:
            logging.info(f"Overriding default mem_n_keys with: {args.mem_n_keys}")
            memory_args.mem_n_keys = args.mem_n_keys
        
        if args.mem_v_dim is not None:
            logging.info(f"Overriding default mem_v_dim with: {args.mem_v_dim}")
            memory_args.mem_v_dim = args.mem_v_dim
            
        if memory_args.mem_share_values:
            HashingMemory.reset_shared_state()
        # logging.info(f"Building student with memory at layers: {memory_args.mem_layers}")


    # 4) Instantiate the student *with* or *without* those memory layers
    student, _, _ = open_clip.create_model_and_transforms(
        args.model,
        args.pretrained,
        device=device,
        precision=args.precision,
        output_dict=True,
        memory_args=memory_args,
    )

    if use_mem and memory_args.mem_share_values:
        logging.info("Applying rank-local memory sharing via mp_parallelize_all...")
        # You need to copy the mp_parallelize_all function as well
        student = mp_parallelize_all(student)
    
    # 5) Sanity check: compare what the model expects vs. what we loaded
    model_keys   = set(student.state_dict().keys())
    ckpt_keys    = set(sd.keys())
    missing = sorted(ckpt_keys - model_keys)
    extra   = sorted(model_keys - ckpt_keys)
    logging.info(f"After stripping 'module.':")
    logging.info(f"  Keys in checkpoint not in model: {missing[:10]}{' …' if len(missing)>10 else ''}")
    logging.info(f"  Keys in model not in checkpoint: {extra[:10]}{' …' if len(extra)>10 else ''}")

    # 6) Finally load the weights (strict=True so we catch any *real* mismatches)
    student.load_state_dict(sd, strict=True)
    logging.info(f"Loaded student checkpoint {args.student_ckpt!r} (use_memory={use_mem})")

    student.to(device).eval()
    return student,memory_args

def compare_mlp_vs_memory_parameters(teacher_model, student_model, memory_args):
    """
    Compares the parameter counts of a standard MLP block vs. a HashingMemory block
    and provides a summary of the architectural changes.
    """
    if not (memory_args and memory_args.is_enabled and memory_args.layers):
        logging.warning("Memory args not provided or enabled. Skipping parameter comparison.")
        return

    # --- Step 1: Calculate parameters for a single standard MLP ---
    mlp_params = 0
    # Find the first non-memory MLP block in the teacher to use as a reference
    for module in teacher_model.visual.transformer.resblocks:
        # Check if the mlp attribute is a Sequential block (the standard MLP)
        if isinstance(module.mlp, torch.nn.Sequential):
            for param in module.mlp.parameters():
                mlp_params += param.numel()
            break  # Found one, no need to check others
    
    if mlp_params == 0:
        logging.error("Could not find a standard MLP block in the teacher model.")
        return

    # --- Step 2: Calculate parameters for a single HashingMemory module ---
    memory_params_per_layer = 0
    # Find the first HashingMemory block in the student model
    for module in student_model.visual.transformer.resblocks:
        if isinstance(module.mlp, HashingMemory):
            for param in module.mlp.parameters():
                # This counts keys, query_proj, value_proj, etc. for one layer
                memory_params_per_layer += param.numel()
            break # Found one, no need to check others

    if memory_params_per_layer == 0:
        logging.error("Could not find a HashingMemory block in the student model.")
        return
        
    # Special handling for the SHARED `values` table
    shared_values_params = 0
    if memory_args.mem_share_values:
        # Find the 'values' table and get its parameters.
        # It's better to access it directly via the module to be sure.
        for module in student_model.modules():
            # Check for the first memory layer created (original) to count shared params once
            if isinstance(module, HashingMemory) and module.original:
                 if hasattr(module, 'values') and module.values is not None:
                     shared_values_params = sum(p.numel() for p in module.values.parameters())
                 break
        
        # The params counted in memory_params_per_layer included the shared values once.
        # We subtract it here to get the "per-layer overhead", then add it back once for the total.
        memory_params_per_layer -= shared_values_params

    # --- Step 3: Calculate totals based on the number of replaced layers ---
    try:
        from open_clip.transformer import parse_pk_layers
        # This part requires access to memory_args, which we build in init_student
        memory_layer_indices = parse_pk_layers(memory_args.layers)
        num_replaced_layers = len(memory_layer_indices)
    except (ImportError, TypeError, AttributeError):
        # Fallback parsing if memory_args is None or layers isn't a string
        num_replaced_layers = 0
        if memory_args and isinstance(memory_args.layers, str) and memory_args.layers:
             num_replaced_layers = len(list(map(int, memory_args.layers.split(','))))


    total_mlp_params_removed = mlp_params * num_replaced_layers
    
    # Total memory params = (per-layer overhead * num_layers) + one-time shared values
    total_memory_params_added = (memory_params_per_layer * num_replaced_layers) + shared_values_params if num_replaced_layers > 0 else 0


    # --- Step 4: Print the report ---
    print("\n" + "="*50)
    print("      MLP vs. Memory Parameter Comparison Report")
    print("="*50)
    if memory_args:
        print(f"Configuration: mem_n_keys={memory_args.mem_n_keys}, mem_v_dim={memory_args.mem_v_dim}\n")
    
    print(f"Parameters in one standard MLP block: {mlp_params:,}")
    print(f"Number of MLP blocks replaced:        {num_replaced_layers}")
    print("----------------------------------------------------")
    print(f"Total MLP parameters removed:         {total_mlp_params_removed:,}\n")

    print(f"Parameters in one Memory layer (per-layer overhead): {memory_params_per_layer:,}")
    if memory_args and memory_args.mem_share_values:
        print(f"Parameters in SHARED `values` table (counted once): {shared_values_params:,}")
    print("----------------------------------------------------")
    print(f"Total Memory parameters added:        {total_memory_params_added:,}\n")

    # --- Step 5: Final analysis ---
    param_change = total_memory_params_added - total_mlp_params_removed
    print("="*50)
    print(f"Net Parameter Change: {param_change:+,}")
    print("="*50 + "\n")

@torch.no_grad()
def encode_entire_split(model, val_loader, device, precision):
    all_I, all_T = [], []
    # all_keys, all_exts, all_caps = [], [], []

    model = model.to(device).eval()
    for (images, texts, *_) in tqdm(val_loader, desc="Eval pass"):
        # unpack exactly the tuple your tweaked pipeline yields:
        # images_b, texts_b, keys_b, exts_b = batch
        # images_b, texts_b = batch 
        images = images.to(device, non_blocking=True)
        texts  = texts.to(device, non_blocking=True)

        autocast_enabled = "amp" in precision or "bf16" in precision
        autocast_dtype = torch.bfloat16 if 'bf16' in precision else torch.float16
        with autocast(enabled=autocast_enabled, dtype=autocast_dtype):
        # with autocast(enabled=("amp" in precision or "bf16" in precision)):
            out = model(images, texts)
            I = out["image_features"]
            T = out["text_features"]
        # normalize on GPU then move to CPU
        I = F.normalize(I, dim=-1).cpu()
        T = F.normalize(T, dim=-1).cpu()

        all_I.append(I)
        all_T.append(T)

    I_all = torch.cat(all_I, dim=0)  # (N, D)
    T_all = torch.cat(all_T, dim=0)  # (N, D)
    # return I_all, T_all, all_keys, all_exts, all_caps
    return I_all, T_all

@torch.no_grad()
def encode_asymmetric_split(
    image_encoder_model, text_encoder_model, val_loader, device, precision):
    """
    Encodes the validation split using one model for images and another for text.
    This function replicates the asymmetric evaluation from the training script.
    """
    all_I, all_T = [], []
    image_encoder_model.to(device).eval()
    text_encoder_model.to(device).eval()

    # Determine autocast type
    autocast_enabled = 'amp' in precision or 'bf16' in precision
    autocast_dtype = torch.bfloat16 if 'bf16' in precision else torch.float16
    
    for (images, texts, *_) in tqdm(val_loader, desc="Asymmetric Eval Pass"):
        images = images.to(device) # Dtype will be handled by autocast or model
        texts  = texts.to(device)

        with autocast(enabled=autocast_enabled, dtype=autocast_dtype if autocast_enabled else None):
            # Use the specified model for each modality
            # This directly mirrors the logic in your evaluate() function
            image_features = image_encoder_model.encode_image(images).float()
            text_features  = text_encoder_model.encode_text(texts).float()

        # Normalize on GPU (in float32), then move to CPU
        I = F.normalize(image_features, dim=-1).cpu()
        T = F.normalize(text_features, dim=-1).cpu()

        all_I.append(I)
        all_T.append(T)

    I_all = torch.cat(all_I, dim=0)
    T_all = torch.cat(all_T, dim=0)
    return I_all, T_all

def create_ground_truth_mappings(data_list, image_paths):
    """
    Creates mappings between text indices and image indices.
    """
    # Create a fast lookup from image path to image index
    img_path_to_idx = {path: i for i, path in enumerate(image_paths)}
    
    text_to_image_map = {}  # Map text_idx -> correct_image_idx
    image_to_text_map = {}  # Map image_idx -> [list_of_correct_text_indices]

    for text_idx, item in enumerate(data_list):
        image_path = item['image']
        if image_path in img_path_to_idx:
            image_idx = img_path_to_idx[image_path]
            text_to_image_map[text_idx] = image_idx
            
            if image_idx not in image_to_text_map:
                image_to_text_map[image_idx] = []
            image_to_text_map[image_idx].append(text_idx)
            
    return text_to_image_map, image_to_text_map


def find_text2image_failures(I: torch.Tensor, T: torch.Tensor, K: int, text_to_image_map: dict):
    """
    For each text (j), find its GT image and check if it's in the top-K.
    """
    S = I @ T.t()  # [num_images, num_texts]
    num_texts = T.shape[0]
    fails = []
    for text_idx in range(num_texts):
        # Get the ground-truth image index for this text
        gt_image_idx = text_to_image_map.get(text_idx)
        if gt_image_idx is None:
            continue  # Skip if no mapping exists

        col = S[:, text_idx]  # Similarities of all images to this text
        topk_inds = torch.topk(col, K).indices.tolist()

        if gt_image_idx not in topk_inds:
            sorted_inds = torch.argsort(col, descending=True)
            rank = (sorted_inds == gt_image_idx).nonzero().item()
            fails.append({
                "idx": text_idx, # The index of the text query
                "gt_rank": rank + 1,
                "retrieved": [(i, float(col[i])) for i in topk_inds]
            })
    fails.sort(key=lambda x: x["gt_rank"], reverse=True)
    return fails


def find_image2text_failures(I: torch.Tensor, T: torch.Tensor, K: int, image_to_text_map: dict):
    """
    For each image (i), check if ANY of its GT captions are in the top-K.
    """
    S = I @ T.t()  # [num_images, num_texts]
    num_images = I.shape[0]
    fails = []
    for image_idx in range(num_images):
        # Get the list of ground-truth text indices for this image
        gt_text_indices = image_to_text_map.get(image_idx)
        if not gt_text_indices:
            continue

        row = S[image_idx, :]  # Similarities of this image to all texts
        topk_inds = torch.topk(row, K).indices.tolist()

        # Check if there is any overlap between retrieved and ground truth
        is_hit = any(gt_idx in topk_inds for gt_idx in gt_text_indices)

        if not is_hit:
            # Find the rank of the BEST-scoring ground-truth caption
            sorted_inds = torch.argsort(row, descending=True)
            ranks = [(sorted_inds == gt_idx).nonzero().item() for gt_idx in gt_text_indices]
            best_rank = min(ranks)
            fails.append({
                "idx": image_idx, # The index of the image query
                "gt_rank": best_rank + 1,
                "retrieved": [(j, float(row[j])) for j in topk_inds]
            })
    fails.sort(key=lambda x: x["gt_rank"], reverse=True)
    return fails


def find_retrieval_failures(I, T, K, text_to_image_map, image_to_text_map):
    return {
        "text2image": find_text2image_failures(I, T, K, text_to_image_map),
        "image2text": find_image2text_failures(I, T, K, image_to_text_map),
    }


def filter_failures_by_rank_window(fails, min_rank, max_rank=float('inf')):
    """
    Filters the list of failures to only include those where the
    ground-truth rank falls within a specific window [min_rank, max_rank].
    
    Args:
        fails (list): The original list of failure tuples.
        min_rank (int): The minimum rank (inclusive) to include. Note that the
                        'rank' in the tuple is 0-indexed, so we add 1 for comparison.
        max_rank (int): The maximum rank (inclusive) to include.

    Returns:
        list: A new list of filtered failure tuples.
    """
    filtered = []
    for j, rank, topk in fails:
        # The rank from find_retrieval_failures is 0-indexed. 
        # User-facing ranks are usually 1-indexed. Let's use 1-indexed for clarity.
        gt_rank = rank + 1
        if min_rank <= gt_rank <= max_rank:
            filtered.append((j, rank, topk))
            
    # The 'fails' list is already sorted by worst rank first, so the
    # filtered list will also be sorted correctly.
    return filtered


def dump_failures(fails, keys, caps, out_jsonl):
    with open(out_jsonl, "w") as f:
        for entry in fails:
            j = entry["idx"]
            retrieved_with_context = []
            # 'retrieved' is already a list of tuples (idx, score)
            for r_idx, r_score in entry["retrieved"]:
                retrieved_with_context.append({
                    "idx": r_idx, "key": keys[r_idx], "caption": caps[r_idx], "score": r_score
                })
            
            f.write(json.dumps({
                "idx": j,
                "key": keys[j],
                "caption": caps[j],
                "gt_rank": entry["gt_rank"],
                "retrieved": retrieved_with_context
            }) + "\n")

def plot_failures(fails, keys, caps, val_data_tar, top_n, K):
    # reopen your validation .tar so we can pull out both images AND captions
    tar = tarfile.open(val_data_tar, "r")
    n = min(len(fails), top_n)
    fig, axes = plt.subplots(n, K+2, figsize=(3*(K+2), 3*n))
    for i in range(n):
        j, rank, topk = fails[i]
        txt = caps[j]

        # col 0: caption
        axes[i,0].axis("off")
        axes[i,0].text(0, 0.5, f"{i+1}. {txt}", wrap=True)

        # col 1: GT image
        # member = tar.getmember(f"{keys[j]}.{exts[j]}")
        member = tar.getmember(f"{keys[j]}.jpg")
        data = tar.extractfile(member).read()
        gt = Image.open(io.BytesIO(data)).convert("RGB")
        axes[i,1].imshow(gt)
        axes[i,1].set_title(f"GT (rank {rank+1})")
        axes[i,1].axis("off")

        # cols 2 … K+1: retrieved
        for c,(rid,score) in enumerate(topk, start=2):
            # mem = tar.getmember(f"{keys[rid]}.{exts[rid]}")
            mem = tar.getmember(f"{keys[rid]}.jpg")
            bd = tar.extractfile(mem).read()
            img = Image.open(io.BytesIO(bd)).convert("RGB")
            axes[i,c].imshow(img)
            axes[i,c].set_title(f"#{c-1} ({score:.2f})")
            axes[i,c].axis("off")

    plt.tight_layout()
    plt.show()
    tar.close()




# class AttentionVisualizerWrapper:
#     """A wrapper for a CLIP model to capture attention maps by monkey-patching."""
#     def __init__(self, model):
#         self.model = model
#         self.hooks = []
#         self.attention_maps = []

#     def __enter__(self):
#         self.hooks = []
#         self.attention_maps = []
        
#         def new_attention_forward(module, x):
#             qkv = module.in_proj(x).reshape(x.shape[0], x.shape[1], 3, -1).permute(2, 0, 1, 3)
#             q, k, v = qkv[0], qkv[1], qkv[2]
            
#             attn_output, attn_weights = F.multi_head_attention_forward(
#                 query=q, key=k, value=v,
#                 embed_dim_to_check=module.out_proj.in_features,
#                 num_heads=module.num_heads,
#                 in_proj_weight=None, in_proj_bias=None,
#                 bias_k=None, bias_v=None,
#                 add_zero_attn=False,
#                 dropout_p=0.0,
#                 out_proj_weight=module.out_proj.weight,
#                 out_proj_bias=module.out_proj.bias,
#                 training=False,
#                 need_weights=True,
#                 attn_mask=module.attn_mask,
#             )
#             self.attention_maps.append(attn_weights.detach().cpu())
#             return module.out_proj(attn_output)

#         for module in self.model.visual.modules():
#             if isinstance(module, Attention):
#                 original_forward = module.forward
#                 module.forward = new_attention_forward.__get__(module, Attention)
#                 self.hooks.append((module, original_forward))
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         for module, original_forward in self.hooks:
#             module.forward = original_forward
#         self.hooks = []

#     def get_attention(self, image_tensor):
#         self.attention_maps = []
#         self.model.encode_image(image_tensor.unsqueeze(0))
#         return self.attention_maps

# # class AttentionVisualizerWrapper:
#     """
#     A wrapper for a CLIP model to capture attention maps by monkey-patching
#     the forward method with a manual attention calculation. This bypasses
#     any optimized kernels that might ignore the need_weights=True flag.
#     """

#     def __init__(self, model):
#         self.model = model
#         self.hooks = []
#         self.attention_maps = []

#     def __enter__(self):
#         self.hooks = []
#         self.attention_maps = []

#         # This is our new, robust forward method
#         def new_attention_forward(module, x):
#             # x.shape: [seq_len, batch_size, embed_dim]
#             seq_len, batch_size, embed_dim = x.shape
            
#             # 1. Get Q, K, V from the input projection
#             qkv = module.in_proj(x)
#             qkv = qkv.reshape(seq_len, batch_size, 3, embed_dim).permute(2, 1, 0, 3)
#             q, k, v = qkv[0], qkv[1], qkv[2] # Shape: [batch_size, seq_len, embed_dim]

#             # 2. Reshape for multi-head attention
#             num_heads = module.num_heads
#             head_dim = embed_dim // num_heads
            
#             q = q.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
#             k = k.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 3, 1) # Transpose for matmul
#             v = v.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            
#             # 3. Manual Attention Calculation
#             scale = head_dim ** -0.5
#             attn_weights = (q @ k) * scale
            
#             # Apply attention mask if it exists
#             if module.attn_mask is not None:
#                 # attn_mask shape is [tgt_len, src_len]
#                 # attn_weights shape is [batch_size, num_heads, tgt_len, src_len]
#                 # We need to broadcast the mask
#                 attn_weights += module.attn_mask

#             attn_weights = F.softmax(attn_weights, dim=-1)
            
#             # --- THIS IS THE CRITICAL PART ---
#             # We have computed the weights ourselves, so we can save them.
#             self.attention_maps.append(attn_weights.detach().cpu())
            
#             # 4. Apply attention to values
#             attn_output = (attn_weights @ v.permute(0, 1, 3, 2).transpose(-2,-1)).permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
            
#             # 5. Final output projection
#             attn_output = attn_output.permute(1, 0, 2) # Back to [seq_len, batch_size, embed_dim]
#             return module.out_proj(attn_output)

#         # Find all Attention modules in the visual tower and replace their forward pass
#         for module in self.model.visual.modules():
#             if isinstance(module, Attention):
#                 original_forward = module.forward
#                 module.forward = new_attention_forward.__get__(module, Attention)
#                 self.hooks.append((module, original_forward))
        
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # Restore all the original forward methods
#         for module, original_forward in self.hooks:
#             module.forward = original_forward
#         self.hooks = []

#     def get_attention(self, image_tensor):
#         self.attention_maps = []
#         # Ensure the model is in eval mode for this pass
#         self.model.eval()
#         self.model.encode_image(image_tensor.unsqueeze(0))
#         return self.attention_maps

class AttentionVisualizerWrapper:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_maps = []

    def __enter__(self):
        self.hooks = []
        self.attention_maps = []

        # Patch every ResidualAttentionBlock.attn (a MultiheadAttention)
        for block in self.model.visual.transformer.resblocks:
            attn_mod    = block.attn
            orig_forward = attn_mod.forward

            # Create a patched forward that always returns (out, weights)
            # def patched_forward(this, *args, **kwargs):
            #     # Force PyTorch to return the weights tuple
            #     kwargs = dict(kwargs, need_weights=True)
            #     out, weights = orig_forward(*args, **kwargs)
            #     # Store a CPU copy
            #     self.attention_maps.append(weights.detach().cpu())
            #     return out, weights
            def patched_forward(this, *args, **kwargs):
                # 1) always ask for weights, and do NOT average across heads
                kwargs = dict(kwargs, need_weights=True, average_attn_weights=False)
                out, weights = orig_forward(*args, **kwargs)
                # 2) save the full [B,heads,T,S] map
                self.attention_maps.append(weights.detach().cpu())
                return out, weights

            # Bind it to the instance
            attn_mod.forward = types.MethodType(patched_forward, attn_mod)
            # Remember for cleanup
            self.hooks.append((attn_mod, orig_forward))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original forward methods
        for attn_mod, orig in self.hooks:
            attn_mod.forward = orig
        self.hooks = []

    def get_attention(self, image_tensor):
        """
        Runs a single forward pass of encode_image() and returns the
        list of [layers x heads x tokens x tokens] attention weight tensors.
        """
        # Clear previous
        self.attention_maps = []
        self.model.eval()
        # This will trigger all the patched forwards
        _ = self.model.encode_image(image_tensor.unsqueeze(0))
        return self.attention_maps



def visualize_comparative_attention(
    image_pil, text_query,
    teacher_attention_maps, student_attention_maps,
    output_path, filename_prefix):
    """
    Visualizes and compares Teacher vs. Student attention maps for a single example.
    """
    os.makedirs(output_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Attention Comparison for Query: \"{text_query}\"", fontsize=16, y=0.97)

    # --- Helper function to process and plot one row ---
    def plot_row(ax_row, attention_maps, model_name):
        if not attention_maps:
            logging.warning(f"No attention maps for {model_name}, skipping row.")
            ax_row[0].set_title(f"{model_name} (No Maps)")
            for ax in ax_row: ax.axis('off')
            return

        last_layer_attention = attention_maps[-1].cpu()
        cls_attention = last_layer_attention.mean(dim=1)[0, 0, 1:].detach()
        
        num_patches = cls_attention.shape[0]
        grid_size = int(math.sqrt(num_patches))
        
        if grid_size * grid_size != num_patches:
            logging.warning(f"Cannot form a square grid from {num_patches} patches. Skipping viz for {model_name}.")
            ax_row[0].set_title(f"{model_name} (Grid Error)")
            for ax in ax_row: ax.axis('off')
            return
            
        attention_grid = cls_attention.reshape(grid_size, grid_size)
        
        resized_attention = F.interpolate(
            attention_grid.unsqueeze(0).unsqueeze(0),
            size=image_pil.size, mode='bilinear', align_corners=False
        ).squeeze().numpy()

        ax_row[0].imshow(image_pil)
        ax_row[0].set_title(f"{model_name}: Original Image")
        ax_row[0].axis('off')

        ax_row[1].imshow(resized_attention, cmap='jet')
        ax_row[1].set_title(f"{model_name}: Attention Heatmap")
        ax_row[1].axis('off')

        ax_row[2].imshow(image_pil)
        ax_row[2].imshow(resized_attention, cmap='jet', alpha=0.5)
        ax_row[2].set_title(f"{model_name}: Overlay")
        ax_row[2].axis('off')

    # --- Plot Teacher (FAIL) and Student (SUCCESS) rows ---
    plot_row(axes[0], teacher_attention_maps, "Teacher (FAIL)")
    plot_row(axes[1], student_attention_maps, "Student (SUCCESS)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # --- DYNAMIC FILENAME and TEXT FILE LOGIC ---
    # Create a safe base filename from the prefix
    safe_base_name = "".join([c if c.isalnum() else "_" for c in filename_prefix])
    
    # Save the PNG image
    png_filename = os.path.join(output_path, f"{safe_base_name}.png")
    plt.savefig(png_filename)
    plt.close(fig)
    logging.info(f"Saved comparative attention visualization to {png_filename}")

    # Save the corresponding query in a .txt file
    txt_filename = os.path.join(output_path, f"{safe_base_name}_query.txt")
    with open(txt_filename, 'w') as f:
        f.write(text_query)

def run_attention_visualization(
    student_model, teacher_model,
    gap_indices, text_to_image_map,
    image_paths, text_captions, preprocess_fn, args,dataset_key):
    """
    Generates and saves comparative attention maps for top text2image gap examples.
    """
    logging.info("\n--- Generating Comparative Attention Map Visualizations ---")
    
    # Create a wrapper for each model
    # student_viz_wrapper = AttentionVisualizerWrapper(student_model)
    # teacher_viz_wrapper = AttentionVisualizerWrapper(teacher_model)
    
    clean_dataset_name = dataset_key.replace('retrieval_', '').replace('-', '_')
    # Construct the final output directory path
    viz_dir = os.path.join(args.output_dir, args.attention_dir, clean_dataset_name)
    os.makedirs(viz_dir, exist_ok=True)

    device = torch.device(args.device)
    student_model.to(device)
    teacher_model.to(device)

    # ----------------------------------
    # Get the correct input dtype based on the precision argument
    # autocast = get_autocast(args.precision)

    # Get a list of the student's memory layers
    # student_memory_layers = [
    #     m for m in student_model.modules() if isinstance(m, (HashingMemory, nn.EmbeddingBag))
    # ]

    original_student_dtype = next(student_model.parameters()).dtype
    # ---------------------------

    num_to_viz = min(len(gap_indices), args.num_visualizations)
    logging.info(f"Visualizing top {num_to_viz} text2image gap examples...")

    for i, text_idx in enumerate(gap_indices[:num_to_viz]):
        gt_image_idx = text_to_image_map.get(text_idx)
        if gt_image_idx is None: continue

        image_path = image_paths[gt_image_idx]
        text_query = text_captions[text_idx]
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
            image_tensor = preprocess_fn(image_pil).to(device)

            
            # Get attention maps from BOTH models for the SAME image
            teacher_model.float() # Ensure teacher is in float32
            with torch.no_grad(): 
                with AttentionVisualizerWrapper(teacher_model) as wrapper:
                    teacher_maps = wrapper.get_attention(image_tensor)
                
            # for layer in student_memory_layers:
            #     layer.float()
            student_model.float()
            with torch.no_grad():
                # Run the student forward pass in float32 to avoid the autocast bug
                with AttentionVisualizerWrapper(student_model) as wrapper:
                    student_maps = wrapper.get_attention(image_tensor)
            
            # --- Restore original precision for student model ---
            # student_model.to(dtype=get_input_dtype(args.precision) or torch.float32)
            student_model.to(dtype=original_student_dtype)

            # --- 3. Plotting ---
            prefix = f"gap_example_{i+1:02d}_{os.path.basename(image_path)}"
            visualize_comparative_attention(
                image_pil, text_query,
                teacher_maps, student_maps,
                viz_dir, prefix
            )

        except Exception as e:
            logging.error(f"Failed to generate visualization for {image_path}: {e}", exc_info=True)



def save_gap_comparison_to_folders(
    mode, gap_indices, teacher_fails_data,
    I_student, T_teacher_student,
    keys, caps, val_tar_path, K, top_n, output_dir):
    """
    Saves a comparison for 'gap' examples to a structured folder layout.
    For each example, it creates a directory containing:
    - The query (image or text).
    - The ground truth (image or text).
    - A subfolder for the Teacher's FAILED retrieval.
    - A subfolder for the Student's SUCCESSFUL retrieval.
    """
    if not gap_indices:
        logging.warning(f"No 'gap' examples found for mode '{mode}'. Skipping save.")
        return

    n_to_save = min(len(gap_indices), top_n)
    logging.info(f"Saving top {n_to_save} '{mode}' gap examples to individual folders...")

    # The main directory for this analysis type, e.g., ".../gap_image2text/"
    main_mode_dir = os.path.join(output_dir, f"gap_{mode}")
    os.makedirs(main_mode_dir, exist_ok=True)

    tar = tarfile.open(val_tar_path, "r")
    S_student = I_student @ T_teacher_student.t()

    for i, idx in enumerate(gap_indices[:n_to_save]):
        # --- 1. Gather Data for this Example ---
        teacher_entry = next(f for f in teacher_fails_data if f["idx"] == idx)
        
        # Ground truth info
        gt_key = keys[idx]
        gt_caption = caps[idx]

        # Teacher's failed retrieval
        teacher_rank = teacher_entry["gt_rank"]
        teacher_retrieved = teacher_entry["retrieved"] # List of (idx, score)

        # Student's successful retrieval (calculated on the fly)
        if mode == 'text2image':
            scores = S_student[:, idx]
        else: # image2text
            scores = S_student[idx, :]
        
        sorted_inds = torch.argsort(scores, descending=True)
        student_rank = (sorted_inds == idx).nonzero().item() + 1
        student_topk_inds = sorted_inds[:K].tolist()
        student_retrieved = [(r_idx, scores[r_idx].item()) for r_idx in student_topk_inds]

        # --- 2. Create Directory Structure ---
        example_dir = os.path.join(main_mode_dir, f"{i+1:02d}_{gt_key}")
        teacher_dir = os.path.join(example_dir, "teacher_retrieval_FAIL")
        student_dir = os.path.join(example_dir, "student_retrieval_SUCCESS")
        os.makedirs(teacher_dir, exist_ok=True)
        os.makedirs(student_dir, exist_ok=True)
        
        # --- 3. Save Files Based on Retrieval Mode ---
        if mode == 'text2image':
            # The query is a caption
            with open(os.path.join(example_dir, "query_caption.txt"), "w") as f:
                f.write(f"Query Caption:\n\n{gt_caption}\n\n")
                f.write("--- Ranks ---\n")
                f.write(f"Teacher (FAIL): Ground truth image was rank {teacher_rank}\n")
                f.write(f"Student (SUCCESS): Ground truth image is rank {student_rank}\n")
            
            # The ground truth is an image
            gt_member = tar.getmember(f"{gt_key}.jpg")
            gt_img = Image.open(io.BytesIO(tar.extractfile(gt_member).read())).convert("RGB")
            gt_img.save(os.path.join(example_dir, "ground_truth_image.jpg"))

            # Save retrieved images for both models
            for rank_num, (ret_idx, score) in enumerate(teacher_retrieved, 1):
                ret_key = keys[ret_idx]
                img = Image.open(io.BytesIO(tar.extractfile(f"{ret_key}.jpg").read())).convert("RGB")
                img.save(os.path.join(teacher_dir, f"rank_{rank_num:02d}_score_{score:.3f}_{ret_key}.jpg"))
            
            for rank_num, (ret_idx, score) in enumerate(student_retrieved, 1):
                ret_key = keys[ret_idx]
                img = Image.open(io.BytesIO(tar.extractfile(f"{ret_key}.jpg").read())).convert("RGB")
                img.save(os.path.join(student_dir, f"rank_{rank_num:02d}_score_{score:.3f}_{ret_key}.jpg"))

        elif mode == 'image2text':
            # The query is an image
            query_img_member = tar.getmember(f"{gt_key}.jpg")
            query_img = Image.open(io.BytesIO(tar.extractfile(query_img_member).read())).convert("RGB")
            query_img.save(os.path.join(example_dir, "query_image.jpg"))
            
            # The ground truth is a caption
            with open(os.path.join(example_dir, "ground_truth_caption.txt"), "w") as f:
                f.write(f"Ground Truth Caption:\n\n{gt_caption}\n\n")
                f.write("--- Ranks ---\n")
                f.write(f"Teacher (FAIL): This caption was rank {teacher_rank}\n")
                f.write(f"Student (SUCCESS): This caption is rank {student_rank}\n")

            # Save retrieved captions for both models into summary text files
            with open(os.path.join(teacher_dir, "retrieved_captions.txt"), "w") as f:
                f.write("--- Teacher's Top-5 Retrieved Captions (FAIL) ---\n\n")
                for rank_num, (ret_idx, score) in enumerate(teacher_retrieved, 1):
                    f.write(f"Rank #{rank_num} (Score: {score:.3f}) ---\n{caps[ret_idx]}\n\n")

            with open(os.path.join(student_dir, "retrieved_captions.txt"), "w") as f:
                f.write("--- Student's Top-5 Retrieved Captions (SUCCESS) ---\n\n")
                for rank_num, (ret_idx, score) in enumerate(student_retrieved, 1):
                    f.write(f"Rank #{rank_num} (Score: {score:.3f}) ---\n{caps[ret_idx]}\n\n")

    tar.close()
    logging.info(f"Successfully saved gap comparison examples to '{main_mode_dir}'")

def load_failures(path):
    """Returns a list of dicts, each with keys 'idx','gt_rank','retrieved',…"""
    return [json.loads(l) for l in open(path, "r")]

def save_failures(mode, fails, keys, caps, val_data_tar, top_n, K, out_dir):
    """
    Instead of plotting, dump the top_n failures into folders under out_dir.
    Each folder is named "{i+1:02d}_{keys[j]}" and contains:
      - caption.txt
      - gt.jpg
      - retrieved_1.jpg ... retrieved_K.jpg
    """
    
    if mode not in ["text_to_image", "image_to_text"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'text_to_image' or 'image_to_text'.")

    # 1) open the shard
    tar = tarfile.open(val_data_tar, "r")

    # 2) make sure out_dir exists
    os.makedirs(out_dir, exist_ok=True)

    # 3) Iterate through the worst failures
    n = min(len(fails), top_n)
    for i in range(n):
        # 'j' is the index of the ground-truth query item
        j, rank, topk = fails[i]
        gt_key = keys[j]
        gt_caption = caps[j]

        # 3a) Make case directory, named after the ground-truth item's key
        case_dir = os.path.join(out_dir, f"{i+1:02d}_{gt_key}")
        os.makedirs(case_dir, exist_ok=True)

        if mode == "text_to_image":
            # Save the ground-truth query text
            with open(os.path.join(case_dir, "query_caption.txt"), "w", encoding="utf-8") as f:
                f.write(f"GT Rank: {rank+1}\n\n{gt_caption}")
            
            # Save the ground-truth target image
            member = tar.getmember(f"{gt_key}.jpg")
            data = tar.extractfile(member).read()
            gt_img = Image.open(io.BytesIO(data)).convert("RGB")
            gt_img.save(os.path.join(case_dir, "ground_truth_image.jpg"))
            
            # Save the top-K *retrieved images*
            for rank_idx, (retrieved_idx, score) in enumerate(topk, start=1):
                ret_key = keys[retrieved_idx]
                ret_member = tar.getmember(f"{ret_key}.jpg")
                ret_data   = tar.extractfile(ret_member).read()
                ret_img    = Image.open(io.BytesIO(ret_data)).convert("RGB")
                filename = f"retrieved_{rank_idx:02d}_img_{ret_key}_score_{score:.3f}.jpg"
                ret_img.save(os.path.join(case_dir, filename))

        elif mode == "image_to_text":
            # Save the ground-truth query image
            member = tar.getmember(f"{gt_key}.jpg")
            data = tar.extractfile(member).read()
            gt_img = Image.open(io.BytesIO(data)).convert("RGB")
            gt_img.save(os.path.join(case_dir, "query_image.jpg"))

            # Save the ground-truth target caption
            with open(os.path.join(case_dir, "ground_truth_caption.txt"), "w", encoding="utf-8") as f:
                f.write(f"GT Rank: {rank+1}\n\n{gt_caption}")

            # Save the top-K *retrieved captions* as text files
            retrieved_captions_content = []
            for rank_idx, (retrieved_idx, score) in enumerate(topk, start=1):
                ret_caption_text = caps[retrieved_idx]
                retrieved_captions_content.append(
                    f"--- Rank #{rank_idx} (Score: {score:.3f}) ---\n{ret_caption_text}\n"
                )
            
            with open(os.path.join(case_dir, "retrieved_captions.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(retrieved_captions_content))

    tar.close()
    print(f"Saved top {n} '{mode}' failures under {out_dir}")

def save_gap_comparison_from_disk(
    mode, gap_indices, teacher_fails_data,
    I_student, I_teacher, T_teacher,
    img_keys, text_captions, K, top_n, output_dir,
    text_to_image_map, image_to_text_map ):
    """Saves gap analysis by reading images directly from their full paths."""
    if not gap_indices:
        logging.warning(f"No 'gap' examples found for mode '{mode}'. Skipping save.")
        return

    n_to_save = min(len(gap_indices), top_n)
    logging.info(f"Saving top {n_to_save} '{mode}' gap examples from disk...")

    main_mode_dir = os.path.join(output_dir, f"gap_{mode}")
    os.makedirs(main_mode_dir, exist_ok=True)

    S_student = I_student @ T_teacher.t()
    S_teacher = I_teacher @ T_teacher.t()

    # The loop variable is the index of the QUERY (text_idx for t2i, image_idx for i2t)
    for i, query_idx in enumerate(gap_indices[:n_to_save]):
        teacher_entry = next((f for f in teacher_fails_data if f["idx"] == query_idx), None)
        if teacher_entry is None: continue

        # --- THE CORE FIX: Use the mappings to find the ground truth ---
        if mode == 'text2image':
            text_idx = query_idx
            gt_image_idx = text_to_image_map.get(text_idx)
            if gt_image_idx is None: continue
            
            gt_img_path = img_keys[gt_image_idx]
            gt_caption = text_captions[text_idx]
            
            # Student's successful retrieval
            scores_student = S_student[:, text_idx]
            sorted_inds = torch.argsort(scores_student, descending=True)
            student_rank = (sorted_inds == gt_image_idx).nonzero().item() + 1
            student_topk_inds = sorted_inds[:K].tolist()

            # --- Teacher's failed retrieval ---
            scores_teacher = S_teacher[:, text_idx]
            teacher_topk_inds = torch.argsort(scores_teacher, descending=True)[:K].tolist()


        elif mode == 'image2text':
            image_idx = query_idx
            gt_text_indices = image_to_text_map.get(image_idx)
            if not gt_text_indices: continue
            
            gt_img_path = img_keys[image_idx]
            # For simplicity, we'll just show the first ground-truth caption
            gt_caption = text_captions[gt_text_indices[0]]

            # Student's successful retrieval
            scores_student = S_student[image_idx, :]
            sorted_inds = torch.argsort(scores_student, descending=True)
            student_ranks = [(sorted_inds == gt_idx).nonzero().item() for gt_idx in gt_text_indices]
            student_rank = min(student_ranks) + 1
            student_topk_inds = sorted_inds[:K].tolist()

             # --- Teacher's failed retrieval ---
            scores_teacher = S_teacher[image_idx, :]
            teacher_topk_inds = torch.argsort(scores_teacher, descending=True)[:K].tolist()


        # --- The rest of the saving logic remains the same ---
        base_name = os.path.splitext(os.path.basename(gt_img_path))[0]
        example_dir = os.path.join(main_mode_dir, f"{i+1:02d}_{base_name}")
        teacher_dir = os.path.join(example_dir, "teacher_retrieval_FAIL")
        student_dir = os.path.join(example_dir, "student_retrieval_SUCCESS")
        os.makedirs(teacher_dir, exist_ok=True)
        os.makedirs(student_dir, exist_ok=True)
        
        if mode == 'text2image':
            with open(os.path.join(example_dir, "query_caption.txt"), "w") as f:
                f.write(f"Query: {gt_caption}\nTeacher Rank: {teacher_entry['gt_rank']}\nStudent Rank: {student_rank}")
            Image.open(gt_img_path).convert("RGB").save(os.path.join(example_dir, "ground_truth_image.jpg"))
            
            # Save TEACHER's retrieved images
            for rank, ret_idx in enumerate(teacher_topk_inds, 1):
                score = scores_teacher[ret_idx].item()
                Image.open(img_keys[ret_idx]).convert("RGB").save(os.path.join(teacher_dir, f"rank_{rank:02d}_score_{score:.3f}_{os.path.basename(img_keys[ret_idx])}"))

            # Save STUDENT's retrieved images
            for rank, ret_idx in enumerate(student_topk_inds, 1):
                score = scores_student[ret_idx].item()
                Image.open(img_keys[ret_idx]).convert("RGB").save(os.path.join(student_dir, f"rank_{rank:02d}_score_{score:.3f}_{os.path.basename(img_keys[ret_idx])}"))

        elif mode == 'image2text':
            Image.open(gt_img_path).convert("RGB").save(os.path.join(example_dir, "query_image.jpg"))
            with open(os.path.join(example_dir, "ground_truth_caption.txt"), "w") as f:
                f.write(f"GT: {gt_caption}\nTeacher Rank: {teacher_entry['gt_rank']}\nStudent Rank: {student_rank}")

            # Save TEACHER's retrieved captions
            with open(os.path.join(teacher_dir, "retrieved_captions.txt"), "w") as f:
                 f.write(f"--- Teacher's Top-{K} Retrieved Captions (FAIL) ---\n\n")
                 for rank, ret_idx in enumerate(teacher_topk_inds, 1):
                     score = scores_teacher[ret_idx].item()
                     f.write(f"Rank #{rank} (Score: {score:.3f}) ---\n{text_captions[ret_idx]}\n\n")

            # Save STUDENT's retrieved captions
            with open(os.path.join(student_dir, "retrieved_captions.txt"), "w") as f:
                 f.write(f"--- Student's Top-{K} Retrieved Captions (SUCCESS) ---\n\n")
                 for rank, ret_idx in enumerate(student_topk_inds, 1):
                     score = scores_student[ret_idx].item()
                     f.write(f"Rank #{rank}: (Score: {score:.3f}) ---\n{text_captions[ret_idx]}\n\n")


def main_webdataset_analysis(args):
    """The original analysis logic for webdataset .tar files."""
    logging.info("Running in 'webdataset' mode.")
    if not args.val_data:
        raise ValueError("--val-data is required for webdataset mode.")

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.model)
    teacher, preprocess_train, preprocess_val = init_teacher(args, device)
    # 2) build data via get_data()
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=0,
        tokenizer=tokenizer,
    )

    val_loader = data["val"].dataloader

    # # 3) encode everything
    # I_all, T_all = encode_entire_split(
    #     model, val_loader, args.device, args.precision
    # )

    # # 2) open the .tar ONCE to build keys & caps lists
    tar = tarfile.open(args.val_data, "r")
    keys = []
    caps = []
    for mem in tar.getmembers():
        # pick only .jpg entries, in archive order
        if mem.name.endswith(".jpg"):
            stem = mem.name.rsplit(".",1)[0]
            keys.append(stem)
            # grab its caption
            txt_mem = tar.getmember(f"{stem}.txt")
            txt = tar.extractfile(txt_mem).read().decode("utf-8").strip()
            caps.append(txt)
    tar.close()


    # # --- TEACHER (Symmetric) EVALUATION ---
    # # The teacher is evaluated against itself.
    logging.info("Starting symmetric evaluation for the Teacher model...")
    I_teacher, T_teacher = encode_entire_split(teacher, val_loader, device, args.precision)
    fails_teacher = find_retrieval_failures(I_teacher, T_teacher, args.K)
    print("Teacher symmetric evaluation complete.")

    # # --- STUDENT (Asymmetric) EVALUATION ---
    logging.info("Calling the official 'evaluate' function in return_mode to get features...")
    
    # We pass a dummy epoch (e.g., 38) to satisfy the function signature.
    # The val_frequency check inside evaluate() needs to pass.
    # Set args.val_frequency to 1 to guarantee it runs.
    args.rank = 0
    args.val_frequency = 1 
    student, memory_args = init_student(args, device)

    # # # --- COMPARISON FUNCTION ---
    # # # This will print the detailed report to your console.
    # # compare_mlp_vs_memory_parameters(teacher, student, memory_args)
    # # # ----------------------------

    eval_results = evaluate(
        teacher=teacher,
        student=student,
        data=data,
        epoch=args.student_epoch, # Use the epoch you're analyzing
        args=args,
        return_mode=True # <-- ACTIVATE OUR HOOK
    )

    I_student = eval_results["image_features"]
    T_teacher_student = eval_results["text_features"]
    reported_metrics = eval_results["metrics"]

    logging.info("Successfully retrieved features from the 'evaluate' function.")
    print("\n--- OFFICIAL METRICS FROM EVALUATE FUNCTION ---")
    print(json.dumps(reported_metrics, indent=2))

    # --- 5. PERFORM FAILURE ANALYSIS ON THE RETRIEVED FEATURES ---
    logging.info("Performing failure analysis on the retrieved features...")
    fails_student = find_retrieval_failures(I_student, T_teacher_student, args.K)

    print("\n--- METRICS RECALCULATION (SHOULD NOW MATCH) ---")
    num_samples = I_student.shape[0]
    num_fails_i2t = len(fails_student["image2text"])
    num_fails_t2i = len(fails_student["text2image"])
    recalculated_r5_i2t = (num_samples - num_fails_i2t) / num_samples
    recalculated_r5_t2i = (num_samples - num_fails_t2i) / num_samples
    print(f"Total samples: {num_samples}")
    print(f"Image->Text R@{args.K} from script: {recalculated_r5_i2t:.4f} ({num_fails_i2t} failures)")
    print(f"Text->Image R@{args.K} from script: {recalculated_r5_t2i:.4f} ({num_fails_t2i} failures)")

    for model_name, fails_dict in [("teacher", fails_teacher), ("student", fails_student)]:
        for mode, fails_list in fails_dict.items():
            out_path = os.path.join(args.output_dir, f"{model_name}_{mode}_failures_k{args.K}.jsonl")
            dump_failures(fails_list, keys, caps, out_path)
            logging.info(f"Wrote {len(fails_list)} {model_name} {mode} failures to {out_path}")


    # --- PHASE 2: ANALYSIS & PLOTTING ---
    logging.info("\n--- Phase 2: Analyzing the 'Gap' and Plotting ---")
    
    # 1. Find the "gap" where teacher fails and student succeeds
    teacher_i2t_fail_indices = {f['idx'] for f in fails_teacher['image2text']}
    student_i2t_fail_indices = {f['idx'] for f in fails_student['image2text']}
    gap_i2t = sorted(
        list(teacher_i2t_fail_indices - student_i2t_fail_indices),
        key=lambda idx: next(f['gt_rank'] for f in fails_teacher['image2text'] if f['idx'] == idx),
        reverse=True
    )
    logging.info(f"Found {len(gap_i2t)} examples where student fixes teacher's image-to-text failures.")

    teacher_t2i_fail_indices = {f['idx'] for f in fails_teacher['text2image']}
    student_t2i_fail_indices = {f['idx'] for f in fails_student['text2image']}
    gap_t2i = sorted(
        list(teacher_t2i_fail_indices - student_t2i_fail_indices),
        key=lambda idx: next(f['gt_rank'] for f in fails_teacher['text2image'] if f['idx'] == idx),
        reverse=True
    )
    logging.info(f"Found {len(gap_t2i)} examples where student fixes teacher's text-to-image failures.")

    # 2. Plot the gap comparisons using the new function
    # plot_gap_comparison(
    #     mode='image2text',
    #     gap_indices=gap_i2t,
    #     teacher_fails_data=fails_teacher['image2text'],
    #     I_student=I_student,
    #     T_teacher_student=T_teacher_student,
    #     keys=keys, caps=caps, val_tar_path=args.val_data,
    #     K=args.K, top_n=args.top_n, output_dir=args.output_dir
    # )

    save_gap_comparison_to_folders(
        mode='image2text',
        gap_indices=gap_i2t,
        teacher_fails_data=fails_teacher['image2text'],
        I_student=I_student,
        T_teacher_student=T_teacher_student,
        keys=keys, caps=caps, val_tar_path=args.val_data,
        K=args.K, top_n=args.top_n, output_dir=args.output_dir
    )

    # plot_gap_comparison(
    #     mode='text2image',
    #     gap_indices=gap_t2i,
    #     teacher_fails_data=fails_teacher['text2image'],
    #     I_student=I_student,
    #     T_teacher_student=T_teacher_student,
    #     keys=keys, caps=caps, val_tar_path=args.val_data,
    #     K=args.K, top_n=args.top_n, output_dir=args.output_dir
    # )
    save_gap_comparison_to_folders(
        mode='text2image',
        gap_indices=gap_t2i,
        teacher_fails_data=fails_teacher['text2image'],
        I_student=I_student,
        T_teacher_student=T_teacher_student,
        keys=keys, caps=caps, val_tar_path=args.val_data,
        K=args.K, top_n=args.top_n, output_dir=args.output_dir
    )

    logging.info("Analysis complete.")


@torch.no_grad()
def get_aligned_features(student, teacher, img_loader, txt_loader, args):
    """
    Correctly extracts aligned features for both symmetric (teacher) and
    asymmetric (student) evaluation in a single pass over the data.
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # --- 1. Encode all unique images with BOTH models ---
    # This ensures the order is identical.
    teacher.to(device).eval()
    student.to(device).eval()
    
    all_teacher_img_features = []
    all_student_img_features = []
    
    for images, _, _ in tqdm(img_loader, desc="Encoding Images (Teacher & Student)"):
        images = images.to(device=device, dtype=input_dtype)
        with autocast():
            teacher_img_feat = unwrap_model(teacher).encode_image(images, normalize=True)
            student_img_feat = unwrap_model(student).encode_image(images, normalize=True)
        all_teacher_img_features.append(teacher_img_feat.cpu())
        all_student_img_features.append(student_img_feat.cpu())
        
    I_teacher = torch.cat(all_teacher_img_features)
    I_student = torch.cat(all_student_img_features)

    # --- 2. Encode all captions with the TEACHER model ---
    all_text_features = []
    for texts, _ in tqdm(txt_loader, desc="Encoding Texts (Teacher)"):
        texts = texts.to(device=device)
        with autocast():
            text_feat = unwrap_model(teacher).encode_text(texts, normalize=True)
        all_text_features.append(text_feat.cpu())
        
    T_teacher = torch.cat(all_text_features)
    
    return I_teacher, I_student, T_teacher

def main_retrieval_analysis(args):
    """New analysis logic using the external evaluation framework."""
    logging.info("Running in 'retrieval-framework' mode.")
    try:
        from eval_metrics import evaluate 
        from eval_get_data import get_data as get_retrieval_data, read_coco_pairs,read_dci_pairs,read_iiw_pairs,read_flickr_pairs
    except ImportError:
        logging.error("Could not import from 'eval_get_data.py'. Make sure it's in the same directory or accessible in PYTHONPATH.")
        sys.exit(1)

    teacher, _, preprocess_val = init_teacher(args, device)
    student, _ = init_student(args, device)
    tokenizer = get_tokenizer(args.model)

    # ---  Get data using the retrieval framework's function ---
    retrieval_data = get_retrieval_data(args, (preprocess_val, preprocess_val), 0, tokenizer)
    
    if not retrieval_data:
        logging.error("No retrieval datasets were loaded. Check your arguments (e.g., --retrieval-coco).")
        return

    # Dynamically find which dataset was requested on the command line
    dataset_key = next((k for k in retrieval_data if k.startswith('retrieval_') and getattr(args, k.replace('-', '_'), False)), None)
    if not dataset_key:
        logging.error("No retrieval dataset specified or found. Use --retrieval-coco, --retrieval-dci, etc.")
        return

    logging.info(f"Analyzing dataset: {dataset_key}")
    txt_data_info, img_data_info, _, _ = retrieval_data[dataset_key]
    txt_loader, img_loader = txt_data_info.dataloader, img_data_info.dataloader

    # --- Step 3: Dynamically reconstruct the correct metadata ---
    logging.info("Re-parsing annotations to get raw captions and image paths...")
    
    if args.retrieval_coco:
        full_data_list = read_coco_pairs(root_dir=args.coco_data_root_dir, dict_root_dir=None)
    elif args.retrieval_dci:
        full_data_list = read_dci_pairs(root_dir=args.dci_retrieval_dir)
    elif args.retrieval_iiw:
        full_data_list = read_iiw_pairs(root_dir=args.iiw_retrieval_dir, finegrained=args.use_finegrained_iiw)
    elif args.retrieval_flickr:
        full_data_list = read_flickr_pairs(root_dir=args.flickr_data_root_dir, split=args.flickr_val_or_test)
    else:
        # Add elif blocks for other datasets like Flickr if you use them
        raise NotImplementedError(f"Data re-parsing for the active dataset is not implemented in main_retrieval_analysis.")

    # # The order of text captions comes from this re-parsed list
    text_captions = [item['caption'] for item in full_data_list]
    image_paths = [item['image'] for item in img_loader.dataset.img_list]
    logging.info(f"Found {len(text_captions)} captions and {len(image_paths)} unique images.")

    logging.info("Creating ground-truth mappings...")
    text_to_image_map, image_to_text_map = create_ground_truth_mappings(full_data_list, image_paths)

    # --- Step 4: Extract features for both Teacher and Student ---
    # TEACHER (Symmetric): Pass the TEACHER as the main model to evaluate
    # logging.info("--- Generating features for Teacher (Symmetric Eval) ---")
    # teacher_eval_results = evaluate(
    #     teacher, retrieval_data, 0, args, tokenizer=tokenizer, return_features=True
    # )
    # I_teacher = teacher_eval_results["image_features"]
    # T_teacher = teacher_eval_results["text_features"] # These are teacher-encoded texts

    # # STUDENT (Asymmetric): Pass the STUDENT as the main model to evaluate
    # logging.info("--- Generating features for Student (Asymmetric Eval) ---")
    # student_eval_results = evaluate(
    #     student, retrieval_data, 0, args, tokenizer=tokenizer, return_features=True
    # )
    # I_student = student_eval_results["image_features"]
    I_teacher, I_student, T_teacher = get_aligned_features(student, teacher, img_loader, txt_loader, args)
    logging.info(f"Feature extraction complete. I_teacher: {I_teacher.shape}, I_student: {I_student.shape}, T_teacher: {T_teacher.shape}")

    # --- 5. PERFORM FAILURE ANALYSIS ON THE RETRIEVED FEATURES ---
    logging.info("Performing failure analysis on the retrieved features...")
    # Student images vs Teacher text
    fails_student = find_retrieval_failures(I_student, T_teacher, args.K, text_to_image_map, image_to_text_map)
    # Teacher images vs Teacher text
    fails_teacher = find_retrieval_failures(I_teacher, T_teacher, args.K, text_to_image_map, image_to_text_map)

    # --- Step 6: Extract keys and captions for saving results ---
    # # The order of captions comes from the text dataset's data_list
    # text_captions = [item['caption'] for item in data_list]
    # # The order of image paths comes from the image dataset's internal list
    # image_paths = [item['image'] for item in img_loader.dataset.img_list]
    # text_captions = [item['caption'] for item in full_data_list]
    # image_paths = [item['image'] for item in img_loader.dataset.img_list]


    # --- Step 7: Analyze the "Gap" and Save Visualizations ---
    logging.info("\n--- Analyzing the 'Gap' and Saving Examples ---")
    
    teacher_i2t_fail_indices = {f['idx'] for f in fails_teacher['image2text']}
    student_i2t_fail_indices = {f['idx'] for f in fails_student['image2text']}
    gap_i2t = sorted(list(teacher_i2t_fail_indices - student_i2t_fail_indices), key=lambda idx: next(f['gt_rank'] for f in fails_teacher['image2text'] if f['idx'] == idx), reverse=True)

    # 1. Find the "gap" where teacher fails and student succeeds
    teacher_i2t_fail_indices = {f['idx'] for f in fails_teacher['image2text' ]}
    student_i2t_fail_indices = {f['idx'] for f in fails_student['image2text' ]}
    gap_i2t = sorted(
        list(teacher_i2t_fail_indices - student_i2t_fail_indices),
        key=lambda idx: next(f['gt_rank' ] for f in fails_teacher['image2text'] if f['idx'] == idx),
        reverse=True
    )

    logging. info(f"Found {len(gap_i2t)} examples where student fixes teacher's image-to-text failures.")

    teacher_t2i_fail_indices = {f['idx'] for f in fails_teacher['text2image' ]}
    student_t2i_fail_indices = {f['idx'] for f in fails_student['text2image' ]}
    gap_t2i = sorted(
        list(teacher_t2i_fail_indices - student_t2i_fail_indices),
        key=lambda idx: next(f['gt_rank' ] for f in fails_teacher['text2image' ] if f['idx' ] == idx),
        reverse=True
    )

    logging.info(f"Found {len(gap_t2i)} examples where student fixes teacher's text-to-image failures.")

    save_gap_comparison_from_disk(
        mode='image2text',
        gap_indices=gap_i2t,
        teacher_fails_data=fails_teacher['image2text'],
        I_student=I_student,
        I_teacher=I_teacher,
        T_teacher=T_teacher,
        img_keys=image_paths,
        text_captions=text_captions,
        K=args.K,
        top_n=args.top_n,
        output_dir=args.output_dir,
        text_to_image_map=text_to_image_map,  # <-- Pass map
        image_to_text_map=image_to_text_map   # <-- Pass map
    )

    save_gap_comparison_from_disk(
        mode='text2image',
        gap_indices=gap_t2i,
        teacher_fails_data=fails_teacher['text2image'],
        I_student=I_student,
        I_teacher=I_teacher,
        T_teacher=T_teacher,
        img_keys=image_paths,
        text_captions=text_captions,
        K=args.K,
        top_n=args.top_n,
        output_dir=args.output_dir,
        text_to_image_map=text_to_image_map,  # <-- Pass map
        image_to_text_map=image_to_text_map   # <-- Pass map
    )

    # ---Run Comparative Attention Visualization if requested ---
    if args.visualize_attention:
        # We pass BOTH models to the visualization function
        run_attention_visualization(
            student_model=student,
            teacher_model=teacher,
            gap_indices=gap_t2i, # Use the text2image gap indices
            text_to_image_map=text_to_image_map,
            image_paths=image_paths,
            text_captions=text_captions,
            preprocess_fn=preprocess_val,
            args=args,
            dataset_key=dataset_key,
        )

    logging.info("Analysis complete. Gap examples saved to output directory.")

    


if __name__ == "__main__":
    args = parse_args()
    args.rank = 0
    # Also add 'world_size' for completeness, as other distributed functions might need it.
    args.world_size = 1
    args.val_frequency = 1 
    device = args.device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.eval_mode == 'webdataset':
        main_webdataset_analysis(args)
    elif args.eval_mode == 'retrieval-framework':
        main_retrieval_analysis(args)
    else:
        raise ValueError(f"Invalid eval_mode '{args.eval_mode}'. Must be 'webdataset' or 'retrieval-framework'.")
    
    


