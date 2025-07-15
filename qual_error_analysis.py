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

import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, get_model_config
from open_clip_train.main_distill_memory import (
    auto_layers_string,
    mp_parallelize_all,
    build_memory_args_automatically,
    load_weights_with_memory_layers,
)
from open_clip_train.data import get_data 
from open_clip.memory import ProductKeyArgs,HashingMemory
from open_clip_train.train_distill import evaluate
# from open_clip_train.distributed import init_distributed_device  # if you ever want multi‐GPU

def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--checkpoint",   required=True,
    #                help="path to epoch_X.pt")
    p.add_argument("--model",        default="ViT-B-16")
    p.add_argument("--pretrained",   default="openai")
    p.add_argument("--val-data",     required=True,
                   help="webdataset shards, e.g. '/…/val/000000.tar' or '*.tar'")
    p.add_argument("--val-num-samples", type=int, default=None)
    p.add_argument("--dataset-name", required=True,
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
    image_encoder_model, text_encoder_model, val_loader, device, precision
):
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

# def find_text2image_failures(I, T, K):
#     # cosine S = I @ T^T
#     # cosine S[i,j] = <I_i, T_j>
#     S = I @ T.t()
#     N = S.shape[0]
#     fails = []
#     # for each caption index j, check if its matching image j is in top-K
#     # for each text query j, look at column j of S to rank all images
#     for j in range(N):
#         # row = S[j]
#         col = S[:, j]             # sim(image_i, text_j)
#         # topk = torch.topk(row, K, largest=True).indices.tolist()
#         topk = torch.topk(col, K, largest=True).indices.tolist()
#         if j not in topk:
#             # rank of the GT is where in sorted descending
#             # rank = (torch.argsort(row, descending=True) == j).nonzero().item()
#             rank = (torch.argsort(col, descending=True) == j).nonzero().item()
#             # fails.append((j, rank, [(idx, float(row[idx])) for idx in topk]))
#             fails.append((j, rank, [(idx, float(col[idx])) for idx in topk]))
#     # sort by worst first
#     fails.sort(key=lambda x: x[1], reverse=True)
#     return fails


def find_text2image_failures(I: torch.Tensor, T: torch.Tensor, K: int):
    """
    For each text (j), look at similarities to all images S[:,j],
    declare a failure if the true-image index j is not in the top-K.
    """
    S = I @ T.t()           # shape (N_images=N_texts, N_texts)
    N = S.size(0)
    fails = []
    for j in range(N):
        col = S[:, j]                       # similarities from all images → text j
        topk_inds = torch.topk(col, K).indices.tolist()
        if j not in topk_inds:
            # rank of the true image among all images
            sorted_inds = torch.argsort(col, descending=True)
            rank = (sorted_inds == j).nonzero().item()
            fails.append((j, rank, [(i, float(col[i])) for i in topk_inds]))
    fails.sort(key=lambda x: x[1], reverse=True)
    return fails


# def find_image2text_failures(I, T, K):
#     # cosine S[i,j] = <I_i, T_j>
#     S = I @ T.t()
#     N = S.shape[0]
#     fails = []
#     # for each image query i, look at row i to rank all captions
#     for i in range(N):
#         # col = S[:, i]  # similarity of all images to text i
#         row = S[i]
#         # topk = torch.topk(col, K, largest=True).indices.tolist()
#         topk = torch.topk(row, K, largest=True).indices.tolist()
#         if i not in topk:
#             # rank = (torch.argsort(col, descending=True) == i).nonzero().item()
#             rank = (torch.argsort(row, descending=True) == i).nonzero().item()
#             # fails.append((i, rank, [(idx, float(col[idx])) for idx in topk]))
#             fails.append((i, rank, [(idx, float(row[idx])) for idx in topk]))
#     fails.sort(key=lambda x: x[1], reverse=True)
#     return fails

def find_image2text_failures(I: torch.Tensor, T: torch.Tensor, K: int):
    """
    For each image (i), look at similarities to all texts S[i,:],
    declare a failure if the true-text index i is not in the top-K.
    """
    S = I @ T.t()           # shape (N_images, N_texts)
    N = S.size(0)
    fails = []
    for i in range(N):
        row = S[i, :]                      # similarities from image i → all texts
        topk_inds = torch.topk(row, K).indices.tolist()
        if i not in topk_inds:
            sorted_inds = torch.argsort(row, descending=True)
            rank = (sorted_inds == i).nonzero().item()
            fails.append((i, rank, [(j, float(row[j])) for j in topk_inds]))
    fails.sort(key=lambda x: x[1], reverse=True)
    return fails


def find_retrieval_failures(I, T, K):
    return {
        "text2image": find_text2image_failures(I, T, K),
        "image2text": find_image2text_failures(I, T, K),
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
        for j, rank, topk in fails:
            f.write(json.dumps({
                "idx": j,
                "key": keys[j],
                "caption": caps[j],
                "gt_rank": rank+1,
                "retrieved": [
                    {"idx": idx, "key": keys[idx], "caption": caps[idx],"score": score}
                    for idx, score in topk
                ]
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


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # # 1) load your student
    # model, preproc_train, preproc_val = open_clip.create_model_and_transforms(
    #     args.model, args.pretrained,
    #     precision=args.precision,
    #     device=args.device,
    #     output_dict=True
    # )

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

    # 2) open the .tar ONCE to build keys & caps lists
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


    # --- TEACHER (Symmetric) EVALUATION ---
    # The teacher is evaluated against itself.
    logging.info("Starting symmetric evaluation for the Teacher model...")
    I_teacher, T_teacher = encode_entire_split(teacher, val_loader, device, args.precision)
    fails_teacher = find_retrieval_failures(I_teacher, T_teacher, args.K)
    print("Teacher symmetric evaluation complete.")

    # --- STUDENT (Asymmetric) EVALUATION ---
    logging.info("Calling the official 'evaluate' function in return_mode to get features...")
    
    # We pass a dummy epoch (e.g., 38) to satisfy the function signature.
    # The val_frequency check inside evaluate() needs to pass.
    # Set args.val_frequency to 1 to guarantee it runs.
    args.rank = 0
    args.val_frequency = 1 
    student, memory_args = init_student(args, device)

    # --- COMPARISON FUNCTION ---
    # This will print the detailed report to your console.
    compare_mlp_vs_memory_parameters(teacher, student, memory_args)
    # ----------------------------

    eval_results = evaluate(
        teacher=teacher,
        student=student,
        data=data,
        epoch=args.student_epoch, # Use the epoch you're analyzing
        args=args,
        return_mode=True # <-- ACTIVATE OUR HOOK
    )

    I_student = eval_results["image_features"]
    T_teacher = eval_results["text_features"]
    reported_metrics = eval_results["metrics"]

    logging.info("Successfully retrieved features from the 'evaluate' function.")
    print("\n--- OFFICIAL METRICS FROM EVALUATE FUNCTION ---")
    print(json.dumps(reported_metrics, indent=2))

    # --- 5. PERFORM FAILURE ANALYSIS ON THE RETRIEVED FEATURES ---
    logging.info("Performing failure analysis on the retrieved features...")
    fails_student = find_retrieval_failures(I_student, T_teacher, args.K)
    
    # This check should now pass with flying colors.
    print("\n--- METRICS RECALCULATION (SHOULD NOW MATCH) ---")
    num_samples = I_student.shape[0]
    num_fails_i2t = len(fails_student["image2text"])
    num_fails_t2i = len(fails_student["text2image"])
    recalculated_r5_i2t = (num_samples - num_fails_i2t) / num_samples
    recalculated_r5_t2i = (num_samples - num_fails_t2i) / num_samples
    print(f"Total samples: {num_samples}")
    print(f"Image->Text R@{args.K} from script: {recalculated_r5_i2t:.4f} ({num_fails_i2t} failures)")
    print(f"Text->Image R@{args.K} from script: {recalculated_r5_t2i:.4f} ({num_fails_t2i} failures)")


    # 4) find failures
    # failures = find_text2image_failures(I_all, T_all, args.K)
    # print(f"{len(failures)} misses out of {I_all.shape[0]}")
    max_rank_val = args.max_rank if args.max_rank is not None else float('inf')
    final_i2t_fails = filter_failures_by_rank_window(
        fails_student["image2text"], 
        min_rank=args.min_rank, 
        max_rank=max_rank_val
    )
    
    final_t2i_fails = filter_failures_by_rank_window(
        fails_student["text2image"],
        min_rank=args.min_rank,
        max_rank=max_rank_val
    )

    final_i2t_fails_teacher = filter_failures_by_rank_window(
        fails_teacher["image2text"],
        min_rank=args.min_rank,
        max_rank=max_rank_val
    )

    final_t2i_fails_teacher = filter_failures_by_rank_window(
        fails_teacher["text2image"],
        min_rank=args.min_rank,
        max_rank=max_rank_val
    )

    # Use a dynamic output directory name
    rank_str = f"rank_{args.min_rank}"
    if args.max_rank:
        rank_str += f"-{args.max_rank}"
    else:
        rank_str += "-inf"

    # 5) dump JSONL
    # dump_failures(failures, keys, caps, args.out_jsonl)
    # print(f"Written failures to {args.out_jsonl}")
    # for model_label, fails_dict in [("student", fails_student),("teacher", fails_teacher)]:
    #     for mode, fails in fails_dict.items():       # mode in {"text2image","image2text"}
    #         out_path = f"{model_label}_{mode}_{args.out_jsonl}"
    #         dump_failures(fails, keys, caps, out_path)
    #         print(f"Wrote {model_label} {mode} failures to {out_path}")

    # 6) plot top‐N worst
    # plot_failures(failures, keys, caps, args.val_data, args.top_n, args.K)

    # 7) save top‐N failures to folders
    # --- Save Student Failures ---
    #Uncomment if you want general top 10 *worst* failures for the student model
    # save_failures(
    #     mode="text_to_image",
    #     fails=fails_student["text2image"],
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir="student_text2image_failures"
    # )
    # save_failures(
    #     mode="image_to_text",
    #     fails=final_i2t_fails,
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir=f"student_i2t_failures_{rank_str}"
    # )
    #Uncomment if you want general top 10 *worst* failures for the student model
    # save_failures(
    #     mode="image_to_text",
    #     fails=fails_student["image2text"],
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir="student_image2text_failures"
    # )
    # save_failures(
    #     mode="text_to_image",
    #     fails=final_t2i_fails,
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir=f"student_t2i_failures_{rank_str}"
    # )
    # # --- Save Teacher Failures (Optional, but good for comparison) ---
    #Uncomment if you want general top 10 *worst* failures for the teacher
    # save_failures(
    #     mode="text_to_image",
    #     fails=fails_teacher["text2image"],
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir="teacher_text2image_failures"
    # )

    # save_failures(
    #     mode="image_to_text",
    #     fails=final_i2t_fails_teacher,
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir=f"teacher_i2t_failures_{rank_str}"
    # )

    #Uncomment if you want general top 10 *worst* failures for the teacher 
    # save_failures(
    #     mode="image_to_text",
    #     fails=fails_teacher["image2text"],
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir="teacher_image2text_failures"
    # )

    # save_failures(
    #     mode="text_to_image",
    #     fails=final_t2i_fails_teacher,
    #     keys=keys, caps=caps, val_data_tar=args.val_data,
    #     top_n=args.top_n, K=args.K,
    #     out_dir=f"teacher_t2i_failures_{rank_str}"
    # )
    


