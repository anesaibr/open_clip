import json
import logging
import math
import os
import time
import re
from unicodedata import normalize

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from typing import Any, Dict, Optional, Tuple, Union

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def format_metrics(metrics):
    """Formats a metrics dictionary into a readable, structured, multi-line string."""
    
    # This dictionary will hold the structured data
    grouped_results = {}

    for full_key, v in metrics.items():
        # Skip epoch key as we'll handle it separately
        if full_key == 'epoch':
            grouped_results['General'] = {'General': {'epoch': v}}
            continue
            
        # Split the key into components
        parts = full_key.split('_')
        
        # Find the dataset name (starts with retrieval_)
        dataset_start = next((i for i, part in enumerate(parts) if part.startswith('retrieval')), None)
        if dataset_start is None:
            # Not a retrieval metric
            grouped_results.setdefault('General', {}).setdefault('General', {})[full_key] = v
            continue
            
        dataset_key = '_'.join(parts[dataset_start:parts.index('text')] if 'text' in parts else parts[dataset_start:parts.index('image')])
        metric_type = 'Text-to-Image' if 'text_to_image' in full_key else 'Image-to-Text'
        
        # Extract the metric name
        if 'R@' in full_key:
            metric_name = full_key.split('_')[-1]  # Gets R@1, R@5, etc.
        elif 'mean_rank' in full_key:
            metric_name = 'mean_rank'
        elif 'median_rank' in full_key:
            metric_name = 'median_rank'
        elif 'num_text_samples' in full_key:
            metric_name = 'text_samples'
        elif 'num_image_samples' in full_key:
            metric_name = 'image_samples'
        else:
            metric_name = full_key.split('_')[-1]
        
        # Organize in the grouped_results structure
        grouped_results.setdefault(dataset_key, {}).setdefault(metric_type, {})[metric_name] = v
        
        # Also store sample counts if available
        if 'num_text_samples' in full_key:
            grouped_results.setdefault(dataset_key, {}).setdefault('Samples', {})['text'] = v
        if 'num_image_samples' in full_key:
            grouped_results.setdefault(dataset_key, {}).setdefault('Samples', {})['image'] = v

    # --- Build the output string ---
    output_lines = [f"Evaluation results for Epoch {metrics.get('epoch', '?')}:"]
    
    # Handle general metrics first
    if 'General' in grouped_results:
        general_metrics = grouped_results['General'].get('General', {})
        if general_metrics:
            output_lines.append("\nGeneral Metrics:")
            for name, value in general_metrics.items():
                output_lines.append(f"  {name.replace('_', ' ').title():<20}: {value:.4f}")

    # Process retrieval datasets
    retrieval_keys = [k for k in grouped_results.keys() if k.startswith('retrieval_')]
    for dataset_key in sorted(retrieval_keys):
        title = dataset_key.replace('retrieval_', '').replace('_', ' ').title()
        output_lines.append(f"\n{title}:")
        
        dataset_metrics = grouped_results[dataset_key]
        
        # Show sample counts first if available
        if 'Samples' in dataset_metrics:
            samples = dataset_metrics['Samples']
            output_lines.append("  Samples:")
            if 'image' in samples:
                output_lines.append(f"    Images: {int(samples['image'])}")
            if 'text' in samples:
                output_lines.append(f"    Texts: {int(samples['text'])}")
        
        # Show retrieval metrics
        for metric_type in ['Image-to-Text', 'Text-to-Image']:
            if metric_type in dataset_metrics:
                output_lines.append(f"  {metric_type}:")
                metrics = dataset_metrics[metric_type]
                
                # Display R@k metrics together
                r_metrics = {k: v for k, v in metrics.items() if k.startswith('R@')}
                if r_metrics:
                    r_line = "    "
                    for r in sorted(r_metrics.keys()):  # Sort R@1, R@5, R@10
                        r_line += f"{r}: {r_metrics[r]:.4f}  "
                    output_lines.append(r_line.strip())
                
                # Display rank metrics
                if 'mean_rank' in metrics:
                    output_lines.append(f"    Mean Rank: {metrics['mean_rank']:.1f}")
                if 'median_rank' in metrics:
                    output_lines.append(f"    Median Rank: {int(metrics['median_rank'])}")

    return "\n".join(output_lines)

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    zero_shot_metrics = zero_shot_eval(
        model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        if 'retrieval_coco' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_coco']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_coco', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
        if 'retrieval_flickr' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_flickr']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_flickr', model, txt_loader, img_loader, img2txt_dict, txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)
        if 'retrieval_cc3m_train' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_cc3m_train']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_cc3m_train', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_docci' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_docci']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_docci', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_urban_1k' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_urban_1k']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_urban_1k', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_iiw' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_iiw']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_iiw', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_dci' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_dci']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_dci', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_sharegpt4v-1k' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_sharegpt4v-1k']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_sharegpt4v-1k', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

        if 'retrieval_sharegpt4v-10k' in data:
            txt_data, img_data, img2txt_dict, txt2img_dict = data['retrieval_sharegpt4v-10k']
            txt_loader, img_loader = txt_data.dataloader, img_data.dataloader
            metrics = retrieval_on_split('retrieval_sharegpt4v-10k', model, txt_loader, img_loader, img2txt_dict,
                                         txt2img_dict,
                                         args, epoch, metrics, device, input_dtype, autocast)

    if not metrics:
        return metrics

    # logging.info(
    #     f"Eval Epoch: {epoch} "
    #     + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    # )

    formatted_metrics_str = format_metrics(metrics)
    # logging.info(f"Evaluation results for Epoch {epoch}:{formatted_metrics_str}")
    logging.info(formatted_metrics_str)

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        # Convert numpy and torch types to native Python types for JSON serialization
        json_metrics = {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                       for k, v in metrics.items()}

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(json_metrics))
            f.write("\n")

    # if args.wandb:
    #     assert wandb is not None, 'Please install wandb.'
    #     if 'train' in data:
    #         dataloader = data['train'].dataloader
    #         num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    #         step = num_batches_per_epoch * epoch
    #     else:
    #         step = None
    #     log_data['epoch'] = epoch
    #     wandb.log(log_data, step=step)

    return metrics


def remap_indices(merged_img_ids, cap_ids, img2txt_dict, txt2img_dict):
    """
    params:
    merged_img_ids: tensor of shape (M, D)
    cap_ids: tensor of shape (N) (But the ordering might be random)
    img2txt_dict: dict mapping each img_id to a list of cap_ids
    txt2img_dict: dict mappint each cap_id to an img_id (a list of one element)
    text_features: tensor of shape (N, D)
    """
    # so now ideally the cap_ids should be (0, ...N), so do the text_features
    # step2: re-index the merged_image_ids and re-do the mapping in the dict.
    # As the original image ids might just be random numbers, they don't represent the real ordering.

    img_id_mapping = {old_id.item(): new_idx for new_idx, old_id in enumerate(merged_img_ids)}
    reindexed_img_ids = torch.tensor([img_id_mapping[img_id.item()] for img_id in merged_img_ids])

    # Update the img2txt_dict and txt2img_dict with new indices
    new_img2txt_dict = {img_id_mapping[img_id]: [cap_id for cap_id in cap_id_list]
                        for img_id, cap_id_list in img2txt_dict.items()}

    new_txt2img_dict = {cap_id: img_id_mapping[txt2img_dict[cap_id][0]]
                        for cap_id in txt2img_dict.keys()}

    return new_img2txt_dict, new_txt2img_dict

def compute_retrieval(similarity_scores, txt2img, img2txt):
    if isinstance(similarity_scores, tuple):
        i2t_similarity_score, t2i_similarity_score = similarity_scores
    else:
        # Otherwise, treat similarity_scores as a single matrix for t2i
        t2i_similarity_score = similarity_scores.t()
        i2t_similarity_score = similarity_scores

    t2i_ranks = torch.zeros(t2i_similarity_score.shape[0])

    for index, score in enumerate(t2i_similarity_score):
        inds = torch.argsort(score, descending=True)
        t2i_ranks[index] = torch.where(inds == txt2img[index])[0][0]

    # Compute metrics
    tr1 = len(torch.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
    tr5 = len(torch.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
    tr10 = len(torch.where(t2i_ranks < 10)[0]) / len(t2i_ranks)
    t2i_report_dict = {
        "text_to_image_R@1": tr1,
        "text_to_image_R@5": tr5,
        "text_to_image_R@10": tr10,
        "text_to_image_mean_rank": t2i_ranks.mean().item() + 1,
        "text_to_image_median_rank": np.floor(np.median(t2i_ranks.numpy())) + 1
    }

    # comput image -> text
    i2t_ranks = torch.zeros(i2t_similarity_score.shape[0])
    for index, score in enumerate(i2t_similarity_score):
        inds = torch.argsort(score, descending=True)
        # Score
        rank = 1e10
        for i in img2txt[index]:
            tmp = torch.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        i2t_ranks[index] = rank

    # Compute metrics
    ir1 = len(torch.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
    ir5 = len(torch.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
    ir10 = len(torch.where(i2t_ranks < 10)[0]) / len(i2t_ranks)

    i2t_report_dict = {
        "image_to_text_R@1": ir1,
        "image_to_text_R@5": ir5,
        "image_to_text_R@10": ir10,
        "image_to_text_mean_rank": i2t_ranks.mean().item() + 1,
        "image_to_text_median_rank": np.floor(np.median(i2t_ranks.numpy())) + 1
    }
    metrics = {**t2i_report_dict, **i2t_report_dict}
    return metrics

def retrieval_on_split(keyword, model, txt_loader, img_loader, img2txt_dict, txt2img_dict, args, epoch, metrics, device,
                       input_dtype, autocast):
    num_txt_samples = txt_loader.num_samples
    num_img_samples = img_loader.num_samples
    all_image_features, all_text_tokens, all_text_features = [], [], []
    all_local_text_tokens = []
    all_img_ids, all_cap_ids = [], []

    with torch.no_grad():
        # first loop over the text dataloader to store all text embeddings
        #for i, batch in tqdm(enumerate(txt_loader), total=len(txt_loader), desc="Processing Texts"):
        for i, batch in enumerate(txt_loader):
            texts, cap_id = batch
            texts = texts.to(device=device, non_blocking=True)
            with autocast():
                if args.inference_with_flair:
                    # DEBUG: capture everything encode_text returns
                    # outputs = unwrap_model(model).encode_text(texts, normalize=False)
                    # logging.info(
                    #     "DEBUG: encode_text returned %d values: %s",
                    #     len(outputs),
                    #     [getattr(o, "shape", type(o)) for o in outputs]
                    # )
                    global_text_token, local_text_tokens = unwrap_model(model).encode_text(texts, normalize=False)
                    global_text_token, local_text_tokens = unwrap_model(model).text_post(
                        global_text_token), unwrap_model(model).text_post(local_text_tokens)
                    text_features = F.normalize(global_text_token, dim=-1)
                    all_text_tokens.append(global_text_token.squeeze(1))  # GPU
                    all_local_text_tokens.append(local_text_tokens)  # GPU
                else:
                    text_features = unwrap_model(model).encode_text(texts, normalize=True)

                all_text_features.append(text_features.detach().cpu())  # cpu list of N, each of shape (B, D)
                all_cap_ids.append(cap_id.detach().cpu())
        all_text_features_tensor = torch.cat(all_text_features)  # (N, 512)
        cap_ids = torch.cat(all_cap_ids)

        
        if args.inference_with_flair:
            mode = "inference_with_flair"
            all_text_tokens_tensor = torch.cat(all_text_tokens)  # on GPU
            all_local_text_tokens_tensor = torch.cat(all_local_text_tokens)

            similarity_scores, img_ids = compute_similarity_scores_attn_pool(
                model, img_loader, all_text_features_tensor, all_text_tokens_tensor, device, input_dtype, autocast, mode
            )
        else:
            similarity_scores, img_ids = compute_similarity_scores_original_clip(model, img_loader,
                                                                                    all_text_features_tensor, device,
                                                                                    input_dtype,
                                                                                    autocast,
                                                                                    mode='original_clip')

        new_img2txt_dict, new_txt2img_dict = remap_indices(merged_img_ids=img_ids, cap_ids=cap_ids,
                                                           img2txt_dict=img2txt_dict, txt2img_dict=txt2img_dict)

        retrieval_metrics = compute_retrieval(similarity_scores=similarity_scores,
                                              txt2img=new_txt2img_dict,
                                              img2txt=new_img2txt_dict)

        if keyword != '':
            temp_retrieval_metrics = {}
            keyword = keyword + '_'
            for k, v in retrieval_metrics.items():
                temp_retrieval_metrics[keyword + k] = v
            retrieval_metrics = temp_retrieval_metrics

        if "epoch" in metrics:  # we only need one epoch information
            metrics.update(
                {**retrieval_metrics,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )
        else:
            metrics.update(
                {**retrieval_metrics,
                 f"epoch": epoch,
                 f"{keyword}num_text_samples": num_txt_samples,
                 f"{keyword}num_image_samples": num_img_samples
                 }
            )

    return metrics


def compute_similarity_scores_original_clip(model, img_loader, all_text_features_tensor, device, input_dtype,
                                            autocast, mode='original_clip'):
    all_image_features = []
    all_img_ids = []

    for i, batch in enumerate(img_loader):
        images, img_id = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        all_img_ids.append(img_id.detach().cpu())
     
        with autocast():
            if mode == 'original_clip':
                image_features = unwrap_model(model).encode_image(images, normalize=True)
            elif mode == 'imgcon':
                _, local_image_tokens = unwrap_model(model).encode_image(images)
                local_image_tokens = unwrap_model(model).image_post(local_image_tokens)
                image_features = unwrap_model(model).visual_proj(local_image_tokens.mean(dim=1, keepdim=True), local_image_tokens, local_image_tokens)
                image_features = image_features.squeeze(1)
                image_features = F.normalize(image_features, dim=-1)
            logit_scale = unwrap_model(model).logit_scale.exp()
            all_image_features.append(image_features.detach().cpu())

    all_image_features_tensor = torch.cat(all_image_features)
    img_ids = torch.cat(all_img_ids)

    similarity_scores = logit_scale.cpu() * all_image_features_tensor @ all_text_features_tensor.t()
    return similarity_scores, img_ids

def compute_similarity_scores_attn_pool(model, img_loader, all_text_features_tensor, all_text_tokens_tensor, device,
                                        input_dtype,
                                        autocast, mode):
    logits_per_image_list = []
    all_img_ids = []

    for i, batch in enumerate(img_loader):
        images, img_id = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        all_img_ids.append(img_id.detach().cpu())
        with autocast():
            if mode == 'inference_with_flair':
                _, image_embeddings = unwrap_model(model).encode_image(images, normalize=False)
                image_embeddings = unwrap_model(model).image_post(image_embeddings)  # down proj to 256
                img_features_after_conditioning = unwrap_model(model).visual_proj(
                    all_text_tokens_tensor.unsqueeze(0),
                    image_embeddings,
                    image_embeddings
                )
                img_features_after_conditioning = F.normalize(img_features_after_conditioning, dim=-1).detach().cpu()
                embed_dim = img_features_after_conditioning.shape[-1]
                img_features_after_conditioning = img_features_after_conditioning.contiguous().view(-1, embed_dim)
            else:
                embed_dim = all_text_features_tensor.shape[-1]
                img_features_after_conditioning = unwrap_model(model).visual_proj(
                    all_text_tokens_tensor.unsqueeze(0),
                    image_embeddings,
                    image_embeddings
                ).detach().cpu().contiguous().view(-1, embed_dim)

            logit_scale = unwrap_model(model).logit_scale.exp()
            logits_per_image = (logit_scale.cpu() * torch.einsum('ij,ij->i', img_features_after_conditioning,
                                                                 all_text_features_tensor)).unsqueeze(0).detach().cpu()
        logits_per_image_list.append(logits_per_image)

    img_ids = torch.cat(all_img_ids)  # shape (M)
    similarity_scores = torch.cat(logits_per_image_list)  # shape (M, N)
    return similarity_scores, img_ids