import json
import logging
import math
import os
import time
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from open_clip_train.train_distill import get_clip_metrics_chunked_further


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


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # Monitor GPU memory usage at the start of the epoch
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        logging.info(f"Epoch {epoch} start - GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            # Logging memory after forward pass (Rank 0 only)
            if args.rank == 0 and i % 10 == 0:  # Log every 10 batches
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logging.info(
                    f"Rank 0 - Batch {i}: "
                    f"Allocated: {allocated:.2f}GB, "
                    f"Reserved: {reserved:.2f}GB"
                )

            backward(total_loss, scaler)

            # Log memory after backward (Rank 0 only)
            if args.rank == 0 and i % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logging.info(
                    f"Rank 0 - Post-Backward {i}: "
                    f"Allocated: {allocated:.2f}GB, "
                    f"Reserved: {reserved:.2f}GB"
                )
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for
def extract_coco_image_id(url):
   match = re.search(r'(\d{12})\.jpg$', url)
   return int(match.group(1)) if match else None

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        val_loader = data['val'].dataloader
        # num_samples = 0
        samples_per_val = val_loader.num_samples

        # # FIXME this does not scale past small eval datasets
        # # all_image_features @ all_text_features will blow up memory and compute very quickly
        # cumulative_loss = 0.0
        # cumulative_gen_loss = 0.0
        # all_image_features, all_text_features = [], []

        # ─── CASE 1: COCO (1 image → 5 captions) ───
        if args.dataset_name.lower() == "mscoco":
            # --- STEP A: One pass over val WebDataset to extract raw features per (image, caption) and collect keys ---
            all_I_dict  = {}    # maps COCO key → one image feature vector (D,)
            all_T_list  = []    # list of all caption features (D,) in the order they appear
            all_TX_keys = []    # parallel list of strings: key of each caption in all_T_list
            with torch.inference_mode():
                for images_b, texts_b, urls_b in val_loader:
                    images_b = images_b.to(device=device, dtype=input_dtype) #TODO: add non_blocking=True if needed?
                    texts_b  = texts_b .to(device=device)

                    with autocast():
                        model_out = model(images_b, texts_b)
                        image_features = model_out["image_features"]
                        text_features = model_out["text_features"]
                        logit_scale = model_out["logit_scale"].mean()

                    # 1. Append each caption’s feature + COCO image-id (extracted from URL) to our lists
                    for b_idx, url_str  in enumerate(urls_b):
                        # Extract the 12-digit COCO ID from the URL, e.g. "000000109976"
                        img_id_int = extract_coco_image_id(url_str)
                        img_id_str = str(img_id_int).zfill(12)
                        all_T_list.append(text_features[b_idx].cpu())  
                        all_TX_keys.append(img_id_str)
                        # 2. For each item in batch, store that image’s feature once (if new)
                        if img_id_str not in all_I_dict:
                            all_I_dict[img_id_str] = image_features[b_idx].cpu()
                        
            # Now all_I_dict has exactly one feature per unique image key.
            img_keys = list(all_I_dict.keys())  
            all_I = torch.stack([all_I_dict[k] for k in img_keys], dim=0)  # shape (N_img, D)
            # all_T_list is length N_txt = total number of captions = 5 * N_img (for COCO)
            all_T = torch.stack(all_T_list, dim=0)  # shape (N_txt, D)

            # --- DEBUG: sanity checks ---
            norms_I = all_I.norm(dim=-1)
            norms_T = all_T.norm(dim=-1)
            logging.info(f"[DEBUG] Unique images={len(img_keys)}, captions={len(all_T_list)}")
            logging.info(f"[DEBUG] Image norms: mean={norms_I.mean():.4f}, std={norms_I.std():.4f}")
            logging.info(f"[DEBUG] Text norms:  mean={norms_T.mean():.4f}, std={norms_T.std():.4f}")
            m = min(len(img_keys), len(all_T_list), 100)
            diag_cos = torch.einsum("id,id->i", all_I[:m], all_T[:m])
            logging.info(f"[DEBUG] Mini diag cos,{m} pairs: mean={diag_cos.mean():.4f}, min={diag_cos.min():.4f}, max={diag_cos.max():.4f}")

            # --- STEP B: Build img2txt_dict and txt2img_dict mappings ---
            N_img = len(img_keys)
            N_txt = len(all_T_list)
            # txt2img: maps each caption index j -> an image index i
            txt2img_dict = {}
            # img2txt: maps each image index i -> list of caption indices [j1,j2,...,j5]
            img2txt_dict = {i: [] for i in range(N_img)}

            for caption_idx, key_str in enumerate(all_TX_keys):
                image_idx = img_keys.index(key_str)  # find which image-key this caption belongs to
                txt2img_dict[caption_idx] = image_idx
                img2txt_dict[image_idx].append(caption_idx)

            # --- STEP C: Compute retrieval metrics exactly as retrieval_on_split does internally ---
            with torch.no_grad():
                # Move logit_scale to CPU (since all_I/all_T are CPU)
                logit_scale = logit_scale.cpu()
                # full similarity: (N_img, N_txt)
                S = logit_scale * (all_I @ all_T.T)  
                # 1) image -> text R@1,5,10
                hits_i2t = {1: 0, 5: 0, 10: 0}
                for i in range(N_img):
                    row = S[i]
                    for k in (1,5,10):
                        topk = torch.topk(row, k=k)[1]
                        if any(int(j) in img2txt_dict[i] for j in topk):
                            hits_i2t[k] += 1
                image_to_text_R1  = hits_i2t[1]  / N_img
                image_to_text_R5  = hits_i2t[5]  / N_img
                image_to_text_R10 = hits_i2t[10] / N_img

                # 2) text -> image R@1,5,10
                hits_t2i = {1: 0, 5: 0, 10: 0}
                for j in range(N_txt):
                    col = S[:, j]
                    for k in (1,5,10):
                        topk = torch.topk(col, k=k)[1]
                        if txt2img_dict[j] in topk:
                            hits_t2i[k] += 1
                text_to_image_R1  = hits_t2i[1]  / N_txt
                text_to_image_R5  = hits_t2i[5]  / N_txt
                text_to_image_R10 = hits_t2i[10] / N_txt

            val_metrics = {
                "image_to_text_R@1":  image_to_text_R1,
                "image_to_text_R@5":  image_to_text_R5,
                "image_to_text_R@10": image_to_text_R10,
                "text_to_image_R@1":  text_to_image_R1,
                "text_to_image_R@5":  text_to_image_R5,
                "text_to_image_R@10": text_to_image_R10,
                # "clip_val_loss":      None  # if you also want to report a loss you'd have to compute it in a separate pass
            }

            metrics.update(val_metrics)
        # ─── CASE 2: Standard retrieval (1 image → 1 caption) ───
        elif args.dataset_name.lower() == "sharegpt4v":
            all_image_features, all_text_features = [], []
            cumulative_loss = 0.0
            num_samples = 0

            with torch.no_grad():
                for i, (images_b, texts_b) in enumerate(val_loader):
                    images_b = images_b.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts_b  = texts_b.to(device=device, non_blocking=True)

                    with autocast():
                        model_out       = model(images_b, texts_b)
                        image_features  = model_out["image_features"]  # (B, D)
                        text_features   = model_out["text_features"]   # (B, D)
                        logit_scale     = model_out["logit_scale"].mean()

                        # Contrastive loss 
                        logits_per_image = logit_scale * (image_features @ text_features.t())
                        logits_per_text  = logits_per_image.t()
                        B = images_b.shape[0]
                        labels = torch.arange(B, device=device)
                        total_loss = (
                            F.cross_entropy(logits_per_image, labels) +
                            F.cross_entropy(logits_per_text, labels)
                        ) / 2.0

                    # Accumulate features on CPU in matching order
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    cumulative_loss += total_loss.item() * B
                    num_samples += B
                    if is_master(args) and (i % 100) == 0:
                        avg_loss = cumulative_loss / num_samples
                        logging.info(f"Eval Epoch: {epoch} [{num_samples}/{samples_per_val}]\tClip Loss: {avg_loss:.6f}")

            # Stack so that index‐i image ↔ index‐i text
            all_I = torch.cat(all_image_features, dim=0)  # (N, D)
            all_T = torch.cat(all_text_features, dim=0)   # (N, D)
            final_logit_scale = logit_scale.cpu()

            # Compute retrieval‐R@k on aligned pairs
            val_metrics = get_clip_metrics_chunked_further(
                all_I,
                all_T,
                final_logit_scale,
                args=args,
                chunk_size=512,
                device=device,
            )
            avg_loss = cumulative_loss / num_samples
            metrics.update({
                **val_metrics,
                "clip_val_loss": avg_loss,
                "num_samples": num_samples,
                "epoch": epoch,
            })

        else:
            raise ValueError(f"Unknown dataset_name: {args.dataset_name!r} (must be 'mscoco' or 'sharegpt4v')")
                        

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale,calculate_mean_median_rank=False):
    metrics = {}
    image_features = image_features.cpu()
    text_features = text_features.cpu()
    logit_scale = logit_scale.cpu() # Ensure logit_scale is also CPU scalar/tensor

    logits_per_image = (logit_scale * image_features @ text_features.t()).detach()
    logits_per_text = logits_per_image.t().detach()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    # Ensure ground_truth is on CPU
    ground_truth = torch.arange(len(text_features), device=image_features.device).view(-1, 1) # Create on same device then move if needed, or just CPU

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach() # No need for .detach().cpu() if already on CPU and not requiring grad
        if calculate_mean_median_rank: # Control with the new argument
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def get_clip_metrics_chunked_further(image_features_cpu, text_features_cpu, logit_scale_cpu, chunk_size=512, device='cuda', args=None):
    metrics = {}
    num_images = image_features_cpu.shape[0]
    num_texts = text_features_cpu.shape[0]

    # This function assumes num_images == num_texts for standard retrieval metrics
    # and that they are paired (i-th image corresponds to i-th text).
    if num_images != num_texts:
        logging.warning("Number of images and texts differ; standard paired retrieval metrics might be misleading.")
        # Fallback or error
        return {"error": "Image and text counts differ"}


    # Accumulators for top-k hits
    # For image_to_text:
    i2t_correct_at_k = {k: 0 for k in [1, 5, 10]}
    # For text_to_image:
    t2i_correct_at_k = {k: 0 for k in [1, 5, 10]}

    # --- Image to Text Retrieval ---
    # Query with image chunks, gallery is all texts
    text_features_gpu = text_features_cpu.to(device) # Gallery on GPU
    logit_scale_gpu = logit_scale_cpu.to(device)

    for i in range(0, num_images, chunk_size):
        img_chunk_cpu = image_features_cpu[i:i + chunk_size]
        img_chunk_gpu = img_chunk_cpu.to(device)
        current_batch_size = img_chunk_gpu.shape[0]

        with torch.no_grad(), get_autocast(args.precision, device_type=device.type)():
            # Logits for current image chunk against ALL texts
            # Shape: (current_batch_size, num_all_texts)
            chunk_logits_gpu = logit_scale_gpu * img_chunk_gpu @ text_features_gpu.t()

        # Ground truth for this chunk (assuming paired data)
        # For image i in the chunk, the correct text is text i (globally)
        gt_for_chunk = torch.arange(i, i + current_batch_size, device=device)

        # Get top K predictions for each image in the chunk
        # We only need top-K for R@K, not full argsort if K is small
        # For R@10, we need at least top 10
        # For mean/median rank, full argsort is still needed.
        # Let's do full argsort for now on the chunk logits (moved to CPU for argsort)
        ranking = torch.argsort(chunk_logits_gpu.cpu(), descending=True, dim=-1) # (current_batch_size, num_all_texts)

        for j in range(current_batch_size): # Iterate over images in the current chunk
            actual_target_idx = gt_for_chunk[j].item() # The global index of the correct text
            # Find where this actual_target_idx appears in the sorted list of predictions for image j
            preds_for_this_image = ranking[j]
            # Example: if preds_for_this_image = [target_idx, other1, other2, ...], then rank is 0
            # rank_of_target = (preds_for_this_image == actual_target_idx).nonzero(as_tuple=True)[0].item()
            # The above can be slow. A more direct way to find the rank:
            try:
                 rank_of_target = torch.where(preds_for_this_image == actual_target_idx)[0].item()
            except IndexError: # Should not happen if target is in the text set
                 rank_of_target = float('inf')


            for k_val in i2t_correct_at_k.keys():
                if rank_of_target < k_val:
                    i2t_correct_at_k[k_val] += 1
        
        del img_chunk_gpu, chunk_logits_gpu
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    del text_features_gpu # Free gallery text features from GPU

    for k_val in i2t_correct_at_k.keys():
        metrics[f"image_to_text_R@{k_val}"] = i2t_correct_at_k[k_val] / num_images if num_images > 0 else 0.0


    # --- Text to Image Retrieval (Symmetric Logic) ---
    image_features_gpu = image_features_cpu.to(device) # Gallery on GPU

    for i in range(0, num_texts, chunk_size):
        txt_chunk_cpu = text_features_cpu[i:i + chunk_size]
        txt_chunk_gpu = txt_chunk_cpu.to(device)
        current_batch_size = txt_chunk_gpu.shape[0]

        with torch.no_grad(), get_autocast(args.precision, device_type=device.type)():
            # Logits for current text chunk against ALL images
            # Shape: (current_batch_size, num_all_images)
            chunk_logits_gpu = logit_scale_gpu * txt_chunk_gpu @ image_features_gpu.t()

        gt_for_chunk = torch.arange(i, i + current_batch_size, device=device)
        ranking = torch.argsort(chunk_logits_gpu.cpu(), descending=True, dim=-1)

        for j in range(current_batch_size):
            actual_target_idx = gt_for_chunk[j].item()
            try:
                rank_of_target = torch.where(ranking[j] == actual_target_idx)[0].item()
            except IndexError:
                rank_of_target = float('inf')

            for k_val in t2i_correct_at_k.keys():
                if rank_of_target < k_val:
                    t2i_correct_at_k[k_val] += 1
        
        del txt_chunk_gpu, chunk_logits_gpu
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    del image_features_gpu

    for k_val in t2i_correct_at_k.keys():
        metrics[f"text_to_image_R@{k_val}"] = t2i_correct_at_k[k_val] / num_texts if num_texts > 0 else 0.0

    # Mean/Median rank are harder to compute accurately without the full matrix or more complex acc.
    # For now, focusing on R@k which is more critical and easier to chunk.
    # You could collect all ranks and then compute mean/median if CPU RAM allows storing all ranks.

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
