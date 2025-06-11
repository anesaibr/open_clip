import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn import MSELoss

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast



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
        # total_loss.backward()
        total_loss.backward(retain_graph=True) #enables tracing past the first crash if needed



def train_one_epoch(model,student_model,data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)


    # (1) Turn on anomaly detection to identify which backward operation yields NaN/inf
    torch.autograd.set_detect_anomaly(True)


    # Debug check: teacher_out and student_out stats
    def debug_tensor_stats(name, t):
        # prints min, max, mean, std
        print(f"{name}: shape={t.shape}, "
                f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
                f"mean={t.mean().item():.6f}, std={t.std().item():.6f}")
        
    # If you want to single-batch debug:
    DEBUG_SINGLE_BATCH = False

    # Monitor GPU memory usage at the start of the epoch
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        logging.info(f"Epoch {epoch} start - GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    student_model = student_model.to(device=device)
    model.eval() # SOLVED: This is true as teacher model should be on eval mode
    student_model.train() # SOLVED: This is true as student model should be on train mode
    
    # if args.distill:
    #     dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_features_teacher, accum_features_student = [], {}, {}

    losses_m = {}
    losses = {}
    loss_mse = MSELoss(reduction='mean')
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, _ = batch # EXTRACT IMAGES ONLY
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        # texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                with torch.no_grad():
                    teacher_out = unwrap_model(model).encode_image(images).detach() 

                # (A) -- Student forward
                student_out = unwrap_model(student_model).encode_image(images) # Fixed: for distributed training

                # print("teacher_out is type:", type(teacher_out), "shape:", teacher_out.shape if isinstance(teacher_out, torch.Tensor) else None)
                # print('student_out is type:', type(student_out), "shape:", student_out.shape if isinstance(student_out, torch.Tensor) else None)
                
                # teacher stats
                # debug_tensor_stats("teacher_out", teacher_out)
                # student stats
                # debug_tensor_stats("student_out", student_out)

                # (B) -- Compute Distillation Loss
                if args.loss_type == 'mse':
                    total_loss = loss_mse(teacher_out, student_out)
                elif args.loss_type == 'l2':
                    total_loss = torch.norm(teacher_out - student_out, p=2, dim=1).mean()
                elif args.loss_type == 'cosine':
                    tearcher_out_norm = teacher_out / torch.norm(teacher_out, p=2, dim=1, keepdim=True)
                    student_out_norm = student_out / torch.norm(student_out, p=2, dim=1, keepdim=True)
                    total_loss = 1 - (tearcher_out_norm * student_out_norm).sum(dim=1).mean()
                else:
                    raise ValueError(f"Invalid loss type: {args.loss_type}")
                # losses = loss(**model_out, output_dict=True)

                # total_loss = sum(losses.values())

                # Debug check: is total_loss itself NaN?
                if torch.isnan(total_loss):
                    print(f"[DEBUG] total_loss is NaN at batch {i}. teacher_out stats above might show why.")
                    # Optionally break or raise an error
                    raise ValueError("NaN in total_loss")
                
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
            # Debug: a sanity check to see if any gradients are NaN or Inf
            for name, param in student_model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"[DEBUG] NaN in grad of: {name}")
                    if torch.isinf(param.grad).any():
                        print(f"[DEBUG] Inf in grad of: {name}")


            # Log memory after backward (Rank 0 only)
            if args.rank == 0 and i % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logging.info(
                    f"Rank 0 - Post-Backward {i}: "
                    f"Allocated: {allocated:.2f}GB, "
                    f"Reserved: {reserved:.2f}GB"
                )
            # If debugging single batch only:
            if DEBUG_SINGLE_BATCH:
                print("[DEBUG] Breaking after single batch debug.")
                break
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    with torch.no_grad():
                        teacher_out = unwrap_model(model).encode_image(images)
                    student_out = unwrap_model(student_model).encode_image(images) # Fixed: for distributed training

                    # Debug stats
                    # debug_tensor_stats("teacher_out", teacher_out)
                    # debug_tensor_stats("student_out", student_out)

                    # Accumulate the features and images for the last accum_freq batches.
                    if "image_features" not in accum_features_teacher:
                        accum_features_teacher["image_features"] = [teacher_out]
                        accum_features_student["image_features"] = [student_out]
                    else:
                        accum_features_teacher["image_features"].append(teacher_out)
                        accum_features_student["image_features"].append(student_out)


                accum_images.append(images)
                # accum_texts.append(texts)

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
                # texts = accum_texts[j]
                with autocast():
                    with torch.no_grad():
                        teacher_out = unwrap_model(model).encode_image(images)
                    student_out = unwrap_model(student_model).encode_image(images)
                    # print("teacher_out is type:", type(teacher_out), "shape:", teacher_out.shape if isinstance(teacher_out, torch.Tensor) else None)
                    # print('student_out is type:', type(student_out), "shape:", student_out.shape if isinstance(student_out, torch.Tensor) else None)

                    # same debugging + loss calc logic
                    # debug_tensor_stats("teacher_out", teacher_out)
                    # debug_tensor_stats("student_out", student_out)


                    if args.loss_type == 'mse':
                        total_loss = loss_mse(teacher_out, student_out)
                    elif args.loss_type == 'l2':
                        total_loss = torch.norm(teacher_out - student_out, p=2, dim=1).mean()
                    elif args.loss_type == 'cosine':
                        tearcher_out_norm = teacher_out / torch.norm(teacher_out, p=2, dim=1, keepdim=True)
                        student_out_norm = student_out / torch.norm(student_out, p=2, dim=1, keepdim=True)
                        total_loss = 1 - (tearcher_out_norm * student_out_norm).sum(dim=1).mean()
                    else:
                        raise ValueError(f"Invalid loss type: {args.loss_type}")

                    
                    # losses = loss(**model_out, output_dict=True)
                    # total_loss = sum(losses)  # TODO check if this should be commented out

                    if torch.isnan(total_loss):
                        print(f"[DEBUG] total_loss is NaN in accum freq pass at batch {i}, sub-batch {j}.")
                        # Optionally break
                        raise ValueError("NaN in total_loss accum pass")

                    losses["loss"] = total_loss

                backward(total_loss, scaler)

                # Debug: a sanity check to see if any gradients are NaN or Inf
                for name, param in student_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"[DEBUG] NaN in grad of: {name}")
                        if torch.isinf(param.grad).any():
                            print(f"[DEBUG] Inf in grad of: {name}")
            
            if DEBUG_SINGLE_BATCH:
                print("[DEBUG] Breaking after single batch debug.")
                break

            
        #TODO: Decide if gradient clipping should be applied to Student model instead (or as well)
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    # torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    # torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_features_teacher, accum_features_student = [], {}, {}

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

            # logit_scale_scalar = logit_scale.item()
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
                f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
                # f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                # "scale": logit_scale_scalar,
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


def evaluate(teacher, student,data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    teacher.eval()
    student.eval()

    # --- TODO:Keep zero_shot_eval? ---
    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    # metrics.update(zero_shot_metrics)
    logging.info("Skipping zero-shot evaluation for image encoder comparison.")

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # dataloader = data['val'].dataloader
        # num_samples = 0
        val_loader = data['val'].dataloader
        samples_per_val = val_loader.num_samples

        if args.dataset_name.lower() == "mscoco":
            # --- STEP A: 1→5 grouping by COCO ID from URL ---
            all_I_dict, all_T_list, all_TX_keys = {}, [], []
            with torch.inference_mode():
                for images_b, texts_b, urls_b in val_loader:
                    images_b = images_b.to(device, dtype=input_dtype)
                    texts_b  = texts_b.to(device)
                    with autocast():
                        # student encodes images, teacher encodes texts
                        I = unwrap_model(student).encode_image(images_b).float()
                        T = unwrap_model(teacher).encode_text(texts_b).float()
                        scale = unwrap_model(teacher).logit_scale.mean()
                    # collect features + group by 12-digit COCO ID
                    for i in range(len(urls_b)):
                        img_id = extract_coco_image_id(urls_b[i])
                        key   = str(img_id).zfill(12)
                        all_T_list.append(T[i].cpu())
                        all_TX_keys.append(key)
                        if key not in all_I_dict:
                            all_I_dict[key] = I[i].cpu()

            # stack into (N_img, D) and (N_txt, D)
            img_keys = list(all_I_dict.keys())
            all_I = torch.stack([all_I_dict[k] for k in img_keys], dim=0)
            all_T = torch.stack(all_T_list, dim=0)

            # build txt2img/img2txt
            N_img, N_txt = len(img_keys), len(all_T_list)
            txt2img = {}
            img2txt = {i: [] for i in range(N_img)}
            for j, key in enumerate(all_TX_keys):
                i = img_keys.index(key)
                txt2img[j] = i
                img2txt[i].append(j)

            # compute full (N_img × N_txt) similarity on CPU
            with torch.no_grad():
                S = scale.cpu() * (all_I @ all_T.T)
                # image→text R@k
                hits_i2t = {k: 0 for k in (1, 5, 10)}
                for i in range(N_img):
                    row = S[i]
                    for k in hits_i2t:
                        if any(int(j) in img2txt[i] for j in row.topk(k).indices):
                            hits_i2t[k] += 1
                # text→image R@k
                hits_t2i = {k: 0 for k in (1, 5, 10)}
                for j in range(N_txt):
                    col = S[:, j]
                    for k in hits_t2i:
                        if txt2img[j] in col.topk(k).indices:
                            hits_t2i[k] += 1

            val_metrics = {
                "image_to_text_R@1":  hits_i2t[1]  / N_img,
                "image_to_text_R@5":  hits_i2t[5]  / N_img,
                "image_to_text_R@10": hits_i2t[10] / N_img,
                "text_to_image_R@1":  hits_t2i[1]  / N_txt,
                "text_to_image_R@5":  hits_t2i[5]  / N_txt,
                "text_to_image_R@10": hits_t2i[10] / N_txt,
            }
            metrics.update(val_metrics)

        elif args.dataset_name.lower() == "sharegpt4v":
            # --- Paired 1→1 retrieval ---
            all_I_feats, all_T_feats = [], []
            cum_loss, num_samples = 0.0, 0
            with torch.no_grad():
                for i, (images_b, texts_b) in enumerate(val_loader):
                    images_b = images_b.to(device, dtype=input_dtype)
                    texts_b  = texts_b.to(device)
                    with autocast():
                        I = unwrap_model(student).encode_image(images_b).float()
                        T = unwrap_model(teacher).encode_text(texts_b).float()
                        scale = unwrap_model(teacher).logit_scale.mean()
                        logits_i2t = scale * (I @ T.t())
                        logits_t2i = logits_i2t.t()
                        B = images_b.size(0)
                        labels = torch.arange(B, device=device)
                        loss = (F.cross_entropy(logits_i2t, labels)
                                + F.cross_entropy(logits_t2i, labels)) / 2
                    all_I_feats.append(I.cpu())
                    all_T_feats.append(T.cpu())
                    cum_loss += loss.item() * B
                    num_samples += B
            all_I = torch.cat(all_I_feats, dim=0)
            all_T = torch.cat(all_T_feats, dim=0)
            val_metrics = get_clip_metrics_chunked_further(
                all_I, all_T, scale.cpu(),
                args=args, chunk_size=512, device=device
            )
            val_metrics["clip_val_loss"] = cum_loss / num_samples
            val_metrics["num_samples"]     = num_samples
            val_metrics["epoch"]           = epoch
            metrics.update(val_metrics)

        else:
            raise ValueError(f"Unknown dataset_name={args.dataset_name!r}")

    else:
        return {}

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


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
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
