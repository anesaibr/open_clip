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

            

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip_norm, norm_type=2.0)
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
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        # cumulative_loss = 0.0
        # cumulative_gen_loss = 0.0
        # all_image_features, all_text_features = [], []

        all_teacher_img_feat, all_student_img_feat = [], []
        all_student_txt_feat, all_teacher_txt_feat = [], []
        need_teacher_text = True # Set to False if you only care about student performance + similarity
        
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                batch_size = images.shape[0]
                with autocast():
                    # model_out = model(images, texts)
                    # image_features = student_model(images)["image_features"].float()
                    # text_features = unwrap_model(model).encode_text(texts).float()
                    # logit_scale = unwrap_model(model).logit_scale
                    # model_out = {"image_features": image_features, "text_features": text_features, "logit_scale": logit_scale}
                    teacher_image_features = unwrap_model(teacher).encode_image(images, normalize=True)
                    student_image_features = unwrap_model(student).encode_image(images, normalize=True)
                    all_teacher_img_feat.append(teacher_image_features.cpu())
                    all_student_img_feat.append(student_image_features.cpu())

                    student_text_features = unwrap_model(student).encode_text(texts, normalize=True)
                    all_student_txt_feat.append(student_text_features.cpu())

                    if need_teacher_text:
                        teacher_text_features = unwrap_model(teacher).encode_text(texts,normalize=True)
                        all_teacher_txt_feat.append(teacher_text_features.cpu())

                    # --- Removed CLIP logit and loss calculation ---
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    # all_image_features.append(image_features.cpu())
                    # all_text_features.append(text_features.cpu())
                    # logit_scale = logit_scale.mean()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # logits_per_text = logits_per_image.t()

                    # batch_size = images.shape[0]
                    # labels = torch.arange(batch_size, device=device).long()
                    # total_loss = (
                    #     F.cross_entropy(logits_per_image, labels) +
                    #     F.cross_entropy(logits_per_text, labels)
                    # ) / 2


                #TODO: Decide if maybe_compute_generative_loss is still relevant.
                # gen_loss = maybe_compute_generative_loss(model_out)

                # cumulative_loss += total_loss * batch_size
                # num_samples += batch_size
                # if is_master(args) and (i % 100) == 0:
                #     logging.info(
                #         f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                #         f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    # if gen_loss is not None:
                    #     cumulative_gen_loss += gen_loss * batch_size
                    #     logging.info(
                    #         f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            # val_metrics = get_clip_metrics(
            #     image_features=torch.cat(all_image_features),
            #     text_features=torch.cat(all_text_features),
            #     logit_scale=logit_scale.cpu(),
            # )
            # loss = cumulative_loss / num_samples
            # metrics.update(
            #     {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            # )
            # if gen_loss is not None:
            #     gen_loss = cumulative_gen_loss / num_samples
            #     metrics.update({"val_generative_loss": gen_loss.item()})


        # --- Concatenate Features ---
        teacher_img_feat = torch.cat(all_teacher_img_feat)
        student_img_feat = torch.cat(all_student_img_feat)
        student_txt_feat = torch.cat(all_student_txt_feat)
        if need_teacher_text:
            teacher_txt_feat = torch.cat(all_teacher_txt_feat)
        
        # --- 1. Image Encoder Similarity Metrics ---
        cos = F.cosine_similarity(F.normalize(teacher_img_feat.float()), F.normalize(student_img_feat.float()), dim=-1)
        metrics["similarity_mean"] = cos.mean().item()
        metrics["similarity_median"] = cos.median().item()

        # --- 2. Student Standalone CLIP Performance Metrics ---
        s_scale = unwrap_model(student).logit_scale.exp().cpu() # Get student scale
        student_clip_metrics = get_clip_metrics(
            image_features=student_img_feat.float(),
            text_features=student_txt_feat.float(),
            logit_scale=s_scale
        )
        # Add a prefix to distinguish these clearly
        metrics.update({f"student_clip_{k}": v for k,v in student_clip_metrics.items()})

        # --- 3. Optional: Teacher Baseline CLIP Performance Metrics ---
        if need_teacher_text:
            t_scale = unwrap_model(teacher).logit_scale.exp().cpu()  # Get teacher scale
            teacher_clip_metrics = get_clip_metrics(
                image_features=teacher_img_feat.float(),
                text_features=teacher_txt_feat.float(),
                logit_scale=t_scale
            )
            # Add a prefix for the baseline
            metrics.update({f"teacher_clip_{k}": v for k,v in teacher_clip_metrics.items()})



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


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
