import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial


import numpy as np
import torch
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP # Import DDP

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, get_model_config
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
# from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync

from open_clip_train.train_distill import train_one_epoch, evaluate
from open_clip.memory import ProductKeyArgs,HashingMemory

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def auto_layers_string(n_blocks, skip_first=True, step=2):
    """
    Generate a string like '2,4,6,8' for every 'step' layer up to n_blocks-1.
    If skip_first=True, skip layer 0 (and 1) so we start from layer 2, for instance.
    """
    start = 2 if skip_first else 0
    layer_indices = [i for i in range(start, n_blocks, step)]
    return ",".join(str(idx) for idx in layer_indices)


def build_memory_args_automatically(vision_blocks_count):
    # Injecting memory every 2 layers from layer 2 onwards:
    layers_str = auto_layers_string(vision_blocks_count, skip_first=True, step=2)
    # Then build your normal ProductKeyArgs
    mem_args = ProductKeyArgs(
        is_enabled=True,
        layers=layers_str,
        mem_n_keys=8,  #make sure 'mem_n_keys **2 mod BLOCK_SIZE(8) == 0'
        mem_heads=2,
        mem_knn=4,
        mem_k_dim=256,
        mem_v_dim=256,  #Replacing -1 with  a power of two for v_dim , reducing from 512 to 256
        mem_share_values=True,  #  if False ---> each layer gets its own memory table
    )
    return mem_args

def mp_parallelize_all(model):
        """
        Minimal function to replicate the original logic that calls `mp_parallelize`
        on each memory layer so it reassigns `self.values` from the global.
        """
        # We no longer need a mesh or distributed config
        for name, submodule in model.named_modules(): # Instead of referencing `model.layers`, doing a submodule scan
            if isinstance(submodule, HashingMemory): # More specific check
                logging.debug(f"[parallelize_model] calling mp_parallelize on submodule {name}")
                # Pass None for args not needed by the simplified mp_parallelize
                submodule.mp_parallelize(None, None, None, torch.float32) # Or appropriate dtype
        return model

def load_weights_with_memory_layers(student_model, teacher_state_dict, memory_args):
        """
        Loads weights from a teacher state_dict into a student model,
        skipping incompatible layers (like HashingMemory replacing MLP).

        Args:
            student_model (nn.Module): The student model instance (with memory layers).
            teacher_state_dict (dict): The state dictionary from the pre-trained teacher.
            memory_args (ProductKeyArgs): The configuration for memory layers,
                                        needed to know which layers were replaced.
        """
        student_sd = student_model.state_dict()
        new_student_sd = student_sd.copy() # Start with student's own initialized weights

        # Identify layers where MLP was replaced by HashingMemory in the student
        memory_layer_indices = set()
        if memory_args and memory_args.is_enabled and memory_args.layers:
            # Assuming parse_pk_layers returns the set of indices where memory is used
            # If parse_pk_layers is not accessible here, replicate its logic or pass the indices
            try:
                from open_clip.transformer import parse_pk_layers # Adjust import if needed
                memory_layer_indices = parse_pk_layers(memory_args.layers)
            except ImportError:
                # Basic parsing if the function isn't available directly
                logging.warning("parse_pk_layers not found. Using basic parsing for memory_args.layers.")
                try:
                    if isinstance(memory_args.layers, str) and memory_args.layers.lower() != 'none':
                        memory_layer_indices = set(map(int, memory_args.layers.split(',')))
                    elif isinstance(memory_args.layers, (list, tuple)):
                        memory_layer_indices = set(memory_args.layers)
                except Exception as e:
                    logging.error(f"Could not parse memory_args.layers: {memory_args.layers}. Error: {e}")
                    memory_layer_indices = set() # Fallback to empty set

        logging.info(f"Memory layers active at indices: {memory_layer_indices}")

        loaded_count = 0
        skipped_mlp_count = 0
        skipped_other_count = 0
        mismatched_shape_count = 0

        for key, teacher_param in teacher_state_dict.items():
            # Focus only on the visual part if necessary (adjust prefix if needed)
            if not key.startswith('visual.'):
                # If you only want to load visual weights, uncomment the next line
                # continue
                pass # Process all weights for now

            is_mlp_key_in_memory_layer = False
            # Check if this key belongs to an MLP block that *should have been* replaced by memory
            # Example key: "visual.transformer.resblocks.10.mlp.c_fc.weight"
            if '.mlp.' in key:
                try:
                    # Extract the layer index from the key
                    parts = key.split('.')
                    resblock_idx = parts.index('resblocks')
                    layer_idx_str = parts[resblock_idx + 1]
                    layer_idx = int(layer_idx_str)
                    if layer_idx in memory_layer_indices:
                        is_mlp_key_in_memory_layer = True
                except (ValueError, IndexError, TypeError):
                    # Handle cases where parsing fails, unlikely for valid keys
                    pass

            if is_mlp_key_in_memory_layer:
                # This key belongs to an MLP in the teacher, but the corresponding
                # layer in the student has HashingMemory instead. Skip loading.
                skipped_mlp_count += 1
                # logging.debug(f"Skipping MLP key from memory layer {layer_idx}: {key}")
                continue

            # Now, check if the key exists in the student and shapes match
            if key in student_sd:
                student_param = student_sd[key]
                if teacher_param.shape == student_param.shape:
                    new_student_sd[key] = teacher_param.clone() # Use clone to avoid aliasing
                    loaded_count += 1
                    # logging.debug(f"Loaded key: {key} with shape {teacher_param.shape}")
                else:
                    logging.warning(
                        f"Shape mismatch for key {key}: "
                        f"Teacher shape {teacher_param.shape}, Student shape {student_param.shape}. Skipping."
                    )
                    mismatched_shape_count += 1
            else:
                # This key exists in the teacher but not the student (should be rare if not MLP)
                skipped_other_count += 1
                logging.warning(f"Key {key} found in teacher but not in student (and not skipped MLP). Skipping.")


        # Load the prepared state dictionary into the student model
        # Using strict=True is a good check here. It ensures that all keys defined
        # in the student model are present in `new_student_sd`. Since we started
        # with `student_sd.copy()`, this should always pass unless something
        # fundamental changed in the model structure unexpectedly.
        student_model.load_state_dict(new_student_sd, strict=True)

        logging.info(f"Weight loading complete.")
        logging.info(f" Weights loaded: {loaded_count}")
        logging.info(f" MLP weights skipped (replaced by memory): {skipped_mlp_count}")
        logging.info(f" Other weights skipped (missing in student): {skipped_other_count}")
        logging.info(f" Weights skipped (shape mismatch): {mismatched_shape_count}")
        logging.info(f" Total keys in teacher: {len(teacher_state_dict)}")
        logging.info(f" Total keys in student: {len(student_sd)}")


def main(args):
    args = parse_args(args)
    print('Current Arguments:', args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    logging.info(f"Rank {args.rank}: Initialized device. args.distributed is set to: {args.distributed}")

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"distill_memory_{model_name_safe}",
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
            f"loss_{args.loss_type}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (Rank: {args.rank}, local {args.local_rank}), World Size: {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    ############################
    # 1) TEACHER MODEL
    ############################
    logging.info(f"Rank {args.rank}: Creating teacher model {args.model}...")
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,                  
        args.pretrained,             
        device=device,               
        precision=args.precision,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        memory_args=None, # crucially, NOT providing memory_args here, so it remains None.
        **model_kwargs
    )
    model.to(device)
    logging.info(f"Rank {args.rank}: Teacher model created.")

    ############################
    # 2) STUDENT MODEL
    ############################
    teacher_model_cfg = get_model_config(args.model)
    vision_blocks_count = teacher_model_cfg["vision_cfg"]["layers"]
    memory_args = None
    if args.use_memory:
        logging.info(f"Rank {args.rank}: Building memory arguments automatically...")
        memory_args = build_memory_args_automatically(vision_blocks_count)
    
    #  --- Add HashingMemory State Reset ---
    if args.use_memory and memory_args.mem_share_values:
        logging.info(f"Rank {args.rank}: Resetting HashingMemory shared state...")
        HashingMemory.reset_shared_state() # Crucial before creating student

    logging.info(f"Rank {args.rank}: Creating student model {args.model} {'with' if args.use_memory else 'without'} memory...")

    student_model,student_preprocess_train, student_preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        memory_args=memory_args, # <--- This time we include the memory arguments
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )
    logging.info(f"Rank {args.rank}: Student model created.")

    # If using memory, we need to load the weights from the teacher model into the student model
    if args.use_memory:
        logging.info(f"Rank {args.rank}: Loading teacher weights into student...")
        teacher_state_dict = model.state_dict() # Get teacher state AFTER moving to device
        load_weights_with_memory_layers(student_model, teacher_state_dict, memory_args)
        logging.info(f"Rank {args.rank}: Teacher weights loaded.")

        # Freezing all parameters
        for param in student_model.parameters():
            param.requires_grad = False
        
        # Unfreezing only memory layers
        for module in student_model.modules():
            if isinstance(module, HashingMemory):
                for param in module.parameters():
                    param.requires_grad = True # Unfreeze memory

        # Logging total and trainable parameter counts
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        logging.info(f"Rank {args.rank}: Student - Total params: {total_params}, Trainable (memory): {trainable_params}")

    student_model.to(device)
    logging.info(f"Rank {args.rank}: Student model moved to {device}.")

    #logging which parameters are frozen
    # for name, param in student_model.named_parameters():
    #     if not param.requires_grad:
    #         logging.info(f"Parameter '{name}' is frozen (requires_grad=False)")

    # Applying Rank-Local Sharing via mp_parallelize_all (AFTER moving to device)
    if args.use_memory and memory_args.mem_share_values:
        logging.info(f"Rank {args.rank}: Applying rank-local memory sharing via mp_parallelize_all...")
        student_model = mp_parallelize_all(student_model) # Finalizes rank-local state
        logging.info(f"Rank {args.rank}: Rank-local sharing applied.")

    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()
        student_model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(student_model)}")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")


    # --- DDP Synchronization ---
    if args.distributed and not args.horovod:
        logging.info(f"Rank {args.rank}: Starting state_dict synchronization...")
        dist.barrier() # Wait for all ranks to reach this point (model created, moved, rank-local sharing done)

        # Use a list container; rank 0 puts state, others put placeholder
        sync_list = [student_model.state_dict() if args.rank == 0 else {}]
        dist.broadcast_object_list(sync_list, src=0) # Broadcast from rank 0

        if args.rank != 0:
            logging.info(f"Rank {args.rank}: Receiving and loading synchronized model state...")
            student_model.load_state_dict(sync_list[0]) # Load the state received from rank 0
            logging.info(f"Rank {args.rank}: State loaded.")
        else:
            logging.info("Rank 0: Broadcast complete.")

        dist.barrier() # Wait for all ranks to finish loading
        logging.info(f"Rank {args.rank}: State synchronization complete.")
    # --- End DDP Synchronization ---

    #  --- DDP Wrapping --- 
    # Wrap models with DDP *after* synchronization
    if args.distributed and not args.horovod:
        logging.info(f"Rank {args.rank}: Wrapping models with DDP...")
        if args.use_bn_sync:
            logging.info(f"Rank {args.rank}: Applying SyncBatchNorm...")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)
        logging.info(f"Rank {args.rank}: DDP wrapping complete.")
    #  --- End DDP Wrapping --- 

    # ----- create distributed_args via Memory Logic/META paper IF FSDP enabled ----
    # distributed_args = DistributedArgs()
    # distributed_args.dp_shard = 4        # shard across 4 GPUs
    # distributed_args.dp_replicate = 1    # only 1 node
    # distributed_args.tp_size = 1         # no tensor parallel
    # distributed_args.memory_parallel_size = 1  # no separate memory parallel
    # distributed_args.fsdp_type = "full_shard"  # fully shard to reduce memory
    # distributed_args.model_dtype = 'bf16'     # store model weights in precision
    # distributed_args.selective_activation_checkpointing = False
    # distributed_args.checkpoint_each_layer = False

    # # Create or load the device_mesh object
    # device_mesh = get_device_mesh(distributed_args)

    # # Define param_dtype or retrieve from your config
    # param_dtype = torch.float16 if args.precision == 'fp16' else torch.float32

    # model = parallelize_model(
    #     model,
    #     device_mesh,
    #     model_args=None,        # or an actual struct
    #     distributed_args=distributed_args,
    #     fsdp_grouping_plan=None,  # or pass if needed
    # )

    # # Log GPU memory usage immediately after parallelization
    # for i in range(torch.cuda.device_count()):
    #     allocated = torch.cuda.memory_allocated(i) / 1e9
    #     reserved = torch.cuda.memory_reserved(i) / 1e9
    #     logging.info(f"After parallelization - GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    #--------------------------
    
    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        # Create optimizer using the potentially DDP-wrapped student_model.parameters()
        # Filter parameters to only include those with requires_grad=True
        trainable_params = [p for p in student_model.parameters() if p.requires_grad]
        logging.info(f"Rank {args.rank}: Creating optimizer for {len(trainable_params)} trainable parameters.")

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            # named_parameters = list(model.named_parameters())
            # gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            # rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            named_trainable_parameters = [(n,p) for n, p in student_model.named_parameters() if p.requires_grad] # Use wrapped model
            gain_or_bias_params = [p for n, p in named_trainable_parameters if exclude(n, p)]
            rest_params = [p for n, p in named_trainable_parameters if include(n, p)]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            if is_master(args):
                defaults = copy.deepcopy(optimizer.defaults)
                defaults['weight_decay'] = args.wd
                defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
                logging.info(
                    f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
                )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None  # Disable gradient scaling for BF16 and other precisions

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            student_sd = checkpoint["state_dict"]
            is_student_ddp = hasattr(student_model, 'module') # Check if student_model is already wrapped in DDP at this point
            has_module_prefix = all(k.startswith('module.') for k in student_sd.keys()) # Check if state_dict keys have the prefix
            
            if is_student_ddp and not has_module_prefix:
                # If model is DDP but checkpoint isn't, add prefix
                student_sd = {'module.' + k: v for k, v in student_sd.items()}
                logging.info("Added 'module.' prefix to student state_dict keys for DDP loading.")

            elif not args.distributed and next(iter(student_sd.items()))[0].startswith('module'):
                student_sd = {k[len('module.'):]: v for k, v in student_sd.items()}
                logging.info("Removed 'module.' prefix from student state_dict keys for non-DDP loading.")

            # model.load_state_dict(sd)
            
            # Load the state into the student model instance
            try:
                student_model.load_state_dict(student_sd, strict=True)
                logging.info(f"Successfully loaded state_dict into student_model.")
            except RuntimeError as e:
                logging.error(f"Error loading student state_dict: {e}")
                logging.error("This might indicate a mismatch between the current student model definition and the one saved in the checkpoint.")
                raise e # Re-raise the error after logging

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    
    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_data(
        args,
        (preprocess_train, preprocess_val), # Use teacher preprocess? TODO:Check consistency
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    logging.info(f"Rank {args.rank}: Starting training loop from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        # train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        train_one_epoch(model,student_model,data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            # evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)
            evaluate(model,student_model,data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)


        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                # "state_dict": original_model.state_dict(),
                "state_dict": student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    
    # Final cleanup: destroy process group (safe shutdown)
    if torch.distributed.is_initialized():
        logging.info(f"Rank {args.rank}: Waiting at final barrier before destroying process group...")
        torch.distributed.barrier()  # <-- required for safe shutdown!
        logging.info(f"Rank {args.rank}: Destroying process group...")
        torch.distributed.destroy_process_group()
        logging.info("Destroyed distributed process group successfully.")

    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
