import glob
import logging
import os
import re
import subprocess
import sys
import random
import argparse
import json
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from huggingface_hub import hf_hub_download
from copy import deepcopy

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


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
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


def download_weights_from_hf(model_repo, filename):
    # Define the custom cache directory relative to the current script
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    local_path = hf_hub_download(repo_id=model_repo, filename=filename, cache_dir=cache_dir)
    return local_path



def run_evaluation(args, create_model_and_transforms, get_tokenizer, get_model_config):
    # args = parse_args(args)
    # parser = combine_parsers()
    # args = parser.parse_args(args)
    # args = get_parser().parse_args(args)

    from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
    from open_clip_train.logger import setup_logging
    from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
    from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync

    from open_clip.memory import ProductKeyArgs, HashingMemory
    from open_clip_train.main_distill_memory import build_memory_args_automatically, mp_parallelize_all
    from eval_metrics import evaluate
    from eval_get_data import get_data


    if 'timm' not in args.opt:
        # set default opt params based on model name (only if timm optimizer not used)
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if not args.baseline and args.distilled_model_path is None:
        raise ValueError("--distilled_model_path must be provided unless running in --baseline only mode.")


    is_flair_baseline = args.baseline == 'flair'
    # create_model_and_transforms, get_tokenizer, get_model_config = get_factories_and_tokenizer(is_flair_baseline) # Get the correct factory functions based on the mode.
    
    
    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
            args.name = "-".join([
                "eval",
                date_str,
                f"model_{model_name_safe}",
            ])

    log_base_path = os.path.join(args.logs_dir, args.name)
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = 'out.log'
    args.log_path = os.path.join(log_base_path, log_filename)
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    setup_logging(args.log_path, logging.INFO)
    logging.info("Starting evaluation run.")
    
    # Setup reporting destinations
    use_wandb = 'wandb' in args.report_to
    use_tensorboard = 'tensorboard' in args.report_to
    writer = None
    if use_tensorboard:
        assert tensorboard is not None, "Tensorboard not installed."
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard")
        os.makedirs(args.tensorboard_path, exist_ok=True)
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if use_wandb:
        assert wandb is not None, "WandB not installed."
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            config=vars(args),
        )

    # random_seed(args.seed, 0)
    # 1. Creating the BASE model. This provides the text tower and logit scale.

    # Determining the source of pretrained weights
    pretrained_path = args.pretrained
    model_kwargs = {}

    if is_flair_baseline:
        # If running FLAIR baseline, download the weights and use the local path
        # The 'pretrained' argument for create_model_and_transforms can be a local path
        pretrained_path = download_weights_from_hf(model_repo=args.huggingface_repo_name,filename=args.huggingface_model_name)
        logging.info(f"FLAIR baseline mode: Using downloaded weights from {pretrained_path}")
        model_kwargs['init_logit_bias'] = 0.0 # The FLAIR checkpoint is SigLIP-style and contains a 'logit_bias'.
        

    logging.info(f"Loading base model '{args.model}' with pretrained weights from: '{pretrained_path}'")
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=pretrained_path,
        precision=args.precision,
        device=device,
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
        **model_kwargs,
    )

    if args.baseline is None:
        # 2. Creating the memory-augmented STUDENT model instance to act as a temporary container.
        logging.info("Preparing to load memory-augmented vision encoder.")
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


        student_model , _, _ = create_model_and_transforms(
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
        # — if we're using shared‐value memory, parallelize all HashingMemory submodules —
        if args.use_memory and memory_args.mem_share_values:
            logging.info(f"Rank {args.rank}: Applying mp_parallelize_all to model…")
            student_model = mp_parallelize_all(student_model)
            logging.info(f"Rank {args.rank}: mp_parallelize_all complete.")

        # load the checkpoint of the student model
        ckpt = torch.load(args.distilled_model_path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)

        # If you used DDP when saving, strip off any "module." prefixes:
        sd = { (k[len("module."): ] if k.startswith("module.") else k):v
            for k,v in sd.items() }

        # Now load, strict=True will verify every weight matches
        model_keys   = set(student_model.state_dict().keys())
        ckpt_keys    = set(sd.keys())
        missing = sorted(ckpt_keys - model_keys)
        extra   = sorted(model_keys - ckpt_keys)
        student_model.load_state_dict(sd, strict=True)
        logging.info(f"Loaded student checkpoint {args.distilled_model_path!r} (use_memory={args.use_memory})")

        # 3. Perform the "model surgery".
        logging.info("Transplanting memory-augmented vision encoder into the base model...")
        model.visual = deepcopy(student_model.visual)
        model.to(device)
        logging.info("Transplant complete. The final composite model is ready for evaluation.")

        # Clean up the temporary student model
        del student_model
        
    else:
        # If running in baseline-only mode, we just use the base model as is.
        logging.info("Running in baseline-only mode. Using the base model without memory augmentation.")
    
    
    start_epoch = 0
    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (None, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    if not any(k.startswith(('val', 'retrieval', 'sharegpt4v')) for k in data.keys()):
        logging.error("No evaluation datasets were specified or found. Please use flags like --retrieval_coco or --zeroshot-val.")
        return

    # Evaluate.
    args.save_logs = args.logs_dir and args.logs_dir.lower() != 'none' and is_master(args)
    metrics = evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)

    # --- 6. Log and Save Results ---
    if metrics:
        logging.info("Evaluation finished.")
        # Convert numpy and torch types to native Python types for JSON serialization
        serializable_metrics = {
            k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
            for k, v in metrics.items()
        }
        # Log to console
        print("\n--- Final Metrics ---")
        print(json.dumps(serializable_metrics, indent=4))
        
        # Save to JSON file
        if 'json' in args.report_to:
            results_file = os.path.join(log_base_path, "results.json")
            with open(results_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            logging.info(f"Results saved to {results_file}")
            if use_wandb:
                # wandb.save(results_file)
                wandb.save(results_file, base_path=args.logs_dir)

    else:
        logging.warning("Evaluation completed, but no metrics were returned.")
        
    if use_wandb and is_master(args):
        wandb.finish()

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def get_parser():
    """Defines command-line arguments strictly for evaluation."""
    parser = argparse.ArgumentParser(description="OpenCLIP Memory-Augmented Model Evaluation")
    
    # --- General Configuration from OpenCLIP params---
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--use-memory", 
        action='store_true', 
        default=False, 
        help="Enable memory layers.Ignored if --baseline-only is set.",
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )

    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )

    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )

    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )

    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )

    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum (for timm optimizers).")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--opt", type=str, default='adamw',
        help="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}']."
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)


    # --- Core Model Configuration ---
    # parser.add_argument("--baseline-only",action='store_true',default=False, help="Run evaluation on a standard, non-augmented CLIP model. This will ignore --distilled_model_path and --use-memory.")
    parser.add_argument(
        "--baseline",
        type=str,
        nargs='?', # Makes the argument optional
        const="clip", # Value if flag is present without an argument (e.g., --baseline)
        default=None, # Value if flag is not present
        help="Run a baseline evaluation. Can be 'clip' for a standard OpenCLIP model "
             "or 'flair' to download a pre-trained FLAIR model from Hugging Face.",
    )
    parser.add_argument("--distilled_model_path", type=str, default=None, help="Path to a specific student model checkpoint (.pt) file to load.")
    parser.add_argument("--huggingface-repo-name", type=str, default='', help="Hugging Face repo to download weights from.")
    parser.add_argument("--huggingface-model-name", type=str, default='', help="Filename of the model in the Hugging Face repo.")
    parser.add_argument("--inference-with-flair", action='store_true', default=False, help="If set, use the FLAIR library for inference. This is only relevant if --baseline is set to 'flair'.")

    # --- Data and Benchmark Specification ---
    parser.add_argument("--data-root-dir", type=str, default='', help="Root directory to your dataset, especially the COCO dataset.")
    parser.add_argument("--coco-data-root-dir", type=str, default='', help="Root directory to the COCO dataset.")
    parser.add_argument("--flickr-data-root-dir", type=str, default='', help="Root directory to the flickr datasets (but we simply use the root of the whole dataset).")
    parser.add_argument("--sharegpt4v-retrieval-dir", type=str, default='', help="Root directory to the share4v dataset.")
    parser.add_argument("--dci-retrieval-dir", type=str, default='', help="Root directory to the train dci dataset.")
    parser.add_argument("--iiw-retrieval-dir", type=str, default='', help="Root directory to the image in words dataset.")
    parser.add_argument("--docci-retrieval-dir", type=str, default='', help="Root directory to fine-grained docci retrieval.")
    parser.add_argument("--urban-1k-retrieval-dir", type=str, default='', help="Root directory to fine-grained urban-1k retrieval.")
    parser.add_argument("--retrieval-coco", action="store_true", default=False, help="Enable COCO retrieval task.")
    parser.add_argument("--retrieval-dci", action="store_true", default=False, help="Enable DCI retrieval task.")
    parser.add_argument("--retrieval-iiw", action="store_true", default=False, help="Enable IIW retrieval task.")
    parser.add_argument("--retrieval-sharegpt4v-1k", action="store_true", default=False, help="Enable ShareGPT4V retrieval (1k size).")
    parser.add_argument("--retrieval-sharegpt4v-10k", action="store_true", default=False, help="Enable ShareGPT4V retrieval (10k size).")
    parser.add_argument("--retrieval-flickr", action="store_true", default=False, help="Enable Flickr retrieval task.")
    parser.add_argument("--retrieval-urban-1k", action="store_true", default=False, help="Enable Urban-1k retrieval task.")
    parser.add_argument("--retrieval-docci", action="store_true", default=False, help="Enable DOCCI retrieval task.")
    parser.add_argument("--zeroshot-eval-datasets", type=str, default=None, help="Datasets that you want to do retrieval.")
   
    # Add other evaluation flags here as needed, following the pattern in flair/data.py
    parser.add_argument("--use_finegrained_iiw",default=True,action="store_true",
        help="If set to true, under the condition that we enable iiw, we further use the fine-grained iiw mode.")
    parser.add_argument("--flickr-val-or-test",type=str,default='val', choices=['val', 'testing'],
        help="Which dataset to be used for inference, default choices are val or test.")
    parser.add_argument("--dict-root-dir",type=str,default=None,help="Path to the preprocessed dictionaries to filter the dataset.")
    # --- Logging and Reporting ---

    parser.add_argument("--logs-dir", type=str, default="logs/", help="Directory to save logs and results.")
    parser.add_argument("--report-to", default='json', nargs='+', choices=['json', 'tensorboard', 'wandb'], help="Where to report results.")
    parser.add_argument("--wandb-project-name", type=str, default="memory-evaluation", help="Name of the W&B project.")
    
    return parser


# if __name__ == "__main__":
#     main(sys.argv[1:])