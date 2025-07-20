import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
import sys
import os
import argparse


def get_factories_and_tokenizer(is_flair_baseline):
    """
    Dynamically modifies sys.path to import from the correct library (standard or FLAIR)
    and returns the necessary factory and tokenizer functions.
    """
    if is_flair_baseline:
        # The path to the FLAIR fork's src directory
        lib_path = os.path.join(os.path.dirname(__file__), 'flair_lib/flair/src')
        logging.info(f"Dynamically importing from FLAIR library: {lib_path}")
        # Point at the Flair‑fork’s src folder
        # lib_path = os.path.join(this_dir, 'flair_lib/flair/src')
        sys.path.insert(0, lib_path)
        # Flair’s entrypoint is flair.factory, not open_clip
        from flair.factory import create_model_and_transforms, get_tokenizer, get_model_config
    else:
        # The path to main open_clip src directory
        lib_path = os.path.join(os.path.dirname(__file__), 'src')
        logging.info(f"Dynamically importing from standard OpenCLIP library: {lib_path}")
        # Now, perform the import. Python will use the path we just added.
        from open_clip import create_model_and_transforms, get_tokenizer, get_model_config

    # Prepend the chosen library path to Python's search path.
    # insert(0, ...) ensures it's the first place Python looks.
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    
    
    
    return create_model_and_transforms, get_tokenizer, get_model_config

def main():
    # We need to get the parser from the logic file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--baseline', type=str, nargs='?', const="clip", default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Determine which library to use
    is_flair_baseline = pre_args.baseline == 'flair'
    
    # Get the correct factory functions
    create_model_and_transforms, get_tokenizer, get_model_config = get_factories_and_tokenizer(is_flair_baseline)

    # Now that the path is set and factories are ready, import and run the main logic
    from eval_memory import get_parser, run_evaluation
    parser = get_parser()
    args = parser.parse_args()
    run_evaluation(args, create_model_and_transforms, get_tokenizer, get_model_config)


if __name__ == "__main__":
    main()