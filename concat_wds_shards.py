# concatenate_wds_shards_v2.py (no re-keying)
import os
import tarfile
import io
import shutil
from tqdm import tqdm
import logging
import sys
import glob # Added glob back

# Configuration for concatenation
INPUT_SHARDS_DIR = "/scratch-shared/Sharegpt4v/wds_multitar_v1" # Where your small shards are
OUTPUT_CONCAT_DIR = "/scratch-shared/Sharegpt4v/wds_multitar_v1_concat_10k_v2" # New output
LOGICAL_SHARD_SIZE_TARGET = 10000
# This is an estimate; actual samples per small shard might vary slightly for the last one.
# It's used to group small shards for concatenation.
APPROX_SAMPLES_IN_SMALL_SHARDS = 1000 # What SAMPLES_PER_DISPATCH_CHUNK was

def concatenate_shards_no_rekey(small_shard_paths, output_large_shard_path):
    """
    Concatenates multiple small WebDataset shards into a single larger one
    by copying members directly, assuming their base keys are already globally unique.
    """
    logging.info(f"Concatenating {len(small_shard_paths)} small shards into {output_large_shard_path} (no re-keying)")
    os.makedirs(os.path.dirname(output_large_shard_path), exist_ok=True)
    
    samples_in_current_large_shard = 0
    
    with tarfile.open(output_large_shard_path, "w") as tf_out:
        for small_shard_path in tqdm(small_shard_paths, desc=f"Processing for {os.path.basename(output_large_shard_path)}", leave=False):
            try:
                with tarfile.open(small_shard_path, "r") as tf_in:
                    for member_in in tf_in:
                        if not member_in.isfile(): # Should not happen if created correctly
                            continue
                        
                        # Extract member data and add to new tar
                        # The member_in.name is already like "sa_140340.jpg" or "sa_140340.txt"
                        member_data_stream = tf_in.extractfile(member_in)
                        if member_data_stream:
                            # Create a new TarInfo object to ensure it's clean, copy relevant attributes
                            info_out = tarfile.TarInfo(name=member_in.name)
                            info_out.size = member_in.size
                            info_out.mtime = member_in.mtime # Preserve modification time if desired
                            info_out.mode = member_in.mode   # Preserve mode if desired
                            # Add other attributes if necessary: info_out.uid, info_out.gid, etc.
                            
                            tf_out.addfile(info_out, member_data_stream)
                            
                            # Count samples (assume each .jpg or image type corresponds to one sample)
                            if member_in.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                                samples_in_current_large_shard += 1
                        else:
                            logging.warning(f"Could not extract member {member_in.name} from {small_shard_path}")
            except Exception as e:
                logging.error(f"Error processing small shard {small_shard_path}: {e}", exc_info=True)
    
    logging.info(f"Finished creating {output_large_shard_path} with approx. {samples_in_current_large_shard} samples.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

    if not os.path.isdir(INPUT_SHARDS_DIR):
        logging.error(f"Input directory not found: {INPUT_SHARDS_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_CONCAT_DIR, exist_ok=True)

    all_small_shard_files = sorted(glob.glob(os.path.join(INPUT_SHARDS_DIR, "*.tar")))
    if not all_small_shard_files:
        logging.error(f"No .tar files found in {INPUT_SHARDS_DIR}")
        sys.exit(1)
    
    logging.info(f"Found {len(all_small_shard_files)} small shards to process.")

    if APPROX_SAMPLES_IN_SMALL_SHARDS <= 0:
        logging.error("APPROX_SAMPLES_IN_SMALL_SHARDS must be positive.")
        sys.exit(1)
        
    small_shards_per_large_shard = LOGICAL_SHARD_SIZE_TARGET // APPROX_SAMPLES_IN_SMALL_SHARDS
    if LOGICAL_SHARD_SIZE_TARGET % APPROX_SAMPLES_IN_SMALL_SHARDS != 0:
        logging.warning(f"LOGICAL_SHARD_SIZE_TARGET ({LOGICAL_SHARD_SIZE_TARGET}) is not an even multiple of APPROX_SAMPLES_IN_SMALL_SHARDS ({APPROX_SAMPLES_IN_SMALL_SHARDS}). Last concatenated shard might be smaller/larger than target if grouping is strict.")

    if small_shards_per_large_shard == 0:
        small_shards_per_large_shard = 1 # Ensure at least one small shard is processed if target is smaller
        logging.warning(f"LOGICAL_SHARD_SIZE_TARGET is smaller than APPROX_SAMPLES_IN_SMALL_SHARDS. Each output shard will contain one input shard.")
        
    logging.info(f"Each large shard will be made from approximately {small_shards_per_large_shard} small shards.")

    output_large_shard_idx = 0
    for i in range(0, len(all_small_shard_files), small_shards_per_large_shard):
        chunk_of_small_shards = all_small_shard_files[i : i + small_shards_per_large_shard]
        if not chunk_of_small_shards:
            continue

        output_large_shard_name = f"{output_large_shard_idx:06d}.tar"
        output_large_shard_path = os.path.join(OUTPUT_CONCAT_DIR, output_large_shard_name)
        
        concatenate_shards_no_rekey(chunk_of_small_shards, output_large_shard_path)
        output_large_shard_idx += 1
        
    logging.info("Concatenation complete.")