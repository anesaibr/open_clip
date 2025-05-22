import os
import tarfile
import pickle
import logging
import sys
import time

TAR_FILES_TO_INDEX = {
    "sam": "/projects/0/prjs1465/ShareGPT4V/sam.tar",
    "coco": "/projects/0/prjs1465/ShareGPT4V/coco.tar",
    "llava": "/projects/0/prjs1465/ShareGPT4V/llava.tar",
}
INDEX_OUTPUT_DIR = "/projects/0/prjs1465/ShareGPT4V/tar_indexes" # Create this directory

def build_and_save_index(source_tag, tar_path):
    pid = os.getpid() # Not strictly needed here but for consistency if used in MP
    logging.info(f"[{source_tag.upper()}_INDEXER] Starting index build for: {tar_path}")
    
    member_map = {} # member.name -> TarInfo object
    if not os.path.exists(tar_path):
        logging.error(f"[{source_tag.upper()}_INDEXER] Tar file not found: {tar_path}")
        return

    try:
        start_time = time.time()
        tf = tarfile.open(tar_path, "r")
        logging.info(f"[{source_tag.upper()}_INDEXER] Opened {tar_path}. Iterating members...")
        
        count = 0
        for memberinfo in tf: # TarInfo objects should be picklable
            # We only care about files for extraction, but map all for completeness if needed
            # if memberinfo.isfile(): # Or map all member types
            member_map[memberinfo.name] = memberinfo # Store the TarInfo object directly
            count +=1
            if count % 100000 == 0:
                logging.info(f"[{source_tag.upper()}_INDEXER] ...processed {count} members for index...")
        tf.close()
        duration = time.time() - start_time
        logging.info(f"[{source_tag.upper()}_INDEXER] Built index with {len(member_map)} entries in {duration:.2f}s.")

        os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)
        index_file_path = os.path.join(INDEX_OUTPUT_DIR, f"{source_tag}_index.pkl")
        with open(index_file_path, "wb") as f_out:
            pickle.dump(member_map, f_out)
        logging.info(f"[{source_tag.upper()}_INDEXER] Saved index to {index_file_path}")

    except Exception as e:
        logging.error(f"[{source_tag.upper()}_INDEXER] Error building index for {tar_path}: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Could run these in parallel if desired, but sequential is fine for a one-off
    for tag, path in TAR_FILES_TO_INDEX.items():
        build_and_save_index(tag, path)
    logging.info("All indexing tasks submitted/completed.")