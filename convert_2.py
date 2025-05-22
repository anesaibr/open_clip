import os
import tarfile
import csv
import pickle
import multiprocessing as mp
from tqdm import tqdm
from functools import lru_cache
from collections import defaultdict
import time
import io
import logging 
import pandas as pd
from itertools import islice
from multiprocessing import Manager
import queue
import sys


# Configuration
# INPUT_SHARDS = "/scratch-shared/Sharegpt4v/data/sam"  # Original image shards
LARGE_SAM_TAR_PATH = '/projects/0/prjs1465/ShareGPT4V/sam.tar'
CSV_PATH = "/projects/0/prjs1465/ShareGPT4V/sharegpt4v/train_A.csv"                # Caption CSV file
# INDEX_PATH = "/projects/0/prjs1465/ShareGPT4V/sam_index.pkl"  # Your existing index
OUTPUT_DIR = "/scratch-shared/Sharegpt4v/wds_final_v1"             # New WebDataset shards
SAMPLES_PER_OUTPUT_SHARD = 10000                           # Adjust based on desired shard size
NUM_WRITER_WORKERS = max(1, mp.cpu_count() // 2) # e.g., half the cores for writing                                    # Adjust based on CPU cores
CSV_SEP = "\t"  # Critical fix: tab separator
IMG_COL = "filepath"
CAP_COL = "title"
# How many (img_data, caption) items to buffer in main before dispatching to a writer worker
SAMPLES_PER_DISPATCH_CHUNK = 1000 # Reduced for OOM mitigation

WRITER_QUEUE_SIZE = 10000  #2000   # Adjust based on memory and performance needs
#------------------------------------------------------------

def csv_rows(csv_path, img_col, cap_col, sep, has_header):
    """Yield (row_idx, img_path, caption) lazily."""
    read_opts = {'sep': sep}
    if has_header:
        read_opts['header'] = 0 # Use the first row as header
        # No 'names' needed if header exists
    else:
        read_opts['header'] = None
        read_opts['names'] = [img_col, cap_col]

    logging.info(f"Streaming CSV with options: {read_opts}")
    processed_rows_count = 0
    try:
        # Use iterator=True and chunksize for memory efficiency
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=10000, iterator=True, **read_opts)):
             logging.debug(f"Processing CSV chunk {chunk_idx}")
             # Check column names after reading first chunk if header was expected
             if chunk_idx == 0 and has_header:
                 if img_col not in chunk.columns:
                      raise ValueError(f"Image key '{img_col}' not found in CSV header: {chunk.columns.tolist()}")
                 if cap_col not in chunk.columns:
                      raise ValueError(f"Caption key '{cap_col}' not found in CSV header: {chunk.columns.tolist()}")

             for row in chunk.itertuples(index=True): # index=True gives global row index
                img_path_val = getattr(row, img_col)
                # Need to handle potential missing data in captions
                caption_val = getattr(row, cap_col, None) # Use default None if col missing (shouldn't happen with names)
                if pd.isna(caption_val):
                    caption_val = "" # Replace NaN/NaT with empty string
                # yield row.Index, getattr(row, img_col), caption_val
                yield processed_rows_count, str(img_path_val), str(caption_val)
                processed_rows_count += 1
    except Exception as e:
        logging.error(f"Error reading or processing CSV {csv_path}: {e}")
        raise # Re-raise error to stop the process
    logging.info(f"Finished streaming CSV. Yielded {processed_rows_count} rows.")



def create_shard(args_create):
    output_shard_file_idx, samples_for_shard = args_create
    # samples_for_shard is now a list of (img_filename_basename, img_data, caption)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{output_shard_file_idx:06d}.tar")
    # logging.debug(f"[WRITER_POOL_WORKER {os.getpid()}] Creating {output_path} with {len(samples_for_shard)} samples.")

    try:
        with tarfile.open(output_path, "w") as tf_out:
            for item_idx_in_shard, (img_filename_basename, img_data, caption) in enumerate(samples_for_shard):
                # CHANGE 1: Use actual filename (basename) as the key, remove extension for .txt
                base_key, _ = os.path.splitext(img_filename_basename) # e.g., "sa_140340" from "sa_140340.jpg"
                
                img_member_name = f"{base_key}.jpg" # Or keep original: img_filename_basename
                txt_member_name = f"{base_key}.txt"

                img_info = tarfile.TarInfo(img_member_name)
                img_info.size = len(img_data)
                tf_out.addfile(img_info, io.BytesIO(img_data))

                caption_data = caption.encode("utf-8")
                txt_info = tarfile.TarInfo(txt_member_name)
                txt_info.size = len(caption_data)
                tf_out.addfile(txt_info, io.BytesIO(caption_data))
    except Exception as e:
        logging.error(f"[WRITER_POOL_WORKER {os.getpid()}] ERROR creating shard {output_path}: {e}", exc_info=True)
    return output_path

# --- Tar Reader Process Target Function ---
def tar_reader_process(tar_path, image_to_first_caption_map, writer_queue, total_unique_images_in_csv):
    pid = os.getpid()
    logging.info(f"[TAR_READER {pid}] Starting. Tar: {tar_path}. Targetting {total_unique_images_in_csv} unique images.")
    processed_tar_members = 0
    relevant_tar_members_found = 0
    items_put_on_queue = 0
    last_log_time = time.time()

    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf: # tf is iterable, good for sequential access
                processed_tar_members += 1
                if not member.isfile():
                    continue
                
                member_basename = os.path.basename(member.name)

                if member_basename in image_to_first_caption_map:
                    relevant_tar_members_found += 1
                    # CHANGE 2: image_to_first_caption_map now stores only the first caption directly
                    first_caption = image_to_first_caption_map[member_basename]
                    try:
                        # Log progress periodically for long tar scans
                        current_time = time.time()
                        if current_time - last_log_time > 60: # Log every 60 seconds
                            logging.info(f"[TAR_READER {pid}] Progress: Scanned {processed_tar_members} tar members. "
                                         f"Found {relevant_tar_members_found}/{total_unique_images_in_csv} relevant images. "
                                         f"Queued {items_put_on_queue} img-cap pairs.")
                            last_log_time = current_time

                        with tf.extractfile(member) as f_member:
                            img_data = f_member.read()
                        
                        # for cap_idx, cap in enumerate(captions):
                        #     # Wait if queue is full; high timeout implies it should eventually clear
                        #     writer_queue.put((img_data, cap), timeout=600) 
                        #     items_put_on_queue += 1

                        # CHANGE 1: Put (img_filename_basename, img_data, caption) onto the queue
                        writer_queue.put((member_basename, img_data, first_caption), timeout=600)
                        items_put_on_queue += 1

                            
                    except queue.Full:
                        logging.error(f"[TAR_READER {pid}] Writer queue full for extended period processing {member.name}. Writers might be too slow or stuck.")
                        # Decide on recovery: break, retry, or raise to terminate tar_reader
                        raise # Reraise to terminate this process if queue is persistently full
                    except Exception as e_extract:
                        logging.error(f"[TAR_READER {pid}] Error extracting/queueing {member.name}: {e_extract}", exc_info=True)
                
    except Exception as e_tar:
        logging.critical(f"[TAR_READER {pid}] Critical error reading tar {tar_path}: {e_tar}", exc_info=True)
    finally:
        logging.info(f"[TAR_READER {pid}] Finished scan. Processed {processed_tar_members} tar members. "
                     f"Found {relevant_tar_members_found} relevant images. Queued {items_put_on_queue} img-caption pairs.")
        logging.info(f"[TAR_READER {pid}] Sending {NUM_WRITER_WORKERS} sentinel values to writer queue.")
        for i in range(NUM_WRITER_WORKERS):
            try:
                writer_queue.put(None, timeout=60) # Sentinel
            except queue.Full:
                logging.warning(f"[TAR_READER {pid}] Writer queue full sending sentinel {i+1}/{NUM_WRITER_WORKERS}. Main loop might not see all sentinels.")
        logging.info(f"[TAR_READER {pid}] Exiting.")



# --- Main Process ---
if __name__ == "__main__":
    # mp.set_start_method('fork')  # Add this line
    mp.set_start_method('spawn', force=True)

    # Configure logging for the main process (do this early)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] # Ensure it goes to stdout
    )
    logging.info(f"Script starting. Num writer workers: {NUM_WRITER_WORKERS}. \
                 Writer queue size: {WRITER_QUEUE_SIZE}. Samples per dispatch chunk: {SAMPLES_PER_DISPATCH_CHUNK}. \
                 Target samples per WDS shard: {SAMPLES_PER_OUTPUT_SHARD}.")

    # 1. Build image_to_captions map from CSV
    # image_to_captions = defaultdict(list)
    # logging.info("Building image-to-caption map from CSV...")
    image_to_first_caption = {}
    logging.info("Building image-to-FIRST-caption map from CSV...")
    total_csv_img_caption_pairs = 0
    unique_images_in_csv = set()

    # Ensure csv_rows is defined globally
    for row_idx, img_path_csv, caption_csv in csv_rows(CSV_PATH, IMG_COL, CAP_COL, CSV_SEP, has_header=True):
        total_csv_img_caption_pairs +=1
        fname_csv = os.path.basename(str(img_path_csv))
        unique_images_in_csv.add(fname_csv)
        if fname_csv not in image_to_first_caption: # Only store the first encountered caption (a.k.a longest caption)
            image_to_first_caption[fname_csv] = caption_csv
        # image_to_captions[fname_csv].append(caption_csv)
        
    total_img_caption_pairs_for_processing = len(image_to_first_caption) 
    logging.info(f"Processed {total_csv_img_caption_pairs} CSV rows. Built map for {len(image_to_first_caption)} unique image files (using first caption found).")

    if not image_to_first_caption:
        logging.error("Image-to-caption map is empty. Check CSV path and format.")
        sys.exit(1)
    if not os.path.exists(LARGE_SAM_TAR_PATH):
        logging.error(f"Large SAM TAR file not found: {LARGE_SAM_TAR_PATH}")
        sys.exit(1)

    # Distribute work in large chunks
    # task_queue = mp.Queue()
    # result_queue = mp.Queue()
    manager = Manager()
    # task_queue = manager.Queue()
    # result_queue = manager.Queue()
    processed_data_queue = manager.Queue(maxsize=WRITER_QUEUE_SIZE)  # This queue will hold (img_data, caption) pairs produced by tar_reader

    # 2. Start the Tar Reader Process
    tar_reader = mp.Process(
        target=tar_reader_process,
        args=(LARGE_SAM_TAR_PATH, image_to_first_caption, processed_data_queue, len(image_to_first_caption))
    )
    tar_reader.start()
    logging.info(f"Tar reader process ({tar_reader.pid}) started.")
        
    # 3. Start Writer Pool (or dedicated writer processes) to consume from processed_data_queue
    # Using a Pool for create_shard is simpler here.
    writer_pool = mp.Pool(NUM_WRITER_WORKERS)
    logging.info(f"Writer pool with {NUM_WRITER_WORKERS} workers started.")

    writer_concurrency_limit = NUM_WRITER_WORKERS + 4 
    writer_semaphore = mp.Semaphore(writer_concurrency_limit)
    logging.info(f"Writer pool concurrency limited by semaphore to {writer_concurrency_limit} tasks.")

    output_samples_buffer = []
    output_phys_shard_idx_counter  = 0
    items_dequeued_for_writing = 0
    sentinels_received = 0
    
    # tqdm for dequeued items, total is number of img-caption pairs from CSV
    with tqdm(total=total_img_caption_pairs_for_processing, desc="Unique Imgs to WDS", unit="img") as pbar:
        try:
            while True: # Loop until all sentinels received and tar_reader joined and queue empty
                try:
                    data_item = processed_data_queue.get(timeout=120) # CHANGE 1: data_item is now (img_filename_basename, img_data, caption)

                    if data_item is None:
                        sentinels_received += 1
                        logging.info(f"Received sentinel {sentinels_received}/{NUM_WRITER_WORKERS} from tar_reader.")
                        if sentinels_received >= NUM_WRITER_WORKERS:
                            logging.info("All expected sentinels received. Tar reader should be finishing.")
                            # Wait for tar_reader to actually finish before breaking main consumption loop.
                            # The queue might still have data even after all sentinels are received if sentinels were queued first.
                            if not tar_reader.is_alive() and processed_data_queue.empty(): # Final check
                                logging.info("All sentinels received, tar_reader not alive, and queue empty. Breaking consumption loop.")
                                break
                        continue # Continue to ensure all sentinels are processed or queue is drained

                    # img_data, caption = data_item
                    # output_samples_buffer.append((img_data, caption))
                    output_samples_buffer.append(data_item) # data_item is the tuple
                    items_dequeued_for_writing +=1
                    pbar.update(1)

                    if len(output_samples_buffer) >= SAMPLES_PER_DISPATCH_CHUNK:
                        samples_to_write = output_samples_buffer[:SAMPLES_PER_DISPATCH_CHUNK]
                        
                        writer_semaphore.acquire()
                        
                        def_err_callback = lambda e, s=writer_semaphore, idx=output_phys_shard_idx_counter: (
                            logging.error(f"ERROR in writer task for output phys_shard {idx}: {e}", exc_info=True), s.release())
                        def_succ_callback = lambda _, s=writer_semaphore: s.release()
                        
                        writer_pool.apply_async(create_shard,
                                                args=((output_phys_shard_idx_counter, samples_to_write),),
                                                callback=def_succ_callback,
                                                error_callback=def_err_callback)
                        output_phys_shard_idx_counter += 1
                        output_samples_buffer = output_samples_buffer[SAMPLES_PER_DISPATCH_CHUNK:]
                        # logging.debug(f"Submitted output shard {webdataset_shards_created}. Buffer: {len(output_samples_buffer)}")

                except queue.Empty:
                    logging.debug("Data queue empty on timeout. Checking tar_reader status...")
                    all_sentinels_in = sentinels_received >= NUM_WRITER_WORKERS
                    tar_reader_dead_and_queue_empty = (not tar_reader.is_alive()) and processed_data_queue.empty()

                    if all_sentinels_in and tar_reader_dead_and_queue_empty:
                        logging.info("Queue empty, all sentinels received, and tar_reader confirmed finished. Exiting consumption loop.")
                        break
                    if not tar_reader.is_alive() and processed_data_queue.empty(): # Fallback
                        logging.info("Tar_reader not alive and queue empty (fallback check). Exiting consumption loop.")
                        break
                except Exception as e_main_loop:
                    logging.error(f"Error in main data consumption loop: {e_main_loop}", exc_info=True)
                    break
            
            # Drain any stragglers after loop break (e.g., if break was due to sentinels but queue not fully empty)
            logging.info("Main consumption loop exited. Draining any remaining items from queue...")
            while True:
                try:
                    data_item = processed_data_queue.get(timeout=10) # Short timeout for draining
                    if data_item is None: continue # Skip any remaining sentinels
                    
                    # img_data, caption = data_item
                    # output_samples_buffer.append((img_data, caption))
                    output_samples_buffer.append(data_item)
                    items_dequeued_for_writing +=1
                    pbar.update(1)

                    if len(output_samples_buffer) >= SAMPLES_PER_DISPATCH_CHUNK:
                        samples_to_write = output_samples_buffer[:SAMPLES_PER_DISPATCH_CHUNK]
                        writer_semaphore.acquire()

                        def_err_callback = lambda e, s=writer_semaphore, idx=output_phys_shard_idx_counter: (
                            logging.error(f"ERROR in writer task for output phys_shard {idx} (drain): {e}", exc_info=True), s.release())
                        def_succ_callback = lambda _, s=writer_semaphore: s.release()

                        writer_pool.apply_async(create_shard, args=((output_phys_shard_idx_counter, samples_to_write),),
                                                callback=def_succ_callback, error_callback=def_err_callback)
                        output_phys_shard_idx_counter += 1
                        output_samples_buffer = output_samples_buffer[SAMPLES_PER_DISPATCH_CHUNK:]
                except queue.Empty:
                    logging.info("Queue confirmed empty during drain.")
                    break
                except Exception as e_drain:
                    logging.error(f"Error draining queue: {e_drain}", exc_info=True)
                    break


            if output_samples_buffer:
                logging.info(f"Writing final partial output physical shard {output_phys_shard_idx_counter} with {len(output_samples_buffer)} samples.")
                writer_semaphore.acquire()

                def_err_callback = lambda e, s=writer_semaphore, idx=output_phys_shard_idx_counter: (
                    logging.error(f"ERROR in final writer task for output phys_shard {idx}: {e}", exc_info=True), s.release())
                def_succ_callback = lambda _, s=writer_semaphore: s.release()
                
                writer_pool.apply_async(create_shard,
                                        args=((output_phys_shard_idx_counter, output_samples_buffer),),
                                        callback=def_succ_callback, error_callback=def_succ_callback) # Error in original: should be error_callback
                output_phys_shard_idx_counter += 1

        finally:
            pbar.close()
            logging.info("Ensuring tar_reader process has finished...")
            if tar_reader.is_alive():
                tar_reader.join(timeout=300) 
            if tar_reader.is_alive():
                logging.warning("Tar reader process did not terminate cleanly after join. Forcing.")
                tar_reader.terminate()
                tar_reader.join() # Wait for termination
            logging.info("Tar reader process has finished.")

            logging.info("Closing writer pool and waiting for all writing tasks to complete...")
            logging.info(f"Waiting for up to {writer_concurrency_limit} writer tasks to release semaphore...")
            all_sem_released = True
            for i in range(writer_concurrency_limit): 
                if not writer_semaphore.acquire(timeout=60): # Wait longer for tasks to finish
                    logging.warning(f"Could not acquire semaphore slot {i+1}/{writer_concurrency_limit} during shutdown. Some tasks might be stuck.")
                    all_sem_released = False
                    break
            if all_sem_released:
                 logging.info("All writer semaphore slots acquired, indicating tasks are complete or have released.")
            
            writer_pool.close()
            writer_pool.join()
            logging.info("Writer pool closed. All writing tasks complete.")

        logging.info(f"Conversion process finished. Total WebDataset shards created: {output_phys_shard_idx_counter}.")
        logging.info(f"Total image-caption pairs processed for writing: {items_dequeued_for_writing} (Expected from CSV: {total_img_caption_pairs_for_processing})")