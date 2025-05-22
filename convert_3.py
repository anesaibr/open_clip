# convert_4.py
import os
import tarfile
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
import time
import io
import logging
import pandas as pd
from multiprocessing import Manager
import queue
import sys
import pickle

# --- Configuration ---
# Base path where individual dataset tar files are located
DATASET_TARS_BASE_PATH = '/projects/0/prjs1465/ShareGPT4V/' 
# Names of the dataset tar files
TAR_FILES = {
    "sam": os.path.join(DATASET_TARS_BASE_PATH, "sam.tar"),
    "coco": os.path.join(DATASET_TARS_BASE_PATH, "coco.tar"),
    "llava": os.path.join(DATASET_TARS_BASE_PATH, "llava.tar"),
    # Add other datasets and their tar files here if needed
}
# CSV_PATH = "/projects/0/prjs1465/ShareGPT4V/sharegpt4v/train_A.csv"
# OUTPUT_DIR = "/scratch-shared/Sharegpt4v/wds_multitar_v1"
CSV_PATH  = "/projects/0/prjs1465/ShareGPT4V/sharegpt4v/train_B.csv"
OUTPUT_DIR = "/projects/0/prjs1465/ShareGPT4V/wds_sharegpt4v/trainB"
INDEX_OUTPUT_DIR = "/projects/0/prjs1465/ShareGPT4V/tar_indexes" 
SAMPLES_PER_OUTPUT_SHARD = 10000
SAMPLES_PER_DISPATCH_CHUNK =  10000 #10000

try:
    SLURM_CPUS_ALLOCATED = int(os.environ.get('SLURM_CPUS_PER_TASK', str(mp.cpu_count())))
except ValueError:
    SLURM_CPUS_ALLOCATED = mp.cpu_count()

NUM_WRITER_WORKERS = max(1, SLURM_CPUS_ALLOCATED - 2)
if SLURM_CPUS_ALLOCATED <= 2: NUM_WRITER_WORKERS = 1

CSV_SEP = "\t"
IMG_COL = "filepath"
CAP_COL = "title"
WRITER_QUEUE_SIZE = 2000

# Path prefix in CSV to strip to get member path for tar files
CSV_PATH_PREFIX_TO_STRIP = "/projects/0/prjs1465/ShareGPT4V/data/"

# --- Helper Functions (csv_rows, create_shard - largely unchanged) ---
def csv_rows(csv_path, img_col, cap_col, sep, has_header):
    # ... (keep your existing csv_rows function) ...
    read_opts = {'sep': sep, 'on_bad_lines': 'warn'}
    if has_header:
        read_opts['header'] = 0
    else:
        read_opts['header'] = None
        read_opts['names'] = [img_col, cap_col]

    logging.info(f"Streaming CSV '{csv_path}' with options: {read_opts}")
    processed_rows_count = 0
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=10000, iterator=True, **read_opts)):
            if chunk_idx == 0 and has_header:
                if img_col not in chunk.columns:
                    raise ValueError(f"Image key '{img_col}' not found in CSV header: {chunk.columns.tolist()}")
                if cap_col not in chunk.columns:
                    raise ValueError(f"Caption key '{cap_col}' not found in CSV header: {chunk.columns.tolist()}")

            for row_tuple in chunk.itertuples(index=False):
                try:
                    img_path_val = getattr(row_tuple, img_col)
                    caption_val = getattr(row_tuple, cap_col, "")
                except AttributeError:
                    logging.error(f"Cannot access columns '{img_col}', '{cap_col}' in row: {row_tuple}")
                    continue

                if pd.isna(caption_val): caption_val = ""
                yield processed_rows_count, str(img_path_val), str(caption_val)
                processed_rows_count += 1
    except Exception as e:
        logging.error(f"Error reading/processing CSV {csv_path} (around row ~{processed_rows_count}): {e}", exc_info=True)
        raise
    logging.info(f"Finished streaming CSV. Yielded {processed_rows_count} rows.")


def create_shard(args_create):
    output_shard_file_idx, samples_for_shard = args_create
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{output_shard_file_idx:06d}.tar")

    try:
        with tarfile.open(output_path, "w") as tf_out:
            for item_idx_in_shard, (img_filename_basename, img_data, caption) in enumerate(samples_for_shard):
                base_key, _ = os.path.splitext(img_filename_basename)
                img_member_name = f"{base_key}.jpg"
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

# --- Multi-Tar Aware Reader Process ---
def multi_tar_reader_process(
    image_info_map_param, 
    writer_queue,
    source_tar_files_config, 
    num_sentinels_to_send
):
    pid = os.getpid()
    # --- BEGIN Logging Reconfiguration for Spawned Process ---
    # Format includes the process ID to distinguish logs from this reader.
    log_format = f'%(asctime)s - %(levelname)s - MT_READER {pid} - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)], # Send to stdout
        force=True # Necessary if basicConfig might be called by other modules imported by the child.
    )
    # --- END Logging Reconfiguration ---

    logging.info(f"Starting. Processing {len(image_info_map_param)} entries from CSV map.")

    # 1. Open all tar files and build member maps
    tar_file_objects = {}
    tar_member_maps = {} 

    for source_tag, tar_path in source_tar_files_config.items():
        
        index_file_path = os.path.join(INDEX_OUTPUT_DIR, f"{source_tag}_index.pkl")
        if not os.path.exists(tar_path):
            logging.warning(f"[MT_READER {pid}] Source tar for '{source_tag}' not found: {tar_path}")
            continue
        if not os.path.exists(index_file_path):
            logging.warning(f"[MT_READER {pid}] Pre-built index for '{source_tag}' not found: {index_file_path}. This source will be slow or skipped.")
            # Optionally, could fall back to building it on the fly here, but that defeats the purpose
            continue
        try:
            logging.info(f"[MT_READER {pid}] Loading pre-built index for '{source_tag}' from {index_file_path}...")
            # sys.stdout.flush() # Not needed if logging is configured for stdout and flushes
            start_load_time   = time.time()
            with open(index_file_path, "rb") as f_idx:
                member_map = pickle.load(f_idx)
            tar_member_maps[source_tag] = member_map
            logging.info(f"[MT_READER {pid}] Loaded index for '{source_tag}' ({len(member_map)} entries) in {time.time() - start_load_time:.2f}s.")
            
            tar_file_objects[source_tag] = tarfile.open(tar_path, "r")
            logging.info(f"[MT_READER {pid}] Opened tar file for '{source_tag}': {tar_path}")

            # tar_file_objects[source_tag] = tf
            # member_map_build_start_time = time.time()
            # member_map = {}
            # member_count_for_map = 0
            # last_map_log_time = time.time()

            # # Iterate to build map (this is the potentially long part for each tar)
            # logging.info(f"Iterating members of '{source_tag}.tar' to build map...")
            # for memberinfo in tf: 
            #     if memberinfo.isfile():
            #         member_map[memberinfo.name] = memberinfo # Store full member name as key
            #     member_count_for_map += 1
            #     if member_count_for_map % 100000 == 0: # Log every 100k members scanned
            #         current_map_log_time = time.time()
            #         if current_map_log_time - last_map_log_time > 30: # Or at least every 30s
            #             logging.info(f"...mapping '{source_tag}': processed {member_count_for_map} members for map...")
            #             last_map_log_time = current_map_log_time
            
            # tar_member_maps[source_tag] = member_map
            # logging.info(f"Mapped {len(member_map)} file members for '{source_tag}' (scanned {member_count_for_map} total tar entries) in {time.time() - member_map_build_start_time:.2f}s.")
            # # sys.stdout.flush() # Not needed

        except Exception as e_load_index:
            logging.error(f"[MT_READER {pid}] Failed to load index or open tar for '{source_tag}': {e_load_index}", exc_info=True)
            if source_tag in tar_file_objects and tar_file_objects[source_tag]:
                try: tar_file_objects[source_tag].close()
                except: pass
            if source_tag in tar_file_objects: del tar_file_objects[source_tag]
            if source_tag in tar_member_maps: del tar_member_maps[source_tag]

    items_put_on_queue = 0
    processed_csv_entries = 0
    missing_in_tar_count = 0
    not_found_log_count = 0
    MAX_NOT_FOUND_LOGS = 20 # Limit how many "not found" messages we print to avoid flooding

    logging.info(f"Starting main extraction loop. Will iterate {len(image_info_map_param)} CSV entries.")
    # 2. Iterate through CSV-derived map and extract
    for csv_image_path, (first_caption, source_tag, member_path_in_tar) in tqdm(
        image_info_map_param.items(), desc="Reading images from tars", unit="image", position=0, leave=True
    ): # position and leave for tqdm in subprocess
        processed_csv_entries += 1
        img_data = None

        if source_tag not in tar_file_objects or source_tag not in tar_member_maps:
            missing_in_tar_count +=1
            if not_found_log_count < MAX_NOT_FOUND_LOGS:
                 logging.warning(f"Source '{source_tag}' for {csv_image_path} not available (tar not opened/mapped).")
                 not_found_log_count+=1
            continue

        tf_source = tar_file_objects[source_tag]
        member_map_source = tar_member_maps[source_tag]

        if member_path_in_tar in member_map_source:
            member_info = member_map_source[member_path_in_tar]
            try:
                with tf_source.extractfile(member_info) as f_member:
                    img_data = f_member.read()
                
                if img_data:
                    img_filename_basename = os.path.basename(csv_image_path)
                    writer_queue.put((img_filename_basename, img_data, first_caption), timeout=600)
                    items_put_on_queue += 1
                else:
                    logging.warning(f"Extracted empty data for {member_path_in_tar} from {source_tag}.tar")
                    missing_in_tar_count +=1

            except queue.Full:
                logging.error(f"Writer queue full (size {writer_queue.qsize()}) processing {member_path_in_tar}. Terminating.")
                raise # This will terminate the reader process
            except Exception as e_extract:
                logging.error(f"Error extracting {member_path_in_tar} from {source_tag}.tar: {e_extract}", exc_info=True)
                missing_in_tar_count +=1
        else:
            missing_in_tar_count +=1
            if not_found_log_count < MAX_NOT_FOUND_LOGS:
                logging.warning(f"Member '{member_path_in_tar}' (from CSV: {csv_image_path}) not found in map for '{source_tag}.tar'.")
                # For specific debugging:
                # if "000000125860.jpg" in member_path_in_tar:
                #     logging.info(f"DEBUG: '{member_path_in_tar}' specifically not found in map for {source_tag}. Keys sample: {list(member_map_source.keys())[:5]}")
                not_found_log_count+=1
            if not_found_log_count == MAX_NOT_FOUND_LOGS:
                logging.warning("Reached max 'not found' log messages. Further misses will not be logged individually.")


    # 3. Cleanup and send sentinels
    logging.info(f"Finished processing CSV map entries. Total items queued: {items_put_on_queue}. Entries missing/error: {missing_in_tar_count} out of {processed_csv_entries} CSV entries attempted.")
    for source_tag_close, tf_obj_close in tar_file_objects.items(): # Use different var names for clarity
        try:
            tf_obj_close.close()
            logging.info(f"Closed tar for '{source_tag_close}'.")
        except Exception as e_close:
            logging.error(f"Error closing tar for '{source_tag_close}': {e_close}")

    logging.info(f"Sending {num_sentinels_to_send} sentinel values to writer queue.")
    for i in range(num_sentinels_to_send):
        try:
            writer_queue.put(None, timeout=60)
        except queue.Full:
            logging.warning(f"Writer queue full sending sentinel {i+1}/{num_sentinels_to_send}.")
    logging.info(f"Exiting.")


# --- Main Script ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Script starting. Num writer workers: {NUM_WRITER_WORKERS}. Writer queue size: {WRITER_QUEUE_SIZE}. Samples per dispatch chunk: {SAMPLES_PER_DISPATCH_CHUNK}.")

    # 1. Build image_info_map: csv_original_path -> (first_caption, source_tag, member_path_in_tar)
    #    Also count total expected items for processing.
    image_info_map = {} 
    logging.info("Building image_info_map from CSV...")
    total_csv_rows_read = 0
    expected_items_to_process = 0

    for _, img_path_csv, caption_csv in csv_rows(CSV_PATH, IMG_COL, CAP_COL, CSV_SEP, has_header=False): #Change to to True if CSV has header
        total_csv_rows_read += 1
        
        # Determine source_tag and member_path_in_tar
        # This logic assumes CSV paths start with CSV_PATH_PREFIX_TO_STRIP
        # and the part after that is "<dataset_name>/<path_within_dataset_tar>"
        
        path_relative_to_data_dir = None
        if img_path_csv.startswith(CSV_PATH_PREFIX_TO_STRIP):
            path_relative_to_data_dir = img_path_csv[len(CSV_PATH_PREFIX_TO_STRIP):] # e.g., "coco/train2017/000000125860.jpg"
        else:
            logging.warning(f"CSV path '{img_path_csv}' does not start with expected prefix '{CSV_PATH_PREFIX_TO_STRIP}'. Cannot determine source tar. Skipping.")
            continue

        parts = path_relative_to_data_dir.split(os.sep, 1)
        source_tag_candidate = parts[0].lower() # "coco", "sam", "llava"
        
        member_path_in_tar = path_relative_to_data_dir # This IS the member path, e.g. "coco/train2017/000..."

        if source_tag_candidate in TAR_FILES:
            # Only add if it's the first time we see this full CSV path (to handle first caption logic)
            if img_path_csv not in image_info_map:
                image_info_map[img_path_csv] = (caption_csv, source_tag_candidate, member_path_in_tar)
                expected_items_to_process +=1
        else:
            logging.warning(f"Unknown source tag '{source_tag_candidate}' derived from path '{img_path_csv}'. Check TAR_FILES config. Skipping.")
            
    logging.info(f"Processed {total_csv_rows_read} CSV rows. Built map for {len(image_info_map)} unique image paths to process.")

    if not image_info_map:
        logging.error("Image info map is empty. Check CSV paths, format, content, and CSV_PATH_PREFIX_TO_STRIP.")
        sys.exit(1)
    for src_tag, tar_fpath in TAR_FILES.items(): # Check existence of specified tar files
        if not os.path.exists(tar_fpath):
            logging.warning(f"Configured source tar file for '{src_tag}' not found: {tar_fpath}. This source will be unavailable.")


    manager = Manager()
    processed_data_queue = manager.Queue(maxsize=WRITER_QUEUE_SIZE)
    
    # Start the Multi-Tar Reader Process
    reader_process = mp.Process( # Renamed from tar_reader
        target=multi_tar_reader_process,
        args=(image_info_map, processed_data_queue, TAR_FILES, NUM_WRITER_WORKERS)
    )
    reader_process.start()
    logging.info(f"Multi-Tar reader process ({reader_process.pid}) started.")

    # ... (Rest of the main script: writer_pool, main consumption loop, cleanup) ...
    # This part remains largely the same, as it consumes (img_filename_basename, img_data, caption)
    # from processed_data_queue and dispatches to create_shard.
    writer_pool = mp.Pool(NUM_WRITER_WORKERS)
    logging.info(f"Writer pool with {NUM_WRITER_WORKERS} workers started.")

    writer_concurrency_limit = NUM_WRITER_WORKERS + 4 
    writer_semaphore = mp.Semaphore(writer_concurrency_limit)
    logging.info(f"Writer pool concurrency limited by semaphore to {writer_concurrency_limit} tasks.")

    output_samples_buffer = []
    output_phys_shard_idx_counter = 0 
    items_dequeued_for_writing = 0
    sentinels_received = 0

    with tqdm(total=expected_items_to_process, desc="Images to WDS", unit="img") as pbar:
        try:
            while True:
                try:
                    data_item = processed_data_queue.get(timeout=120)

                    if data_item is None:
                        sentinels_received += 1
                        logging.info(f"Received sentinel {sentinels_received}/{NUM_WRITER_WORKERS} from reader.")
                        if sentinels_received >= NUM_WRITER_WORKERS:
                            logging.info("All expected sentinels received from reader.")
                            if not reader_process.is_alive() and processed_data_queue.empty():
                                logging.info("Reader exited and queue empty after all sentinels. Breaking consumption loop.")
                                break
                        continue

                    # data_item is (img_filename_basename, img_data, caption)
                    output_samples_buffer.append(data_item)
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

                except queue.Empty:
                    logging.debug("Data queue empty on timeout. Checking reader status...")
                    all_sentinels_in = sentinels_received >= NUM_WRITER_WORKERS
                    reader_dead_and_queue_empty = (not reader_process.is_alive()) and processed_data_queue.empty()

                    if all_sentinels_in and reader_dead_and_queue_empty:
                        logging.info("Queue empty, all sentinels received, and reader confirmed finished. Exiting consumption loop.")
                        break
                    if not reader_process.is_alive() and processed_data_queue.empty(): 
                        logging.info("Reader not alive and queue empty (fallback check). Exiting consumption loop.")
                        break
                except Exception as e_main_loop:
                    logging.error(f"Error in main data consumption loop: {e_main_loop}", exc_info=True)
                    break
            
            logging.info("Main consumption loop exited. Draining any remaining items from queue...")
            while True: 
                try:
                    data_item = processed_data_queue.get(timeout=10) 
                    if data_item is None: continue
                    
                    output_samples_buffer.append(data_item)
                    items_dequeued_for_writing +=1
                    pbar.update(1)

                    if len(output_samples_buffer) >= SAMPLES_PER_DISPATCH_CHUNK:
                        samples_to_write = output_samples_buffer[:SAMPLES_PER_DISPATCH_CHUNK]
                        writer_semaphore.acquire()
                        # ... (callbacks and apply_async as above) ...
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
                # ... (callbacks and apply_async as above) ...
                def_err_callback = lambda e, s=writer_semaphore, idx=output_phys_shard_idx_counter: (
                    logging.error(f"ERROR in final writer task for output phys_shard {idx}: {e}", exc_info=True), s.release())
                def_succ_callback = lambda _, s=writer_semaphore: s.release()
                writer_pool.apply_async(create_shard,
                                        args=((output_phys_shard_idx_counter, output_samples_buffer),),
                                        callback=def_succ_callback, error_callback=def_err_callback)
                output_phys_shard_idx_counter += 1

        finally:
            pbar.close()
            logging.info("Ensuring reader process has finished...")
            if reader_process.is_alive():
                reader_process.join(timeout=300) 
            if reader_process.is_alive():
                logging.warning("Reader process did not terminate cleanly after join. Forcing.")
                reader_process.terminate()
                reader_process.join()
            logging.info("Reader process has finished.")

            logging.info("Closing writer pool and waiting for all writing tasks to complete...")
            logging.info(f"Waiting for up to {writer_concurrency_limit} writer tasks to release semaphore (max 60s per task)...")
            all_sem_acquired_at_shutdown = True
            for i in range(writer_concurrency_limit): 
                if not writer_semaphore.acquire(timeout=60): 
                    logging.warning(f"Could not acquire writer semaphore slot {i+1}/{writer_concurrency_limit} during shutdown. Some tasks might still be running or stuck.")
                    all_sem_acquired_at_shutdown = False
                    break
            if all_sem_acquired_at_shutdown:
                 logging.info("All writer semaphore slots acquired, indicating tasks are complete or have released their slots.")
            # Release them back if acquired for check (not strictly necessary before close/join if callbacks are robust)
            # for i in range(writer_concurrency_limit):
            #     try: writer_semaphore.release()
            #     except ValueError: break # If already fully released

            writer_pool.close()
            writer_pool.join()
            logging.info("Writer pool closed. All writing tasks complete.")

        logging.info(f"Conversion process finished. Total physical WebDataset shards created: {output_phys_shard_idx_counter}.")
        logging.info(f"Total image-caption pairs processed for writing: {items_dequeued_for_writing} (Expected unique images from CSV: {expected_items_to_process})")