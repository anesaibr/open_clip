import logging
import sys

def setup_logging(log_file, level, include_host=False):
    # 1. Get the root logger instance
    root_logger = logging.getLogger()

    # 2. --- REMOVE EXISTING HANDLERS --- 
    # This is the crucial step to prevent duplicates
    if root_logger.hasHandlers():
        # Iterate over a copy list [:] so we can remove items
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # 3. Set the desired level ON THE ROOT LOGGER
    root_logger.setLevel(level)
    
    # 4. Create the custom formatter
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    # logging.root.setLevel(level)
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # for logger in loggers:
    #     logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

