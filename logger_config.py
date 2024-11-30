import logging
from datetime import datetime

def setup_logger(log_file_name=None, log_level=logging.INFO):
    """
    Sets up logging configuration.
    
    Args:
        log_file_name (str): Name of the log file (optional). Defaults to 'app_<timestamp>.log'.
        log_level (int): Logging level (default: logging.INFO).
    """
    if log_file_name is None:
        # Use a default name if not provided
        log_file_name = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    logging.info(f"Logging initialized. Log file: {log_file_name}")
