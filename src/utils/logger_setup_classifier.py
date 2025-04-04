import logging

def get_logger(ch_log_level: int = logging.INFO, fh_log_level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with console and file handlers.

    Args:
        ch_log_level (int): Logging level for the console handler. Default is logging.INFO.
        fh_log_level (int): Logging level for the file handler. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(ch_log_level)
    
    # File Handler
    fh = logging.FileHandler('training.log')
    fh.setLevel(fh_log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger