from loguru import logger
from .coordinator import get_rank

def log_rank0(msg):
    if get_rank() == 0:
        logger.info(f"{msg}") 
