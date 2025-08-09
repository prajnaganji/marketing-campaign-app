
from loguru import logger
import sys

# Configure Loguru
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")

# File logging
logger.add("logs.log", rotation="1 MB", retention=3, level="INFO", enqueue=True, compression="zip")
