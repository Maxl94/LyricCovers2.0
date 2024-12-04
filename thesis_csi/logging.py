import logging
from datetime import datetime
from typing import Optional


def get_logger(name: str = "ai4mplus", level: int = logging.DEBUG) -> logging.Logger:
    """Get logger.

    Args:
        name (str): Logger name.
        level (int): Logger level.

    Returns:
        logging.Logger: Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


def progress_bar(
    iterable,
    total_steps: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    message: str = "Processing",
    interval: int = 1,
):
    """
    A wrapper around an iterable that logs progress as a progress bar.

    Args:
        iterable: Iterable object.
        total_steps: Total number of steps. Defaults to None.
        logger: Logger object. Defaults to None.
        message: Message to log. Defaults to "Processing".
        interval: Interval to log progress. Defaults to 1.

    Yields:
        Any: Item from the iterable.
    """
    start_time = datetime.now()

    if total_steps is None:
        try:
            total_steps = len(iterable)
        except TypeError:
            total_steps = None

    if logger is None:
        logger = get_logger(__name__)
        logger.setLevel(logging.INFO)

    for current_step, item in enumerate(iterable, start=1):
        if current_step % interval == 0 or (total_steps and current_step == total_steps):
            # Time elapsed since the start
            delta = datetime.now() - start_time
            time_str = f"{delta.days}d " + (datetime.min + delta).strftime("%H:%M:%S")

            if total_steps:
                # Calculate and log remaining time if total_steps is provided
                time_rem = delta / current_step * (total_steps - current_step)
                time_rem_str = f"{time_rem.days}d " + (datetime.min + time_rem).strftime(
                    "%H:%M:%S"
                )
                logger.info(f"{message} [{current_step}/{total_steps}] {time_str}/{time_rem_str}")
            else:
                # Log progress without remaining time if total_steps is not provided
                logger.info(f"{message} [{current_step}] {time_str}")

        yield item
