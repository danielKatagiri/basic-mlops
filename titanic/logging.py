import logging


def get_logger() -> logging.Logger:
    """
    Return a logger
    Return:
        (logging.Logger): Logger object
    """
    logger = logging.getLogger("defaultLogger")
    sh = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    fmt_date = "%Y-%m-%dT%T%Z"
    formatter = logging.Formatter(fmt, fmt_date)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    return logger


LOGGER = get_logger()
