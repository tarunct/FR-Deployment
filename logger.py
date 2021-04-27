import logging
import sys
from logging.handlers import TimedRotatingFileHandler


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def get_logger(file, name='Rotating Log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_handler = TimedRotatingFileHandler(file, when="d", interval=1, backupCount=0)
    log_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(filename)s;%(message)s")
    log_handler.setFormatter(formatter)

    logger.addHandler(log_handler)

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger
