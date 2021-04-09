import logging

import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def get_logger(name: str = "BERT-PT"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(tqdm_handler)

    return logger
