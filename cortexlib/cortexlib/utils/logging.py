import logging
from colorlog import ColoredFormatter


class Logger:
    def __init__(self):
        SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
        logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

        def success(self, message, *args, **kwargs):
            if self.isEnabledFor(SUCCESS_LEVEL):
                self._log(SUCCESS_LEVEL, message, args, **kwargs)

        logging.Logger.success = success

        logger = logging.getLogger()

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'bold_white',
                    'SUCCESS':  'bold_green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'bold_red',
                }
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.INFO)
        logger.propagate = False
        self.logger = logger

    def progress(self, msg):
        self.logger.info(f"⏳ {msg}...")

    def success(self, msg):
        self.logger.success(f"✅ {msg}!")

    def info(self, msg):
        self.logger.info(f"ℹ️ {msg}")

    def warning(self, msg):
        self.logger.warning(f"⚠️ {msg}")

    def error(self, msg, exc_info=True):
        self.logger.error(f"❌ {msg}", exc_info=exc_info)

    def debug(self, msg):
        self.logger.debug(f"🐛 {msg}")

    def critical(self, msg):
        self.logger.critical(f"💥 {msg}")
