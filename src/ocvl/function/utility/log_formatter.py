# Source - https://stackoverflow.com/a
# Posted by Sergey Pleshakov, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-10, License - CC BY-SA 4.0
# Modified slighly for my use.

import logging

from colorama import Fore


class LogFormatter(logging.Formatter):

    grey = "\x1b[38;0m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;0m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    basic_format = "%(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: Fore.GREEN + basic_format + reset,
        logging.WARNING: Fore.YELLOW + basic_format + reset,
        logging.ERROR: Fore.RED + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
