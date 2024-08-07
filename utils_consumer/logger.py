import logging
from typing import Any
from pathlib import Path

class Logger(object):
    def __init__(self, name: str = None):
        """
        Initializes a new instance of the Logger class.

        Args:
            name (str, optional): The name of the logger. Defaults to None.

        Initializes the logger with the specified name and sets the log level to DEBUG.
        Sets the log formatter to include the timestamp, name, level, and message.
        Ensures that the logger does not have duplicate handlers by adding console and file handlers if necessary.

        Returns:
            None
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Ensure handlers are not duplicated
        if not self.logger.hasHandlers():
            self._add_console_handler()
            self._add_file_handler()

    def _add_console_handler(self):
        """
        Adds a console handler to the logger.

        This function creates a new `StreamHandler` object and sets its formatter to `self.formatter`.
        It then adds the console handler to the logger using the `addHandler` method.

        Parameters:
            None

        Returns:
            None
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """
        Adds a file handler to the logger.

        This function creates a `logging.FileHandler` object with the log file path
        constructed using the logger's name and the directory "logs". The file handler
        is then configured with the formatter and added to the logger.

        Parameters:
            None

        Returns:
            None
        """
        log_dir = Path(r"logs")
        log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        log_file_path = log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Returns the logger object associated with this instance.

        :return: The logger object.
        :rtype: logging.Logger
        """
        return self.logger
    
    def log(self, level: str, message: str, error: Any = None, params: Any = None):
        """
        Logs a message with the specified level and optional error and params.

        Args:
            level (str): The log level (e.g., 'debug', 'info', 'warning', 'error', 'critical').
            message (str): The log message.
            error (Any, optional): The error object to include in the log message. Defaults to None.
            params (Any, optional): The additional parameters to include in the log message. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        if error:
            message = f"{message} | Error: {error}"
        if params:
            message = f"{message} | Params: {params}"

        log_method = getattr(self.logger, level.lower(), None)
        if callable(log_method):
            log_method(f"{message}\n\n")
        else:
            self.logger.error(f"Invalid log level: {level}. Message: {message}")



