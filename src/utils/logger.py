"""
Logging system for SR_Models.
Creates timestamped log files in separate info and error directories.
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
import colorama

PROJECT_ROOT = Path(__file__).parent.parent.parent


class SRLogger:
    """Custom logger for Super-Resolution models with separate info/error logs."""

    def __init__(self,
                 name: str = "SR_Models",
                 log_dir: str = "logs",
                 console: bool = True,
                 file: bool = True,
                 level: str = "INFO"):
        """
        Initialize logger with separate info and error handlers.

        Args:
            name: Logger name
            log_dir: Base directory for log files
            console: Whether to log to console
            file: Whether to log to files
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.log_dir = PROJECT_ROOT / log_dir
        self.console = console
        self.file = file
        self.level = getattr(logging, level.upper())

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels

        # Skip re‑configuration if this process already did it
        if getattr(self.logger, "_sr_configured", False):
            return

        self.logger._sr_configured = True

        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handlers only in the main process to avoid countless empty files
        is_main = (os.getpid() == os.getppid()) or (os.getenv("PYTHON_MAIN_PROCESS", "1") == "1")

        # Setup handlers
        if self.console:
            self._setup_console_handler()
        if self.file and is_main:
            self._setup_file_handlers()

    def _setup_console_handler(self):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)

        # Add colors for different levels
        class ColoredFormatter(logging.Formatter):
            """Custom formatter with colors."""

            if colorama:
                COLORS = {
                    'DEBUG': colorama.Fore.CYAN,
                    'INFO': colorama.Fore.GREEN,
                    'WARNING': colorama.Fore.YELLOW,
                    'ERROR': colorama.Fore.RED,
                    'CRITICAL': colorama.Fore.MAGENTA,
                }
                RESET = colorama.Style.RESET_ALL
            else:
                COLORS = {
                    'DEBUG': '\\033[36m',  # Cyan
                    'INFO': '\\033[32m',  # Green
                    'WARNING': '\\033[33m',  # Yellow
                    'ERROR': '\\033[31m',  # Red
                    'CRITICAL': '\\033[35m',  # Magenta
                }
                RESET = '\\033[0m'

            def __init__(self, fmt, datefmt=None):
                super().__init__(fmt, datefmt)
                self.default_formatter = logging.Formatter(fmt, datefmt)

            def format(self, record):
                # Store original levelname to restore it later
                original_levelname = record.levelname

                # Add color to the levelname for console output
                log_color = self.COLORS.get(original_levelname, self.RESET)
                record.levelname = f"{log_color}{original_levelname}{self.RESET}"

                # Format the record using the default formatter
                formatted_message = self.default_formatter.format(record)

                # Restore the original levelname so other handlers (like file handlers)
                # receive the unmodified record.
                record.levelname = original_levelname

                return formatted_message

        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_handlers(self):
        """Setup separate file handlers for info and error logs."""
        # Re‑use one timestamp for the whole run (all processes)
        timestamp = os.environ.get("SR_LOG_TIMESTAMP")

        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.environ["SR_LOG_TIMESTAMP"] = timestamp

        # Create log directories
        info_dir = Path(self.log_dir) / 'info'
        error_dir = Path(self.log_dir) / 'error'
        info_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        # Info handler (DEBUG, INFO levels)
        info_file = info_dir / f"{timestamp}_info.log"
        info_handler = logging.FileHandler(info_file, mode='a', encoding='utf-8')
        info_handler.setLevel(logging.DEBUG)
        info_handler.setFormatter(self.detailed_formatter)
        info_handler.addFilter(lambda record: record.levelno < logging.WARNING)
        self.logger.addHandler(info_handler)

        # Error handler (WARNING, ERROR, CRITICAL levels)
        error_file = error_dir / f"{timestamp}_error.log"
        error_handler = logging.FileHandler(error_file, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(error_handler)

        # Store file paths
        self.info_file = str(info_file)
        self.error_file = str(error_file)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        """Log error message with optional exception info."""
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False):
        """Log critical message with optional exception info."""
        self.logger.critical(message, exc_info=exc_info)

    def log_exception(self, message: str = "An exception occurred"):
        """Log exception with full traceback."""
        self.logger.error(message, exc_info=True)

    @contextmanager
    def catch_errors(self, message: str = "Error occurred", reraise: bool = True):
        """
        Context manager to catch and log errors.

        Args:
            message: Error message prefix
            reraise: Whether to re-raise the exception after logging

        Example:
            with logger.catch_errors("Failed to load model"):
                model = load_model()
        """
        try:
            yield
        except Exception as e:
            self.error(f"{message}: {str(e)}", exc_info=True)
            if reraise:
                raise

    def log_metrics(self, metrics: dict, step: Optional[int] = None, prefix: str = ""):
        """
        Log metrics in a formatted way.

        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch number
            prefix: Prefix for the log message
        """
        metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in metrics.items()])

        if step is not None:
            message = f"{prefix} [Step {step}] {metric_str}"
        else:
            message = f"{prefix} {metric_str}"

        self.info(message.strip())

    def log_training_start(self, model_name: str, config: dict):
        """Log training start with configuration summary."""
        self.info("=" * 80)
        self.info(f"Starting training for {model_name.upper()}")
        self.info("=" * 80)
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 80)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.info(f"Epoch {epoch}/{total_epochs} started")

    def log_epoch_end(self, epoch: int, metrics: dict, time_elapsed: float):
        """Log epoch end with metrics."""
        self.info(f"Epoch {epoch} completed in {time_elapsed:.2f}s")
        self.log_metrics(metrics, prefix="  Metrics:")

    def close(self):
        """Close all handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_logger: Optional[SRLogger] = None


def get_logger(name: str = "SR_Models",
               log_dir: str = "logs",
               console: bool = True,
               file: bool = True,
               level: str = "INFO") -> SRLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        log_dir: Base directory for log files
        console: Whether to log to console
        file: Whether to log to files
        level: Logging level

    Returns:
        Logger instance
    """
    global _logger

    if _logger is None:
        _logger = SRLogger(name, log_dir, console, file, level)

    return _logger


def setup_logger(config: dict) -> SRLogger:
    """
    Setup logger from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Logger instance
    """
    return get_logger(
        name="SR_Models",
        log_dir=config.get('output', {}).get('log_dir', 'logs'),
        console=config.get('logging', {}).get('console', True),
        file=config.get('logging', {}).get('file', True),
        level=config.get('logging', {}).get('level', 'INFO')
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = get_logger()

    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test metrics logging
    metrics = {
        'loss': 0.0234,
        'psnr': 28.45,
        'ssim': 0.892
    }
    logger.log_metrics(metrics, step=10, prefix="Training")

    # Test exception logging
    try:
        1 / 0
    except Exception:
        logger.log_exception("Division by zero error")

    # Test context manager
    with logger.catch_errors("Failed to process", reraise=False):
        raise ValueError("Test error")

    logger.info("Logger test completed")
