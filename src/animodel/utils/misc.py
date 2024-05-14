import logging
import time

import hydra
from omegaconf import OmegaConf


def init_logger(logging_level="INFO"):
    logging_config = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",  # Specify date format without milliseconds
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": f"{logging_level}",
            },
        },
        "root": {"handlers": ["console"], "level": f"{logging_level}"},
    }
    # Apply the logging configuration
    logging.config.dictConfig(logging_config)

    # Getting a logger with __name__
    logger = logging.getLogger(__name__)

    return logger


class Timer:
    def __init__(self, name=None, log_fn=print, print_start=False):
        self.name = name
        self.log_fn = log_fn
        self.print_start = print_start

    def __enter__(self):
        if self.name is not None and self.print_start:
            self.log_fn(f"Timing [{self.name}]")
        self.start_time = time.time()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        name_str = f"[{self.name}] " if self.name is not None else ""
        self.log_fn(f"{name_str}Time lapsed: {time.time() - self.start_time:.1f}s")


def parse_configs_and_instantiate(target):
    """
    Parses command line arguments and instantiates the target function. The command line arguments
    can contain an argument `configs=...`, which should list configuration files separated by
    commas. In that case, it merges these files, and instantiates the target function using
    the combined configuration settings. Command line arguments can override settings specified
    in the configuration files.

    Args:
        target: The target function to instantiate.
    """
    # Parse command line arguments - expecting list of configuration files as input argument configs=...
    config = OmegaConf.from_cli()

    # Load configurations from the specified files
    if "configs" in config:
        # Split the 'configs' argument into a list
        config_files = config.configs.split(",")

        # Remove the 'configs' argument from the command line arguments - not used in the main function
        config.pop("configs")

        # Load and merge configurations from the specified files
        file_configs = OmegaConf.create()
        for filename in config_files:
            file_config = OmegaConf.load(filename)
            file_configs = OmegaConf.merge(file_configs, file_config)

        # Merge command line arguments into the YAML configuration
        # Command line arguments will override YAML configuration if provided
        config = OmegaConf.merge(file_configs, config)

    config = OmegaConf.create({"_target_": target, **config})
    # Instantiate the function with arguments from the configuration
    hydra.utils.instantiate(config)
