import logging
import subprocess


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def run_command(command: list[str]) -> None:
    """
    Run a shell command and raise an error if it fails.

    Args:
        command: List of command arguments to execute

    Raises:
        RuntimeError: If the command fails, with stderr output in the error message
    """
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(command)} failed with error:\n{result.stderr}"
        )


def get_logger(name: str, level: str = "INFO"):
    """
    Set up and return a simple console logger.

    Args:
        name: Name for the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logger instance configured with console output
    """

    # Create logger
    logger = logging.getLogger(name)

    # Add console handler if logger doesn't have handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    return logger


def check_ffmpeg_installed() -> None:
    """
    Check if ffmpeg is installed and accessible from PATH.

    Raises:
        RuntimeError: If ffmpeg is not found or not working properly
    """
    try:
        # Try to run ffmpeg -version
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg is installed but not working properly. Error: {result.stderr}"
            ) from None

    except FileNotFoundError as err:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "- MacOS: brew install ffmpeg\n"
            "- Windows: Download from https://ffmpeg.org/download.html"
        ) from err
