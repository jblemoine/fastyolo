import logging
import subprocess


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB color values to hexadecimal color code.

    Parameters
    ----------
    r : int
        Red color value (0-255)
    g : int
        Green color value (0-255)
    b : int
        Blue color value (0-255)

    Returns
    -------
    str
        Hexadecimal color code in format '#RRGGBB'
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def setup_logger(name: str, level: str = "INFO"):
    """Set up and return a simple console logger.

    Parameters
    ----------
    name : str
        Name for the logger
    level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL), by default "INFO"

    Returns
    -------
    logging.Logger
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
    """Check if ffmpeg is installed and accessible from PATH.

    Raises
    ------
    RuntimeError
        If ffmpeg is not found or not working properly
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
