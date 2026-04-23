# core/logger.py
import logging
import sys
import io
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with consistent formatting.
    Handles Windows console encoding (cp1252) safely.
    All Unicode characters render without errors.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler with safe Unicode encoding ─────────────
    # On Windows, stdout uses cp1252 by default
    # which cannot encode many Unicode characters (arrows, etc.)
    # Wrap in utf-8 with 'replace' fallback for safety
    try:
        if hasattr(sys.stdout, 'buffer'):
            # Standard case: stdout has an underlying buffer
            utf8_stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding = 'utf-8',
                errors   = 'replace',
                line_buffering = True,
            )
            console_handler = logging.StreamHandler(utf8_stdout)
        else:
            # Fallback: already wrapped or special stdout
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.stream.reconfigure(
                encoding='utf-8', errors='replace'
            ) if hasattr(
                console_handler.stream, 'reconfigure'
            ) else None
    except Exception:
        # Last resort: plain stdout, errors silently replaced
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler — always utf-8, no encoding issues ────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / "raptor_rag.log",
        encoding = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger