from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _resolve_repo_root() -> Path:
    """Walk up the directory tree until we find the resource folders."""
    for candidate in PACKAGE_ROOT.parents:
        if (candidate / "models").is_dir() and (candidate / "data").is_dir():
            return candidate
    return PACKAGE_ROOT.parent


REPO_ROOT = _resolve_repo_root()
DEFAULT_HISTORY_LENGTH = 16
POINTER_GESTURE_ID = 2

__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "DEFAULT_HISTORY_LENGTH",
    "POINTER_GESTURE_ID",
]
