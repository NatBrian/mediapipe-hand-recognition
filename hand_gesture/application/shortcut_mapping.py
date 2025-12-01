"""Simple mapping of gestures to keyboard shortcuts (cross-platform).

This file maps gesture names to keyboard shortcuts for easy customization.
Modify the SHORTCUT_MAPPING dictionary to change which shortcuts are triggered.
Automatically adapts to Mac or Windows.
"""
import platform
from typing import Dict, List, Optional

# Detect platform
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# Mapping of gesture names to keyboard shortcuts
# Format: "gesture_name": ["key1", "key2", ...] where keys are pyautogui key names
# For Mac: 'command' = Cmd, 'option' = Alt, 'ctrl' = Control
# For Windows: 'win' = Windows key, 'ctrl' = Control, 'alt' = Alt
SHORTCUT_MAPPING: Dict[str, List[str]] = {
    # Keypoint gestures (static hand signs) - ACTIVE
    "Open": ["command", "tab"] if IS_MAC else ["alt", "tab"],  # App Switcher (Mac: Cmd+Tab, Windows: Alt+Tab)
    "Pointer": None,  # No shortcut (used for tracking/index finger movement)
    "OK": ["command", "space"] if IS_MAC else ["win", "s"],  # Search (Mac: Spotlight, Windows: Search)
    "Peace": ["ctrl", "up"] if IS_MAC else ["win", "tab"],  # Mission Control/Task View (Mac: Ctrl+Up, Windows: Win+Tab)
    
    # COMMENTED OUT - Uncomment to enable
    # "Close": ["command", "w"] if IS_MAC else ["ctrl", "w"],  # Close window/tab
    # "Metal": ["command", "`"] if IS_MAC else ["alt", "tab"],  # Switch between windows of same app
    
    # Point history gestures (dynamic finger movements) - COMMENTED OUT
    # "Stop": ["escape"],  # Escape key (same on both platforms)
    # "Clockwise": ["command", "right"] if IS_MAC else ["ctrl", "tab"],  # Next tab/window
    # "Counter Clockwise": ["command", "left"] if IS_MAC else ["ctrl", "shift", "tab"],  # Previous tab/window
    # "Move": ["command", "m"] if IS_MAC else ["win", "down"],  # Minimize window
    # "Eight": ["command", "h"] if IS_MAC else ["win", "d"],  # Hide application / Show desktop
}

# Optional: Map gesture IDs to gesture names if needed
# This allows using IDs instead of names
KEYPOINT_GESTURE_NAMES = [
    "Open",      # ID 0
    "Close",     # ID 1
    "Pointer",   # ID 2
    "OK",        # ID 3
    "Peace",     # ID 4
    "Metal",     # ID 5
]

POINT_HISTORY_GESTURE_NAMES = [
    "Stop",              # ID 0
    "Clockwise",         # ID 1
    "Counter Clockwise", # ID 2
    "Move",              # ID 3
    "Eight",             # ID 4
]


def get_shortcut_for_keypoint_gesture(gesture_id: int) -> Optional[List[str]]:
    """Get shortcut keys for a keypoint gesture by ID."""
    if 0 <= gesture_id < len(KEYPOINT_GESTURE_NAMES):
        gesture_name = KEYPOINT_GESTURE_NAMES[gesture_id]
        return SHORTCUT_MAPPING.get(gesture_name)
    return None


def get_shortcut_for_point_history_gesture(gesture_id: int) -> Optional[List[str]]:
    """Get shortcut keys for a point history gesture by ID."""
    if 0 <= gesture_id < len(POINT_HISTORY_GESTURE_NAMES):
        gesture_name = POINT_HISTORY_GESTURE_NAMES[gesture_id]
        return SHORTCUT_MAPPING.get(gesture_name)
    return None

