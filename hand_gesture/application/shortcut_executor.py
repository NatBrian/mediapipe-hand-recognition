"""Shortcut executor using pyautogui for cross-platform shortcuts."""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

try:
    import pyautogui
except ImportError:
    pyautogui = None  # type: ignore

from hand_gesture.application.shortcut_mapping import (
    get_shortcut_for_keypoint_gesture,
    get_shortcut_for_point_history_gesture,
)


class ShortcutExecutor:
    """Executes keyboard shortcuts based on recognized gestures."""

    def __init__(self, debounce_seconds: float = 0.3, enabled: bool = True):
        """Initialize the shortcut executor.
        
        Args:
            debounce_seconds: Minimum time between executing the same shortcut (prevents spam)
            enabled: Whether shortcuts are enabled (can be toggled)
        """
        if pyautogui is None:
            raise ImportError(
                "pyautogui is not installed. Install it with: pip install pyautogui"
            )
        
        self.debounce_seconds = debounce_seconds
        self.enabled = enabled
        self._last_executed_shortcut: Optional[tuple] = None
        self._last_execution_time: float = 0.0
        
        # Safety: Set a small pause between pyautogui actions
        pyautogui.PAUSE = 0.05
        
        # Mouse control settings
        self.screen_width, self.screen_height = pyautogui.size()
        self.mouse_enabled = True

    def execute_shortcut(self, keys: List[str]) -> bool:
        """Execute a keyboard shortcut.
        
        Args:
            keys: List of key names (e.g., ["command", "tab"])
            
        Returns:
            True if shortcut was executed, False otherwise
        """
        if not self.enabled or not keys:
            return False

        # Create a unique identifier for this shortcut
        shortcut_id = tuple(sorted(keys))
        current_time = time.time()

        # Debounce: Don't execute the same shortcut too frequently
        if (
            self._last_executed_shortcut == shortcut_id
            and (current_time - self._last_execution_time) < self.debounce_seconds
        ):
            return False

        try:
            # For arrow keys, pyautogui uses 'up', 'down', 'left', 'right'
            # Press all keys simultaneously
            pyautogui.hotkey(*keys)
            self._last_executed_shortcut = shortcut_id
            self._last_execution_time = current_time
            print(f"[Shortcut] ✓ Executed: {keys}")
            return True
        except Exception as e:
            print(f"[Shortcut] ✗ Error executing {keys}: {e}")
            return False

    def execute_keypoint_gesture(self, gesture_id: int) -> bool:
        """Execute shortcut for a keypoint gesture by ID.
        
        Args:
            gesture_id: The gesture ID from the keypoint classifier
            
        Returns:
            True if shortcut was executed, False otherwise
        """
        keys = get_shortcut_for_keypoint_gesture(gesture_id)
        if keys:
            # Debug: print when shortcut is executed
            from hand_gesture.application.shortcut_mapping import KEYPOINT_GESTURE_NAMES
            gesture_name = KEYPOINT_GESTURE_NAMES[gesture_id] if gesture_id < len(KEYPOINT_GESTURE_NAMES) else f"ID{gesture_id}"
            return self.execute_shortcut(keys)
        # Note: Pointer gesture (ID 2) is handled separately for mouse control
        return False

    def execute_point_history_gesture(self, gesture_id: int) -> bool:
        """Execute shortcut for a point history gesture by ID.
        
        Args:
            gesture_id: The gesture ID from the point history classifier
            
        Returns:
            True if shortcut was executed, False otherwise
        """
        keys = get_shortcut_for_point_history_gesture(gesture_id)
        if keys:
            return self.execute_shortcut(keys)
        return False

    def toggle_enabled(self) -> None:
        """Toggle whether shortcuts are enabled."""
        self.enabled = not self.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Set whether shortcuts are enabled."""
        self.enabled = enabled

    def move_mouse(self, landmark_list: Sequence[Sequence[int]], camera_width: int, camera_height: int) -> None:
        """Move mouse cursor based on index finger tip position (landmark index 8).
        
        Args:
            landmark_list: List of landmark coordinates from MediaPipe
            camera_width: Width of the camera frame
            camera_height: Height of the camera frame
        """
        if not self.mouse_enabled or not self.enabled:
            return
        
        # Index finger tip is landmark index 8
        if len(landmark_list) > 8:
            index_finger_x = landmark_list[8][0]
            index_finger_y = landmark_list[8][1]
            
            # Map camera coordinates to screen coordinates
            # Invert X because camera is mirrored
            screen_x = int((camera_width - index_finger_x) / camera_width * self.screen_width)
            screen_y = int(index_finger_y / camera_height * self.screen_height)
            
            # Clamp to screen bounds
            screen_x = max(0, min(screen_x, self.screen_width - 1))
            screen_y = max(0, min(screen_y, self.screen_height - 1))
            
            try:
                pyautogui.moveTo(screen_x, screen_y, duration=0.0)
            except Exception as e:
                print(f"[Mouse] Error moving mouse: {e}")

