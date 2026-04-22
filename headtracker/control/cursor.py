"""Mouse cursor controller driven by head angles."""
import numpy as np
import pyautogui

from .filters import precision_curve, soft_deadzone

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


class CursorController:
    """Maps filtered head angles to screen coordinates and moves the mouse.

    Angles flow through:
      neutral subtraction → soft deadzone → precision curve → exponential smoothing
    """

    CORNER_MARGIN = 2
    WARMUP_FRAMES = 6

    def __init__(
        self,
        screen_w,
        screen_h,
        max_yaw_deg=14.0,
        max_pitch_deg=8.0,
        curve_knee=0.65,
        curve_knee_out=0.65,
        deadzone_inner_yaw=0.0,
        deadzone_outer_yaw=0.0,
        deadzone_inner_pitch=0.0,
        deadzone_outer_pitch=0.0,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.max_yaw_deg = max_yaw_deg
        self.max_pitch_deg = max_pitch_deg
        self.curve_knee = curve_knee
        self.curve_knee_out = curve_knee_out
        self.dz_inner_yaw = deadzone_inner_yaw
        self.dz_outer_yaw = deadzone_outer_yaw
        self.dz_inner_pitch = deadzone_inner_pitch
        self.dz_outer_pitch = deadzone_outer_pitch

        self.neutral_pitch = 0.0
        self.neutral_yaw = 0.0
        self.is_calibrated = False

        self._curr_x = screen_w / 2.0
        self._curr_y = screen_h / 2.0
        self._warmup_count = self.WARMUP_FRAMES

    def calibrate(self, pitch, yaw):
        self.neutral_pitch = pitch
        self.neutral_yaw = yaw
        self.is_calibrated = True

    def reset_warmup(self):
        self._warmup_count = self.WARMUP_FRAMES

    def update(self, pitch, yaw):
        """Compute cursor position from head angles and move the mouse.

        Returns (delta_pitch, delta_yaw) after deadzone for HUD display.
        """
        m = self.CORNER_MARGIN
        sw, sh = self.screen_w, self.screen_h

        delta_yaw = soft_deadzone(
            yaw - self.neutral_yaw, self.dz_inner_yaw, self.dz_outer_yaw)
        delta_pitch = soft_deadzone(
            pitch - self.neutral_pitch, self.dz_inner_pitch, self.dz_outer_pitch)

        norm_x = float(np.clip(delta_yaw / self.max_yaw_deg, -1.0, 1.0))
        norm_y = float(np.clip(delta_pitch / self.max_pitch_deg, -1.0, 1.0))

        mapped_x = precision_curve(norm_x, self.curve_knee, self.curve_knee_out)
        mapped_y = precision_curve(norm_y, self.curve_knee, self.curve_knee_out)

        target_x = float(np.clip(sw / 2 + mapped_x * (sw / 2), m, sw - 1 - m))
        target_y = float(np.clip(sh / 2 - mapped_y * (sh / 2), m, sh - 1 - m))

        dist = np.hypot(target_x - self._curr_x, target_y - self._curr_y)
        alpha = float(np.clip(dist / 100.0, 0.20, 0.55))
        self._curr_x += (target_x - self._curr_x) * alpha
        self._curr_y += (target_y - self._curr_y) * alpha
        if dist < 2.0:
            self._curr_x, self._curr_y = target_x, target_y

        self._curr_x = float(np.clip(self._curr_x, m, sw - 1 - m))
        self._curr_y = float(np.clip(self._curr_y, m, sh - 1 - m))

        if self._warmup_count > 0:
            self._warmup_count -= 1
        else:
            pyautogui.moveTo(int(self._curr_x), int(self._curr_y))

        return delta_pitch, delta_yaw
