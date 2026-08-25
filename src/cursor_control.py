"""
cursor_control.py

Maps a predicted class label to a fixed-magnitude movement vector and
updates a simulated cursor's 2D position. [ASSUMPTION — mapping/speed
values are our own design choice; see config.CLASS_TO_VECTOR and
config.CURSOR_STEP_SPEED. This produces STEPPED (discrete) cursor
movement, as discussed: classification-driven control does not
reproduce the smooth continuous movement of the original dataset's
regression-based decoders unless further smoothing/hybrid refinement
is added as a project extension.]
"""

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CursorController:
    def __init__(self, start_pos=(0.0, 0.0)):
        self.x, self.y = start_pos
        self.history = [(self.x, self.y)]

    def step(self, predicted_class: str, smoothing: float = 0.0):
        """
        Move the cursor one step in the direction of predicted_class.

        smoothing: optional exponential smoothing factor in [0,1) applied
        to the movement vector across consecutive steps, to reduce jerkiness.
        0.0 = no smoothing (pure discrete stepping). [ASSUMPTION / optional extension]
        """
        vec = config.CLASS_TO_VECTOR.get(predicted_class, (0.0, 0.0))
        dx = vec[0] * config.CURSOR_STEP_SPEED
        dy = vec[1] * config.CURSOR_STEP_SPEED

        if smoothing > 0 and len(self.history) > 1:
            prev_x, prev_y = self.history[-1]
            prev_dx = self.x - prev_x
            prev_dy = self.y - prev_y
            dx = smoothing * prev_dx + (1 - smoothing) * dx
            dy = smoothing * prev_dy + (1 - smoothing) * dy

        new_x = np.clip(self.x + dx, config.WORKSPACE_MIN, config.WORKSPACE_MAX)
        new_y = np.clip(self.y + dy, config.WORKSPACE_MIN, config.WORKSPACE_MAX)

        self.x, self.y = new_x, new_y
        self.history.append((self.x, self.y))
        return self.x, self.y

    def reset(self, start_pos=(0.0, 0.0)):
        self.x, self.y = start_pos
        self.history = [(self.x, self.y)]
