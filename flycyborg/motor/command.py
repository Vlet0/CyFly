"""
command.py — WheelCommand dataclass for a differential-drive robot.

A differential-drive robot has two independently controlled wheels.
We expose three intuitive axes and convert to left/right PWM:

    forward  ∈ [-1, 1]   — longitudinal speed
    turn     ∈ [-1, 1]   — yaw rate (+1 = left, -1 = right)
    brake    ∈ [ 0, 1]   — braking signal (overrides forward)

AI2-THOR action mapping:
    |forward| > MOVE_THRESHOLD  → MoveAhead / MoveBack
    |turn|    > TURN_THRESHOLD  → RotateLeft / RotateRight
    brake > BRAKE_THRESHOLD     → Stop (stay)
"""

from __future__ import annotations

from dataclasses import dataclass, field

MOVE_THRESHOLD: float = 0.15
TURN_THRESHOLD: float = 0.15
BRAKE_THRESHOLD: float = 0.6

# AI2-THOR discrete actions
ACTIONS = ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Pass"]


@dataclass
class WheelCommand:
    forward: float = 0.0   # [-1, 1]
    turn: float = 0.0      # [-1, 1]; positive = turn left
    brake: float = 0.0     # [0, 1]
    note: str = ""

    def clamp(self) -> "WheelCommand":
        return WheelCommand(
            forward=float(max(-1.0, min(1.0, self.forward))),
            turn=float(max(-1.0, min(1.0, self.turn))),
            brake=float(max(0.0, min(1.0, self.brake))),
            note=self.note,
        )

    def to_thor_action(self) -> str:
        """Convert continuous command to the best discrete AI2-THOR action."""
        if self.brake > BRAKE_THRESHOLD:
            return "Pass"
        if abs(self.turn) > TURN_THRESHOLD:
            return "RotateLeft" if self.turn > 0 else "RotateRight"
        if abs(self.forward) > MOVE_THRESHOLD:
            return "MoveAhead" if self.forward > 0 else "MoveBack"
        return "Pass"

    @staticmethod
    def stop(note: str = "") -> "WheelCommand":
        return WheelCommand(forward=0.0, turn=0.0, brake=1.0, note=note)

    @staticmethod
    def ahead(note: str = "") -> "WheelCommand":
        return WheelCommand(forward=1.0, turn=0.0, note=note)
