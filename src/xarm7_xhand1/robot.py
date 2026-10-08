"""Composite xArm7 + XHand1 connection and observation layer."""
from dataclasses import dataclass
import time
from .arm import XArm7, XArmState
from .config import HardwareConfig
from .hand import XHand1, XHandState

@dataclass(frozen=True)
class RobotState:
    arm: XArmState
    hand: XHandState

class XArm7XHand1:
    def __init__(self, config: HardwareConfig, *, allow_arm_motion: bool = False, allow_hand_motion: bool = False):
        self.config = config
        self.arm, self.hand = XArm7(config, allow_motion=allow_arm_motion), XHand1(config, allow_motion=allow_hand_motion)
    def connect(self) -> None:
        self.arm.connect()
        try: self.hand.connect()
        except Exception: self.arm.close(); raise
    def read_state(self) -> RobotState:
        state, now = RobotState(self.arm.read_state(), self.hand.read_state()), time.monotonic()
        if now - state.arm.monotonic_time > self.config.max_state_age_s: raise RuntimeError("xArm state is stale")
        if now - state.hand.monotonic_time > self.config.max_state_age_s: raise RuntimeError("XHand state is stale")
        return state
    def hold(self) -> None: self.arm.hold(); self.hand.hold()
    def close(self) -> None:
        errors = []
        for component in (self.hand, self.arm):
            try: component.close()
            except Exception as exc: errors.append(exc)
        if errors: raise RuntimeError(f"Errors while closing robot: {errors}")
    def __enter__(self): self.connect(); return self
    def __exit__(self, *_): self.close()
