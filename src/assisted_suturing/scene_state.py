"""Validation-only SceneState contract. This module never controls a robot."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SceneQuality(str, Enum):
    PASS = "PASS"
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


ObjectType = Literal["needle", "suture", "wound_edge", "tool_tip", "marker"]


@dataclass(frozen=True)
class TrackedObject:
    object_id: str
    object_type: ObjectType
    track_id: int
    confidence: float
    visible: bool
    occluded: bool

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.visible and self.occluded:
            raise ValueError("an object cannot be visible and occluded simultaneously")


@dataclass(frozen=True)
class SceneState:
    frame_id: int
    timestamp_monotonic_ns: int
    calibration_version: str
    scene_quality: SceneQuality
    objects: tuple[TrackedObject, ...]

    def validate(self) -> None:
        if self.frame_id < 0 or self.timestamp_monotonic_ns < 0:
            raise ValueError("frame and timestamp values must be non-negative")
        if not self.calibration_version:
            raise ValueError("calibration_version is required")
        for obj in self.objects:
            obj.validate()
