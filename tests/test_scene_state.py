import unittest

from assisted_suturing.scene_state import SceneQuality, SceneState, TrackedObject


class SceneStateTest(unittest.TestCase):
    def test_valid_scene_state(self) -> None:
        state = SceneState(
            frame_id=1,
            timestamp_monotonic_ns=1,
            calibration_version="test-1",
            scene_quality=SceneQuality.PASS,
            objects=(TrackedObject("needle-1", "needle", 1, 0.95, True, False),),
        )
        state.validate()


    def test_invalid_confidence_is_rejected(self) -> None:
        obj = TrackedObject("needle-1", "needle", 1, 1.01, True, False)
        with self.assertRaises(ValueError):
            obj.validate()
