from __future__ import annotations

import unittest

from app.processor import process_image
from tests.smoke_test import build_challenging_bottle, build_distractor_bottle, build_sideways_bottle


class ProcessorGeometryTests(unittest.TestCase):
    def test_process_uses_bottle_first_on_mild_rotation(self) -> None:
        content, expected = build_challenging_bottle()
        result = process_image(content, detector_backend="heuristic")
        meta = result.metadata

        self.assertTrue(meta.bottle_found)
        self.assertTrue(meta.crop_found)
        self.assertGreater(meta.bottle_confidence, 1.4)
        self.assertGreater(meta.detection_confidence, 2.0)
        self.assertGreaterEqual(abs(meta.bottle_rotation_degrees), max(0.8, expected["rotation_degrees"] * 0.4))
        self.assertGreaterEqual(result.crop_bgr.shape[1], result.crop_bgr.shape[0])

    def test_process_reorients_sideways_bottle(self) -> None:
        content, expected = build_sideways_bottle()
        result = process_image(content, detector_backend="heuristic")
        meta = result.metadata

        self.assertTrue(meta.bottle_found)
        self.assertTrue(meta.crop_found)
        self.assertGreater(meta.bottle_confidence, 1.2)
        self.assertGreater(meta.detection_confidence, 1.8)
        self.assertGreater(abs(meta.bottle_rotation_degrees), expected["rotation_degrees"] * 0.45)
        self.assertGreaterEqual(result.crop_bgr.shape[1], result.crop_bgr.shape[0])
        self.assertLess(meta.crop_box["x2"] - meta.crop_box["x1"], meta.oriented_width)
        self.assertLess(meta.crop_box["y2"] - meta.crop_box["y1"], meta.oriented_height)

    def test_process_prefers_date_over_right_side_distractor(self) -> None:
        content, expected = build_distractor_bottle()
        result = process_image(content, detector_backend="heuristic")
        meta = result.metadata

        self.assertTrue(meta.bottle_found)
        self.assertTrue(meta.crop_found)
        self.assertGreater(meta.detection_confidence, 2.0)
        self.assertGreater(abs(meta.bottle_rotation_degrees), expected["rotation_degrees"] * 0.35)

        bottle_x1 = meta.bottle_box["x1"]
        bottle_y1 = meta.bottle_box["y1"]
        bottle_x2 = meta.bottle_box["x2"]
        bottle_y2 = meta.bottle_box["y2"]
        crop_x1 = meta.crop_box["x1"]
        crop_y1 = meta.crop_box["y1"]
        crop_x2 = meta.crop_box["x2"]
        crop_y2 = meta.crop_box["y2"]

        bottle_w = bottle_x2 - bottle_x1
        bottle_h = bottle_y2 - bottle_y1
        crop_center_x = ((crop_x1 + crop_x2) / 2.0 - bottle_x1) / max(1.0, bottle_w)
        crop_center_y = ((crop_y1 + crop_y2) / 2.0 - bottle_y1) / max(1.0, bottle_h)
        crop_width_ratio = (crop_x2 - crop_x1) / max(1.0, bottle_w)
        crop_height_ratio = (crop_y2 - crop_y1) / max(1.0, bottle_h)

        self.assertGreater(crop_center_x, 0.28)
        self.assertLess(crop_center_x, 0.64)
        self.assertGreater(crop_center_y, 0.22)
        self.assertLess(crop_center_y, 0.56)
        self.assertGreater(crop_width_ratio, 0.18)
        self.assertGreater(crop_height_ratio, 0.06)


if __name__ == "__main__":
    unittest.main()
