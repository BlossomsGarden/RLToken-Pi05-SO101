import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.distributed_protocol import check_rows, check_state, decode_frame, encode_frame
from scripts.select_red_green_annotations import select


class DistributedContractTests(unittest.TestCase):
    def test_action_shapes_and_nonfinite(self):
        self.assertEqual(check_rows([[1, 2, 3, 4, 5, 6]]).shape, (1, 6))
        for invalid in ([[1, 2]], [[float("nan")] * 6], [[0] * 6] * 11):
            with self.assertRaises(ValueError):
                check_rows(invalid)
        with self.assertRaises(ValueError):
            check_state([0] * 5)

    def test_camera_frame_transport(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(decode_frame(encode_frame(image)).shape, image.shape)
        with self.assertRaises(ValueError):
            encode_frame(image.astype(np.float32))

    def test_select_only_verified_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"episode_indices": list(range(30))}))
            annotations = base / "annotations.json"
            annotations.write_text(json.dumps({"annotations": [
                {"episode": 0, "start": 1, "end": 2},
                {"episode": 29, "start": 3, "end": 4},
                {"episode": 30, "start": 1, "end": 2},
            ]}))
            self.assertEqual([row["episode"] for row in select(annotations, manifest)["annotations"]], [0, 29])


if __name__ == "__main__":
    unittest.main()
