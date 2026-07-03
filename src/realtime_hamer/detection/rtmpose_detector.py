"""RTMPose wholebody hand detector with person tracking (from hamer-demo)."""

import numpy as np
from rtmlib import Wholebody, PoseTracker


def create_detector(det_frequency: int = 10, mode: str = "lightweight", device: str = "cuda"):
    """Create a hand bbox detector using RTMPose wholebody + onnxruntime-gpu."""
    pose_model = PoseTracker(
        Wholebody,
        det_frequency=det_frequency,
        mode=mode,
        backend="onnxruntime",
        device=device,
    )

    def detector(frame):
        all_keypoints, all_scores = pose_model(frame)
        boxes = []
        is_right = []

        num_hand_keypoints = 21
        l_start_id = 91
        r_start_id = 91 + num_hand_keypoints

        def pose_to_box(keypoints, scores):
            valid_mask = scores > 0.5
            if not np.any(valid_mask):
                return None, 0.0

            valid_keypoints = keypoints[valid_mask]
            x = valid_keypoints[:, 0]
            y = valid_keypoints[:, 1]
            score = float(scores[valid_mask].mean())
            x_min, y_min, x_max, y_max = x.min(), y.min(), x.max(), y.max()
            width = x_max - x_min
            height = y_max - y_min
            if width > height:
                diff = (width - height) / 2
                y_min -= diff
                y_max += diff
            else:
                diff = (height - width) / 2
                x_min -= diff
                x_max += diff
            return [x_min, y_min, x_max, y_max], score

        for keypoints, scores in zip(all_keypoints, all_scores):
            l_keypoints = keypoints[l_start_id : l_start_id + num_hand_keypoints]
            r_keypoints = keypoints[r_start_id : r_start_id + num_hand_keypoints]
            l_scores = scores[l_start_id : l_start_id + num_hand_keypoints]
            r_scores = scores[r_start_id : r_start_id + num_hand_keypoints]

            for hand_idx, (kpts, scs) in enumerate(
                [(l_keypoints, l_scores), (r_keypoints, r_scores)]
            ):
                box, _score = pose_to_box(kpts, scs)
                if box is not None and (box[2] - box[0]) > 0 and (box[3] - box[1]) > 0:
                    boxes.append(box)
                    is_right.append(hand_idx == 1)

        return boxes, is_right

    return detector
