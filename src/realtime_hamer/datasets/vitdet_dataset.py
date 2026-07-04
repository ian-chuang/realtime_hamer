"""Minimal hand-crop dataset for HaMeR inference (no training deps)."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch
from yacs.config import CfgNode

DEFAULT_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255.0 * np.array([0.229, 0.224, 0.225])


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    if target_aspect_ratio is None:
        return input_shape
    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    return np.array([w_new, h_new])


def _rotate_2d(pt: np.ndarray, rot_rad: float) -> np.ndarray:
    x, y = pt[0], pt[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)


def _gen_trans_from_patch_cv(
    c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot
):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_downdir = _rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    src = np.stack(
        [src_center, src_center + src_downdir, src_center + src_rightdir]
    ).astype(np.float32)
    dst = np.stack(
        [
            dst_center,
            dst_center + np.array([0, dst_height * 0.5], dtype=np.float32),
            dst_center + np.array([dst_width * 0.5, 0], dtype=np.float32),
        ]
    ).astype(np.float32)
    return cv2.getAffineTransform(src, dst)


def _crop_patch(img, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip):
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img.shape[1] - c_x - 1
    trans = _gen_trans_from_patch_cv(
        c_x, c_y, bb_width, bb_height, patch_width, patch_height, 1.0, 0.0
    )
    return cv2.warpAffine(
        img,
        trans,
        (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def prepare_hand_batch(
    cfg: CfgNode,
    img_cv2: np.ndarray,
    box: np.ndarray,
    is_right: bool,
    device: torch.device,
    rescale_factor: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Fast single-hand crop → model batch (no DataLoader overhead)."""
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
    box = box.astype(np.float32)
    center = (box[2:4] + box[0:2]) / 2.0
    scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
    bbox_shape = cfg.MODEL.get("BBOX_SHAPE", None)
    bbox_size = float(expand_to_aspect_ratio(scale * 200, target_aspect_ratio=bbox_shape).max())

    flip = not is_right
    cvimg = img_cv2
    downsampling_factor = (bbox_size / img_size) / 2.0
    if downsampling_factor > 1.1:
        k = max(3, int(downsampling_factor) | 1)
        cvimg = cv2.GaussianBlur(cvimg, (k, k), (downsampling_factor - 1) / 2)

    patch = _crop_patch(cvimg, center[0], center[1], bbox_size, bbox_size, img_size, img_size, flip)
    patch = np.transpose(patch[:, :, ::-1], (2, 0, 1)).astype(np.float32)
    for c in range(3):
        patch[c] = (patch[c] - mean[c]) / std[c]

    h, w = img_cv2.shape[:2]
    return {
        "img": torch.from_numpy(patch).unsqueeze(0).to(device, non_blocking=True),
        "box_center": torch.tensor([[center[0], center[1]]], dtype=torch.float32, device=device),
        "box_size": torch.tensor([bbox_size], dtype=torch.float32, device=device),
        "img_size": torch.tensor([[float(w), float(h)]], dtype=torch.float32, device=device),
        "right": torch.tensor([1.0 if is_right else 0.0], dtype=torch.float32, device=device),
    }


class ViTDetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        img_cv2: np.ndarray,
        boxes: np.ndarray,
        right: np.ndarray,
        rescale_factor: float = 2.5,
        train: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert train is False, "ViTDetDataset is only for inference"
        self.cfg = cfg
        self.img_cv2 = img_cv2
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)

        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        center = self.center[idx].copy()
        scale = self.scale[idx]
        bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=bbox_shape).max()
        patch_width = patch_height = self.img_size
        flip = self.right[idx] == 0

        cvimg = self.img_cv2.copy()
        downsampling_factor = (bbox_size * 1.0) / patch_width / 2.0
        if downsampling_factor > 1.1:
            k = max(3, int(downsampling_factor) | 1)
            cvimg = cv2.GaussianBlur(cvimg, (k, k), (downsampling_factor - 1) / 2)

        img_patch_cv = _crop_patch(
            cvimg,
            center[0],
            center[1],
            bbox_size,
            bbox_size,
            patch_width,
            patch_height,
            flip,
        )
        img_patch = np.transpose(img_patch_cv[:, :, ::-1], (2, 0, 1)).astype(np.float32)
        for n_c in range(3):
            img_patch[n_c] = (img_patch[n_c] - self.mean[n_c]) / self.std[n_c]

        return {
            "img": img_patch,
            "personid": int(self.personid[idx]),
            "box_center": self.center[idx].copy(),
            "box_size": bbox_size,
            "img_size": 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]]),
            "right": self.right[idx].copy(),
        }
