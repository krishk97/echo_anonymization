"""Tests for the masking and resizing helpers in src/image_utils.py."""

import numpy as np

from src.image_utils import crop_and_scale, mask_outside_ultrasound_video


def _moving_disc(
    video: np.ndarray, cy: int, cx: int, r: int, rng: np.random.Generator
) -> None:
    """Fills a disc of `video` with per-frame random noise, in place."""
    n, h, w, _ = video.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    video[:, disc] = rng.integers(40, 255, size=(n, h, w, 1), dtype=np.uint8)[:, disc]


def test_single_mask_keeps_largest_moving_region() -> None:
    """With two moving regions, the single mask is the hull of the larger one."""
    rng = np.random.default_rng(0)
    video = np.zeros((20, 120, 200, 3), dtype=np.uint8)
    _moving_disc(video, 60, 60, 40, rng)  # large sector
    _moving_disc(video, 60, 160, 20, rng)  # smaller moving patch
    masked = mask_outside_ultrasound_video(video, single_mask=True)
    assert masked[:, 60, 60].any()
    assert not masked[:, 60, 160].any()


def test_crop_and_scale_near_square_frame_not_black() -> None:
    """A padding that rounds to 0 (225x224 frame) no longer returns a black image."""
    img = np.full((225, 224, 3), 200, dtype=np.uint8)
    out = crop_and_scale(img, res=(224, 224), zoom=0.0)
    assert out.shape == (224, 224, 3)
    assert out.min() == 200


def test_crop_and_scale_crops_to_aspect_ratio() -> None:
    """A tall frame is center-cropped to a square before resizing."""
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    img[50:250] = 255  # the central 200x200 square
    out = crop_and_scale(img, res=(100, 100), zoom=0.0)
    assert out.shape == (100, 100, 3)
    assert out.min() == 255
