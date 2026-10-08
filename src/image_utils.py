import os 
import numpy as np
from PIL import Image
import cv2


def save_video_cv2(frames: np.ndarray, filepath: str, fps: float = 30.0, is_color: bool = True, codec: str = "MJPG") -> None:
    """Saves a video array using OpenCV.

    Args:
        frames (np.ndarray): Video frames with shape (N, H, W, C).
        filepath (str): Output filepath. Should end with ".avi".
        fps (float): Frames per second.
        is_color (bool): Whether to write as color video.
        codec (str): FourCC codec string (e.g., "MJPG", "XVID").

    Raises:
        ValueError: If `frames` has invalid shape or values.
        RuntimeError: If the video writer cannot be opened.
    """
    if not isinstance(frames, np.ndarray):
        raise TypeError("`frames` must be a NumPy array.")

    if frames.ndim != 4:
        raise ValueError(f"Expected `frames` shape (N, H, W, C), got {frames.shape}.")

    n_frames, height, width, channels = frames.shape
    if n_frames == 0:
        raise ValueError("`frames` must contain at least one frame.")

    if is_color:
        if channels != 3:
            raise ValueError(f"Color video requires C=3, got C={channels}.")
    else:
        if channels != 1:
            raise ValueError(f"Grayscale video requires C=1, got C={channels}.")

    if fps <= 0:
        raise ValueError("`fps` must be > 0.")

    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            frame_min = float(np.nanmin(frames))
            frame_max = float(np.nanmax(frames))
            if frame_min >= 0.0 and frame_max <= 1.0:
                frames = (frames * 255.0).round().astype(np.uint8)
            else:
                frames = np.clip(frames, 0, 255).round().astype(np.uint8)
        else:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(filepath, fourcc, float(fps), (int(width), int(height)), bool(is_color))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {filepath}")

    try:
        for i in range(n_frames):
            frame = frames[i]
            if is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = frame[..., 0]
            writer.write(frame)
    finally:
        writer.release()


def mask_outside_ultrasound_video(original_pixels: np.array, single_mask=False, crop_to_mask=False) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.
    Adapted from EchoPrime: https://github.com/echonet/EchoPrime/blob/main/video_utils.py
    Args:
        vid (np.ndarray): A numpy array representing the video frames in RGB format. FxHxWxC.
        single_mask (bool): If True, create a single convex hull mask for the entire video.
        crop_to_mask (bool): If True, crop the video to the bounding box of the mask.
    Returns:
        np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    vid = np.copy(original_pixels)
    ##################### CREATE MASK #####################
    # Create gray scale verison of the video
    vid_gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in vid], axis=0)

    # Get mean and std of frames 
    frame_mean = vid_gray.astype(np.float32).mean(axis=0)
    frame_std = vid_gray.astype(np.float32).std(axis=0)

    # Erode std of video to get rid of small variations (e.g. EKG tracing, varying text like HR/FPS)
    kernel = np.ones((3, 3), np.uint8)
    frame_std_eroded = cv2.erode(frame_std.astype(np.uint8), kernel, iterations=10)

    # Make binary and get overlap of mean and std frames to find mask of ultrasound region.
    frame_mean = np.where(frame_mean > 0, 1, 0)
    frame_std_eroded = np.where(frame_std_eroded > 0, 1, 0)
    frame_overlap = np.add(frame_mean, frame_std_eroded)
    frame_overlap = np.where(frame_overlap > 1, 1, 0)

    # Dilate
    kernel = np.ones((3,3), np.uint8)
    frame_overlap = cv2.dilate(frame_overlap.astype(np.uint8), kernel, iterations=10).astype(np.uint8)

    # Fill everything that's outside the mask sector with some other number like 100
    cv2.floodFill(frame_overlap, None, (0,0), 100)
    # make all non-100 values 255. The rest are 0
    frame_overlap = np.where(frame_overlap!=100, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
    if single_mask: 
        frame_overlap = np.zeros_like(frame_overlap)
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
    else: 
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
    frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
    # Fill everything that's outside hull with some other number like 100
    cv2.floodFill(frame_overlap, None, (0, 0), 100)
    # make all non-100 values 255. The rest are 0
    frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0), dtype=bool)

    # Store the dimension values
    # Apply the mask to every frame and channel (changing in place)
    vid = np.stack([
        cv2.bitwise_and(frame.astype(np.uint8),
                        frame.astype(np.uint8),
                        mask=frame_overlap.astype(np.uint8))
        for frame in vid
    ])
    if crop_to_mask: 
        vid = vid[:, 
                  np.argwhere(frame_overlap).min(axis=0)[0]:np.argwhere(frame_overlap).max(axis=0)[0], 
                  np.argwhere(frame_overlap).min(axis=0)[1]:np.argwhere(frame_overlap).max(axis=0)[1]]

    return vid


def crop_and_scale(img, res=(224, 224), zoom=0.1, interpolation=Image.Resampling.BICUBIC):
    """
    Crop and scale an image.
    Source code: https://github.com/echonet/EchoPrime/blob/main/video_utils.py
    Args:
        img (np.ndarray): Image to crop and scale. Shape: (H, W, C).
        res (tuple): Resolution to scale to.
        zoom (float): Zoom factor.
        interpolation: Interpolation method.
    Returns:
        np.ndarray: Cropped and scaled image.
    """
    in_res = img.shape[:2]
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out: # if input aspect ratio is taller than output aspect ratio, crop height
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[:, padding:-padding]
    if zoom != 0:
        pad_x = round(int(in_res[1] * zoom))
        pad_y = round(int(in_res[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    # img = cv2.resize(img, res, interpolation=interpolation)
    # Use PIL resize instead of cv2.resize to antialiasing issues
    img = Image.fromarray(img).resize((res[1], res[0]), resample=interpolation)
    return np.asarray(img)