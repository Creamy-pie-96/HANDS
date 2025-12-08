import numpy as np
import time
from collections import deque
from typing import Iterable, Tuple, Union


def landmarks_to_array(landmarks: Iterable) -> np.ndarray:
    """Convert an iterable of landmarks with .x and .y into an Nx2 NumPy array.

    Args:
        landmarks: iterable of objects with `.x` and `.y` (normalized 0..1)

    Returns:
        np.ndarray of shape (N, 2) dtype float with columns (x, y).
    """
    arr = np.array([[lm.x, lm.y] for lm in landmarks], dtype=float)
    return arr


def normalized_to_pixels(
    norm_xy: Union[Tuple[float, float], np.ndarray], frame_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Map normalized coordinates (0..1) to pixel coordinates and clip to frame bounds.

    Accepts a single point `(x,y)` or an array of points shape `(N,2)`.

    Args:
        norm_xy: (2,) or (N,2) array-like with values in 0..1
        frame_shape: frame shape as returned by `frame.shape` (height, width, ...)

    Returns:
        np.ndarray of ints with same leading shape as `norm_xy`, mapped to pixels.
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    arr = np.asarray(norm_xy, dtype=float)

    # Handle single point (2,) -> convert to (1,2) for unified processing
    single = False
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("norm_xy must be shape (2,) or (N,2)")
        arr = arr.reshape((1, 2))
        single = True

    arr_px = np.empty_like(arr)
    arr_px[..., 0] = arr[..., 0] * w
    arr_px[..., 1] = arr[..., 1] * h

    # clip to valid pixel indices
    arr_px[..., 0] = np.clip(arr_px[..., 0], 0, w - 1)
    arr_px[..., 1] = np.clip(arr_px[..., 1], 0, h - 1)

    arr_px = arr_px.astype(int)
    return arr_px[0] if single else arr_px


def euclidean(a, b):
    """Euclidean distance between points.

    - If `a` and `b` are 1-D points, returns a scalar.
    - If arrays of points, returns distances per-row.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.linalg.norm(a - b, axis=-1)


class EWMA:
    """Exponential weighted moving average for smoothing 1-D or 2-D points.

    Example:
        s = EWMA(alpha=0.2)
        smoothed = s.update([x, y])
    """

    def __init__(self, alpha: float = 0.2, init: Union[None, Iterable] = None) -> None:
        self.alpha = float(alpha)
        self.value = None if init is None else np.array(init, dtype=float)

    def update(self, x: Iterable) -> np.ndarray:
        x = np.array(x, dtype=float)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value






__all__ = [
    "landmarks_to_array",
    "normalized_to_pixels",
    "euclidean",
    "EWMA",

]

