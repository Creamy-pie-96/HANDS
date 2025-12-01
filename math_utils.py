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


# I will never ever use it! As EWMA is far superior. It's just here because I already had created it
class MovingAverage:
    """Simple moving average buffer (keeps last N samples)."""

    def __init__(self, n: int = 5) -> None:
        self.n = int(n)
        self.buf = deque(maxlen=self.n)

    def update(self, x: Iterable) -> np.ndarray:
        self.buf.append(np.array(x, dtype=float))
        return np.mean(self.buf, axis=0)


class ClickDetector:
    """Relative-only pinch/click detector.

    This detector operates on a normalized distance value (unitless) where
    distances are expressed relative to the image diagonal (0..~1). Call
    `pinched(dist_rel)` with `dist_rel = pixel_dist / image_diag_px`.

    Rationale: using a distance normalized by the image diagonal makes the
    detection invariant to camera resolution and hand distance from camera.
    """

    def __init__(self, thresh_rel: float = 0.055, hold_frames: int = 3, cooldown_s: float = 0.4) -> None:
        # thresh_rel: fraction of image diagonal (e.g. 0.08 ~= 8% of diagonal)
        self.thresh_rel = float(thresh_rel)
        self.hold_frames = int(hold_frames)
        self.cooldown_s = float(cooldown_s)
        self._count = 0
        self._last_time = -999.0

    def pinched(self, dist_rel: float) -> bool:
        """Return True when a pinch/click is detected using relative distance.

        Args:
            dist_rel: distance between index and thumb normalized by image diagonal
        """
        now = time.time()
        if now - self._last_time < self.cooldown_s:
            return False

        if float(dist_rel) <= self.thresh_rel:
            self._count += 1
            if self._count >= self.hold_frames:
                self._last_time = now
                self._count = 0
                return True
            return False
        else:
            self._count = 0
            return False


__all__ = [
    "landmarks_to_array",
    "normalized_to_pixels",
    "euclidean",
    "EWMA",
    "MovingAverage",
    "ClickDetector",
]

