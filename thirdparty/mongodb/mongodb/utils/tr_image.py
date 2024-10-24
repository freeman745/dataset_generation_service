from typing import Optional, Tuple, Union
import numpy as np


NUMERIC_TYPE = Union[int, float]


def ti_normalize_range(
    img: np.ndarray,
    new_min: NUMERIC_TYPE,
    new_max: NUMERIC_TYPE,
    min: Optional[NUMERIC_TYPE] = None,
    max: Optional[NUMERIC_TYPE] = None,
    dtype=None,
) -> np.ndarray:
    if np.isnan(new_min):
        new_min = 0
    assert new_min <= new_max, (new_min, new_max)
    if min is None:
        min = np.min(img)
    if max is None:
        max = np.max(img)
    assert min <= max, (min, max)
    if max == min:
        if max <= 0:
            return (np.ones_like(img) * new_min).astype(img.dtype)
        else:
            return (np.ones_like(img) * new_max).astype(img.dtype)

    img_01 = (img - min) / (max - min)  # range [0, 1]
    img = (img_01 * (new_max - new_min)) + new_min  # range [new_min, new_max]
    if dtype is not None:
        img = img.astype(dtype)
    return img


def ti_normalize_8b(img, nan=0):
    if np.any(np.isnan(img)):
        if nan == "min":
            nan = np.nanmin(img)
        elif nan == "max":
            nan = np.nanmax(img)
        img = np.nan_to_num(img, nan=nan)
    return ti_normalize_range(img, new_min=0, new_max=255, dtype="uint8")


def get_dsize(img: np.ndarray, dsize: Optional[Union[float, Tuple[int]]] = None):
    or_size = img.shape[1], img.shape[0]
    if isinstance(dsize, float):
        dsize = [int(s * dsize) for s in or_size]
        return dsize
    if isinstance(dsize, int):
        max_size = max(or_size)
        factor = dsize / max_size
        return get_dsize(img, factor)
    return dsize


def rescale_array(data, factor=0.5, mode="nearest"):
    from scipy.ndimage import zoom

    if mode == "nearest":
        order = 0
    elif mode == "bilinear":
        order = 1
    elif mode == "bicubic":
        order = 3
    else:
        raise ValueError(
            "Unsupported mode. Choose from 'bilinear', 'nearest', 'bicubic'."
        )
    return zoom(data, factor, order=order)
