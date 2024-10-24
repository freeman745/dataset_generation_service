from typing import Tuple, Optional
import numpy as np


class BBox:
    def __init__(
        self,
        rectangle: Tuple[int],
        score: Optional[float] = None,
        label: Optional[str] = None,
        from_wd: bool = False,
    ):
        self.score = score
        self.label = label
        if from_wd:
            self.coords = self.bbox_wd2pts(rectangle)
        else:
            self.coords = coords

    @staticmethod
    def bbox_wd2pts(rectangle: Tuple[int]) -> Tuple[int]:
        try:
            return (
                rectangle[0],
                rectangle[1],
                rectangle[0] + rectangle[2],
                rectangle[1] + rectangle[3],
            )
        except:
            return tuple()

    def _key(self):
        return (self.score, *self.coords)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other) -> bool:
        if isinstance(other, BBox):
            return self._key() == other._key()
        return NotImplemented

    def __repr__(self) -> str:
        desc = ""
        if self.score is not None:
            desc += f"score: {self.score}, "
        desc += f"coords: {self.coords}"
        if self.label is not None:
            desc += f", label: {self.label}"
        return desc


def crop_image(image: np.ndarray, bbox: BBox):
    x_min, y_min, x_max, y_max = bbox.coords
    return image[y_min:y_max, x_min:x_max]
