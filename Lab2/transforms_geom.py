from typing import Tuple

from PIL import Image
import random


def hflip(img: Image.Image, uv: Tuple[float, float]) -> Tuple[Image.Image, Tuple[float, float]]:
    """Horizontal flip with coordinate update.

    Assumes the image is square (227×227) as pre-resized by the dataset.
    u' = (W - 1) - u
    v' = v
    """
    w, h = img.size
    img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
    u, v = uv
    u2 = (w - 1) - u
    return img2, (u2, v)


def maybe_hflip(p: float):
    """Factory returning a callable that randomly flips with prob p."""

    def _apply(img: Image.Image, uv: Tuple[float, float]):
        if random.random() < p:
            return hflip(img, uv)
        return img, uv

    return _apply

