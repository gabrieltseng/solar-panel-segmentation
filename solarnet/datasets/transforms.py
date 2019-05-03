"""
Image transformations, along with their corresponding
mask transformations (if applicable)
"""

import numpy as np
from typing import Tuple, Optional


def no_change(image: np.ndarray,
              mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                          Optional[np.ndarray]]:
    if mask is None: return image
    return image, mask


def horizontal_flip(image: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                                Optional[np.ndarray]]:
    # input: image[channels, height, width]
    image = image[:, :, ::-1]
    if mask is None: return image

    mask = mask[:, ::-1]
    return image, mask


def vertical_flip(image: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                              Optional[np.ndarray]]:
    # input: image[channels, height, width]
    image = image[:, ::-1, :]
    if mask is None: return image

    mask = mask[::-1, :]
    return image, mask


def colour_jitter(image: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray,
                                                              Optional[np.ndarray]]:
    _, height, width = image.shape
    zitter = np.zeros_like(image)

    for channel in range(zitter.shape[0]):
        noise = np.random.randint(0, 30, (height, width))
        zitter[channel, :, :] = noise

    image = image + zitter
    if mask is None: return image
    return image, mask
