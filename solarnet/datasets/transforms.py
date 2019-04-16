"""
Image transformations, along with their corresponding
mask transformations (if applicable)
"""

import numpy as np


def no_change(image, mask=None):
    if mask is None: return image
    return image, mask


def horizontal_flip(image, mask=None):
    # input: image[channels, height, width]
    image = image[:, :, ::-1]
    if mask is None: return image

    mask = mask[:, ::-1]
    return image, mask


def vertical_flip(image, mask=None):
    # input: image[channels, height, width]
    image = image[:, ::-1, :]
    if mask is None: return image

    mask = mask[::-1, :]
    return image, mask


def colour_jitter(image, mask=None):
    _, height, width = image.shape
    zitter = np.zeros_like(image)

    for channel in range(zitter.shape[0]):
        noise = np.random.randint(0, 30, (height, width))
        zitter[channel, :, :] = noise

    image = image + zitter
    if mask is None: return image
    return image, mask
