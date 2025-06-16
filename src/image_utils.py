"""Image utility functions."""

import cv2
import numpy as np


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to LAB color space."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """Convert an LAB image to RGB color space."""
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def rgb2yuv(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to YUV color space."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)


def yuv2rgb(yuv: np.ndarray) -> np.ndarray:
    """Convert a YUV image to RGB color space."""
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def srgb2lin(srgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image to linear RGB color space."""
    srgb = srgb.astype(float) / 255.0
    return np.where(
        srgb <= 0.0404482362771082,
        srgb / 12.92,
        np.power(((srgb + 0.055) / 1.055), 2.4),
    )


def lin2srgb(lin: np.ndarray) -> np.ndarray:
    """Convert a linear RGB image to sRGB color space."""
    return 255 * np.where(
        lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin
    )


def get_luminance(
    linear_image: np.ndarray, luminance_conversion=[0.2126, 0.7152, 0.0722]
) -> np.ndarray:
    """Calculate the luminance of a linear RGB image.

    Args:
        linear_image: Linear RGB image.
        luminance_conversion: Coefficients for converting RGB to luminance.

    Returns:
        np.ndarray: Luminance channel of the image.
    """
    return np.sum([[luminance_conversion]] * linear_image, axis=2)


def take_luminance_from_first_chroma_from_second(
    luminance: np.ndarray,
    chroma: np.ndarray,
    mode: str = "lab",
    scaling_factor: int = 1,
) -> np.ndarray:
    """Replace the luminance channel of the first image with the luminance channel of the second image.
    Args:
        luminance: Luminance image.
        chroma: Chroma image.
        mode: Color space mode to use ('lab', 'yuv', 'luminance').
        scaling_factor: Scaling factor for luminance adjustment."""
    assert luminance.shape == chroma.shape, f"{luminance.shape=} != {chroma.shape=}"
    if mode == "lab":
        lab = rgb2lab(chroma)
        lab[:, :, 0] = rgb2lab(luminance)[:, :, 0]
        return lab2rgb(lab)
    if mode == "yuv":
        yuv = rgb2yuv(chroma)
        yuv[:, :, 0] = rgb2yuv(luminance)[:, :, 0]
        return yuv2rgb(yuv)
    if mode == "luminance":
        lluminance = srgb2lin(luminance)
        lchroma = srgb2lin(chroma)
        return lin2srgb(
            np.clip(
                lchroma
                * (
                    (get_luminance(lluminance) / (get_luminance(lchroma)))
                    ** scaling_factor
                )[:, :, np.newaxis],
                0,
                1,
            )
        )
