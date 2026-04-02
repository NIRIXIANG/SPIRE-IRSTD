"""Visualize IR target points on images."""
import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image

POINT_COLOR = (240, 2, 127)  # single-class highlight


def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 2,
                   draw_text: bool = False,
                   draw_scores: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10):
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text or draw_scores:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse(
                [point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                fill=POINT_COLOR,
                outline=(255, 255, 255),
            )
            if draw_scores:
                draw.text((point[0] - r - 4, point[1] - r - 10), text=f"{score:.2f}", font=font)

    return img
