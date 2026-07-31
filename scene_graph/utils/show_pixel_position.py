#!/usr/bin/env python3
"""Interactively inspect image pixel coordinates and colors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show mouse pixel coordinates over an image."
    )
    parser.add_argument("image", type=Path, help="Image file to inspect.")
    parser.add_argument(
        "--window-name",
        default="pixel position",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale. Coordinates are still reported in original image pixels.",
    )
    return parser.parse_args()


def draw_status(image: np.ndarray, text: str) -> np.ndarray:
    display = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    padding = 8
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    cv2.rectangle(
        display,
        (0, 0),
        (text_width + 2 * padding, text_height + baseline + 2 * padding),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        display,
        text,
        (padding, padding + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return display


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale must be positive")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    height, width = image.shape[:2]
    status_text = f"image: {width}x{height} | move mouse, press q/esc to quit"
    state = {"display": draw_status(image, status_text)}

    def on_mouse(event, x, y, flags, userdata):
        del flags, userdata
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            return

        orig_x = int(round(x / args.scale))
        orig_y = int(round(y / args.scale))
        orig_x = max(0, min(width - 1, orig_x))
        orig_y = max(0, min(height - 1, orig_y))
        b, g, r = image[orig_y, orig_x].tolist()
        text = f"x={orig_x}, y={orig_y} | RGB=({r}, {g}, {b})"
        print(text, flush=True)

        marked = image.copy()
        cv2.drawMarker(
            marked,
            (orig_x, orig_y),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        state["display"] = draw_status(marked, text)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window_name, on_mouse)

    while True:
        cv2.imshow(args.window_name, resize_for_display(state["display"], args.scale))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyWindow(args.window_name)


if __name__ == "__main__":
    main()
