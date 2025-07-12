"""Utility helpers for video processing."""

import cv2


def open_video(path: str):
    """Yield frames from a video file."""
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def create_writer(path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
