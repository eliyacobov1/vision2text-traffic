"""Image caption generation for traffic scenes."""

from typing import Optional

import cv2
from PIL import Image
from transformers import pipeline


class CaptionGenerator:
    """Generate natural-language descriptions using a vision-language model."""

    def __init__(self, model: str = "nlpconnect/vit-gpt2-image-captioning", device: int = -1) -> None:
        """Initialize captioning pipeline.

        Args:
            model: Model name or path for the HuggingFace image-to-text pipeline.
            device: Device index (``-1`` for CPU).
        """
        self.pipe = pipeline("image-to-text", model=model, device=device)

    def caption(self, frame) -> str:
        """Generate caption for a frame.

        Args:
            frame: BGR image array.

        Returns:
            Generated caption string.
        """
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.pipe(image)[0]
        return result["generated_text"].strip()
