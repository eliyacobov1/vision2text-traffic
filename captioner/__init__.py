try:  # pragma: no cover - optional heavy dependency
    from .generate_caption import CaptionGenerator
except Exception:  # pragma: no cover
    CaptionGenerator = None  # type: ignore

from .simple_captioner import SimpleCaptioner
from .transformer_captioner import VisionLanguageModel

