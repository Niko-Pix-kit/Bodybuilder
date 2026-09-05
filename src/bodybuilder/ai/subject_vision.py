"""Local, task-based subject/part localization. Predictions are not verified facts."""
from __future__ import annotations

import gc
import math
import threading
from dataclasses import dataclass
from typing import Any

from PIL import Image

from bodybuilder.ai.base import BackendFatalError, GenerationCancelled


@dataclass(frozen=True)
class Region:
    label: str
    box: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        left, top, right, bottom = self.box
        return max(0, right - left) * max(0, bottom - top)


def parse_regions(answer: dict[str, Any], size: tuple[int, int]) -> list[Region]:
    """Accept Florence's OD, dense-caption and open-vocabulary result schemas."""
    boxes = answer.get("bboxes", [])
    labels = answer.get("bboxes_labels", answer.get("labels", []))
    result = []
    for label, box in zip(labels, boxes, strict=False):
        if not isinstance(label, str) or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            coordinates = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in coordinates):
            continue
        left, top, right, bottom = coordinates
        clipped = (max(0, round(left)), max(0, round(top)),
                   min(size[0], round(right)), min(size[1], round(bottom)))
        region = Region(label.strip().lower(), clipped)
        if region.label and clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            result.append(region)
    return result


class SubjectVision:
    """Native Florence-2 on CPU, leaving accelerator memory to SDXL.

    Downloads weights, not remote Python code. No photographs are transmitted.
    It is deliberately a detector/description tool, not an identity recognizer.
    """
    model_id = "florence-community/Florence-2-base"

    def __init__(self, cancel_event: threading.Event, log=lambda _text: None) -> None:
        self.cancel_event = cancel_event
        self.log = log
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise GenerationCancelled("Subject analysis cancelled")

    def prepare(self) -> None:
        self.check_cancelled()
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
            self.log("Loading local subject/part detector (Florence-2). First use downloads its weights.")
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=False)
            self.model = Florence2ForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float32, use_safetensors=True,
                trust_remote_code=False, attn_implementation="eager").eval().to("cpu")
            self.torch = torch
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            self.close()
            raise BackendFatalError(
                "The local subject detector could not load. Update requirements.txt in the same "
                "Python environment and check the model download. No generic substitute was used. "
                + str(exc)) from exc

    def task(self, image: Image.Image, task: str, text: str = "") -> dict[str, Any]:
        self.prepare()
        from transformers import StoppingCriteria, StoppingCriteriaList

        event = self.cancel_event
        class CancelAtToken(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return event.is_set()

        inputs = self.processor(text=task + text, images=image.convert("RGB"), return_tensors="pt")
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                      num_beams=1, stopping_criteria=StoppingCriteriaList([CancelAtToken()]))
        self.check_cancelled()
        decoded = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(decoded, task=task, image_size=image.size)
        result = parsed.get(task, {})
        return result if isinstance(result, dict) else {}

    def detect(self, image: Image.Image) -> list[Region]:
        return parse_regions(self.task(image, "<OD>"), image.size)

    def describe_regions(self, image: Image.Image) -> list[Region]:
        return parse_regions(self.task(image, "<DENSE_REGION_CAPTION>"), image.size)

    def locate(self, image: Image.Image, label: str) -> list[Region]:
        return parse_regions(self.task(image, "<OPEN_VOCABULARY_DETECTION>", label), image.size)

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.torch = None
        gc.collect()
