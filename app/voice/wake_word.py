"""
Wake-word detection backed by openWakeWord (Apache-2.0, local ONNX
models, no account/API key of any kind - genuinely local and free,
unlike commercial wake-word SDKs that require a signed-up access key even
on their "free" tier). Ships a pretrained "hey_jarvis" model - matching
the assistant's name (Jarvis) - that the user downloads once (see
app/voice/download_wakeword_model.py) - never fetched automatically by
this module.

openWakeWord's own `Model.predict()` maintains a small rolling prediction
buffer per loaded model to reduce false positives; `process_frame` resets
it immediately after a detection so the same utterance can't fire twice
in a row (see WakeWordProvider.process_frame's contract in
app/voice/base.py).
"""

import os

import numpy as np
from openwakeword.model import Model

from app.voice.base import WakeWordError, WakeWordProvider

#: openWakeWord's own recommended chunk size: 1280 samples = 80ms at
#: 16kHz. Longer chunks reduce CPU usage; shorter ones reduce latency -
#: this is the documented sweet spot, not a tuned choice of ours.
_FRAME_LENGTH = 1280


class OpenWakeWordProvider(WakeWordProvider):
    """WakeWordProvider implementation backed by a single openWakeWord
    ONNX model (e.g. the pretrained `hey_jarvis_v0.1.onnx`)."""

    frame_length = _FRAME_LENGTH

    def __init__(self, wakeword_model_path: str, threshold: float = 0.5):
        if not wakeword_model_path:
            raise WakeWordError(
                "No wake-word model configured - set "
                "JARVIS_VOICE_WAKEWORD_MODEL_PATH (see .env.example)."
            )
        if not os.path.exists(wakeword_model_path):
            raise WakeWordError(
                f"Wake-word model not found at '{wakeword_model_path}' - "
                "run `python -m app.voice.download_wakeword_model` first."
            )
        try:
            self._model = Model(
                wakeword_models=[wakeword_model_path], inference_framework="onnx"
            )
        except Exception as exc:  # noqa: BLE001 - openWakeWord raises plain Exception
            raise WakeWordError(
                f"Failed to load wake-word model '{wakeword_model_path}': {exc}. "
                "Make sure `python -m app.voice.download_wakeword_model` has been "
                "run (it also fetches the required feature models)."
            ) from exc

        self.threshold = threshold
        self._model_key = os.path.splitext(os.path.basename(wakeword_model_path))[0]

    def process_frame(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16)
        try:
            predictions = self._model.predict(samples)
        except Exception as exc:  # noqa: BLE001 - isolate a single bad frame
            raise WakeWordError(f"Wake-word detection failed: {exc}") from exc

        score = predictions.get(self._model_key, 0.0)
        detected = score >= self.threshold
        if detected:
            self._model.reset()
        return detected
