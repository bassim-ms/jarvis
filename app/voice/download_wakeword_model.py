"""
Run this once to download the pretrained "hey_jarvis" wake-word model
(and openWakeWord's required feature/melspectrogram models) from
openWakeWord's own GitHub release assets - a one-time, $0, local file
download, never a runtime API call:

    python -m app.voice.download_wakeword_model

"hey_jarvis" is one of openWakeWord's own built-in pretrained models
(alongside "alexa", "hey_mycroft", "hey_rhasspy", ...) - verified present
in the installed openwakeword package's own MODELS registry, not a
model this project invented or trained.

Downloads into openWakeWord's own package resources directory by
default. After it finishes, point JARVIS_VOICE_WAKEWORD_MODEL_PATH (see
.env.example) at the printed `hey_jarvis_v0.1.onnx` path.
"""

import os

from openwakeword.utils import download_models


def main() -> None:
    print("[jarvis] downloading the 'hey_jarvis' wake-word model...")
    download_models(model_names=["hey_jarvis"])

    from openwakeword import MODELS

    onnx_path = MODELS["hey_jarvis"]["download_url"].replace(".tflite", ".onnx")
    onnx_filename = os.path.basename(onnx_path)

    print("[jarvis] done. Set JARVIS_VOICE_WAKEWORD_MODEL_PATH to the full path of:")
    print(f"           {onnx_filename}")
    print("         (see the console output above for exactly where it was saved).")


if __name__ == "__main__":
    main()
