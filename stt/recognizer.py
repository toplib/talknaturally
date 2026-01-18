import json
from vosk import Model, KaldiRecognizer
from pathlib import Path

from .config import SAMPLE_RATE
from .exceptions import ModelNotFoundError


class SpeechRecognizer:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise ModelNotFoundError(f"Model not found: {model_path}")

        self.model = Model(str(model_path))
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)

    def accept_audio(self, data: bytes):
        if self.recognizer.AcceptWaveform(data):
            return json.loads(self.recognizer.Result()), True
        else:
            return json.loads(self.recognizer.PartialResult()), False
