import numpy as np
from faster_whisper import WhisperModel
from .config import MODEL_NAME, SAMPLE_RATE

class WhisperRecognizer:
    def __init__(self, model_name: str = MODEL_NAME, buffer_seconds: float = 5.0):
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.buffer_seconds = buffer_seconds
        self.buffer = np.zeros((0,), dtype=np.float32)  # empty 1D array

    def add_audio(self, audio_chunk: np.ndarray):
        """Add new audio chunk to buffer"""
        audio_flat = audio_chunk.flatten()
        self.buffer = np.concatenate([self.buffer, audio_flat])

    def transcribe_buffer(self):
        """Transcribe current buffer and clear it"""
        if len(self.buffer) == 0:
            return ""

        segments, _ = self.model.transcribe(self.buffer, beam_size=5, word_timestamps=False)
        text = " ".join(segment.text for segment in segments)

        # Reset buffer after transcription
        self.buffer = np.zeros((0,), dtype=np.float32)
        return text.strip()
