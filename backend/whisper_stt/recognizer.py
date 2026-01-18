import numpy as np
from faster_whisper import WhisperModel
from .config import SAMPLE_RATE, BLOCK_SIZE


class WhisperRecognizer:
    def __init__(
        self,
        model_name="small",
        silence_threshold=0.0015,
        min_speech_sec=1.0,
        silence_sec=0.5,          # 👈 CRITICAL
        max_buffer_sec=6.0,
    ):
        self.debug = False
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=6,
        )

        self.silence_threshold = silence_threshold
        self.min_samples = int(min_speech_sec * SAMPLE_RATE)
        self.max_samples = int(max_buffer_sec * SAMPLE_RATE)
        self.silence_blocks_required = int(
            (silence_sec * SAMPLE_RATE) / BLOCK_SIZE
        )

        self.buffer = np.empty((0,), dtype=np.float32)

        self.in_speech = False
        self.silence_blocks = 0

    def add_audio(self, chunk: np.ndarray):
        audio = chunk[:, 0] if chunk.ndim == 2 else chunk
        energy = np.mean(audio ** 2)

        self.buffer = np.append(self.buffer, audio)
        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples :]

        if self.debug:
            state = "SPEECH" if self.in_speech else "SILENCE"
            print(
                f"ENERGY={energy:.4f} "
                f"STATE={state} "
                f"BUF={len(self.buffer)/SAMPLE_RATE:.1f}s "
                f"SILENCE_BLK={self.silence_blocks}",
                end="\r",
            )

        if energy > self.silence_threshold:
            self.in_speech = True
            self.silence_blocks = 0
            return None

        if self.in_speech:
            self.silence_blocks += 1
            if (
                self.silence_blocks >= self.silence_blocks_required
                and len(self.buffer) >= self.min_samples
            ):
                self.in_speech = False
                self.silence_blocks = 0
                if self.debug:
                    print("\n🧠 Transcribing…")
                return self._transcribe()

        return None


    def _transcribe(self):
        segments, _ = self.model.transcribe(
            self.buffer,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
        )

        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        self.buffer = np.empty((0,), dtype=np.float32)
        return text
