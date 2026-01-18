import os
import sounddevice as sd
import soundfile as sf
from neuttsair.neutts import NeuTTS

class RealTimeTTS:
    def __init__(self, backbone="neuphonic/neutts-nano", codec_repo="neuphonic/neucodec"):
        # Initialize NeuTTS
        self.tts = NeuTTS(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec_repo,
            codec_device="cpu",
        )

    def encode_reference(self, ref_audio_path):
        """Prepare reference audio for style/voice"""
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
        return self.tts.encode_reference(ref_audio_path)

    def speak(self, input_text, ref_codes, ref_text, sample_rate=24000):
        """Generate audio and play immediately"""
        wav = self.tts.infer(input_text, ref_codes, ref_text)
        # Play audio immediately
        sd.play(wav, samplerate=sample_rate, blocking=True)
        return wav
