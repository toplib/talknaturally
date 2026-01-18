import queue
import sounddevice as sd

class MicrophoneStream:
    def __init__(self, sample_rate: int, block_size: int, channels: int):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.queue = queue.Queue()

    def _callback(self, indata, frames, time, status):
        self.queue.put(indata.copy())

    def __enter__(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop()
        self.stream.close()

    def read(self):
        return self.queue.get()
