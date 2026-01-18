from whisper_stt import WhisperRecognizer, MicrophoneStream, SAMPLE_RATE, BLOCK_SIZE, CHANNELS

recognizer = WhisperRecognizer(buffer_seconds=5.0)

print("🎤 Speak (Ctrl+C to stop)")

with MicrophoneStream(SAMPLE_RATE, BLOCK_SIZE, CHANNELS) as mic:
    try:
        while True:
            audio = mic.read()  # shape (block, 1)
            recognizer.add_audio(audio)

            # Only transcribe when buffer is ~5 sec or more
            if len(recognizer.buffer) >= SAMPLE_RATE * 5:
                text = recognizer.transcribe_buffer()
                if text:
                    print("📝", text)

    except KeyboardInterrupt:
        # Transcribe any leftover audio
        leftover = recognizer.transcribe_buffer()
        if leftover:
            print("📝", leftover)
        print("\n👋 Stopped cleanly.")
