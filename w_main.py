from llm import LLMClient
from whisper_stt import WhisperRecognizer, MicrophoneStream, SAMPLE_RATE, BLOCK_SIZE, CHANNELS

# Initialize modules
llm = LLMClient()
recognizer = WhisperRecognizer()

print("🎤 Speak (Ctrl+C to stop)")

chat_history = [
    {
        "role": "system",
        "content": (
            "You are an English speaking coach. "
            "Correct grammar and reply naturally. "
            "Keep responses short and conversational."
        ),
    }
]

try:
    with MicrophoneStream(SAMPLE_RATE, BLOCK_SIZE, CHANNELS) as mic:
        while True:
            audio = mic.read()
            text = recognizer.add_audio(audio)

            if text:  # Whisper returns text when ready
                print(f"\n👤: {text}")
                print("🤖: ", end="", flush=True)

                # Append user input to chat history
                chat_history.append({"role": "user", "content": text})

                # Stream LLM response
                assistant_response = ""
                for token in llm.chat_stream(chat_history):
                    print(token, end="", flush=True)
                    assistant_response += token

                print("\n")

                # Append assistant response to chat history
                chat_history.append({"role": "assistant", "content": assistant_response})

except KeyboardInterrupt:
    print("\n👋 Stopped cleanly.")
