from llm import LLMClient
from stt import (
    SpeechRecognizer,
    MicrophoneStream,
    MODEL_PATH,
    SAMPLE_RATE,
    BLOCK_SIZE,
    CHANNELS,
)

# Initialize modules
llm = LLMClient()
recognizer = SpeechRecognizer(MODEL_PATH)

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

with MicrophoneStream(SAMPLE_RATE, BLOCK_SIZE, CHANNELS) as mic:
    try:
        while True:
            data = mic.read()
            result, is_final = recognizer.accept_audio(data)

            if is_final and result.get("text"):
                user_text = result["text"]
                print(f"\n👤: {user_text}")
                print("🤖: ", end="", flush=True)

                # Append user input to chat history
                chat_history.append({"role": "user", "content": user_text})

                # Stream LLM response
                assistant_response = ""
                for token in llm.chat_stream(chat_history):
                    print(token, end="", flush=True)
                    assistant_response += token

                print("\n")

                # Append assistant response to chat history
                chat_history.append({"role": "assistant", "content": assistant_response})

            elif "partial" in result:
                print("PARTIAL:", result["partial"], end="\r")

    except KeyboardInterrupt:
        print("\n👋 Stopped cleanly.")
