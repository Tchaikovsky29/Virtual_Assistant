# Virtual Assistant

A Python voice-based virtual assistant that combines Groq-hosted LLM reasoning, speech recognition, text-to-speech, web search, image generation, Python execution, and a persistent Pinecone memory store.

## Features

- Continuous voice interaction using a microphone
- ReAct-style tool calling with LangChain and Groq
- Spoken responses using StreamElements TTS
- Conversation memory during a session
- Current-time lookup
- DuckDuckGo web search
- Python expression execution
- Pinecone hybrid search for personal memory
- Text-to-image generation through Pollinations AI
- Configurable alternative speech recognition and TTS implementations

## Project Structure

| File | Purpose |
| --- | --- |
| `main.py` | Starts the assistant, handles speech input/output, and registers the LangChain tools |
| `tools.py` | Implements search, memory, time, exit, and image-generation tools |
| `try.py` | Tests StreamElements text-to-speech voices |
| `alt.txt` | Notes and example code for Whisper, ElevenLabs, pyttsx3, and pydub alternatives |
| `example.env` | Template for API keys |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10 or newer is recommended
- A working microphone and speaker
- A Groq API key
- A Pinecone API key
- A Hugging Face token (for the embeddings model)
- Internet access for the LLM, search, embeddings, TTS, and image generation

The project also depends on system libraries used by PyAudio and FFmpeg. On Debian/Ubuntu, PortAudio development headers can be installed with:

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev ffmpeg
```

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd Virtual_Assistant
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Create a `.env` file using `example.env`:

```bash
cp example.env .env
```

Add the required credentials:

```env
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_optional_elevenlabs_api_key
```

`ELEVENLABS_API_KEY` is only needed if you choose the alternative ElevenLabs implementation shown in `alt.txt`.

## Pinecone Setup

Create a serverless or pod-based Pinecone index named `assistant` with:

- **Dimensions:** `384`
- **Metric:** `dotproduct`

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional vectors. The project combines dense embeddings with Pinecone's default BM25 sparse encoder for hybrid retrieval.

## Running the Assistant

Before first run, create the directories used for temporary audio and generated images:

```bash
mkdir -p VA RA
```

Then start the assistant:

```bash
python main.py
```

Speak your request after the terminal prints `listening...`. Say goodbye or a similar parting phrase to let the `Exit` tool stop the process.

To test a text-to-speech voice:

```bash
python try.py
```

## Available Tools

- **Time** — reports the local system time
- **Search** — searches the web with DuckDuckGo
- **Human_input** — asks a spoken question and captures the user's response
- **python_repl** — evaluates a Python expression or command
- **UpdateDB** — stores relevant personal information in Pinecone
- **Retriever** — retrieves relevant information from Pinecone
- **Exit** — ends the conversation
- **Generate_image** — generates an image and saves it as `RA/image.jpg`

## Audio Configuration

The active speech stack is:

- Speech recognition: Google Speech Recognition
- Text-to-speech: StreamElements
- Audio playback: `rpaudio`

Examples for using Whisper, ElevenLabs, pyttsx3, or pydub are available in `alt.txt` and `try.py`. Voice selection can be changed in `main.py` and `try.py` through `stream_elements.Voice` values.

## Notes and Known Limitations

- Speech recognition and image generation require network access.
- Pinecone memory is global to the configured index and is not isolated by user by default.
- The temporary audio file is stored at `VA/input.wav`; generated speech is stored at `VA/output.mp3`.
- Google Speech Recognition is used without an API key in the current implementation, subject to its availability and usage limits.
- The Groq model is currently configured in `main.py` as `llama3-70b-8192`; model availability may change.
- Image output is opened locally when `PIL.Image.show()` is supported, and is also saved to `RA/image.jpg`.
- Error handling is intentionally minimal and may vary by operating system and hardware.

## Contributing

Contributions are welcome. Bug fixes, cross-platform audio support, stronger error handling, model/configuration improvements, and test coverage are especially useful.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and test them locally.
4. Open a pull request describing the problem and solution.

## License

See `LICENSE` for license information.
