# LLM Voice Chat

This is a small project that demonstrates a voice-enabled chat interface using Streamlit, Hugging Face Transformers, and TTS (text-to-speech).

## Features
- Chat with an LLM (Qwen/Qwen3-1.7B by default)
- Text-to-speech for assistant responses
- Simple, interactive UI

## Why it may run slower
- The LLM model is large and runs on CPU by default (unless you have GPU support)
- Model loading and inference can be slow, especially on first run
- TTS model also requires time to generate audio
- Streamlit UI updates and audio playback add some latency

## Solutions to improve speed
- Use a smaller or quantized model if available
- Run on a machine with a GPU (and install CUDA dependencies)
- Keep the app running to avoid repeated model loading
- Optimize the code for batch processing or asynchronous calls

## Requirements
- Python 3.8+
- streamlit
- transformers
- sounddevice
- numpy
- mamba-ssm (for some models)

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run: `streamlit run linked_pipes.py`
