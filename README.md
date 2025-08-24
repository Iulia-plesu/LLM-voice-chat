
# LLM Voice Chat

LLM Voice Chat is a small project that lets you interact with a large language model (LLM) using a chat interface and hear its responses with text-to-speech. It uses Streamlit for the UI and Hugging Face Transformers for both text generation and speech synthesis.

## Features
- Chat with an LLM (Qwen/Qwen3-1.7B by default)
- Hear responses read aloud using TTS
- Simple, interactive web interface

## Getting Started
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run linked_pipes.py`

## Requirements
- Python 3.8+
- streamlit
- transformers
- sounddevice
- numpy
- mamba-ssm (for some models)

## Performance Notes
Depending on your hardware and setup, you may notice the app runs a bit slower. This is due to:
- The LLM model is large and runs on CPU by default (unless you have GPU support)
- Model loading and inference can be slow, especially on first run
- TTS model also requires time to generate audio
- Streamlit UI updates and audio playback add some latency

### How to Improve Speed
- Use a smaller or quantized model if available
- Run on a machine with a GPU (and install CUDA dependencies)
- Keep the app running to avoid repeated model loading
- Optimize the code for batch processing or asynchronous calls
