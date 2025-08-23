import sounddevice as sd
import numpy as np
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
output1 = pipe(messages)


if isinstance(output1, list):
    assistant_reply = None
    for item in output1:
        if isinstance(item, dict) and item.get('role') == 'assistant':
            assistant_reply = item.get('content', str(item))
            break
    if assistant_reply is None:
        assistant_reply = str(output1)
    generated_text = assistant_reply
elif isinstance(output1, dict):
    generated_text = output1.get('content', str(output1))
else:
    generated_text = str(output1)



pipe2 = pipeline("text-to-speech", model="facebook/mms-tts-eng")

output2 = pipe2(generated_text)

print("Text-generation output:", generated_text)
print("Type of generated_text:", type(generated_text))
print("Text-to-speech output:", output2)

audio = output2.get('audio')
sampling_rate = output2.get('sampling_rate')
if sampling_rate is None or not isinstance(sampling_rate, (int, float)) or sampling_rate < 1000:
    sampling_rate = 22050 

if audio is not None:
    audio_np = np.array(audio, dtype=np.float32)
    print(f"Audio shape: {audio_np.shape}, min: {audio_np.min()}, max: {audio_np.max()}")

    
    if np.abs(audio_np).max() > 1.0:
        audio_np = audio_np / np.abs(audio_np).max()
        print("Audio normalized to [-1, 1] range.")
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    print(f"Playing audio with sampling rate: {sampling_rate}")
    sd.play(audio_np, samplerate=int(sampling_rate))
    sd.wait()
else:
    print("No audio found in output.")