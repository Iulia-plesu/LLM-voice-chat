import streamlit as st
import time
from transformers import pipeline
import sounddevice as sd
import numpy as np

def run_chat():
    if "pipe" not in st.session_state:
        st.session_state.pipe = pipeline("text-generation", model="Qwen/Qwen3-1.7B")
    if "pipe2" not in st.session_state:
        st.session_state.pipe2 = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]
        
    messages = st.session_state.get("messages", [])
    from concurrent.futures import ThreadPoolExecutor
    def render_message(message):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if len(messages) > 4:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(render_message, messages))
    else:
        for message in messages:
            render_message(message)
   
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
       
        with st.chat_message("user"):
            st.markdown(prompt)
       
        pipe = st.session_state.pipe
        pipe2 = st.session_state.pipe2
       
        try:
            # Use messages format for better answer extraction
            messages = [
                {"role": "user", "content": prompt},
            ]
            output1 = pipe(messages)
            
            # Debug: Print the structure to understand what we're getting
            print("Raw output:", output1)
            
            assistant_content = None
            
            # Handle the complex nested structure
            if isinstance(output1, list) and len(output1) > 0:
                # Check if it's the format: [{'generated_text': [messages]}]
                first_item = output1[0]
                if isinstance(first_item, dict) and 'generated_text' in first_item:
                    generated_text = first_item['generated_text']
                    if isinstance(generated_text, list):
                        # Look for the assistant's response in the generated text
                        for msg in generated_text:
                            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                assistant_content = msg.get('content', '').strip()
                                break
                    else:
                        assistant_content = str(generated_text).strip()
                # Fallback: check if the first item is directly an assistant message
                elif isinstance(first_item, dict) and first_item.get('role') == 'assistant':
                    assistant_content = first_item.get('content', '').strip()
            
            # If we still don't have content, try other extraction methods
            if not assistant_content:
                if isinstance(output1, dict):
                    assistant_content = output1.get('content', str(output1)).strip()
                else:
                    assistant_content = str(output1).strip()
            
            # Remove <think>...</think> section if present and clean up the response
            import re
            # Remove thinking tags (case insensitive, multiline)
            assistant_content = re.sub(r'<think>.*?</think>', '', assistant_content, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # Remove any lines that start with thinking patterns
            lines = assistant_content.split('\n')
            cleaned_lines = []
            skip_thinking = False
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith('<think>') or 'think>' in line.lower():
                    skip_thinking = True
                    continue
                elif line.lower().endswith('</think>') or '</think>' in line.lower():
                    skip_thinking = False
                    continue
                elif skip_thinking:
                    continue
                elif line and not line.lower().startswith('hmm') and not line.lower().startswith('okay, the user'):
                    cleaned_lines.append(line)
            
            # Join the cleaned lines and remove extra whitespace
            assistant_content = ' '.join(cleaned_lines).strip()
            
            # If the content starts with reasoning text, try to find the actual answer
            if assistant_content.lower().startswith(('okay, the user', 'hmm,', 'i need to think', 'first, i remember')):
                # Look for sentences that seem like actual answers (not reasoning)
                sentences = assistant_content.split('. ')
                actual_answer = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if (not sentence.lower().startswith(('okay, the user', 'hmm,', 'i need to think', 'first, i remember', 'wait, the user', 'the question is')) 
                        and len(sentence) > 10):
                        actual_answer.append(sentence)
                
                if actual_answer:
                    assistant_content = '. '.join(actual_answer).strip()
                    if not assistant_content.endswith('.'):
                        assistant_content += '.'
            
            # Final check - if content is empty, use a default message
            if not assistant_content:
                assistant_content = "I'm having trouble generating a response. Please try again."
                
        except Exception as e:
            assistant_content = f"Error: {e}"
        
        # Show only the assistant's 'content' part
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in assistant_content.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # TTS: play only the assistant's 'content' part
        try:
            output2 = pipe2(assistant_content)
            audio = output2.get('audio')
            sampling_rate = output2.get('sampling_rate') or 22050
            if audio is not None:
                audio_np = np.array(audio, dtype=np.float32)
                if np.abs(audio_np).max() > 1.0:
                    audio_np = audio_np / np.abs(audio_np).max()
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()
                sd.play(audio_np, samplerate=int(sampling_rate))
                sd.wait()
        except Exception as e:
            print(f"TTS error: {e}")
        
        # Add only the assistant's 'content' part to the session state
        st.session_state.messages.append({"role": "assistant", "content": assistant_content})

if __name__ == "__main__":
    run_chat()