import gradio as gr
import sys
import os
import traceback
import tempfile
import shutil
import numpy as np
from typing import Dict, List, Union, Any
from pathlib import Path
import scipy.io.wavfile

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agents.manager_agent import ManagerAgent

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agents.manager_agent import ManagerAgent

class AadhaarChatInterface:
    def __init__(self):
        self.manager_agent = ManagerAgent()
        self.chat_history = []
        
        # Get supported languages
        self.languages = self.manager_agent.multilingual_agent.supported_languages
        self.language_names = {lang: info['name'] for lang, info in self.languages.items()}

    def process_query(self, message: str, audio_data: any, source_language: str, target_language: str, history: list) -> list:
        try:
            if audio_data:
                # Handle audio input (both upload and recording)
                if isinstance(audio_data, tuple):
                    audio_samples, sampling_rate = audio_data
                    
                    # Create a temporary WAV file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_path = temp_file.name

                    # Convert and save audio data using scipy
                    try:
                        import scipy.io.wavfile as wav
                        if isinstance(audio_samples, np.ndarray):
                            # Ensure the data is in the right format
                            audio_samples = (audio_samples * 32767).astype(np.int16)
                            wav.write(temp_path, sampling_rate, audio_samples)
                        elif isinstance(audio_samples, str) and os.path.exists(audio_samples):
                            # Copy existing file
                            shutil.copy2(audio_samples, temp_path)
                        
                        # Process audio with temporary file
                        response = self.manager_agent.process_multilingual_query(
                            audio_file=temp_path,
                            source_language=source_language,
                            target_language=target_language
                        )
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    response = self.manager_agent.process_multilingual_query(
                        audio_file=audio_data,
                        source_language=source_language,
                        target_language=target_language
                    )
                
                query = response.get("original_query", "")
                
            elif message:
                # Process text input
                response = self.manager_agent.process_multilingual_query(
                    query=message,
                    source_language=source_language,
                    target_language=target_language
                )
                query = message
            else:
                return history
            
            # Format response for chat history
            if not history:
                history = []
                
            history.append([query, response["answer"]])
            
            # Add debug info if available
            debug_info = ""
            if "confidence" in response:
                debug_info += f"Confidence: {response['confidence']:.2f}\n"
            if "sources" in response:
                debug_info += "Sources:\n" + "\n".join(response.get("sources", []))
            
            if debug_info:
                history.append([None, f"Debug Info:\n{debug_info}"])
                
            return history

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(f"Error details: {str(e)}")
            traceback.print_exc()  # Print full traceback
            if not history:
                history = []
            history.append([message if message else "Audio Query", error_message])
            return history

    def process_uploaded_audio(audio_input, source_lang, target_lang, history):
        if audio_input is not None:
            # audio_input is tuple (file_path, sampling_rate)
            return self.process_query(None, audio_input, source_lang, target_lang, history)
        return history

    def process_recorded_audio(audio_input, source_lang, target_lang, history):
        if audio_input is not None:
            # audio_input is tuple (file_path, sampling_rate)
            return self.process_query(None, audio_input, source_lang, target_lang, history)
        return history
    
    def create_interface(self):
        with gr.Blocks(title="Aadhaar Customer Service Assistant") as interface:
            gr.Markdown("# Aadhaar Customer Service Assistant")
            
            with gr.Row():
                source_language = gr.Dropdown(
                    choices=list(self.language_names.keys()),
                    value="english",
                    label="Input Language"
                )
                target_language = gr.Dropdown(
                    choices=list(self.language_names.keys()),
                    value="english",
                    label="Output Language"
                )

            chatbot = gr.Chatbot(
                [],
                height=400
            )

            with gr.Row():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Type your question here...",
                    scale=4
                )

            with gr.Row():
                # Audio recording component
                mic_audio = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record Audio"
                )

                # Audio upload component (changed from File to Audio)
                upload_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload Audio File"
                )

            def clear_chat():
                return []

            def process_text(message, source_lang, target_lang, history):
                try:
                    if message:
                        response = self.manager_agent.process_multilingual_query(
                            query=message,
                            source_language=source_lang,
                            target_language=target_lang
                        )
                        history = history or []
                        history.append((message, response["answer"]))
                        return history, ""
                except Exception as e:
                    history = history or []
                    history.append((message, f"Error: {str(e)}"))
                    return history, ""

            def process_audio(audio_path, source_lang, target_lang, history):
                try:
                    if audio_path:
                        # Process audio query
                        response = self.manager_agent.process_multilingual_query(
                            audio_file=audio_path,
                            source_language=source_lang,
                            target_language=target_lang
                        )
                        
                        history = history or []
                        
                        # Show original transcribed text in source language
                        if "text" in response:
                            source_text = response["text"]
                            history.append((f"Question ({source_lang}): {source_text}", None))
                        
                        # Show English translation if source language is not English
                        if source_lang != "english" and "original_text" in response:
                            english_text = response["original_text"]
                            history.append((f"Question (English): {english_text}", None))
                        
                        # Show answer in target language
                        if "answer" in response:
                            history.append((None, response["answer"]))
                        
                        # Show confidence and sources if available
                        debug_info = ""
                        if "confidence" in response:
                            debug_info += f"Confidence: {response['confidence']:.2f}\n"
                        if "sources" in response:
                            debug_info += "Sources:\n" + "\n".join(response.get("sources", []))
                        
                        if debug_info:
                            history.append((None, f"Debug Info:\n{debug_info}"))
                            
                        return history
                        
                except Exception as e:
                    history = history or []
                    history.append(("Audio Processing Error", f"Error: {str(e)}"))
                    return history

            # Text input handling
            txt.submit(
                fn=process_text,
                inputs=[
                    txt,
                    source_language,
                    target_language,
                    chatbot
                ],
                outputs=[chatbot, txt]
            )

            # Audio recording handling
            mic_audio.stop_recording(
                fn=process_audio,
                inputs=[
                    mic_audio,
                    source_language,
                    target_language,
                    chatbot
                ],
                outputs=[chatbot]
            )

            # Audio upload handling
            upload_audio.change(  # Changed from upload to change
                fn=process_audio,
                inputs=[
                    upload_audio,
                    source_language,
                    target_language,
                    chatbot
                ],
                outputs=[chatbot]
            )

            # Clear button
            clear = gr.Button("Clear")
            clear.click(
                fn=clear_chat,
                inputs=None,
                outputs=chatbot
            )

        return interface

def main():
    chat_app = AadhaarChatInterface()
    interface = chat_app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=4
    )

if __name__ == "__main__":
    main()