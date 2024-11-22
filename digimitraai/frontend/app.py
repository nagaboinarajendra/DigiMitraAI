import streamlit as st
from typing import Dict
import sys
import os
from pathlib import Path
import traceback
import sounddevice as sd
import wavio
import numpy as np
import tempfile
from scipy.io import wavfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from agents.manager_agent import ManagerAgent

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'manager_agent' not in st.session_state:
        try:
            st.session_state.manager_agent = ManagerAgent()
            st.session_state.debug_info = []
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            st.session_state.debug_info = [f"Initialization error: {str(e)}"]

def display_message(role: str, content: str, metadata: Dict = None, message_idx: int = 0):
    """Display a message in the chat interface"""
    with st.chat_message(role):
        st.markdown(content)
        if metadata:
            with st.expander("Show Details", expanded=False):
                if "sources" in metadata:
                    st.info("Sources:\n" + "\n".join(metadata["sources"]))
                if "confidence" in metadata:
                    st.info(f"Confidence: {metadata['confidence']:.2f}")

def record_audio():
    """Record audio from microphone"""
    try:
        # List available audio devices
        devices = sd.query_devices()
        input_device = None
        
        # Find first input device
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_device = i
                break
        
        if input_device is None:
            st.error("No input device found")
            return None

        fs = 44100  # Sample rate
        seconds = 10  # Maximum recording duration
        
        st.session_state.recording = True
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, device=input_device)
        st.info("🎤 Recording... Click 'Stop Recording' when finished")
        sd.wait()
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            wavio.write(temp_audio.name, recording, fs, sampwidth=2)
            return temp_audio.name
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None
    finally:
        st.session_state.recording = False

def process_audio_response(audio_file):
    """Process audio file and get response"""
    try:
        response = st.session_state.manager_agent.process_query(None, audio_file)
        
        # Clear any previous responses for the same query
        query_text = response.get('text', 'Audio processed')
        st.session_state.messages = [msg for msg in st.session_state.messages 
                                   if msg['role'] != 'assistant' or 
                                   'previous_query' not in msg or 
                                   msg['previous_query'] != query_text]
        
        # Add the new response
        st.session_state.messages.extend([
            {
                "role": "user",
                "content": f"Audio Query: {query_text}"
            },
            {
                "role": "assistant",
                "content": response["answer"],
                "metadata": response,
                "previous_query": query_text
            }
        ])
        return True
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return False

def main():
    st.title("Aadhaar Customer Service Assistant")
    
    try:
        initialize_session_state()
        
        # Audio input options
        col1, col2 = st.columns(2)
        
        with col1:
            audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
        
        with col2:
            # Microphone recording
            if not st.session_state.recording:
                if st.button("🎤 Start Recording", key="start_rec"):
                    st.session_state.recording = True
                    with st.spinner("Recording..."):
                        audio_path = record_audio()
                        if audio_path:
                            with open(audio_path, 'rb') as audio:
                                process_audio_response(audio)
                            os.unlink(audio_path)
            else:
                if st.button("⏹️ Stop Recording", key="stop_rec"):
                    st.session_state.recording = False
        
        # Process uploaded audio
        if audio_file:
            process_audio_response(audio_file)
        
        # Text input
        if prompt := st.chat_input("Type your question here"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                response = st.session_state.manager_agent.process_query(prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": response,
                    "previous_query": prompt
                })
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
        
        # Display chat history
        for idx, message in enumerate(st.session_state.messages):
            display_message(
                message["role"],
                message["content"],
                message.get("metadata"),
                message_idx=idx
            )

        # Display debug information
        if st.session_state.debug_info:
            with st.expander("Debug Information", expanded=False):
                for info in st.session_state.debug_info:
                    st.text(info)
    
    except Exception as e:
        error_msg = f"Application error: {str(e)}\n{traceback.format_exc()}"
        st.error(error_msg)
        if 'debug_info' in st.session_state:
            st.session_state.debug_info.append(error_msg)

if __name__ == "__main__":
    main()