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

# Get absolute path to project root and add to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Add these imports
from agents.manager_agent import ManagerAgent
from utils.credentials_handler import CredentialsHandler
from utils.language_handler import LanguageHandler
from agents.multilingual_agent import MultilingualAgent

def get_language_options(manager):
    """Get supported languages directly from multilingual agent"""
    try:
        if hasattr(manager, 'multilingual_agent'):
            return manager.multilingual_agent.supported_languages
        return {'english': {'code': 'en', 'name': 'English', 'translate_code': 'en'}}
    except Exception as e:
        print(f"Error getting language options: {e}")
        return {'english': {'code': 'en', 'name': 'English', 'translate_code': 'en'}}


def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        try:
            print("Starting initialization...")
            st.session_state.messages = []
            st.session_state.recording = False
            st.session_state.audio_data = None
            st.session_state.debug_info = []
            
            # Initialize manager agent
            print("Initializing manager agent...")
            manager = ManagerAgent()
            st.session_state.manager_agent = manager
            
            # Set language preferences
            st.session_state.language_preferences = {
                'source_language': 'english',
                'target_language': 'english'
            }
            
            # Mark initialization as complete
            st.session_state.initialized = True
            print("Initialization complete")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            st.error(f"Error initializing agent: {str(e)}")
            st.session_state.debug_info = [f"Initialization error: {str(e)}"]
            st.session_state.initialized = False

def display_message(role: str, content: str, metadata: Dict = None, message_idx: int = 0):
    """Display a message in the chat interface with language support"""
    with st.chat_message(role):
        # Display translation info if available
        if metadata and metadata.get("is_translation"):
            st.markdown("*Translation:*")
        
        # Display main content
        st.markdown(content)
        
        # Show additional details in expander
        if metadata and not metadata.get("is_translation"):
            with st.expander("Show Details", expanded=False):
                if "original_answer" in metadata:
                    st.info(f"Original (English): {metadata['original_answer']}")
                if "sources" in metadata:
                    st.info("Sources:\n" + "\n".join(metadata["sources"]))
                if "confidence" in metadata:
                    st.info(f"Confidence: {metadata['confidence']:.2f}")
                if "source_language" in metadata:
                    st.info(f"Input Language: {metadata['source_language'].capitalize()}")
                if "target_language" in metadata:
                    st.info(f"Output Language: {metadata['target_language'].capitalize()}")

def record_audio():
    """Record audio with language support"""
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


def process_multilingual_audio(audio_file, source_language: str, target_language: str):
    """Process audio file with language support"""
    try:
        st.session_state.debug_info.append("Processing multilingual audio file")
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
        
        try:
            response = st.session_state.manager_agent.process_multilingual_query(
                audio_file=audio_path,
                source_language=source_language,
                target_language=target_language
            )
            
            # Add transcription to chat
            if response.get("original_query"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Audio Query in {source_language.capitalize()}:\n{response['original_query']}",
                    "metadata": {"is_audio": True}
                })
            
                # Add translation if languages are different
                if source_language != 'english':
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Translated to English: {response['translated_query']}",
                        "metadata": {"is_translation": True}
                    })
            
            # Add response to chat
            if response.get("answer"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": response
                })
                
                # Play audio response if available
                if "audio_response" in response:
                    st.audio(response["audio_response"])
                    
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            
        finally:
            # Cleanup temporary file
            if 'audio_path' in locals():
                os.unlink(audio_path)
                
    except Exception as e:
        st.error(f"Error handling audio file: {str(e)}")

def main():
    st.title("Aadhaar Customer Service Assistant")

    try:
        initialize_session_state()
        
        # Get languages directly from multilingual agent
        if 'manager_agent' in st.session_state:
            languages = st.session_state.manager_agent.multilingual_agent.supported_languages
        else:
            languages = {'english': {'code': 'en', 'name': 'English', 'translate_code': 'en'}}
            st.warning("Using fallback language options")

        # Language selection
        col1, col2 = st.columns(2)
        with col1:
            source_language = st.selectbox(
                "Select Input Language",
                options=list(languages.keys()),
                format_func=lambda x: languages[x]['name']
            )
        with col2:
            target_language = st.selectbox(
                "Select Output Language",
                options=list(languages.keys()),
                format_func=lambda x: languages[x]['name']
            )

        # Audio input options
        col3, col4 = st.columns(2)
        
        with col3:
            audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
        
        with col4:
            if not st.session_state.recording:
                if st.button("🎤 Start Recording", key="start_rec"):
                    audio_path = record_audio()
                    if audio_path:
                        with open(audio_path, 'rb') as audio:
                            process_multilingual_audio(
                                audio,
                                source_language,
                                target_language
                            )
                        os.unlink(audio_path)
            else:
                st.button("⏹️ Stop Recording", key="stop_rec")
        
        # Process uploaded audio
        if audio_file:
            process_multilingual_audio(
                audio_file,
                source_language,
                target_language
            )
        
        # Text input
        if prompt := st.chat_input(f"Type your question in {languages[source_language]['name']}"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                response = st.session_state.manager_agent.process_multilingual_query(
                    query=prompt,
                    source_language=source_language,
                    target_language=target_language
                )
                
                # Display original and translated if different languages
                if source_language != 'english':
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Original: {response['original_query']}\nTranslated: {response['translated_query']}",
                        "is_translation": True
                    })
                
                # Add response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": response
                })
                
                # Play audio response if available
                if "audio_response" in response:
                    st.audio(response["audio_response"])
                    
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