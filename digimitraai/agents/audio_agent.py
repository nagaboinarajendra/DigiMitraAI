# agents/audio_agent.py

from typing import Dict, List
import openai
import os
from pathlib import Path
import tempfile

class AudioAgent:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = self.api_key

    def process_audio(self, audio_file, language: str = None) -> Dict:
        """
        Process audio file and convert to text using OpenAI's Whisper API
        
        Args:
            audio_file: Audio file object from Streamlit
            language: Optional language code for transcription
        
        Returns:
            Dict containing transcribed text and metadata
        """
        try:
            # Save audio file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name

            # Use OpenAI's Whisper API
            with open(tmp_path, "rb") as audio:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language=language
                )

            return {
                "success": True,
                "text": transcript.text,
                "language": language or "auto",
                "confidence": 0.9  # Whisper API doesn't return confidence, using default
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0
            }
        
        finally:
            # Clean up temporary file
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
    
    def validate_audio(self, audio_file) -> Dict:
        """Validate audio file before processing"""
        # Get file extension
        ext = Path(audio_file.name).suffix.lower()
        
        # Check supported formats
        if ext not in ['.wav', '.mp3', '.m4a']:
            return {
                "valid": False,
                "error": f"Unsupported format {ext}. Supported formats: .wav, .mp3, .m4a"
            }
        
        # Check file size (10MB limit for Whisper API)
        if audio_file.size > 10 * 1024 * 1024:  # 10MB in bytes
            return {
                "valid": False,
                "error": "File size exceeds 10MB limit"
            }
        
        return {"valid": True}

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", 
            "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "iw", "uk", "el", "ms", 
            "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", 
            "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", 
            "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
            "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
            "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
            "ba", "jw", "su"
        ]