from typing import Dict, Optional
from utils.language_handler import LanguageHandler
import tempfile
import os

class MultilingualAgent:
    def __init__(self, google_credentials_path: str = None):
        """Initialize multilingual processing agent"""
        self.language_handler = LanguageHandler(google_credentials_path)

    def process_audio_query(self, audio_file, source_language: str) -> Dict:
        """Process audio query in source language"""
        try:
            # Get language code
            lang_code = self.language_handler.supported_languages[source_language]['code']
            
            # Convert speech to text
            stt_result = self.language_handler.speech_to_text(audio_file, lang_code)
            
            if not stt_result["success"]:
                return stt_result
                
            text = stt_result["text"]
            
            # Translate to English if not already in English
            if source_language != 'english':
                translation = self.language_handler.translate_text(
                    text,
                    source_language,
                    'english'
                )
                
                if not translation["success"]:
                    return translation
                    
                return {
                    "success": True,
                    "text": text,  # Original text
                    "translated_text": translation["text"],  # English translation
                    "confidence": stt_result["confidence"],
                    "source_language": source_language
                }
            
            return {
                "success": True,
                "text": text,
                "translated_text": text,  # Same as text for English
                "confidence": stt_result["confidence"],
                "source_language": source_language
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def translate_response(self, response: str, target_language: str) -> Dict:
        """Translate response to target language"""
        try:
            if target_language == 'english':
                return {
                    "success": True,
                    "text": response,
                    "language": "english"
                }
            
            translation = self.language_handler.translate_text(
                response,
                'english',
                target_language
            )
            
            return translation
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def generate_audio_response(self, text: str, language: str) -> Dict:
        """Generate audio response in target language"""
        try:
            audio_content, result = self.language_handler.text_to_speech(text, language)
            
            if not result["success"]:
                return result
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                temp_audio.write(audio_content)
                audio_path = temp_audio.name
            
            return {
                "success": True,
                "audio_path": audio_path,
                "language": language
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_supported_languages(self) -> Dict:
        """Get list of supported languages"""
        # Directly access the supported_languages dictionary from language_handler
        return self.language_handler.supported_languages