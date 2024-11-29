from google.cloud import speech_v1
from google.cloud import texttospeech_v1
from google.cloud import translate_v2 as translate
import os
import tempfile
from typing import Dict, Tuple
import soundfile as sf
import json

class LanguageHandler:
    def __init__(self, google_credentials_path: str = None):
        """Initialize language handling services"""
        if google_credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
        
        self.speech_client = speech_v1.SpeechClient()
        self.tts_client = texttospeech_v1.TextToSpeechClient()
        self.translate_client = translate.Client()
        
        # Updated language codes with simplified speech recognition codes
        self.supported_languages = {
            'malayalam': {
                'code': 'en',  # Simplified to 'en' for speech recognition
                'translate_code': 'ml',
                'name': 'Malayalam',
                'voice': 'ml-IN-Standard-A',
                'speech_code': 'ml-IN'
            },
            'hindi': {
                'code': 'hi',
                'translate_code': 'hi',
                'name': 'Hindi',
                'voice': 'hi-IN-Standard-A',
                'speech_code': 'hi-IN'
            },
            'tamil': {
                'code': 'ta',
                'translate_code': 'ta',
                'name': 'Tamil',
                'voice': 'ta-IN-Standard-A',
                'speech_code': 'ta-IN'
            },
            'telugu': {
                'code': 'te',
                'translate_code': 'te',
                'name': 'Telugu',
                'voice': 'te-IN-Standard-A',
                'speech_code': 'te-IN'
            },
            'english': {
                'code': 'en',  # Simplified to 'en'
                'translate_code': 'en',
                'name': 'English',
                'voice': 'en-IN-Standard-A',
                'speech_code': 'en-IN'
            }
        }
    def _print_supported_languages(self):
        """Print languages supported by Google Cloud Speech-to-Text"""
        try:
            parent = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/locations/global"
            request = speech_v1.ListSupportedLanguagesRequest(
                parent=parent,
                display_name_filter="India"
            )
            supported_languages = self.speech_client.list_supported_languages(request)
            print("\nSupported Speech-to-Text languages:")
            for language in supported_languages.languages:
                print(f"- {language.language_code}: {language.display_name}")
        except Exception as e:
            print(f"Error listing supported languages: {e}")

    def verify_language_support(self):
        """Verify language support for speech recognition"""
        try:
            # List supported languages
            supported = self.speech_client.get_supported_languages()
            print("Supported languages:")
            for language in supported.languages:
                print(f"- {language.language_code}")
        except Exception as e:
            print(f"Error checking language support: {str(e)}")


    def speech_to_text(self, audio_file, source_language: str) -> Dict:
        """Convert speech to text in specified language"""
        try:
            # Read audio file
            with open(audio_file, 'rb') as audio:
                content = audio.read()
            
            # Get language info
            lang_info = self.supported_languages.get(source_language)
            if not lang_info:
                return {
                    "success": False,
                    "error": f"Language '{source_language}' not supported",
                    "text": None
                }
            
            # Configure speech recognition
            audio = speech_v1.RecognitionAudio(content=content)
            config = speech_v1.RecognitionConfig(
                language_code="ml-IN",  # Always use 'en' for speech recognition
                enable_automatic_punctuation=True,
                sample_rate_hertz=16000,
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                alternative_language_codes=["en-IN"],
            )
            
            print(f"Attempting speech recognition...")
            
            # Perform recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            if not response.results:
                return {
                    "success": False,
                    "error": "No speech detected",
                    "text": None
                }
            
            # Get transcribed text
            text = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            
            # If source language isn't English, translate the recognized text
            if source_language != 'english':
                translation = self.translate_text(
                    text=text,
                    source_language='english',
                    target_language=source_language
                )
                if translation["success"]:
                    text = translation["text"]
            
            return {
                "success": True,
                "text": text,
                "confidence": confidence,
                "language": source_language
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Speech-to-text error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "text": None
            }

    def translate_text(self, text: str, source_language: str, target_language: str) -> Dict:
        """Translate text between languages"""
        try:
            # Get language codes for translation
            source_code = self.supported_languages[source_language]['translate_code']
            target_code = self.supported_languages[target_language]['translate_code']
            
            # Perform translation
            result = self.translate_client.translate(
                text,
                source_language=source_code,
                target_language=target_code
            )
            
            return {
                "success": True,
                "text": result['translatedText'],
                "source_language": source_language,
                "target_language": target_language
            }
            
        except Exception as e:
            print(f"Translation error: {str(e)}")  # Debug print
            return {
                "success": False,
                "error": str(e),
                "text": None
            }