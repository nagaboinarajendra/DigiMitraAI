import whisper
from google.cloud import translate_v2 as translate
import tempfile
import os
from typing import Dict, Optional, List
import torch
import gc

class MultilingualAgent:
    def __init__(self, google_credentials_path: str = None):
        """Initialize multilingual processing agent"""
        if google_credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
            
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("Loading Whisper model...")
        try:
            # Set environment variables for memory optimization
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            # Configure PyTorch for memory efficiency
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            
            # Use medium model for better accuracy with longer sentences
            self.whisper_model = whisper.load_model(
                "base",
                device="cpu",
                download_root="./models"
            )
            
            # Enhanced prompt templates for each language
            self.prompt_templates = {
                'english': [
                    "This is a question about Aadhaar ID and UIDAI services.",
                    "Common topics: enrollment, verification, mandatory requirements.",
                ],
                'malayalam': [
                    "ആധാർ സംബന്ധിച്ചുള്ള ചോദ്യങ്ങൾ.",  # Questions about Aadhaar
                    "യുഐഡിഎഐ സേവനങ്ങളെക്കുറിച്ചുള്ള വിവരങ്ങൾ.",  # Information about UIDAI services
                ],
                'hindi': [
                    "आधार और यूआईडीएआई सेवाओं के बारे में प्रश्न.",
                    "पंजीकरण, सत्यापन, अनिवार्य आवश्यकताएं.",
                ]
                # Add templates for other languages
            }

            # Optimize model for inference
            self.whisper_model.eval()
            with torch.no_grad():
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print("Whisper model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Whisper model: {str(e)}")
            raise
        
        # Initialize Google Translate
        self.translate_client = translate.Client()
        
        # Define supported languages with Whisper language codes
        self.supported_languages = {
            'malayalam': {
                'code': 'ml',
                'name': 'Malayalam',
                'translate_code': 'ml',
                'whisper_code': 'malayalam',
                'common_words': ['aadhaar', 'aadhar', 'uid', 'uidai'],  # Words to preserve
                'confidence_threshold': 0.6
            },
            'hindi': {
                'code': 'hi',
                'name': 'Hindi',
                'translate_code': 'hi',
                'whisper_code': 'hindi',
                'common_words': ['aadhaar', 'aadhar', 'uid', 'uidai'],  # Words to preserve
                'confidence_threshold': 0.6
            },
            'tamil': {
                'code': 'ta',
                'name': 'Tamil',
                'translate_code': 'ta',
                'whisper_code': 'tamil',
                'common_words': ['aadhaar', 'aadhar', 'uid', 'uidai'],  # Words to preserve
                'confidence_threshold': 0.6
            },
            'telugu': {
                'code': 'te',
                'name': 'Telugu',
                'translate_code': 'te',
                'whisper_code': 'telugu',
                'common_words': ['aadhaar', 'aadhar', 'uid', 'uidai'],  # Words to preserve
                'confidence_threshold': 0.6
            },
            'english': {
                'code': 'en',
                'name': 'English',
                'translate_code': 'en',
                'whisper_code': 'english',
                'common_words': ['aadhaar', 'aadhar', 'uid', 'uidai'],  # Words to preserve
                'confidence_threshold': 0.6
            }
        }

    def _cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio for better recognition"""
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Apply noise reduction
            audio_denoised = librosa.effects.preemphasis(audio)
            
            # Normalize audio
            audio_normalized = librosa.util.normalize(audio_denoised)
            
            # Save processed audio
            temp_path = audio_path + "_processed.wav"
            sf.write(temp_path, audio_normalized, sr)
            
            return temp_path
        except Exception as e:
            print(f"Audio preprocessing failed: {e}")
            return audio_path


    def process_audio_query(self, audio_file, source_language: str) -> Dict:
        """Process audio query with improved language handling"""
        temp_path = None
        try:
            lang_config = self.supported_languages.get(source_language)
            if not lang_config:
                return {
                    "success": False,
                    "error": f"Language {source_language} not supported",
                    "text": None
                }

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                if hasattr(audio_file, 'read'):
                    temp_file.write(audio_file.read())
                else:
                    with open(audio_file, 'rb') as src_file:
                        temp_file.write(src_file.read())
                temp_path = temp_file.name

            try:
                prompts = self.prompt_templates.get(source_language, [""])
                
                # Multiple transcription attempts with different configurations
                transcription_results = []
                
                # Attempt 1: Standard transcription
                result1 = self.whisper_model.transcribe(
                    temp_path,
                    language=lang_config['whisper_code'],
                    task="transcribe",  # First try transcription in source language
                    initial_prompt=prompts[0],
                    temperature=0.0,
                    best_of=3,
                    fp16=False,
                    condition_on_previous_text=True,
                    verbose=True
                )
                transcription_results.append(result1)
                
                # Attempt 2: Direct translation to English
                if source_language != 'english':
                    result2 = self.whisper_model.transcribe(
                        temp_path,
                        language=lang_config['whisper_code'],
                        task="translate",
                        initial_prompt=prompts[0],
                        temperature=0.2,
                        best_of=3,
                        fp16=False
                    )
                    transcription_results.append(result2)
                
                # Get the best result
                best_result = self._select_best_transcription(transcription_results)
                
                if not best_result or not best_result.get("text"):
                    return {
                        "success": False,
                        "error": "No clear speech detected",
                        "text": None
                    }
                
                text = best_result["text"].strip()
                
                # For non-English, get both original and translated text
                if source_language != 'english':
                    # Get original language text
                    original_text = result1["text"].strip()
                    
                    # Use Google Translate for accurate translation
                    translation = self.translate_client.translate(
                        original_text,
                        source_language=lang_config['translate_code'],
                        target_language='en'
                    )
                    english_text = translation['translatedText']
                else:
                    original_text = text
                    english_text = text
                
                # Post-process both texts
                english_text = self._post_process_text(english_text)
                if source_language == 'english':
                    english_text = self._enhance_english_recognition(english_text)
                
                # Double-check translation accuracy
                if source_language != 'english':
                    # Back-translate to verify
                    back_translation = self.translate_client.translate(
                        english_text,
                        source_language='en',
                        target_language=lang_config['translate_code']
                    )
                    
                    # If back-translation is very different, use the direct Whisper translation
                    if self._calculate_similarity(original_text, back_translation['translatedText']) < 0.5:
                        english_text = result2["text"].strip()
                        english_text = self._post_process_text(english_text)

                return {
                    "success": True,
                    "text": original_text,
                    "original_text": english_text,
                    "confidence": best_result.get("confidence", 0.0),
                    "source_language": source_language
                }

            finally:
        # Clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        print(f"Error cleaning up temporary file: {e}")

        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": None
            }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


    def _select_best_transcription(self, results):
        """Select the best transcription result based on multiple criteria"""
        if not results:
            return None
            
        for result in results:
            text = result.get("text", "").lower()
            # Prioritize results containing key terms
            if any(term in text for term in ["aadhaar", "aadhar", "uid", "mandatory"]):
                return result
        
        # Default to the first result if no key terms found
        return results[0]

    def _post_process_text(self, text: str) -> str:
        """Clean up and standardize text"""
        # Common replacements for better accuracy
        replacements = {
            "adhaar": "Aadhaar",
            "adhar": "Aadhaar",
            "aadhar": "Aadhaar",
            "aadharam": "Aadhaar",
            "uid": "UID",
            "uidai": "UIDAI"
        }
        
        text = text.strip()
        for old, new in replacements.items():
            text = text.replace(old.lower(), new)
        
        # Ensure question marks are preserved
        if any(q in text.lower() for q in ["what", "how", "why", "when", "where", "which", "is", "are"]) and not text.endswith("?"):
            text += "?"
            
        return text

    def _enhance_english_recognition(self, text):
        """Enhance English text recognition specifically for Aadhaar domain"""
        # Common misrecognitions and their corrections
        corrections = {
            "a dark": "aadhaar",
            "other": "aadhaar",
            "ada": "aadhaar",
            "adhar": "aadhaar",
            "meditation": "mandatory",
            "mandate": "mandatory",
            "you id": "uid",
            "you ideal": "uidai",
            "you i": "uid"
        }
        
        text_lower = text.lower()
        for wrong, right in corrections.items():
            if wrong in text_lower:
                text = text.lower().replace(wrong, right)
        
        # Ensure proper capitalization
        text = text.capitalize()
        text = text.replace("aadhaar", "Aadhaar")
        text = text.replace("uid", "UID")
        text = text.replace("uidai", "UIDAI")
        
        # Ensure question mark for questions
        if any(q in text.lower() for q in ["what", "how", "why", "when", "where", "which", "is", "are"]) and not text.endswith("?"):
            text += "?"
            
        return text
    
    def translate_text(self, text: str, source_language: str, target_language: str) -> Dict:
        """Translate text between languages"""
        try:
            # Skip translation if languages are the same
            if source_language == target_language:
                return {
                    "success": True,
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language
                }
            
            source_code = self.supported_languages[source_language]['translate_code']
            target_code = self.supported_languages[target_language]['translate_code']
            
            result = self.translate_client.translate(
                text,
                source_language=source_code,
                target_language=target_code
            )
            
            translated_text = self._post_process_text(result['translatedText'])
            
            return {
                "success": True,
                "text": translated_text,
                "source_language": source_language,
                "target_language": target_language
            }
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": None
            }