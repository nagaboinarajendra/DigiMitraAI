import os
import tempfile
import numpy as np
from typing import Dict, List, Optional
import whisper
import wave
import contextlib

class AudioProcessor:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)
        self.supported_formats = {'.wav', '.mp3'}
    
    def validate_audio(self, file_path: str) -> Dict:
        """Validate audio file format and quality"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.supported_formats:
            return {
                "valid": False,
                "error": f"Unsupported format. Supported formats: {', '.join(self.supported_formats)}"
            }
        
        if ext == '.wav':
            return self._validate_wav(file_path)
        
        return {"valid": True}
    
    def _validate_wav(self, file_path: str) -> Dict:
        """Validate WAV file specifications"""
        try:
            with contextlib.closing(wave.open(file_path, 'r')) as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                
                if channels > 2:
                    return {
                        "valid": False,
                        "error": "Too many audio channels. Maximum supported: 2"
                    }
                
                if frame_rate < 8000:
                    return {
                        "valid": False,
                        "error": "Sample rate too low. Minimum: 8000 Hz"
                    }
                
                return {"valid": True}
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid WAV file: {str(e)}"
            }
    
    def process_audio(self, audio_file, language: Optional[str] = None) -> Dict:
        """Process audio file and return transcription"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Validate audio
            validation = self.validate_audio(tmp_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"]
                }
            
            # Transcribe audio
            transcribe_options = {}
            if language:
                transcribe_options["language"] = language
            
            result = self.model.transcribe(tmp_path, **transcribe_options)
            
            # Calculate confidence
            confidence = np.mean([segment["confidence"] for segment in result["segments"]])
            
            return {
                "success": True,
                "text": result["text"],
                "language": result["language"],
                "confidence": confidence,
                "segments": result["segments"]
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages"""
        return self.model.supported_languages()