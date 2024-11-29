import os
from pathlib import Path
from typing import Optional
import json

class CredentialsHandler:
    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
    def verify_credentials(self) -> bool:
        """Verify that credentials file exists and is valid"""
        try:
            if not self.credentials_path:
                raise ValueError("No credentials path provided")
            
            creds_path = Path(self.credentials_path)
            if not creds_path.exists():
                raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
            
            # Verify JSON format
            with open(creds_path, 'r') as f:
                creds = json.load(f)
            
            required_fields = [
                'type', 'project_id', 'private_key_id', 'private_key',
                'client_email', 'client_id'
            ]
            
            for field in required_fields:
                if field not in creds:
                    raise ValueError(f"Missing required field in credentials: {field}")
            
            return True
            
        except Exception as e:
            print(f"Error verifying credentials: {str(e)}")
            return False
    
    def setup_credentials(self) -> bool:
        """Set up credentials in environment"""
        try:
            if not self.verify_credentials():
                return False
            
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            return True
            
        except Exception as e:
            print(f"Error setting up credentials: {str(e)}")
            return False