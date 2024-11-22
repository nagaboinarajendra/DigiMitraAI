import os
from dotenv import load_dotenv
from pathlib import Path
from agents.manager_agent import ManagerAgent
import json

def load_environment():
    """Load environment variables"""
    current_dir = Path(__file__).parent
    env_path = current_dir / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        raise FileNotFoundError("No .env file found!")
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found in environment variables!")

def verify_json_knowledge_base():
    """Verify JSON knowledge base exists and is valid"""
    json_path = Path("data/faqs/consolidated_faqs.json")
    if not json_path.exists():
        raise FileNotFoundError(
            "consolidated_faqs.json not found! Please run the FAQ converter utility first."
        )
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not data.get('faqs'):
                raise ValueError("No FAQs found in JSON file")
            print(f"Found {len(data['faqs'])} FAQs in knowledge base")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON file format")

def main():
    try:
        print("Loading environment variables...")
        load_environment()
        
        print("Verifying JSON knowledge base...")
        verify_json_knowledge_base()
        
        print("Initializing manager agent...")
        manager = ManagerAgent()
        
        print("Initializing knowledge base...")
        manager.initialize_knowledge_base()
        
        print("✅ Knowledge base initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()