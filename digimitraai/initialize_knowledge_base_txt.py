import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from agents.manager_agent import ManagerAgent

def load_environment():
    """Load environment variables from .env file"""
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Look for .env file in current directory
    env_path = current_dir / '.env'
    
    # Load the .env file
    if env_path.exists():
        load_dotenv(env_path)
    else:
        raise FileNotFoundError("No .env file found!")
    
    # Verify OPENAI_API_KEY is set
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found in environment variables!")

def load_faqs():
    """Load FAQ documents from the data directory"""
    faqs = []
    faqs_dir = Path(__file__).parent / "data" / "faqs"
    
    if not faqs_dir.exists():
        raise FileNotFoundError(f"FAQs directory not found at {faqs_dir}")
    
    for filename in os.listdir(faqs_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(faqs_dir, filename), 'r', encoding='utf-8') as file:
                faqs.append(file.read())
    
    if not faqs:
        raise ValueError("No FAQ files found in the data/faqs directory!")
    
    return faqs

def main():
    try:
        # Load environment variables
        print("Loading environment variables...")
        load_environment()
        
        # Initialize manager agent
        print("Initializing manager agent...")
        manager = ManagerAgent()
        
        # Load FAQs
        print("Loading FAQ documents...")
        faqs = load_faqs()
        print(f"Loaded {len(faqs)} FAQ documents")
        
        # Initialize knowledge base
        print("Initializing knowledge base...")
        manager.initialize_knowledge_base(faqs)
        
        print("✅ Knowledge base initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()