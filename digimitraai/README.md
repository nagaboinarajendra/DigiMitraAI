# Aadhaar Customer Service Chatbot

An intelligent chatbot system for Aadhaar-related queries using RAG (Retrieval Augmented Generation) and LLM technologies.

## Features
- JSON-based knowledge base management
- RAG implementation using FAISS vector store
- LLM integration with ChatGPT
- Audio input processing
- Streamlit-based user interface
- FAQ converter utility

## Setup Instructions
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in .env:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
4. Run the FAQ converter utility:
   ```bash
   streamlit run utils/faq_converter_app.py
   ```
5. Initialize the knowledge base:
   ```bash
   python initialize_knowledge_base.py
   ```
6. Run the application:
   ```bash
   streamlit run frontend/app.py
   ```

## Project Structure
- `agents/`: Contains agent implementations (RAG, LLM, Audio)
- `utils/`: Utility functions and helpers
- `data/`: Data storage (FAQs, vector store)
- `frontend/`: Streamlit interface
- `config/`: Configuration files

## Requirements
- Python 3.8+
- See requirements.txt for package dependencies

## Required Installations on Terminal
sudo apt update && sudo apt install -y ffmpeg
pip install librosa soundfile numpy scipy

## License
[Your chosen license]
