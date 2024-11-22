# agents/manager_agent.py

import os
from typing import Dict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
import traceback
from agents.rag_agent import RAGAgent
from agents.llm_agent import LLMAgent
from agents.audio_agent import AudioAgent
import json

class ManagerAgent:
    def __init__(self, 
                 rag_confidence_threshold: float = 0.8,  # Increased from 0.6
                 audio_confidence_threshold: float = 0.7,
                 vector_store_path: str = "data/vector_store"):
        # Load environment variables
        self._load_environment()
        
        # Initialize agents
        try:
            self.rag_agent = RAGAgent(vector_store_path=vector_store_path)
            self.llm_agent = LLMAgent()
            self.audio_agent = AudioAgent()
            self.rag_confidence_threshold = rag_confidence_threshold
            self.audio_confidence_threshold = audio_confidence_threshold
            print("All agents initialized successfully")
        except Exception as e:
            print(f"Error initializing agents: {str(e)}")
            raise

    def _load_environment(self):
        """Load environment variables from .env file"""
        # Get the current directory
        current_dir = Path(__file__).parent.parent
        
        # Look for .env file
        env_path = current_dir / '.env'
        
        # Load the .env file
        if env_path.exists():
            load_dotenv(env_path)
        
        # Verify OPENAI_API_KEY is set
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables!")


    def process_query(self, query: str, audio_file: Optional = None) -> Dict:
        try:
            if audio_file:
                audio_result = self.audio_agent.process_audio(audio_file)
                if not audio_result["success"]:
                    return audio_result
                query = audio_result["text"]

            # Clear previous memory before processing new query
            self.rag_agent.clear_memory()
            self.llm_agent.clear_memory()

            # Process with RAG
            try:
                rag_response = self.rag_agent.process_query(query)
                print(f"RAG Confidence: {rag_response['confidence']}")
                
                # Use RAG response if confidence is high enough
                if rag_response["confidence"] >= self.rag_confidence_threshold:
                    return {**rag_response, "text": query}
                
                # Use RAG response if it's a very close match
                if rag_response["confidence"] >= 0.9:
                    return {**rag_response, "text": query}
                
            except Exception as e:
                print(f"RAG processing error: {str(e)}")
                return self._process_llm_fallback(query)

            # Use LLM if RAG confidence is low
            llm_response = self.llm_agent.process_query(query)
            
            # Only combine if RAG has moderate confidence
            if 0.5 <= rag_response["confidence"] < self.rag_confidence_threshold:
                combined = self._combine_responses(rag_response, llm_response)
                return {**combined, "text": query}
            
            return {**llm_response, "text": query}

        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "source": "Error Handler",
                "confidence": 0.0,
                "text": query if query else "Audio processing failed"
            }

    def _process_audio_input(self, audio_file) -> Dict:
        """Process audio input and handle errors"""
        try:
            validation = self.audio_agent.validate_audio(audio_file)
            if not validation["valid"]:
                return {
                    "success": False,
                    "answer": f"Audio file error: {validation['error']}",
                    "source": "Audio Processing",
                    "confidence": 0.0
                }
            
            audio_result = self.audio_agent.process_audio(audio_file)
            if not audio_result["success"]:
                return {
                    "success": False,
                    "answer": f"Audio processing error: {audio_result['error']}",
                    "source": "Audio Processing",
                    "confidence": 0.0
                }
            
            return audio_result
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return {
                "success": False,
                "answer": "Error processing audio input",
                "source": "Audio Processing",
                "confidence": 0.0
            }

    def _process_llm_fallback(self, query: str) -> Dict:
        """Process query with LLM as fallback"""
        try:
            return self.llm_agent.process_query(query)
        except Exception as e:
            print(f"Error in LLM fallback: {str(e)}")
            return {
                "answer": "I apologize, but I'm having trouble processing your request. Please try again.",
                "source": "Error Handler",
                "confidence": 0.0
            }

    def _combine_responses(self, rag_response: Dict, llm_response: Dict) -> Dict:
        """Combine RAG and LLM responses"""
        combined_answer = f"""Based on our knowledge base and AI analysis:
        
        {rag_response['answer']}
        
        Additionally: {llm_response['answer']}"""
        
        return {
            "answer": combined_answer,
            "sources": rag_response.get("sources", []) + [llm_response["source"]],
            "confidence": max(rag_response["confidence"], llm_response["confidence"])
        }

    def initialize_knowledge_base(self, documents: List[str]) -> None:
        """Initialize the RAG knowledge base with documents"""
        try:
            print("Starting knowledge base initialization...")
            self.rag_agent.initialize_vector_store(documents)
            print("Knowledge base initialized successfully")
        except Exception as e:
            print(f"Error initializing knowledge base: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def update_knowledge_base(self, new_documents: List[str]) -> None:
        """Update the knowledge base with new documents"""
        self.rag_agent.update_knowledge_base(new_documents)