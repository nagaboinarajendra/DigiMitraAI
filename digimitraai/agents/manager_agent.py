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
                 rag_confidence_threshold: float = 0.75,
                 audio_confidence_threshold: float = 0.7,
                 vector_store_path: str = "data/vector_store",
                 json_path: str = "data/faqs/consolidated_faqs.json"):
        # Load environment variables
        self._load_environment()
        
        # Initialize paths
        self.vector_store_path = vector_store_path
        self.json_path = json_path
        
        # Verify JSON knowledge base exists
        self._verify_knowledge_base()
        
        # Initialize agents
        self.rag_agent = RAGAgent(
            vector_store_path=vector_store_path,
            json_path=json_path
        )
        self.llm_agent = LLMAgent()
        self.audio_agent = AudioAgent()
        
        # Set confidence thresholds
        self.rag_confidence_threshold = rag_confidence_threshold
        self.audio_confidence_threshold = audio_confidence_threshold
        
        print("All agents initialized successfully")

    def _load_environment(self):
        """Load environment variables"""
        try:
            current_dir = Path(__file__).parent.parent
            env_path = current_dir / '.env'
            
            if env_path.exists():
                load_dotenv(env_path)
                os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
            else:
                raise ValueError("No .env file found!")
            
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not found in environment variables!")
            
            print("Environment variables loaded successfully")
        except Exception as e:
            print(f"Error loading environment variables: {str(e)}")
            raise

    def _verify_knowledge_base(self):
        """Verify JSON knowledge base exists and is valid"""
        try:
            json_path = Path(self.json_path)
            if not json_path.exists():
                raise FileNotFoundError("Knowledge base JSON file not found!")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data.get('faqs'):
                    raise ValueError("No FAQs found in knowledge base!")
                print(f"Found {len(data['faqs'])} FAQs in knowledge base")
        except Exception as e:
            print(f"Error verifying knowledge base: {str(e)}")
            raise

    def process_query(self, query: str, audio_file: Optional = None) -> Dict:
        """Process a query using the appropriate agent(s)"""
        try:
            if audio_file:
                audio_result = self._process_audio_input(audio_file)
                if not audio_result["success"]:
                    return audio_result
                query = audio_result["text"]

            # Clear previous memory
            self.rag_agent.clear_memory()
            self.llm_agent.clear_memory()

            # Process with RAG
            print(f"\nProcessing query: {query}")
            try:
                rag_response = self.rag_agent.process_query(query)
                confidence = rag_response["confidence"]
                domain_relevant = rag_response["domain_relevant"]
                has_sources = rag_response["has_sources"]
                semantic_match = rag_response.get("semantic_match", 0.0)
                debug_info = rag_response.get("debug_info", {})
                
                print(f"RAG confidence: {confidence}")
                print(f"Domain relevant: {domain_relevant}")
                print(f"Has sources: {has_sources}")
                print(f"Semantic match: {semantic_match}")
                
                # Case 1: Very high confidence RAG response
                if confidence >= self.rag_confidence_threshold and semantic_match >= 0.85:
                    print("Using RAG response (high confidence)")
                    return {**rag_response, "text": query}
                
                # Case 2: Exact match
                if rag_response.get("exact_match", False):
                    print("Using RAG response (exact match)")
                    return {**rag_response, "text": query}
                
                # Case 3: Domain relevant query
                if domain_relevant:
                    print("Domain relevant query, processing...")
                    
                    # Get LLM response
                    llm_response = self.llm_agent.process_query(query)
                    
                    # Combine responses only if RAG has moderately relevant info
                    if has_sources and semantic_match >= 0.6:
                        print("Combining RAG and LLM responses")
                        combined = self._combine_responses(rag_response, llm_response)
                        return {**combined, "text": query}
                    
                    # Use LLM response for domain-relevant queries without good matches
                    print("Using LLM response for domain-relevant query")
                    return {**llm_response, "text": query}
                
                # Case 4: Not domain relevant
                return {
                    "answer": "I'm an Aadhaar assistance chatbot. Could you please ask a question related to Aadhaar services?",
                    "confidence": 1.0,
                    "source": "System",
                    "text": query
                }

            except Exception as e:
                print(f"RAG processing error: {str(e)}")
                if domain_relevant:
                    return self._process_llm_fallback(query)
                raise

        except Exception as e:
            error_msg = f"Error in process_query: {str(e)}"
            print(error_msg)
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "source": "Error Handler",
                "confidence": 0.0,
                "text": query if query else "Audio processing failed"
            }

    def _process_audio_input(self, audio_file) -> Dict:
        """Process audio input and handle errors"""
        try:
            audio_result = self.audio_agent.process_audio(audio_file)
            
            if not audio_result["success"]:
                return {
                    "success": False,
                    "answer": f"Audio processing error: {audio_result.get('error', 'Unknown error')}",
                    "source": "Audio Processing",
                    "confidence": 0.0
                }
            
            if audio_result["confidence"] < self.audio_confidence_threshold:
                return {
                    "success": False,
                    "answer": "The audio wasn't clear enough. Could you please repeat or type your question?",
                    "source": "Audio Processing",
                    "confidence": audio_result["confidence"]
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
        
        Here's response from LLM: {llm_response['answer']}"""
        
        return {
            "answer": combined_answer,
            "sources": rag_response.get("sources", []) + [llm_response["source"]],
            "confidence": max(rag_response["confidence"], llm_response["confidence"])
        }
    def initialize_knowledge_base(self, documents: Optional[List[str]] = None) -> None:
        """Initialize the RAG knowledge base"""
        try:
            print("Starting knowledge base initialization...")
            if documents:
                print("Warning: Direct document initialization is deprecated. Please use the FAQ converter utility to update the JSON knowledge base.")
                
            # Initialize vector store from JSON
            self.rag_agent.initialize_vector_store()
            print("Knowledge base initialized successfully")
        except Exception as e:
            print(f"Error initializing knowledge base: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def update_knowledge_base(self) -> None:
        """Update the knowledge base from JSON"""
        try:
            print("Updating knowledge base from JSON...")
            # Re-initialize vector store with latest JSON data
            self.rag_agent.initialize_vector_store()
            print("Knowledge base updated successfully")
        except Exception as e:
            print(f"Error updating knowledge base: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def force_update_knowledge_base(self, new_documents: List[str]) -> None:
        """
        Legacy method for direct document updates. 
        Not recommended - use FAQ converter utility instead.
        """
        print("Warning: Direct document updates are deprecated. Please use the FAQ converter utility to update the JSON knowledge base.")
        try:
            # Convert documents to JSON format
            from utils.faq_converter import FAQConverter
            converter = FAQConverter()
            for doc in new_documents:
                converter.process_text_content(doc)
            
            # Reinitialize vector store
            self.update_knowledge_base()
        except Exception as e:
            print(f"Error in force update: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise