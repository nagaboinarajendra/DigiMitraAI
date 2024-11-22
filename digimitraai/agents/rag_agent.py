from typing import Dict, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGAgent:
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo", 
                 vector_store_path: str = "data/vector_store",
                 json_path: str = "data/faqs/consolidated_faqs.json"):
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.json_path = json_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Enhanced text splitter for JSON content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.qa_template = """You are an expert Aadhaar customer service assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the context, just say "I don't have enough information to answer that question accurately using the Knowledgebase, hence will utilize LLM to answer your query"
        Try to be as helpful as possible while staying true to the context provided.

        Context: {context}

        Question: {question}

        Answer: """

        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )
        
        self.vector_store = None
        self.qa_chain = None
        self.faq_data = None
        
        self._load_vector_store()

    def _load_json_faqs(self) -> List[Dict]:
        """Load FAQs from JSON file"""
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('faqs', [])
            return []
        except Exception as e:
            print(f"Error loading JSON FAQs: {str(e)}")
            return []

    def _prepare_documents(self, faqs: List[Dict]) -> List[str]:
        """Prepare FAQ documents for vectorization"""
        documents = []
        for faq in faqs:
            # Create a combined text with metadata
            doc_text = f"""Question: {faq['question']}\nAnswer: {faq['answer']}"""
            if 'metadata' in faq:
                doc_text += f"\nMetadata: {json.dumps(faq['metadata'])}"
            documents.append(doc_text)
        return documents

    def _load_vector_store(self):
        """Load or create vector store from JSON FAQs"""
        try:
            # First try to load existing vector store
            vector_store_path = Path(self.vector_store_path)
            if vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._initialize_qa_chain()
                print("Successfully loaded existing vector store")
            else:
                # Initialize from JSON if vector store doesn't exist
                self.initialize_vector_store()
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.vector_store = None

    def _initialize_qa_chain(self):
        """Initialize the QA chain with the vector store"""
        try:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": self.qa_prompt}
            )
            print("Successfully initialized QA chain")
        except Exception as e:
            print(f"Error initializing QA chain: {str(e)}")
            self.qa_chain = None

    def initialize_vector_store(self) -> None:
        """Initialize the FAISS vector store from JSON FAQs"""
        try:
            print("Starting vector store initialization from JSON...")
            
            # Load FAQs from JSON
            faqs = self._load_json_faqs()
            if not faqs:
                raise ValueError("No FAQs found in JSON file")
            
            # Prepare documents
            documents = self._prepare_documents(faqs)
            
            # Split documents into chunks
            texts = self.text_splitter.create_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            # Save vector store
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            
            # Initialize QA chain
            self._initialize_qa_chain()
            
            print("Vector store initialized successfully from JSON")
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            raise

    def _find_exact_faq_match(self, query: str) -> Dict:
        """Find exact or close match in FAQ data"""
        faqs = self._load_json_faqs()
        query_lower = query.lower().strip('?., ')
        
        for faq in faqs:
            question = faq['question'].lower().strip('?., ')
            if query_lower == question:
                return {'match': faq, 'confidence': 1.0}
            elif query_lower in question or question in query_lower:
                return {'match': faq, 'confidence': 0.9}
        
        return {'match': None, 'confidence': 0.0}


    def _is_domain_relevant(self, query: str) -> bool:
        """Check if query is relevant to Aadhaar domain"""
        domain_keywords = [
            'aadhaar', 'aadhar', 'uid', 'uidai', 'biometric', 'enrollment', 'enrolment',
            'demographic', 'authentication', 'ekyc', 'kyc', 'resident', 'virtual id',
            'update', 'correction', 'verification', 'identity', 'card', 'number',
            'unique identification', 'address', 'mobile', 'email', 'fingerprint', 'iris',
            'face', 'photo', 'otp', 'masked', 'mandatory', 'optional', 'register'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in domain_keywords)

    def _calculate_confidence(self, query: str, source_documents) -> Dict:
        """Calculate confidence score with stricter criteria"""
        try:
            # Initialize response
            response = {
                "confidence": 0.0,
                "domain_relevant": False,
                "has_sources": bool(source_documents),
                "exact_match": False,
                "semantic_match": 0.0,
                "debug_info": {}  # Add debug information
            }
            
            # Check domain relevance
            response["domain_relevant"] = self._is_domain_relevant(query)
            
            # If not domain relevant, return low confidence
            if not response["domain_relevant"]:
                return response
            
            # Check for exact FAQ match
            exact_match = self._find_exact_faq_match(query)
            if exact_match['confidence'] > 0.0:
                response.update({
                    "confidence": exact_match['confidence'],
                    "exact_match": True,
                    "semantic_match": 1.0
                })
                return response
            
            # If no sources but domain relevant, return low confidence
            if not source_documents:
                response.update({
                    "confidence": 0.2,
                    "semantic_match": 0.0,
                    "debug_info": {"reason": "no_sources"}
                })
                return response
            
            try:
                # Calculate semantic similarity for each document
                query_embedding = self.embeddings.embed_query(query)
                similarities = []
                matched_texts = []  # Store matched texts for debugging
                
                for doc in source_documents:
                    # Extract the question part if it exists
                    doc_text = doc.page_content
                    if "Q:" in doc_text and "A:" in doc_text:
                        question_part = doc_text.split("A:")[0].replace("Q:", "").strip()
                    else:
                        question_part = doc_text

                    doc_embedding = self.embeddings.embed_query(question_part)
                    similarity = float(cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(doc_embedding).reshape(1, -1)
                    )[0][0])
                    
                    similarities.append(similarity)
                    matched_texts.append({
                        "text": question_part[:100] + "...",  # First 100 chars
                        "similarity": similarity
                    })

                # Get maximum similarity
                max_similarity = max(similarities) if similarities else 0.0
                response["semantic_match"] = max_similarity
                
                # Store debug information
                response["debug_info"].update({
                    "max_similarity": max_similarity,
                    "all_similarities": similarities,
                    "matched_texts": matched_texts,
                    "num_sources": len(source_documents)
                })

                # Strict confidence calculation
                if max_similarity >= 0.95:  # Near-exact match
                    confidence = 0.95
                elif max_similarity >= 0.85:  # Very high similarity
                    confidence = 0.85
                elif max_similarity >= 0.75:  # High similarity
                    confidence = 0.6
                elif max_similarity >= 0.6:  # Moderate similarity
                    confidence = 0.4
                else:  # Low similarity
                    confidence = 0.2  # Force low confidence for poor matches
                
                # Adjust confidence based on the spread of similarities
                relevant_sources = len([s for s in similarities if s > 0.6])
                if relevant_sources == 0:
                    confidence *= 0.5  # Significantly reduce confidence if no relevant sources
                
                response["confidence"] = confidence
                response["debug_info"]["final_confidence"] = confidence
                response["debug_info"]["relevant_sources"] = relevant_sources
                
                # Print detailed debug information
                print("\nConfidence Calculation Debug:")
                print(f"Max Similarity: {max_similarity:.4f}")
                print(f"Relevant Sources: {relevant_sources}")
                print(f"Final Confidence: {confidence:.4f}")
                print("\nTop Matches:")
                for match in sorted(matched_texts, key=lambda x: x['similarity'], reverse=True)[:2]:
                    print(f"- {match['text']} (similarity: {match['similarity']:.4f})")
                
                return response

            except Exception as e:
                print(f"Error calculating similarity: {str(e)}")
                response.update({
                    "confidence": 0.2 if response["domain_relevant"] else 0.0,
                    "debug_info": {"error": str(e)}
                })
                return response
                
        except Exception as e:
            print(f"Error in confidence calculation: {str(e)}")
            return {
                "confidence": 0.0,
                "domain_relevant": False,
                "has_sources": False,
                "exact_match": False,
                "semantic_match": 0.0,
                "debug_info": {"error": str(e)}
            }

    def process_query(self, query: str) -> Dict:
        """Process a query and return response with sources"""
        try:
            if not self.qa_chain:
                self._load_vector_store()
                if not self.qa_chain:
                    raise ValueError("QA chain not initialized. Vector store may be empty.")
            
            # Get response from QA chain
            result = self.qa_chain({
                "question": query,
                "chat_history": []
            })
            
            # Get source documents
            source_docs = result.get("source_documents", [])
            
            # Calculate confidence with detailed context
            confidence_info = self._calculate_confidence(query, source_docs)
            
            print(f"\nQuery: {query}")
            print(f"Confidence Info: {confidence_info}")
            
            return {
                "answer": result["answer"],
                "sources": [doc.page_content for doc in source_docs],
                "confidence": confidence_info["confidence"],
                "domain_relevant": confidence_info["domain_relevant"],
                "has_sources": confidence_info["has_sources"],
                "exact_match": confidence_info["exact_match"],
                "semantic_match": confidence_info["semantic_match"],
                "debug_info": confidence_info.get("debug_info", {})
            }
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()