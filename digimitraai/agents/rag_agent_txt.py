# agents/rag_agent.py

from typing import Dict, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo", vector_store_path: str = "data/vector_store"):
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.qa_template = """You are an expert Aadhaar customer service assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the context, just say "I don't have enough information to answer that question accurately."
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
        
        self._load_vector_store()

    def _load_vector_store(self):
        """Try to load existing vector store"""
        try:
            vector_store_path = Path(self.vector_store_path)
            if vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Only use this if you trust the source
                )
                self._initialize_qa_chain()
                print("Successfully loaded existing vector store")
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

    def initialize_vector_store(self, documents: List[str]) -> None:
        """Initialize the FAISS vector store with FAQ documents"""
        try:
            print("Starting vector store initialization...")
            texts = self.text_splitter.create_documents(documents)
            
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            
            self._initialize_qa_chain()
            print("Vector store initialized successfully")
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            raise

    def _calculate_similarity(self, query: str, doc_content: str) -> float:
        """Calculate semantic similarity between query and document content"""
        try:
            # Get embeddings for query and document
            query_embedding = self.embeddings.embed_query(query)
            doc_embedding = self.embeddings.embed_query(doc_content)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                np.array(query_embedding).reshape(1, -1),
                np.array(doc_embedding).reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def _calculate_confidence(self, query: str, source_documents) -> float:
        """
        Calculate confidence score based on:
        1. Number of relevant sources
        2. Semantic similarity of the best match
        3. Presence of exact question matches
        """
        if not source_documents:
            return 0.0
        
        try:
            # Initialize base confidence from number of sources
            num_sources = len(source_documents)
            base_confidence = min(0.3 + (0.2 * num_sources), 0.9)
            
            # Calculate similarities for each document
            similarities = []
            for doc in source_documents:
                # Check for exact question match
                doc_content = doc.page_content.lower()
                query_lower = query.lower()
                
                # Look for question patterns
                if "q:" in doc_content or "question:" in doc_content:
                    # Extract question part
                    if "q:" in doc_content:
                        doc_question = doc_content.split("q:")[1].split("a:")[0].strip()
                    else:
                        doc_question = doc_content.split("question:")[1].split("answer:")[0].strip()
                    
                    # Check for exact or near match
                    if query_lower == doc_question or query_lower in doc_question or doc_question in query_lower:
                        return 1.0  # Highest confidence for exact matches
                
                # Calculate semantic similarity
                similarity = self._calculate_similarity(query, doc_content)
                similarities.append(similarity)
            
            # Get maximum similarity
            max_similarity = max(similarities) if similarities else 0.0
            
            # Adjust confidence based on similarity
            final_confidence = max(base_confidence, max_similarity)
            
            # Boost confidence if similarity is very high
            if max_similarity > 0.85:
                final_confidence = max(final_confidence, 0.95)
            
            return final_confidence
            
        except Exception as e:
            print(f"Error in confidence calculation: {str(e)}")
            return base_confidence if 'base_confidence' in locals() else 0.0

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
            
            # Calculate confidence using improved method
            confidence = self._calculate_confidence(query, source_docs)
            
            print(f"Query: {query}")
            print(f"Confidence Score: {confidence}")
            print(f"Number of source documents: {len(source_docs)}")
            
            return {
                "answer": result["answer"],
                "sources": [doc.page_content for doc in source_docs],
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()