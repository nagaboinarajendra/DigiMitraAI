import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self, save_path: str = "data/vector_store"):
        self.save_path = save_path
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def process_faqs(self, faq_files: List[str]) -> List[Dict]:
        """Process FAQ files into structured format"""
        processed_faqs = []
        
        for faq_file in faq_files:
            with open(faq_file, 'r') as f:
                content = f.read()
                qa_pairs = content.split('\n\n')
                
                for pair in qa_pairs:
                    if 'Q:' in pair and 'A:' in pair:
                        q, a = pair.split('A:')
                        processed_faqs.append({
                            'question': q.replace('Q:', '').strip(),
                            'answer': a.strip()
                        })
        
        return processed_faqs
    
    def create_vector_store(self, documents: List[str]) -> FAISS:
        """Create and save vector store from documents"""
        texts = self.text_splitter.create_documents(documents)
        vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Save vector store
        os.makedirs(self.save_path, exist_ok=True)
        vector_store.save_local(self.save_path)
        
        return vector_store
    
    def load_vector_store(self) -> FAISS:
        """Load existing vector store"""
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"Vector store not found at {self.save_path}")
        
        return FAISS.load_local(self.save_path, self.embeddings)
    
    def update_vector_store(self, new_documents: List[str]) -> FAISS:
        """Update existing vector store with new documents"""
        try:
            vector_store = self.load_vector_store()
        except FileNotFoundError:
            return self.create_vector_store(new_documents)
        
        texts = self.text_splitter.create_documents(new_documents)
        vector_store.add_documents(texts)
        vector_store.save_local(self.save_path)
        
        return vector_store