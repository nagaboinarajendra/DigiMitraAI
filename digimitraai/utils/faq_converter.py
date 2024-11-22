import streamlit as st
import PyPDF2
import json
import re
from pathlib import Path
from typing import List, Dict, Union
import hashlib
import datetime

class FAQConverter:
    def __init__(self, json_output_path: str = "data/faqs/consolidated_faqs.json"):
        self.json_output_path = Path(json_output_path)
        self.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def generate_faq_id(self, question: str) -> str:
        """Generate a unique ID for a FAQ entry"""
        return hashlib.md5(question.lower().encode()).hexdigest()[:8]
    
    def extract_qa_pairs(self, text: str) -> List[Dict]:
        """Extract Q&A pairs from text content"""
        # Split text into Q&A sections
        qa_sections = re.split(r'(?=Q:|Question:)', text)
        qa_pairs = []
        
        for section in qa_sections:
            if not section.strip():
                continue
                
            # Try to find question and answer
            question_match = re.search(r'(?:Q:|Question:)\s*(.*?)(?=A:|Answer:|$)', section, re.DOTALL)
            answer_match = re.search(r'(?:A:|Answer:)\s*(.*?)(?=Q:|Question:|$)', section, re.DOTALL)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                if question and answer:  # Only add if both question and answer exist
                    faq_entry = {
                        "id": self.generate_faq_id(question),
                        "question": question,
                        "answer": answer,
                        "metadata": {
                            "added_date": datetime.datetime.now().isoformat(),
                            "last_updated": datetime.datetime.now().isoformat(),
                            "source_type": "converted"
                        }
                    }
                    qa_pairs.append(faq_entry)
        
        return qa_pairs

    def read_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def read_txt(self, txt_file) -> str:
        """Read text from TXT file"""
        return txt_file.getvalue().decode('utf-8')
    
    def load_existing_faqs(self) -> List[Dict]:
        """Load existing FAQs from JSON file"""
        if self.json_output_path.exists():
            with open(self.json_output_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    return data.get('faqs', [])
                except json.JSONDecodeError:
                    st.warning("Error reading existing FAQ file. Starting fresh.")
                    return []
        return []
    
    def merge_faqs(self, existing_faqs: List[Dict], new_faqs: List[Dict]) -> List[Dict]:
        """Merge new FAQs with existing ones, updating duplicates"""
        faq_dict = {faq['id']: faq for faq in existing_faqs}
        
        for new_faq in new_faqs:
            if new_faq['id'] in faq_dict:
                # Update existing FAQ
                faq_dict[new_faq['id']]['answer'] = new_faq['answer']
                faq_dict[new_faq['id']]['metadata']['last_updated'] = datetime.datetime.now().isoformat()
                faq_dict[new_faq['id']]['metadata']['updated_count'] = \
                    faq_dict[new_faq['id']]['metadata'].get('updated_count', 0) + 1
            else:
                # Add new FAQ
                faq_dict[new_faq['id']] = new_faq
        
        return list(faq_dict.values())
    
    def save_faqs(self, faqs: List[Dict]):
        """Save FAQs to JSON file"""
        output_data = {
            "metadata": {
                "last_updated": datetime.datetime.now().isoformat(),
                "total_faqs": len(faqs)
            },
            "faqs": faqs
        }
        
        with open(self.json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def process_file(self, uploaded_file) -> int:
        """Process uploaded file and return number of FAQs added/updated"""
        try:
            # Extract text based on file type
            if uploaded_file.name.lower().endswith('.pdf'):
                text = self.read_pdf(uploaded_file)
            else:
                text = self.read_txt(uploaded_file)
            
            # Extract Q&A pairs
            new_faqs = self.extract_qa_pairs(text)
            
            if not new_faqs:
                st.error("No valid Q&A pairs found in the document!")
                return 0
            
            # Load and merge with existing FAQs
            existing_faqs = self.load_existing_faqs()
            merged_faqs = self.merge_faqs(existing_faqs, new_faqs)
            
            # Save merged FAQs
            self.save_faqs(merged_faqs)
            
            return len(new_faqs)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            raise

    def process_text_content(self, text_content: str) -> int:
        """Process raw text content and add to JSON knowledge base"""
        try:
            # Extract Q&A pairs
            new_faqs = self.extract_qa_pairs(text_content)
            
            if not new_faqs:
                print("No valid Q&A pairs found in the content!")
                return 0
            
            # Load and merge with existing FAQs
            existing_faqs = self.load_existing_faqs()
            merged_faqs = self.merge_faqs(existing_faqs, new_faqs)
            
            # Save merged FAQs
            self.save_faqs(merged_faqs)
            
            return len(new_faqs)
            
        except Exception as e:
            print(f"Error processing text content: {str(e)}")
            raise