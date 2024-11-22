import streamlit as st
from pathlib import Path
import json
from faq_converter import FAQConverter

def display_existing_faqs(converter: FAQConverter):
    """Display existing FAQs in the consolidated file"""
    faqs = converter.load_existing_faqs()
    
    if not faqs:
        st.info("No FAQs found in the consolidated file.")
        return
    
    st.write(f"Total FAQs: {len(faqs)}")
    
    with st.expander("View Existing FAQs"):
        for faq in faqs:
            st.markdown(f"**Q: {faq['question']}**")
            st.write(f"A: {faq['answer']}")
            st.write(f"Last Updated: {faq['metadata']['last_updated']}")
            st.divider()

def main():
    st.title("FAQ Document Converter")
    st.write("""
    Upload PDF or TXT files containing FAQs in the format:
    ```
    Q: Question text
    A: Answer text
    ```
    The tool will convert and append them to the consolidated JSON knowledge base.
    """)
    
    # Initialize converter
    converter = FAQConverter()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload FAQ Document", 
        type=['pdf', 'txt'],
        help="Upload a PDF or TXT file with questions marked with 'Q:' and answers with 'A:'"
    )
    
    if uploaded_file:
        st.write("File uploaded:", uploaded_file.name)
        
        if st.button("Process File"):
            with st.spinner("Processing file..."):
                try:
                    num_faqs = converter.process_file(uploaded_file)
                    st.success(f"Successfully processed {num_faqs} FAQ entries!")
                    
                    # Show preview of processed FAQs
                    display_existing_faqs(converter)
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    # Display existing FAQs
    st.divider()
    st.subheader("Current Knowledge Base")
    display_existing_faqs(converter)
    
    # Export option
    if st.button("Download Consolidated JSON"):
        try:
            json_path = Path(converter.json_output_path)
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_str = f.read()
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="consolidated_faqs.json",
                    mime="application/json"
                )
            else:
                st.warning("No consolidated JSON file exists yet.")
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")

if __name__ == "__main__":
    main()