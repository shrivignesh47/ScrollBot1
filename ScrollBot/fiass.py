
# pdf_processor.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_vector_store():
    # ✅ Add all your PDFs here
    pdf_paths = [
        r"D:\rinvoq_pi.pdf"
    ]

    raw_docs = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            continue

        print(f"📄 Reading {pdf_path}...")
        try:
            pdf_reader = PdfReader(pdf_path)
            if len(pdf_reader.pages) == 0:
                print(f"⚠️  No pages found in {pdf_path}")
                continue

            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Clean up strange characters or line breaks
                    text = re.sub(r'\s+', ' ', text).strip()
                    raw_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "page": i + 1,
                                "source": os.path.basename(pdf_path)
                            }
                        )
                    )
                else:
                    print(f"⚠️  Empty text on page {i+1} of {pdf_path}")
        except Exception as e:
            print(f"🚨 Error reading {pdf_path}: {str(e)}")

    if not raw_docs:
        print("❌ No document content extracted. Check if PDFs are scanned images.")
        return

    print(f"✅ Extracted {len(raw_docs)} pages from {len(pdf_paths)} PDFs")

    # Split into meaningful chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # Slightly smaller for better retrieval
        chunk_overlap=100,        # Overlap preserves context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✂️  Split into {len(chunks)} chunks")

    # Use consistent embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create and save FAISS index
    try:
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("🎉 FAISS index saved at './faiss_index/'")
        
        # Print sample
        if chunks:
            print("📌 Sample chunk:")
            print("   Source:", chunks[0].metadata)
            print("   Text:", chunks[0].page_content[:150] + "...")
    except Exception as e:
        print("🚨 Error saving FAISS index:", str(e))


if __name__ == "__main__":
    import re  # Added missing import
    create_vector_store()