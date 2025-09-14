"""
    This is a personalized document q&a with local llm developed by Farshad Nozad Heravi (f.n.hervai@gmail.com)
    This app uses ollama as the local llm and pdfplumber as the pdf reader.
    This app is used to answer questions from provided documents and URLs.
"""

import streamlit as st
import PyPDF2
import os
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import sys
import ollama
import time
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import json

# Try to import optional PDF libraries
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    print("pdfplumber not found! please install it: pip install pdfplumber")
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    print("PyMuPDF not found! please install it: pip install pymupdf")
    HAS_PYMUPDF = False

# Configure page
st.set_page_config(
    page_title="Personalized Document Q&A with Local LLM",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'documents' not in st.session_state:
    st.session_state.documents = []  # List of document metadata
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'chunk_vectors' not in st.session_state:
    st.session_state.chunk_vectors = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'urls' not in st.session_state:
    st.session_state.urls = []  # List of URLs to process

@st.cache_data
def get_available_models():
    """Get list of available Ollama models."""
    try:
        # Try to get models using the ollama library
        models_response = ollama.list()
        model_names = []
        
        # Handle different possible response formats
        if isinstance(models_response, dict):
            if 'models' in models_response:
                # Standard format: {'models': [{'name': 'model1'}, {'name': 'model2'}]}
                for model in models_response['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
                    elif isinstance(model, str):
                        model_names.append(model)
            else:
                # Alternative format: direct list or other structure
                for key, value in models_response.items():
                    if isinstance(value, list):
                        model_names.extend(value)
        elif isinstance(models_response, list):
            # Direct list format
            model_names = models_response
        
        # If ollama library fails, try CLI fallback
        if not model_names:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                model_names = [line.split()[0] for line in lines if line.strip()]
        
        return model_names
        
    except subprocess.TimeoutExpired:
        st.error("Ollama command timed out. Check if Ollama is running properly.")
        return []
    except FileNotFoundError:
        st.error("Ollama not found. Please install Ollama first.")
        return []
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.error("Make sure Ollama is running: `ollama serve`")
        
        # Try one more fallback with subprocess
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                model_names = [line.split()[0] for line in lines if line.strip()]
                return model_names
        except:
            pass
            
        return []

def extract_url_text(url: str) -> str:
    """Extract text content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Error processing URL {url}: {str(e)}")
        return ""

def extract_pdf_text(pdf_file) -> str:
    """Extract text from uploaded PDF file with multiple fallback methods."""
    methods_available = ["PyPDF2"]
    if HAS_PDFPLUMBER:
        methods_available.append("pdfplumber")
    if HAS_PYMUPDF:
        methods_available.append("PyMuPDF")
    
    # Method 1: Try pdfplumber first (most reliable)
    if HAS_PDFPLUMBER:
        try:
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"[Page {page_num + 1}] {page_text}\n\n"
                
                if text.strip():
                    return text
        except Exception as e:
            st.warning(f"pdfplumber failed: {str(e)}. Trying next method...")
    
    # Method 2: Try PyMuPDF
    if HAS_PYMUPDF:
        try:
            pdf_file.seek(0)
            pdf_data = pdf_file.read()
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text += f"[Page {page_num + 1}] {page_text}\n\n"
            doc.close()
            
            if text.strip():
                return text
        except Exception as e:
            st.warning(f"PyMuPDF failed: {str(e)}. Trying next method...")
    
    # Method 3: PyPDF2 with recursion fix
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(3000)
    
    try:
        pdf_file.seek(0)
        reader = PyPDF2.PdfReader(pdf_file, strict=False)
        text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"[Page {page_num + 1}] {page_text}\n\n"
            except Exception:
                continue
        
        if text.strip():
            return text
            
    except Exception as e:
        st.error(f"PDF extraction failed: {str(e)}")
        return ""
    finally:
        sys.setrecursionlimit(original_limit)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, doc_type: str = "pdf", doc_name: str = "document") -> List[Tuple[str, int, str, str]]:
    """Split text into chunks with metadata including document type and name."""
    text = clean_text(text)
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_page = 1
    
    for sentence in sentences:
        if '[Page' in sentence:
            try:
                page_match = sentence.split('[Page ')[1].split(']')[0]
                current_page = int(page_match)
            except:
                pass
        
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                clean_chunk = re.sub(r'\[Page \d+\]', '', current_chunk).strip()
                if clean_chunk:
                    chunks.append((clean_chunk, current_page, doc_type, doc_name))
            current_chunk = sentence + ". "
    
    if current_chunk.strip():
        clean_chunk = re.sub(r'\[Page \d+\]', '', current_chunk).strip()
        if clean_chunk:
            chunks.append((clean_chunk, current_page, doc_type, doc_name))
    
    return chunks

def process_pdf_document(pdf_file, doc_name: str = None):
    """Process a single PDF document."""
    try:
        if doc_name is None:
            doc_name = pdf_file.name if hasattr(pdf_file, 'name') else "PDF Document"
        
        text = extract_pdf_text(pdf_file)
        if not text:
            st.error(f"Failed to extract text from {doc_name}")
            return []
        
        chunks = chunk_text(text, doc_type="pdf", doc_name=doc_name)
        if not chunks:
            st.error(f"No content found in {doc_name}")
            return []
        
        return chunks
    except Exception as e:
        st.error(f"Error processing {doc_name}: {str(e)}")
        return []

def process_url_document(url: str):
    """Process a single URL document."""
    try:
        doc_name = urlparse(url).netloc or url
        text = extract_url_text(url)
        if not text:
            st.error(f"Failed to extract text from {url}")
            return []
        
        chunks = chunk_text(text, doc_type="url", doc_name=doc_name)
        if not chunks:
            st.error(f"No content found in {url}")
            return []
        
        return chunks
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")
        return []

def process_all_documents(pdf_files: List = None, urls: List[str] = None):
    """Process multiple PDF files and URLs."""
    try:
        all_chunks = []
        documents = []
        
        # Process PDF files
        if pdf_files:
            for i, pdf_file in enumerate(pdf_files):
                doc_name = pdf_file.name if hasattr(pdf_file, 'name') else f"PDF_{i+1}"
                with st.spinner(f"Processing PDF: {doc_name}..."):
                    chunks = process_pdf_document(pdf_file, doc_name)
                    if chunks:
                        all_chunks.extend(chunks)
                        documents.append({
                            'name': doc_name,
                            'type': 'pdf',
                            'chunks': len(chunks)
                        })
        
        # Process URLs
        if urls:
            for i, url in enumerate(urls):
                doc_name = urlparse(url).netloc or f"URL_{i+1}"
                with st.spinner(f"Processing URL: {doc_name}..."):
                    chunks = process_url_document(url)
                    if chunks:
                        all_chunks.extend(chunks)
                        documents.append({
                            'name': doc_name,
                            'type': 'url',
                            'url': url,
                            'chunks': len(chunks)
                        })
        
        if not all_chunks:
            st.error("No content found in any documents")
            return False
        
        # Update session state
        st.session_state.chunks = all_chunks
        st.session_state.documents = documents
        
        # Create vectorizer and vectors
        chunk_texts = [chunk[0] for chunk in all_chunks]
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1
        )
        
        chunk_vectors = vectorizer.fit_transform(chunk_texts)
        
        st.session_state.vectorizer = vectorizer
        st.session_state.chunk_vectors = chunk_vectors
        st.session_state.document_processed = True
        st.session_state.processing_complete = True
        
        return True
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def search_documents(query: str, top_k: int = 3) -> List[Tuple[str, int, float, str, str]]:
    """Search for relevant document chunks using TF-IDF."""
    if not st.session_state.vectorizer or st.session_state.chunk_vectors is None:
        return []
    
    query_vector = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, st.session_state.chunk_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            chunk_text, page, doc_type, doc_name = st.session_state.chunks[idx]
            results.append((chunk_text, page, similarities[idx], doc_type, doc_name))
    
    return results

def generate_llm_answer(question: str, contexts: List[Tuple[str, int, float, str, str]], model_name: str) -> str:
    """Generate answer using local LLM via Ollama."""
    if not contexts:
        return "I couldn't find relevant information in the documents to answer your question."
    
    # Prepare context
    context_text = ""
    for i, (context, page, score, doc_type, doc_name) in enumerate(contexts[:3]):
        if doc_type == "url":
            context_text += f"Context {i+1} (From {doc_name}):\n{context}\n\n"
        else:
            context_text += f"Context {i+1} (From {doc_name}, Page {page}):\n{context}\n\n"
    
    # Create prompt
    prompt = f"""Based on the following excerpts from documents, answer the question clearly and concisely.

{context_text}

Question: {question}

Instructions:
- Provide a direct answer based on the context provided
- Mention the document name and page number(s) or URL where you found the information
- If the answer is not in the context, say so
- Keep your response concise and factual

Answer:"""

    try:
        with st.spinner(f"Generating answer using {model_name}..."):
            start_time = time.time()
            
            if isinstance(model_name, dict):
                clean_model = model_name.get('name', str(model_name))
            else:
                clean_model = str(model_name).strip().split(' ')[0]
            
            st.write(f"Debug: Using model '{clean_model}'")  # Debug info
            
            # Method 1: Try CLI approach (most reliable)
            try:
                import subprocess
                process = subprocess.Popen(
                    ['ollama', 'run', clean_model],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(input=prompt, timeout=60)
                
                if process.returncode == 0:
                    answer = stdout.strip()
                    end_time = time.time()
                    response_time = end_time - start_time
                    st.caption(f"⏱️ Generated in {response_time:.1f}s using {clean_model} (CLI)")
                    return answer
                else:
                    raise Exception(f"CLI error: {stderr}")
                    
            except Exception as cli_error:
                st.warning(f"CLI method failed: {cli_error}. Trying Python API...")
                
                # Method 2: Try Python API
                try:
                    response = ollama.generate(
                        model=clean_model,
                        prompt=prompt,
                        options={
                            'temperature': 0.1,
                            'top_p': 0.9,
                            'top_k': 40,
                            'num_predict': 256,
                        }
                    )
                    answer = response.get('response', '')
                    end_time = time.time()
                    response_time = end_time - start_time
                    st.caption(f"⏱️ Generated in {response_time:.1f}s using {clean_model} (API)")
                    return answer
                    
                except Exception as api_error:
                    # Method 3: Try chat API
                    try:
                        response = ollama.chat(
                            model=clean_model,
                            messages=[{'role': 'user', 'content': prompt}],
                            options={'temperature': 0.1, 'num_predict': 256}
                        )
                        answer = response['message']['content']
                        end_time = time.time()
                        response_time = end_time - start_time
                        st.caption(f"⏱️ Generated in {response_time:.1f}s using {clean_model} (Chat)")
                        return answer
                    except Exception as chat_error:
                        raise Exception(f"All methods failed. API: {api_error}, Chat: {chat_error}")
            
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        st.markdown("**Troubleshooting:**")
        st.code(f"""
# Check if Ollama is running
ollama serve

# List available models  
ollama list

# Test the specific model
ollama run {clean_model} "Hello"

# Pull the model if missing
ollama pull {clean_model}
        """)
        return "Sorry, I encountered an error while generating the answer."

def main():
    st.title("🤖 Personalized Document Q&A with Local LLM")
    st.markdown("Upload a PDF or add a URL and ask questions using a local LLM via Ollama!")
    
    # Model selection in sidebar
    with st.sidebar:
        st.header("🧠 LLM Configuration")
        
        # Get available models
        available_models = get_available_models()
        st.session_state.available_models = available_models
        
        if available_models:
            selected_model = st.selectbox(
                "Choose LLM Model:",
                available_models,
                help="Select the local model to use for answering questions"
            )
            st.session_state.selected_model = selected_model
            
            # Show model info
            st.success(f"✅ Using: {selected_model}")
        else:
            st.error("❌ No Ollama models found!")
            st.markdown("""
            **Setup Instructions:**
            1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
            2. Start Ollama: `ollama serve`
            3. Download a model: `ollama pull llama3.2:3b`
            4. Refresh this page
            """)
            st.stop()
        
        st.markdown("---")
        
        # Document Management section
        st.header("📚 Document Management")
        
        # Show current documents
        if st.session_state.documents:
            st.subheader("📊 Current Documents")
            for i, doc in enumerate(st.session_state.documents):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if doc['type'] == 'pdf':
                        st.info(f"📄 {doc['name']}: {doc['chunks']} chunks")
                    else:
                        st.info(f"🌐 {doc['name']}: {doc['chunks']} chunks")
                with col2:
                    if st.button("🗑️", key=f"remove_{i}", help="Remove document"):
                        # Remove document and its chunks
                        doc_to_remove = doc['name']
                        st.session_state.chunks = [chunk for chunk in st.session_state.chunks if chunk[3] != doc_to_remove]
                        st.session_state.documents = [d for d in st.session_state.documents if d['name'] != doc_to_remove]
                        
                        # Rebuild vectorizer if there are still documents
                        if st.session_state.chunks:
                            chunk_texts = [chunk[0] for chunk in st.session_state.chunks]
                            vectorizer = TfidfVectorizer(
                                max_features=5000,
                                stop_words='english',
                                ngram_range=(1, 2),
                                max_df=0.95,
                                min_df=1
                            )
                            chunk_vectors = vectorizer.fit_transform(chunk_texts)
                            st.session_state.vectorizer = vectorizer
                            st.session_state.chunk_vectors = chunk_vectors
                        else:
                            st.session_state.document_processed = False
                            st.session_state.vectorizer = None
                            st.session_state.chunk_vectors = None
                        st.rerun()
            
            st.info(f"Total: {len(st.session_state.chunks)} chunks from {len(st.session_state.documents)} documents")
            st.markdown("---")
        
        # Add new documents section
        st.subheader("➕ Add New Documents")
        
        # PDF upload section
        st.markdown("**📄 Add PDF Files**")
        new_pdf_files = st.file_uploader(
            "Choose PDF files to add",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to add to your collection",
            key="new_pdf_uploader"
        )
        
        # URL input section
        st.markdown("**🌐 Add URLs**")
        new_url_input = st.text_area(
            "Enter URLs to add (one per line)",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Enter URLs to analyze, one per line",
            key="new_url_input"
        )
        
        # Process new URLs
        new_urls = []
        if new_url_input:
            new_urls = [url.strip() for url in new_url_input.split('\n') if url.strip()]
            # Validate URLs
            valid_new_urls = []
            for url in new_urls:
                try:
                    result = urlparse(url)
                    if result.scheme and result.netloc:
                        # Check if URL already exists
                        existing_urls = [doc.get('url', '') for doc in st.session_state.documents if doc['type'] == 'url']
                        if url not in existing_urls:
                            valid_new_urls.append(url)
                        else:
                            st.warning(f"URL already added: {url}")
                    else:
                        st.warning(f"Invalid URL: {url}")
                except:
                    st.warning(f"Invalid URL: {url}")
            new_urls = valid_new_urls
        
        # Add documents button
        if (new_pdf_files or new_urls):
            if st.button("🔄 Add to Collection", type="primary"):
                # Process new documents
                new_chunks = []
                new_documents = []
                
                # Process new PDF files
                if new_pdf_files:
                    for pdf_file in new_pdf_files:
                        doc_name = pdf_file.name
                        # Check if PDF already exists
                        existing_names = [doc['name'] for doc in st.session_state.documents]
                        if doc_name not in existing_names:
                            with st.spinner(f"Processing PDF: {doc_name}..."):
                                chunks = process_pdf_document(pdf_file, doc_name)
                                if chunks:
                                    new_chunks.extend(chunks)
                                    new_documents.append({
                                        'name': doc_name,
                                        'type': 'pdf',
                                        'chunks': len(chunks)
                                    })
                        else:
                            st.warning(f"PDF already added: {doc_name}")
                
                # Process new URLs
                if new_urls:
                    for url in new_urls:
                        doc_name = urlparse(url).netloc or f"URL_{len(st.session_state.documents)+1}"
                        with st.spinner(f"Processing URL: {doc_name}..."):
                            chunks = process_url_document(url)
                            if chunks:
                                new_chunks.extend(chunks)
                                new_documents.append({
                                    'name': doc_name,
                                    'type': 'url',
                                    'url': url,
                                    'chunks': len(chunks)
                                })
                
                if new_chunks:
                    # Add new chunks to existing ones
                    st.session_state.chunks.extend(new_chunks)
                    st.session_state.documents.extend(new_documents)
                    
                    # Rebuild vectorizer with all chunks
                    chunk_texts = [chunk[0] for chunk in st.session_state.chunks]
                    
                    vectorizer = TfidfVectorizer(
                        max_features=5000,
                        stop_words='english',
                        ngram_range=(1, 2),
                        max_df=0.95,
                        min_df=1
                    )
                    
                    chunk_vectors = vectorizer.fit_transform(chunk_texts)
                    
                    st.session_state.vectorizer = vectorizer
                    st.session_state.chunk_vectors = chunk_vectors
                    st.session_state.document_processed = True
                    
                    st.success(f"✅ Added {len(new_chunks)} new chunks from {len(new_documents)} documents!")
                    st.rerun()
                else:
                    st.error("Failed to process any new documents")
        
        # Clear all documents button
        if st.session_state.documents:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", help="Remove all documents and start over"):
                st.session_state.document_processed = False
                st.session_state.chunks = []
                st.session_state.documents = []
                st.session_state.urls = []
                st.session_state.vectorizer = None
                st.session_state.chunk_vectors = None
                st.session_state.processing_complete = False
                st.rerun()
        
        # Model recommendations
        if available_models:
            st.markdown("---")
            st.markdown("### 🚀 Model Recommendations")
            st.markdown("""
            **For Q&A tasks:**
            - `llama3.2:3b` - Best balance
            - `phi3:mini` - Fast & accurate
            - `gemma:2b` - Lightweight
            
            **Download more models:**
            ```bash
            ollama pull llama3.2:3b
            ollama pull phi3:mini  
            ollama pull gemma:2b
            ```
            """)
    
    # Main content area
    if not st.session_state.document_processed:
        st.info("👈 Add documents using the sidebar to get started.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔧 Setup Required:
            1. **Install Ollama** (if not done):
               ```bash
               curl -fsSL https://ollama.com/install.sh | sh
               ```
            2. **Start Ollama service**:
               ```bash
               ollama serve
               ```
            3. **Download a model**:
               ```bash
               ollama pull llama3.2:3b
               ```
            """)
        
        with col2:
            st.markdown("""
            ### ✨ Features:
            - 🤖 **Local LLM** - No API costs
            - 📄 **Multi-PDF Support** - Upload multiple PDFs
            - 🌐 **URL Support** - Analyze web content
            - 🔍 **Smart Search** - TF-IDF similarity
            - 📍 **Citations** - Document and page references
            - ⚡ **Fast** - Runs on your hardware
            - 📚 **Document Management** - Add/remove docs anytime
            """)
            
    else:
        st.header("💬 Ask Questions")
        st.info("💡 You can add more documents anytime using the sidebar!")
        
        # Question input
        question = st.text_input(
            "What would you like to know about your documents?",
            placeholder="e.g., What are the main conclusions? Who are the authors?",
            key="question_input"
        )
        
        # Answer generation
        if question and st.session_state.selected_model:
            # Search for relevant chunks
            search_results = search_documents(question, top_k=3)
            
            if search_results:
                # Generate answer using local LLM
                answer = generate_llm_answer(question, search_results, st.session_state.selected_model)
                
                # Display answer
                st.markdown("### 🤖 Answer")
                st.markdown(answer)
                
                # Show sources
                with st.expander("📖 Source Contexts", expanded=False):
                    for i, (context, page, score, doc_type, doc_name) in enumerate(search_results):
                        if doc_type == "url":
                            st.markdown(f"**Context {i+1} - From {doc_name}** (Similarity: {score:.3f})")
                        else:
                            st.markdown(f"**Context {i+1} - From {doc_name}, Page {page}** (Similarity: {score:.3f})")
                        st.markdown(f"_{context[:400]}..._" if len(context) > 400 else f"_{context}_")
                        if i < len(search_results) - 1:
                            st.markdown("---")
            else:
                st.warning("No relevant information found. Try rephrasing your question or using different keywords.")
        
        elif question and not st.session_state.selected_model:
            st.warning("Please select a model from the sidebar first.")
        
        # Document statistics
        if st.session_state.chunks:
            with st.expander("📊 Document Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Chunks", len(st.session_state.chunks))
                with col2:
                    st.metric("Documents", len(st.session_state.documents))
                with col3:
                    pdf_docs = len([doc for doc in st.session_state.documents if doc['type'] == 'pdf'])
                    url_docs = len([doc for doc in st.session_state.documents if doc['type'] == 'url'])
                    st.metric("PDFs/URLs", f"{pdf_docs}/{url_docs}")
                with col4:
                    avg_length = np.mean([len(chunk[0]) for chunk in st.session_state.chunks])
                    st.metric("Avg Chunk Length", f"{avg_length:.0f} chars")
    
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;    
            color: gray;
            text-align: right;
            padding: 10px 70px;
        }
        </style>
        <div class="footer">
            developed by <b>Farshad Nozad Heravi</b>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()