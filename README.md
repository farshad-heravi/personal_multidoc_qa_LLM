# 🤖 Personalized Document Q&A with Local LLM

A powerful multi-document, multi-URL question-answering application that uses local Large Language Models (LLMs) via Ollama to analyze and answer questions based on the provided resources. Built with Streamlit for an intuitive web interface.

![project_local_llm_qa_lq](https://github.com/user-attachments/assets/e6cb0f83-2154-4618-8f70-e1c788f0780c)

## ✨ Key Features

- **🤖 Local LLM Integration** - Uses Ollama for private, cost-free AI responses
- **📄 Multi-PDF Support** - Upload and analyze multiple PDF documents simultaneously
- **🌐 URL Content Analysis** - Extract and analyze content from web URLs
- **🧭 RAG-based pipeline** – combines retrieval and generation for grounded answers.
- **🔍 Intelligent Document Search** - TF-IDF vectorization for relevant context retrieval
- **📍 Source Citations** - Provides document names and page references for answers
- **📚 Dynamic Document Management** - Add/remove documents anytime during your session
- **⚡ Fast Performance** - Runs entirely on your local hardware
- **🔒 Privacy-First** - No data sent to external APIs

## 🛠️ Technologies Used

### Core Framework
- **Streamlit** - Web application framework
- **Python 3.10+** - Programming language

### PDF Processing
- **PyPDF2** - Primary PDF text extraction
- **pdfplumber** - Enhanced PDF parsing (optional)

### Text Processing & Search
- **scikit-learn** - TF-IDF vectorization and similarity search
- **NumPy** - Numerical computations

### LLM Integration
- **Ollama** - Local LLM inference engine

### Web Content Processing
- **requests** - HTTP requests for URL content
- **BeautifulSoup4** - HTML parsing and text extraction

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Download a language model**:
   ```bash
   # Recommended models for Q&A
   ollama pull llama3.2:3b    # Best balance of speed and accuracy
   ollama pull phi3:mini      # Fast and accurate
   ollama pull gemma:2b       # Lightweight option
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd personal_multidoc_qa_llm
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv personal_multidoc_qa_llm_env
   source personal_multidoc_qa_llm_env/bin/activate  # On Windows: personal_multidoc_qa_llm_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📋 Usage Guide

### 1. Model Selection
- Choose your preferred LLM model from the sidebar
- The app automatically detects available Ollama models

### 2. Document Upload
- **PDF Files**: Upload one or multiple PDF documents
- **URLs**: Add web page URLs for content analysis
- Documents are processed and chunked automatically

### 3. Ask Questions
- Type your questions in natural language
- The system finds relevant document sections
- Get AI-generated answers with source citations

### 4. Document Management
- View all loaded documents in the sidebar
- Add new documents anytime without losing existing ones
- Remove individual documents as needed
- Clear all documents to start fresh

## 🔧 Architecture

### Document Processing Pipeline
1. **Text Extraction**: Multiple PDF libraries ensure reliable text extraction
2. **Content Chunking**: Intelligent text segmentation with page tracking
3. **Vectorization**: TF-IDF feature extraction for semantic search
4. **Context Retrieval**: Cosine similarity matching for relevant chunks

### LLM Integration
- **Multi-method Approach**: CLI, Python API, and Chat API fallbacks
- **Error Handling**: Comprehensive error recovery and user guidance
- **Performance Monitoring**: Response time tracking and optimization

## 📊 Supported File Types

- **PDF Documents**: `.pdf` files with text content
- **Web Content**: Any publicly accessible HTTP/HTTPS URLs
- **Text Formats**: Extracted text from various PDF structures

## ⚙️ Configuration

### Model Recommendations
- **llama3.2:3b** - Best overall performance for Q&A tasks
- **phi3:mini** - Fastest response times with good accuracy
- **gemma:2b** - Lightweight option for resource-constrained systems

### Performance Tuning
- **Chunk Size**: Default 800 characters (adjustable in code)
- **Search Results**: Top 3 most relevant contexts
- **TF-IDF Features**: 5000 maximum features with bigrams

## 🐛 Troubleshooting

### Common Issues

1. **No Ollama models found**:
   ```bash
   ollama serve
   ollama pull llama3.2:3b
   ```

2. **PDF extraction fails**:
   - Install optional dependencies: `pip install pdfplumber pymupdf`
   - Try different PDF files or convert to text first

3. **URL content not loading**:
   - Check internet connection
   - Verify URL accessibility
   - Some sites may block automated access

### Debug Information
- The app provides detailed error messages and troubleshooting steps
- Check the sidebar for model status and document statistics
- Use the browser's developer console for additional debugging

## 📁 Project Structure

```
personal_multidoc_qa_llm/
├── main.py              # Main application file
└── requirements.txt     # Python dependencies
```

## 🔒 Privacy & Security

- **Local Processing**: All document analysis happens on your machine
- **No External APIs**: No data sent to third-party services
- **Secure Storage**: Documents are only kept in memory during session
- **Private LLM**: Your questions and documents never leave your system

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Contact

**Developer**: Farshad Nozad Heravi  
**Email**: f.n.heravi@gmail.com

## 📜 License

This project is open source.

---

*Built with ❤️ for private, intelligent document analysis*
