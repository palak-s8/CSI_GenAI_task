# 📚 RAG Chat System with LangChain

A powerful Retrieval-Augmented Generation (RAG) system built with LangChain that allows users to upload PDF documents and chat with them using AI-powered contextual responses.

## ✨ Features

- **📄 PDF Document Processing**: Upload and process multiple PDF files
- **🔍 Semantic Search**: Advanced vector-based document retrieval using HuggingFace embeddings
- **💬 Conversational AI**: Chat interface with SARVAM-m for contextual responses
- **🧠 Memory Management**: Maintains conversation context and history
- **📚 Source Attribution**: View the exact sources used for each answer
- **📱 Responsive Design**: Clean, professional UI that works on all devices
- **⚡ Fast Processing**: Efficient document chunking and vector storage
- **🔄 Persistent Storage**: ChromaDB vector database for document persistence

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- SARVAM AI API subscription key

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `env_example.txt` to `.env`
   - Add your SARVAM AI API key:
     ```
     SARVAM_API_KEY=your_actual_api_key_here
     ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to the displayed URL (usually `http://localhost:8501`)

## 📖 How to Use

### 1. Upload Documents
- Use the sidebar to upload one or more PDF files
- Click "Process Documents" to create embeddings and vector store

### 2. Start Chatting
- Once documents are processed, you can ask questions
- The system will search through your documents and provide contextual answers
- View source documents for each response

### 3. Manage Conversations
- Chat history is maintained throughout the session
- Clear chat history when needed
- Upload new documents at any time

## 🏗️ Architecture

```
User Upload → PDF Processing → Text Chunking → Embeddings → Vector Store (ChromaDB)
                                                                    ↓
User Query → Semantic Search → Context Retrieval → LLM (SARVAM-m) → Response
```

### Components

- **Document Loader**: PyPDF2 for PDF processing
- **Text Splitter**: RecursiveCharacterTextSplitter for optimal chunking
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB for efficient similarity search
- **LLM**: SARVAM-m for response generation
- **Memory**: ConversationBufferMemory for context persistence

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SARVAM_API_KEY` | Your SARVAM AI API subscription key | Required |
| `SARVAM_MODEL_NAME` | SARVAM model to use | `sarvam-m` |
| `SARVAM_TEMPERATURE` | Response creativity (0-1) | `0.7` |
| `SARVAM_MAX_TOKENS` | Maximum response length | `1000` |

### Customization

You can modify the following parameters in `app.py`:

- **Chunk size**: Adjust `chunk_size` in `create_vector_store()` function
- **Search results**: Modify `k` value in retriever configuration
- **Model parameters**: Change temperature, max_tokens, etc.

## 🔧 Technical Details

### Document Processing
- Documents are split into 1000-character chunks with 200-character overlap
- Each chunk maintains metadata including page numbers
- Vector embeddings are created using HuggingFace's sentence-transformers model

### Vector Search
- Uses similarity search to find most relevant document chunks
- Retrieves top 3 most similar chunks for context
- ChromaDB provides fast and efficient similarity search

### Memory Management
- Conversation history is stored in session state
- Previous context is maintained for longer conversations
- Source documents are preserved for transparency

## 🐛 Troubleshooting

### Common Issues

1. **"SARVAM API key not found"**
   - Ensure you have a `.env` file with your API key
   - Check that the key is valid and has sufficient credits

2. **PDF processing errors**
   - Verify PDF files are not corrupted
   - Check file size (very large files may take longer)

3. **Memory issues**
   - Large documents may consume significant memory
   - Consider reducing chunk size for very long documents

4. **Slow responses**
   - First-time processing includes embedding generation
   - Subsequent queries will be faster
   - Check your internet connection for API calls

### Performance Tips

- Use smaller chunk sizes for faster processing
- Limit the number of simultaneous document uploads
- Clear chat history periodically to free memory

## 🔒 Security & Privacy

- **API Keys**: Never commit your `.env` file to version control
- **Data Storage**: Documents are processed locally and stored in ChromaDB
- **SARVAM AI**: Only document content is sent to SARVAM AI for responses
- **Session Data**: Chat history is stored in Streamlit session state (cleared on restart)

## 📱 Mobile Experience

The application is fully responsive and optimized for:
- Desktop browsers
- Tablets
- Mobile devices
- Touch interfaces

## 🚀 Future Enhancements

Potential improvements for future versions:
- Support for more document formats (DOCX, TXT, etc.)
- Advanced filtering and search options
- Export conversation history
- Custom embedding models
- Multi-language support
- Advanced analytics and insights

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [SARVAM AI](https://sarvam.ai/)
- UI framework: [Streamlit](https://streamlit.io/)
- Vector database: [ChromaDB](https://www.trychroma.com/)
- Embeddings: [HuggingFace](https://huggingface.co/)

---

**Happy Document Chatting! 📚💬** 