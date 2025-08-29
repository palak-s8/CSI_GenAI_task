import streamlit as st
import os
from typing import List, Dict, Any
import tempfile
from pathlib import Path
import json

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# SARVAM AI imports
from sarvamai import SarvamAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="RAG Chat System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #000000;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #000000;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def load_environment_variables():
    """Load environment variables from .env file"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if SARVAM API key is set
    api_key = os.getenv('SARVAM_API_KEY')
    if not api_key:
        st.error("⚠️ SARVAM API key not found. Please set SARVAM_API_KEY in your .env file or environment variables.")
        st.stop()
    return api_key

def process_pdf(uploaded_file) -> List[Document]:
    """Process uploaded PDF and return documents"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return documents
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def create_vector_store(documents: List[Document], api_key: str):
    """Create vector store from documents"""
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings using HuggingFace (free alternative)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store using Qdrant
        # Using in-memory collection for simplicity
        collection_name = "documents_collection"
        vector_store = Qdrant.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            force_recreate=True  # Recreate collection each time for demo
        )
        
        return vector_store, len(splits)
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None, 0

def create_conversation_chain(vector_store, api_key: str):
    """Create conversation chain with memory"""
    try:
        # Initialize SARVAM AI client
        client = SarvamAI(
            api_subscription_key=api_key,
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create custom LLM class for SARVAM AI
        class SarvamLLM(Runnable):
            def __init__(self, client):
                self.client = client
                self.output_parser = StrOutputParser()
            
            def invoke(self, input_data, config=None, **kwargs):
                """Main method for LangChain compatibility"""
                try:
                    # Extract the question from the input
                    if isinstance(input_data, dict):
                        question = input_data.get("question", str(input_data))
                    else:
                        question = str(input_data)
                    
                    response = self.client.chat.completions(
                        messages=[{"content": question, "role": "user"}]
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    st.error(f"Error calling SARVAM AI: {str(e)}")
                    return "I apologize, but I encountered an error processing your request."
            
            def __call__(self, prompt, **kwargs):
                """Legacy method for backward compatibility"""
                return self.invoke(prompt, **kwargs)
        
        # Create conversation chain with custom LLM
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=SarvamLLM(client),
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Chat System</h1>', unsafe_allow_html=True)
    
    # Load environment variables
    api_key = load_environment_variables()
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📁 Upload Documents")
        st.markdown("Upload PDF files to start chatting with your documents.")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    all_documents = []
                    for uploaded_file in uploaded_files:
                        documents = process_pdf(uploaded_file)
                        all_documents.extend(documents)
                    
                    if all_documents:
                        # Create vector store
                        vector_store, num_chunks = create_vector_store(all_documents, api_key)
                        
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.documents_loaded = True
                            
                            # Create conversation chain
                            conversation_chain = create_conversation_chain(vector_store, api_key)
                            if conversation_chain:
                                st.session_state.conversation_chain = conversation_chain
                                st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s) into {num_chunks} chunks!")
                            else:
                                st.error("Failed to create conversation chain")
                        else:
                            st.error("Failed to create vector store")
                    else:
                        st.error("No documents could be processed")
        
        # Display current status
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded and ready for chat!")
        else:
            st.info("📁 Please upload PDF documents to begin")
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.markdown("### 💬 Chat with Your Documents")
        
        if not st.session_state.documents_loaded:
            st.info("👆 Please upload PDF documents in the sidebar to start chatting!")
        else:
            # Chat input
            user_question = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What are the main topics discussed in the documents?",
                key="user_input"
            )
            
            if user_question and st.button("Send", type="primary"):
                if st.session_state.conversation_chain:
                    with st.spinner("Thinking..."):
                        try:
                            # Get response from conversation chain
                            response = st.session_state.conversation_chain.invoke({
                                "question": user_question,
                                "chat_history": st.session_state.chat_history
                            })
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "answer": response["answer"],
                                "sources": response.get("source_documents", [])
                            })
                            
                            # Clear input
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error getting response: {str(e)}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 📝 Conversation History")
                for i, chat in enumerate(st.session_state.chat_history):
                    # User message
                    st.markdown(f'<div class="message user-message"><strong>You:</strong> {chat["question"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Assistant message
                    st.markdown(f'<div class="message assistant-message"><strong>Assistant:</strong> {chat["answer"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Sources if available
                    if chat.get("sources"):
                        with st.expander(f"📚 Sources for this answer"):
                            for j, source in enumerate(chat["sources"]):
                                st.markdown(f"**Source {j+1}:**")
                                st.markdown(f"Page: {source.metadata.get('page', 'N/A')}")
                                st.markdown(f"Content: {source.page_content[:200]}...")
                    
                    st.markdown("---")
    
    with col2:
        # Information panel
        st.markdown("### ℹ️ System Information")
        
        if st.session_state.documents_loaded:
            st.success("**Status:** Active")
            st.info(f"**Chat History:** {len(st.session_state.chat_history)} messages")
            
            if st.session_state.vector_store:
                st.info("**Vector Store:** Qdrant")
                st.info("**Embeddings:** HuggingFace (all-MiniLM-L6-v2)")
                st.info("**LLM:** SARVAM-m")
        else:
            st.warning("**Status:** Waiting for documents")
        
        # Instructions
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload PDFs** in the sidebar
        2. **Process documents** to create embeddings
        3. **Ask questions** about your documents
        4. **View sources** for each answer
        """)
        
        # Features
        st.markdown("### ✨ Features")
        st.markdown("""
        - 📄 PDF document processing
        - 🔍 Semantic search
        - 💬 Conversational memory
        - 📚 Source attribution
        - 📱 Responsive design
        """)

if __name__ == "__main__":
    main() 