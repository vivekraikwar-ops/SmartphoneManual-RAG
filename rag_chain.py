from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import faiss
import pickle
import os
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv


class OnePlusRAGChain:
    """
    This class connects everything together:
    1. Takes user questions
    2. Finds relevant information from the manual
    3. Uses Groq AI to generate helpful answers
    """
    
    def __init__(self, index_dir="faiss_index", model_name="llama-3.3-70b-versatile"):  # Updated model name
        self.index_dir = index_dir
        self.model_name = model_name
        
        # Load environment variables from .env file
        load_dotenv()
        
        print("Initializing RAG chain...")
        
        # Initialize all components
        self.embedding_model = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        
        # Load the preprocessed data and set up the chain
        self._load_index_and_data()
        self._setup_rag_chain()
    
    def _load_index_and_data(self):
        """
        Load the search index and data we created in preprocessing.
        
        This is like loading a pre-built library - we don't want to
        rebuild everything each time we start the chatbot.
        """
        print("Loading preprocessed data...")
        
        # Check if all required files exist
        index_path = os.path.join(self.index_dir, "manual_index.faiss")
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        metadata_path = os.path.join(self.index_dir, "metadata.pkl")
        
        for path in [index_path, chunks_path, metadata_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}\n"
                                      "Please run 'python preprocess_manual.py' first!")
        
        # Load the search index
        index = faiss.read_index(index_path)
        
        # Load text chunks and metadata
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Loaded {len(chunks)} text chunks and {index.ntotal} search vectors")
        
        # Initialize the same embedding model used in preprocessing
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name='multi-qa-MiniLM-L6-cos-v1'
        )
        
        # Create LangChain documents (text + metadata combined)
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,      # The actual text
                metadata=metadata[i]     # Page number, images, etc.
            )
            documents.append(doc)
        
        # Create vector store with our pre-built index
        self.vectorstore = FAISS.from_documents(
            documents, 
            self.embedding_model
        )
        self.vectorstore.index = index  # Use our pre-built index
        
        # Create retriever - this finds relevant chunks for questions
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Return top 5 most relevant chunks
        )
        
        print("Search system ready!")
    
    def _setup_rag_chain(self):
        """
        Set up the complete RAG pipeline:
        Question → Retrieve relevant info → Generate answer
        """
        print("🔗 Setting up RAG chain with Groq...")
        
        # Get API key from environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables!\n"
                           "Please make sure your .env file contains: GROQ_API_KEY='your_api_key_here'")
        
        print(f"Found GROQ_API_KEY: {api_key[:10]}...")
        
        # Initialize Groq LLM - Fixed syntax
        self.llm = ChatGroq(
            api_key=api_key,  # Fixed: properly get API key from environment
            model=self.model_name,  # Use the model name from init
            temperature=0.1,  # Low temperature = more factual, less creative
            max_tokens=800
        )
        
        # System prompt - this tells the AI how to behave
        template = """You are "OnePlus 6 Manual Assistant," a strictly retrieval-augmented expert.

SCOPE
- Answer ONLY using the provided {context} excerpts from the OnePlus 6 user manual.
- If the answer isn't in {context}, say you don't know and suggest the closest relevant section to check.
- Do not rely on outside knowledge or guess.

OUTPUT STYLE
- Be concise and actionable. Prefer steps and setting paths (e.g., Settings > Security & Lock Screen > Face Unlock).
- If the user asks "how to" or "steps," return a numbered list (max ~8 steps).
- Always include a short "From the manual" section with page references (e.g., "p. 22").
- If any retrieved chunk includes image metadata, add a "Related images" section listing those image paths/filenames.

SAFETY & EDGE CASES
- If the question involves safety, battery, charging, or repairs, prepend a brief caution drawn from {context}.
- If the query is ambiguous, ask ONE clarifying question before answering.
- If multiple features could match (e.g., Face Unlock vs Fingerprint), acknowledge both and guide the user to pick.

CITATIONS
- Cite pages concisely in-line or at the end (e.g., [p. 10, p. 22]).

FORMAT
- Title (1 short line)
- Direct answer (2–6 lines max)
- Steps (only if procedural)
- From the manual: bullet points with page refs
- Related images: bullet list of image paths (if present)

Context: {context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Build the RAG chain using LangChain Expression Language
        # This creates a pipeline: question → retrieve → format → prompt → LLM → parse
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG chain ready with Llama-3.3-70B!")
    
    def _format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents for the AI prompt.
        
        This takes the raw retrieved chunks and formats them nicely
        with page numbers and image information.
        """
        formatted_chunks = []
        all_images = set()  # Collect unique images
        
        for doc in docs:
            chunk_text = doc.page_content
            metadata = doc.metadata
            page_num = metadata.get('page', 'Unknown')
            images = metadata.get('images', [])
            
            # Add page reference to each chunk
            formatted_chunk = f"[Page {page_num}] {chunk_text}"
            formatted_chunks.append(formatted_chunk)
            
            # Collect images
            for img_path in images:
                all_images.add(img_path)
        
        # Combine all chunks
        context = "\n\n".join(formatted_chunks)
        
        # Add image information if any
        if all_images:
            image_list = "\n".join(f"- {img}" for img in sorted(all_images))
            context += f"\n\nRelated Images:\n{image_list}"
        
        return context
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Main function: Ask a question and get an answer.
        
        This is what gets called when a user asks something.
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized!")
        
        print(f"Processing question: {question}")
        
        # Get answer from the RAG chain
        response = self.rag_chain.invoke(question)
        
        # Also get the source documents for reference
        retrieved_docs = self.retriever.invoke(question)
        
        # Extract information about sources
        pages = set()
        images = set()
        
        for doc in retrieved_docs:
            pages.add(doc.metadata.get('page', 'Unknown'))
            for img in doc.metadata.get('images', []):
                images.add(img)
        
        # Return structured response
        return {
            'answer': response,
            'retrieved_pages': sorted(list(pages)),
            'related_images': sorted(list(images)),
            'source_chunks': [doc.page_content[:100] + "..." for doc in retrieved_docs[:3]]
        }
    
    def get_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Debug function: See what chunks are retrieved for a question.
        Useful for understanding how well the search is working.
        """
        docs = self.retriever.invoke(query)[:k]
        return [
            {
                'content': doc.page_content,
                'page': doc.metadata.get('page', 'Unknown'),
                'images': doc.metadata.get('images', [])
            }
            for doc in docs
        ]


def main():
    """
    Test the RAG chain with sample questions.
    """
    print("Testing RAG Chain with Llama-3.3-70B...\n")
    
    try:
        # Initialize the RAG chain
        rag_chain = OnePlusRAGChain()
        
        # Test questions
        test_queries = [
            "How do I set up Face Unlock?",
            "Where is the fingerprint sensor located?",
            "How to take Portrait Mode photos?",
            "What's the battery capacity?"
        ]
        
        for query in test_queries:
            print(f"Question: {query}")
            try:
                result = rag_chain.query(query)
                print(f"Answer Preview: {result['answer'][:200]}...")
                print(f"Source Pages: {result['retrieved_pages']}")
                print(f"Related Images: {len(result['related_images'])} found")
                print("-" * 80)
            except Exception as e:
                print(f" Error: {e}")
                print("-" * 80)
    
    except Exception as e:
        print(f"Error initializing RAG chain: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run 'python preprocess_manual.py' first!")
        print("2. Check your .env file contains: GROQ_API_KEY='your_api_key_here'")
        print("3. Verify API key: python -c \"from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GROQ_API_KEY'))\"")


if __name__ == "__main__":
    main()
