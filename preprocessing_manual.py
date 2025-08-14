import fitz 
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np
import pickle
from PIL import Image
import io

class OnePlusManualPreprocessor:
    """
    Here we are converting our raw pdf into searchable chunks 
    """
    
    def __init__(self, pdf_path, images_dir="images", index_dir="faiss_index"):
        self.pdf_path = pdf_path
        self.images_dir = images_dir
        self.index_dir = index_dir
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,    # Each chunk will be 500 characters
            chunk_overlap=50,  # 50 characters overlap between chunks to maintain context
            separators=["\n\n", "\n", ". ", " "]  # Split at paragraphs, then sentences
        )
        
        # Storage for our processed data
        self.chunks = []      # Text pieces
        self.embeddings = []  # Numerical representations
        self.metadata = []    # Information about each chunk (page, images, etc.)
    
    def extract_text_and_images(self):
        """
        Step 1: Open the PDF and extract both text and images from each page.
        Why we do this: We need to convert the PDF into text we can search
        and save images that might help explain concepts.
        """
        print(f"Opening PDF: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        
        page_texts = []    # Store text from each page
        page_images = {}   # Store images by page number
        
        # Go through each page in the PDF
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text from this page
            text = page.get_text()
            text = ' '.join(text.split())  # Clean up extra spaces
            
            if text.strip():  # Only save pages with actual text
                page_texts.append({
                    'text': text,
                    'page': page_num + 1  # Use 1-based page numbers (more intuitive)
                })
                print(f"Extracted text from page {page_num + 1}")
            
            # Extract images from this page
            image_list = page.get_images()
            page_image_paths = []
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get the actual image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Save as PNG if it's a normal image (not CMYK color)
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                        img_path = os.path.join(self.images_dir, img_filename)
                        
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        
                        page_image_paths.append(img_path)
                        print(f"Saved image: {img_filename}")
                    
                    pix = None  # Clean up memory
                
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
            
            # Remember which images are on which page
            if page_image_paths:
                page_images[page_num + 1] = page_image_paths
        
        doc.close()
        print(f"Processing complete: {len(page_texts)} pages with text, {len(page_images)} pages with images")
        
        return page_texts, page_images
    
    def create_chunks_with_metadata(self, page_texts, page_images):
        """
        Step 2: Break the text into smaller, searchable chunks.
        
        Why we do this: Large blocks of text are hard to search precisely.
        Smaller chunks allow us to find exactly the right information.
        """
        print("Creating text chunks...")
        
        for page_data in page_texts:
            page_text = page_data['text']
            page_num = page_data['page']
            
            # Split the page text into smaller chunks
            text_chunks = self.text_splitter.split_text(page_text)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                if chunk.strip():  # Only add non-empty chunks
                    # Create metadata - information about this chunk
                    chunk_metadata = {
                        'page': page_num,
                        'chunk_id': f"page_{page_num}_chunk_{chunk_idx + 1}",
                        'images': page_images.get(page_num, [])  # Images from the same page
                    }
                    
                    self.chunks.append(chunk)
                    self.metadata.append(chunk_metadata)
        
        print(f"Created {len(self.chunks)} searchable text chunks")
    
    def create_embeddings(self):
        """
        Step 3: Convert text chunks into numerical vectors (embeddings).
        
        Why we do this: Computers can't directly compare text for meaning,
        but they can compare numbers. Embeddings represent the 'meaning'
        of text as numbers.
        """
        print("Converting text to embeddings...")
        
        if not self.chunks:
            raise ValueError("No text chunks found! Run extract_text_and_images first.")
        
        # Process chunks in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}...")
            
            # Convert text to numbers
            batch_embeddings = self.embedding_model.encode(
                batch_chunks,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity comparison
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(all_embeddings)
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        print(f"   Each text chunk is now represented by {self.embeddings.shape[1]} numbers")
    
    def build_faiss_index(self):
        """
        Step 4: Build a search index for fast similarity search.
        
        Why we do this: When a user asks a question, we need to quickly
        find the most relevant chunks. FAISS makes this search very fast.
        """
        print("Building search index...")
        
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings found! Run create_embeddings first.")
        
        # Create FAISS index - this is like building a search engine
        dimension = self.embeddings.shape[1]  # Number of dimensions in our embeddings
        index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (for similarity)
        
        # Add all our embeddings to the search index
        index.add(self.embeddings.astype(np.float32))
        print(f" Added {index.ntotal} vectors to search index")
        
        # Save everything to disk so we don't have to rebuild it every time
        index_path = os.path.join(self.index_dir, "manual_index.faiss")
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        metadata_path = os.path.join(self.index_dir, "metadata.pkl")
        
        faiss.write_index(index, index_path)
        
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Saved search index to: {index_path}")
        print(f"Saved chunks to: {chunks_path}")
        print(f"Saved metadata to: {metadata_path}")
        
        return index
    
    def process_manual(self):
        """
        Run the complete preprocessing pipeline.
        This is the main function that does everything step by step.
        """
        print("Starting OnePlus 6 manual preprocessing...\n")
        
        # Step 1: Extract text and images
        print("STEP 1: Extracting content from PDF")
        page_texts, page_images = self.extract_text_and_images()
        print()
        
        # Step 2: Create searchable chunks
        print("STEP 2: Creating searchable chunks")
        self.create_chunks_with_metadata(page_texts, page_images)
        print()
        
        # Step 3: Convert to embeddings
        print("STEP 3: Converting text to embeddings")
        self.create_embeddings()
        print()
        
        # Step 4: Build search index
        print("STEP 4: Building search index")
        index = self.build_faiss_index()
        print()
        
        print("Preprocessing completed successfully!")
        print("Next step: Run 'python rag_chain.py' to test the system")
        return index


def main():
    """
    Main function - this runs when you execute the script.
    """
    # Make sure the PDF file exists
    pdf_path = "USER_MANUAL_OP6_FINAL.pdf"  # Name of your PDF file
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found!")
        print("Please make sure the OnePlus 6 manual PDF is in the current directory.")
        return
    
    # Create and run the preprocessor
    preprocessor = OnePlusManualPreprocessor(pdf_path)
    
    try:
        preprocessor.process_manual()
        print("\nSuccess! You can now run the chatbot with: streamlit run app.py")
    except Exception as e:
        print(f"Error during preprocessing: {e}")


if __name__ == "__main__":
    main()
