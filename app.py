import streamlit as st
import os
from PIL import Image
from rag_chain import OnePlusRAGChain
from dotenv import load_dotenv

# Load environment variables (like API keys)
load_dotenv()

# Configure the Streamlit page
st.set_page_config(
    page_title="OnePlus 6 Manual Assistant",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better looks
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #FF6B6B;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #FF6B6B;
}
.stButton button {
    background-color: #FF6B6B;
    color: white;
    border: none;
    border-radius: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_chain():
    """
    Load the RAG chain once and cache it.
    
    @st.cache_resource ensures this only runs once per session,
    not every time the user interacts with the app.
    """
    try:
        print("🔄 Loading RAG chain...")
        return OnePlusRAGChain()
    except Exception as e:
        st.error(f"❌ Error loading RAG chain: {e}")
        st.error("Please make sure you've run 'python preprocess_manual.py' first!")
        return None

def display_image(image_path):
    """
    Display an image if it exists, with error handling.
    """
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            st.warning(f"⚠️ Could not load image {image_path}: {e}")
            return None
    else:
        st.warning(f"⚠️ Image not found: {image_path}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📱 OnePlus 6 Manual Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by Groq AI and built with LangChain*")
    
    # Sidebar with information and sample questions
    with st.sidebar:
        st.header("🔧 How This Works")
        st.markdown("""
        **The Technology:**
        - 🔍 **Retrieval**: Searches the OnePlus 6 manual
        - 🤖 **Groq AI**: Generates natural responses
        - 📊 **Vector Search**: Finds most relevant information
        - 🖼️ **Images**: Shows related diagrams when available
        
        **Features:**
        - ✅ Manual-based answers only (no guessing!)
        - 📄 Page references for verification
        - 🖼️ Related images and diagrams
        - ⚡ Fast responses with Groq
        """)
        
        st.header("📋 Try These Questions")
        sample_questions = [
            "How do I set up Face Unlock?",
            "Where is the fingerprint sensor?",
            "How to use Portrait Mode?",
            "What are the camera specifications?",
            "How to enable Gaming Mode?",
            "What's included in the box?",
            "How to insert SIM cards?",
            "Battery safety precautions?",
            "How to use Alert Slider?",
            "What is OnePlus Switch?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.user_input = question
        
        st.header("🔍 System Status")
        
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """👋 Hello! I'm your OnePlus 6 Manual Assistant, powered by **Groq AI**.

I can help you find specific information from the official OnePlus 6 user manual. I'll provide:
- ✅ Accurate answers based only on the manual
- 📄 Page references for verification  
- 🖼️ Related images when available

What would you like to know about your OnePlus 6?"""
            }
        ]
    
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Load the RAG chain
    rag_chain = load_rag_chain()
    
    if rag_chain is None:
        st.error("⚠️ **System not ready!**")
        st.error("Please run `python preprocess_manual.py` first to process the manual.")
        st.code("python preprocess_manual.py")
        return
    else:
        with st.sidebar:
            st.success("✅ System Ready!")
            st.info("📚 Manual processed and indexed")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display related images if available
            if "images" in message and message["images"]:
                st.subheader("📸 Related Images from Manual")
                cols = st.columns(min(len(message["images"]), 3))  # Max 3 columns
                for idx, img_path in enumerate(message["images"]):
                    with cols[idx % 3]:
                        image = display_image(img_path)
                        if image:
                            st.image(image, 
                                   caption=f"From page {os.path.basename(img_path)}", 
                                   use_container_width=True)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about the OnePlus 6...")
    
    # Handle sample question input from sidebar
    if st.session_state.user_input:
        user_input = st.session_state.user_input
        st.session_state.user_input = ""
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching manual and generating response..."):
                try:
                    # Query the RAG system
                    result = rag_chain.query(user_input)
                    
                    # Display the answer
                    st.markdown(result["answer"])
                    
                    # Display related images if available
                    if result["related_images"]:
                        st.subheader("📸 Related Images from Manual")
                        cols = st.columns(min(len(result["related_images"]), 3))
                        for idx, img_path in enumerate(result["related_images"]):
                            with cols[idx % 3]:
                                image = display_image(img_path)
                                if image:
                                    st.image(image, 
                                           caption=f"From {os.path.basename(img_path)}", 
                                           use_container_width=True)
                    
                    # Show source information in expandable section
                    with st.expander("🔍 Source Information & Debug"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if result["retrieved_pages"]:
                                st.write("**📄 Pages Referenced:**")
                                for page in result["retrieved_pages"]:
                                    st.write(f"• Page {page}")
                        
                        with col2:
                            if result["related_images"]:
                                st.write("**🖼️ Related Images:**")
                                for img in result["related_images"]:
                                    st.write(f"• {os.path.basename(img)}")
                        
                        if result["source_chunks"]:
                            st.write("**📝 Source Text Snippets:**")
                            for i, chunk in enumerate(result["source_chunks"], 1):
                                st.write(f"**{i}.** {chunk}")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "images": result["related_images"]
                    })
                
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Footer with controls
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": """👋 Chat cleared! I'm ready to help with your OnePlus 6 questions.

What would you like to know?"""
                }
            ]
            st.rerun()
    
    with col2:
        st.markdown("**📄 Pages:** 52 total")
    
    with col3:
        st.markdown("**🚀 AI:** Groq (Mixtral)")
    
    with col4:
        st.markdown("**📱 Model:** OnePlus 6")


if __name__ == "__main__":
    main()
