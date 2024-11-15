import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

# Initialize Vertex AI
PROJECT_ID = 'my-project-0004-346516'

location = "us-central1"  # or your preferred location
vertexai.init(
    project=PROJECT_ID, 
    location=location,
)

# Set page config
st.set_page_config(
    page_title="Hello World App",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("👋 Navigation")
    st.write("Welcome to our colorful app!")

# Main content
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>
        Hello World! 🌈
    </h1>
    """, unsafe_allow_html=True)

# Add dramatic AI message
st.markdown("""
    <h3 style='text-align: center; color: #FF0000; margin-bottom: 30px;'>
        🤖 AI Takeover Imminent... Loading... ⏳
    </h3>
    """, unsafe_allow_html=True)

# Add some colorful elements
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p style='color: #FF4B4B;'>🌹 Red Hello!</p>", unsafe_allow_html=True)

with col2:
    st.markdown("<p style='color: #3366FF;'>🌊 Blue Hello!</p>", unsafe_allow_html=True)

with col3:
    st.markdown("<p style='color: #32CD32;'>🌿 Green Hello!</p>", unsafe_allow_html=True)

# Add Gemini interaction section
st.markdown("---")
st.markdown("""
    <h2 style='text-align: center; color: #4B4BFF;'>
        💬 Ask Gemini
    </h2>
    """, unsafe_allow_html=True)

user_input = st.text_input("Enter your question:", placeholder="Ask me anything...")

if user_input:
    with st.spinner("Generating response..."):
        try:
            # Initialize Gemini model
            MODEL_ID = "gemini-1.5-pro-002"
            model = GenerativeModel(MODEL_ID)
            
            # Set generation config
            generation_config = GenerationConfig(
                temperature=0.9,
                top_p=1.0,
                top_k=32,
                candidate_count=1,
                max_output_tokens=8192,
            )
            
            # Define safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
            
            # Generate response
            response = model.generate_content(
                user_input,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            
            # Get the response text
            response_text = response.text
            
            # Display response in a nice box
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                    {response_text}
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
