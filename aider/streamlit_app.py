import streamlit as st
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
import vertexai

# Set page config
st.set_page_config(
    page_title="Gemini Pro Text Generator",
    layout="centered"
)

# Initialize Vertex AI
try:
    vertexai.init(project="your-project-id", location="us-central1")
except Exception as e:
    st.error(f"Failed to initialize Vertex AI: {str(e)}")

# Sidebar
with st.sidebar:
    st.title("Navigation")
    st.write("Welcome to the Gemini Pro Text Generator!")
    
# Main content
st.title("Gemini Pro Text Generator 🤖")

# Initialize the model
MODEL_ID = "gemini-1.5-pro-002"
model = GenerativeModel(MODEL_ID)
example_model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a helpful language translator.",
        "Your mission is to translate text in English to French.",
    ],
)

# Model configuration
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

# Safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# User input
user_input = st.text_input("Enter your text to translate:", "I like bagels.")

if st.button("Translate"):
    # Prepare prompt
    prompt = f"User input: {user_input}\nAnswer:"
    contents = [prompt]
    
    # Display token count
    token_count = example_model.count_tokens(contents)
    st.write(f"Token count: {token_count}")
    
    # Generate response
    with st.spinner("Generating translation..."):
        response = example_model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
    # Display results
    st.subheader("Translation:")
    st.write(response.text)
    
    # Display metadata in an expander
    with st.expander("View Technical Details"):
        st.write("Usage Metadata:", response.to_dict().get("usage_metadata"))
        st.write("Finish Reason:", response.candidates[0].finish_reason)
        st.write("Safety Ratings:", response.candidates[0].safety_ratings)
