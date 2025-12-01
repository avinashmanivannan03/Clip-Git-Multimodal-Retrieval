import streamlit as st
from PIL import Image as PILImage
from utils.clip_utils import CLIPComparator
from utils.llm_utils import LLMComparator
import os
import plotly.graph_objects as go
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Visual-Text Matching Validator",
    page_icon=":mag:",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 5px;
        border: 1px solid #ced4da;
    }
    .stFileUploader>div>div>div>button {
        border-radius: 5px;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.2em;
        border-radius: 3px;
    }
    .image-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .image-item {
        flex: 1 1 200px;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .error-message {
        color: #dc3545;
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_clip_model():
    return CLIPComparator()

@st.cache_resource
def load_llm_model():
    try:
        return LLMComparator()
    except ValueError as e:
        st.warning(str(e))
        return None

clip_comparator = load_clip_model()
llm_comparator = load_llm_model()

# Session state initialization
if 'analyses' not in st.session_state:
    st.session_state.analyses = []  # List of {'image': file, 'text': str, 'results': dict}
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "CLIP (No LLM)"
if 'show_add_another' not in st.session_state:
    st.session_state.show_add_another = False

# Sidebar
with st.sidebar:
    st.title("Settings")
    st.session_state.current_mode = st.radio(
        "Select Mode:",
        ("CLIP (No LLM)", "LLM (Gemini + GIT)"),
        index=0 if st.session_state.current_mode == "CLIP (No LLM)" else 1
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool validates how well a text description matches an image.
    - **CLIP Mode**: Uses the CLIP model for embedding-based similarity
    - **LLM Mode**: Uses Gemini for advanced reasoning with GIT captions
    """)

# Main content
st.title(":mag: Visual-Text Matching Validator")
st.markdown("Upload an image and provide a text description to check how well they match.")

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def display_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 50], 'color': "#FF5252"},
                {'range': [50, 75], 'color': "#FFD740"},
                {'range': [75, 100], 'color': "#4CAF50"}
            ],
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def analyze_image(image_file, text, mode):
    temp_image_path = save_uploaded_file(image_file)
    if not temp_image_path:
        return None
    
    try:
        if mode == "CLIP (No LLM)":
            # Calculate similarity score
            similarity_score = clip_comparator.calculate_similarity(temp_image_path, text)
            
            # Get detailed analysis
            analysis = clip_comparator.detailed_analysis(temp_image_path, text)
            
            results = {
                "score": similarity_score,
                "details": analysis,
                "type": "clip"
            }
            
        elif mode == "LLM (Gemini + GIT)" and llm_comparator is not None:
            # Generate caption
            caption = llm_comparator.generate_caption(temp_image_path)
            
            # Analyze with Gemini
            analysis = llm_comparator.analyze_with_gemini(temp_image_path, text, caption)
            
            if "error" in analysis:
                raise Exception(analysis["error"])
            
            results = {
                "score": analysis["score"],
                "details": analysis,
                "caption": caption,
                "type": "llm"
            }
        
        return results
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

def display_analysis(index, analysis):
    st.markdown(f"---")
    st.subheader(f"Analysis {index + 1}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(analysis['image'], use_container_width=True)
    with col2:
        st.markdown(f"**Text Description:** {analysis['text']}")
    
    if analysis['results']['type'] == "clip":
        # Display CLIP analysis results
        display_score_gauge(analysis['results']['score'])
        
        # Detailed analysis
        st.markdown("### :microscope: Detailed Analysis")
        
        # Word-level similarities
        st.markdown("#### Word-level Similarity Breakdown")
        words, scores = zip(*analysis['results']['details']['word_scores'])
        colors = ["#FF5252" if s < 50 else "#FFD740" if s < 75 else "#4CAF50" for s in scores]
        
        fig_words = go.Figure(go.Bar(
            x=list(words),
            y=list(scores),
            marker_color=colors,
            text=list(scores),
            textposition='auto'
        ))
        fig_words.update_layout(
            xaxis_title="Words",
            yaxis_title="Similarity Score",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_words, use_container_width=True)
        
        # Most/Least matching words
        col_least, col_most = st.columns(2)
        with col_least:
            st.markdown("#### :warning: Least Matching Words")
            for word, score in analysis['results']['details']['least_matching']:
                st.markdown(f"- `{word}`: {score}%")
        
        with col_most:
            st.markdown("#### :white_check_mark: Best Matching Words")
            for word, score in analysis['results']['details']['most_matching']:
                st.markdown(f"- `{word}`: {score}%")
        
        # Matching regions
        st.markdown("#### :frame_with_picture: Matching Regions")
        st.markdown("The text best matches these parts of the image:")
        for region in analysis['results']['details'].get("matching_regions", []):
            st.markdown(f"- {region.capitalize()}")
        
    elif analysis['results']['type'] == "llm":
        # Display LLM analysis results
        display_score_gauge(analysis['results']['score'])
        
        # Show generated caption
        with st.expander(":pencil: Generated Image Caption"):
            st.info(analysis['results']['caption'])
        
        # Display Gemini analysis
        st.markdown("### :microscope: Detailed Analysis")
        
        with st.expander(":white_check_mark: Matching Elements"):
            st.markdown(analysis['results']['details']['matching_elements'] or "No matching elements found")
        
        with st.expander(":warning: Discrepancies"):
            st.markdown(analysis['results']['details']['discrepancies'] or "No discrepancies found")
        
        with st.expander(":thought_balloon: Justification"):
            st.markdown(analysis['results']['details']['justification'] or "No justification provided")
        
        with st.expander(":bulb: Suggestions for Improvement"):
            st.markdown(analysis['results']['details']['suggestions'] or "No suggestions provided")

# Main workflow
if not st.session_state.analyses and not st.session_state.show_add_another:
    # Initial upload and analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader(
            "Upload an image", 
            type=["jpg", "jpeg", "png"],
            help="Upload an image file to analyze",
            key="initial_upload"
        )
    
    with col2:
        st.subheader("Text Input")
        user_text = st.text_area(
            "Enter text description", 
            height=200,
            placeholder="Enter a detailed description of what you expect to see in the image...",
            help="Provide a text description to compare with the image content",
            key="initial_text"
        )
    
    if st.button("Analyze", key="initial_analyze"):
        if uploaded_file and user_text.strip():
            with st.spinner("Analyzing..."):
                results = analyze_image(uploaded_file, user_text, st.session_state.current_mode)
                if results:
                    st.session_state.analyses.append({
                        'image': uploaded_file,
                        'text': user_text,
                        'results': results
                    })
                    st.session_state.show_add_another = True
                    st.rerun()
        else:
            if not uploaded_file:
                st.warning("Please upload an image file")
            if not user_text.strip():
                st.warning("Please enter a text description")

# Display existing analyses
for idx, analysis in enumerate(st.session_state.analyses):
    display_analysis(idx, analysis)

# Add another image option
if st.session_state.show_add_another:
    st.markdown("---")
    st.subheader(":camera: Add Another Image")
    
    with st.form("add_another_form"):
        new_image = st.file_uploader(
            "Upload another image to analyze", 
            type=["jpg", "jpeg", "png"],
            help="Upload another image to analyze",
            key="additional_image"
        )
        
        new_text = st.text_area(
            "Enter text description for new image", 
            height=200,
            placeholder="Enter a description for the new image...",
            help="Provide a text description for the new image",
            key="additional_text"
        )
        
        if st.form_submit_button("Analyze New Image"):
            if new_image and new_text.strip():
                with st.spinner("Analyzing new image..."):
                    results = analyze_image(new_image, new_text, st.session_state.current_mode)
                    if results:
                        st.session_state.analyses.append({
                            'image': new_image,
                            'text': new_text,
                            'results': results
                        })
                        st.rerun()
            else:
                if not new_image:
                    st.warning("Please upload an image file")
                if not new_text.strip():
                    st.warning("Please enter a text description")