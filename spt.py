import streamlit as st
import torch
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder
import tempfile
import os
import warnings
import asyncio
import os
import sys
from pathlib import Path

# Set FFmpeg path
ffmpeg_path = str(Path(__file__).parent / "bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path
os.environ["FFMPEG_BINARY"] = str(Path(ffmpeg_path) / "ffmpeg"

# Verify FFmpeg
try:
    import subprocess
    subprocess.run(["ffmpeg", "-version"], check=True, 
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("FFmpeg successfully configured!")
except Exception as e:
    print(f"FFmpeg error: {str(e)}")
    sys.exit(1)

# Configure environment
os.environ["PATH"] += os.pathsep + r"C:\Users\Arun\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Real-Time Voice-to-Text",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_chunks' not in st.session_state:
    st.session_state.audio_chunks = []

# Title and description
st.title("🎤 Real-Time Voice-to-Text with Whisper")
st.markdown("""
This application provides real-time speech recognition with Hindi-to-English translation capability.
""")

# Sidebar with options
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Model Size",
        ("base", "small", "medium", "large-v3"),
        index=2,
        help="Larger models are more accurate but slower"
    )
    
    translation_mode = st.checkbox(
        "Enable Hindi-to-English Translation",
        value=True,
        help="When enabled, Hindi speech will be automatically translated to English"
    )
    
    st.markdown("---")
    st.markdown("### Performance")
    if torch.cuda.is_available():
        st.success("GPU acceleration available")
    else:
        st.warning("Using CPU - processing may be slow")

# Initialize the Whisper pipeline with continuous processing
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            device=device,
            return_timestamps=True
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def process_audio_chunk(audio_path):
    model = load_model()
    if model is None:
        return None
        
    try:
        # First detect language
        detection_result = model(audio_path, generate_kwargs={"task": "transcribe"})
        detected_language = detection_result.get("language", "en")
        
        # Process based on language and user preference
        if detected_language == "hi" and translation_mode:
            result = model(
                audio_path,
                generate_kwargs={
                    "language": "hi",
                    "task": "translate"
                }
            )
        else:
            result = model(
                audio_path,
                generate_kwargs={
                    "language": "en",
                    "task": "transcribe"
                }
            )
            
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def main():
    model = load_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Live Recording")
        audio_bytes = audio_recorder(
            pause_threshold=1.5,
            sample_rate=16000,
            text="Hold to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            key="recorder"
        )
        
        if audio_bytes:
            st.session_state.is_recording = True
            st.session_state.audio_chunks.append(audio_bytes)
            
            # Process the latest chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            text = process_audio_chunk(tmp_path)
            if text:
                st.session_state.transcription += text + " "
            
            os.unlink(tmp_path)
            
            # Show latest audio chunk
            st.audio(audio_bytes, format="audio/wav")
        else:
            if st.session_state.is_recording:
                st.session_state.is_recording = False
                st.success("Recording completed!")
    
    with col2:
        st.subheader("Transcription Output")
        transcription_display = st.empty()
        transcription_display.text_area(
            "Live Transcription",
            st.session_state.transcription,
            height=300
        )
        
        if st.button("Clear Transcription"):
            st.session_state.transcription = ""
            st.session_state.audio_chunks = []
            transcription_display.empty()

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")

st.markdown("---")
st.markdown("""
### Usage Tips:
- Hold the microphone button to record continuously
- The system will process speech in real-time without 30s limits
- Hindi speech will be translated to English when enabled
- Clear transcription between sessions for best results
""")
