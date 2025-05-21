import streamlit as st
import pickle
import torch
import nltk
import re
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os


# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Label maps
text_label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
audio_label_map = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}

# Load text models
try:
    with open('simple_model.pkl', 'rb') as f:
        simple_model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Text model files ('simple_model.pkl', 'vectorizer.pkl') not found. Please run the training script first.")
    st.stop()
try:
    transformer_model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')
    tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_model')
except Exception as e:
    st.warning(f"Error loading DistilBERT model: {str(e)}. Continuing with simple text model only.")
    transformer_model, tokenizer = None, None

# Load audio model
audio_model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
try:
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name)
except Exception as e:
    st.error(f"Error loading audio model: {str(e)}. Audio predictions will not be available.")
    audio_model, audio_feature_extractor = None, None

# Text prediction function
def predict_text_emotion(sentence, simple_model, vectorizer, transformer_model, tokenizer, label_map):
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    cleaned_sentence = clean_text(sentence)
    # Simple model prediction
    sentence_tfidf = vectorizer.transform([cleaned_sentence])
    simple_pred = simple_model.predict(sentence_tfidf)[0]
    simple_emotion = label_map[simple_pred]
    # Transformer model prediction
    transformer_emotion = None
    if transformer_model and tokenizer:
        inputs = tokenizer(cleaned_sentence, return_tensors='pt', padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = transformer_model(**inputs)
        transformer_pred = torch.argmax(outputs.logits, dim=1).item()
        transformer_emotion = label_map[transformer_pred]
    return simple_emotion, transformer_emotion

# Audio prediction function
def predict_audio_emotion(audio_file_path, fs=16000):
    if not audio_model or not audio_feature_extractor:
        return "Audio model not available."
    try:
        speech, _ = librosa.load(audio_file_path, sr=fs)
        inputs = audio_feature_extractor(speech, sampling_rate=fs, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = audio_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        emotion = audio_model.config.id2label[predicted_ids.item()]
        return emotion
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Live recording function
def record_audio(duration=5, fs=16000, output_file="temp_recording.wav"):
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        sf.write(output_file, audio.squeeze(), fs)
        return output_file
    except Exception as e:
        return f"Error during recording: {str(e)}"

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #4a4a4a !important;
        margin-bottom: 20px !important;
    }
    .subheader {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #4a4a4a !important;
        margin-top: 20px !important;
    }
    .description {
        font-size: 16px !important;
        color: #666666 !important;
        margin-bottom: 30px !important;
    }
    .stButton>button {
        background-color: #4a8cff !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
    }
    .stButton>button:hover {
        background-color: #3a7cdf !important;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #999999;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
    .footer {
        font-size: 14px !important;
        color: #999999 !important;
        margin-top: 40px !important;
        border-top: 1px solid #e0e0e0;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="header">Emotion Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Analyze emotions from text input or audio recordings using advanced machine learning models.</div>', unsafe_allow_html=True)

# Input type selection with icons
input_type = st.radio(
    "Select Input Type:",
    ["Text", "Audio File Upload", "Live Audio Recording"],
    horizontal=True,
    help="Choose the type of input you want to analyze"
)

# Process input based on selection
if input_type == "Text":
    st.markdown('<div class="subheader">Text Emotion Analysis</div>', unsafe_allow_html=True)
    sentence = st.text_area(
        "Enter your text here:",
        placeholder="e.g., I'm feeling really excited about this new opportunity!",
        height=150
    )
    
    if st.button("Analyze Text Emotion"):
        if not sentence.strip():
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text emotion..."):
                simple_emotion, transformer_emotion = predict_text_emotion(
                    sentence, simple_model, vectorizer, transformer_model, tokenizer, text_label_map
                )
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<div class="subheader">Analysis Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="TF-IDF + Logistic Regression", value=simple_emotion.capitalize())
            
            with col2:
                if transformer_emotion:
                    st.metric(label="DistilBERT Model", value=transformer_emotion.capitalize())
                else:
                    st.metric(label="DistilBERT Model", value="Not available")
            
            st.markdown('</div>', unsafe_allow_html=True)

elif input_type == "Audio File Upload":
    st.markdown('<div class="subheader">Audio File Emotion Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">Please upload a WAV audio file (mono, 16kHz recommended)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an audio file:",
        type=["wav"],
        accept_multiple_files=False,
        help="Supported format: WAV, mono channel, 16kHz sample rate"
    )
    
    if uploaded_file and st.button("Analyze Audio Emotion"):
        with st.spinner("Processing audio file..."):
            temp_file_path = "temp_uploaded_audio.wav"
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                emotion = predict_audio_emotion(temp_file_path)
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="subheader">Analysis Result</div>', unsafe_allow_html=True)
                st.metric(label="Detected Emotion", value=emotion.capitalize())
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    st.markdown('<div class="footer">Note: Model accuracy depends on input quality. For audio, clear recordings with minimal background noise yield best results.</div>', unsafe_allow_html=True)
    
elif input_type == "Live Audio Recording":
    st.markdown('<div class="subheader">Live Audio Emotion Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">Ensure your microphone is properly connected and allowed in browser settings</div>', unsafe_allow_html=True)
    
    duration = st.slider(
        "Recording duration (seconds):",
        min_value=3,
        max_value=10,
        value=5,
        help="Longer recordings may provide better results"
    )
    
    if st.button(f"Start {duration}-Second Recording"):
        with st.spinner(f"Recording audio for {duration} seconds..."):
            result = record_audio(duration=duration)
            
            if isinstance(result, str) and "Error" in result:
                st.error(result)
            else:
                temp_file_path = result
                emotion = predict_audio_emotion(temp_file_path)
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="subheader">Analysis Result</div>', unsafe_allow_html=True)
                st.metric(label="Detected Emotion", value=emotion.capitalize())
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

