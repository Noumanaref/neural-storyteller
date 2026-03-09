import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import os
from collections import Counter
import numpy as np
import base64
from io import BytesIO
import time

# ------------------------------------------------------------------
# 0. PAGE CONFIGURATION (MUST BE FIRST)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Vision AI Studio | Professional Image Captioning",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# 1. CLASS DEFINITIONS (UNCHANGED - MATCHES CHECKPOINT)
# ------------------------------------------------------------------

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return str(text).lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]

class Encoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=712):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        out = self.linear(features)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, embed_size=512, hidden_size=712, vocab_size=7737, num_layers=5):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        h_0 = features.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c_0 = features.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        outputs, _ = self.lstm(embeddings, (h_0, c_0))
        outputs = self.linear(outputs)
        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, embed_size=512, hidden_size=712, vocab_size=7737, num_layers=5):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size=2048, hidden_size=hidden_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, features, captions):
        enc_out = self.encoder(features)
        outputs = self.decoder(enc_out, captions)
        return outputs

# ------------------------------------------------------------------
# 2. PREMIUM CSS STYLING
# ------------------------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600;700&display=swap');
        
        /* Global Reset & Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            color: #1a1a2e;
            overflow-x: hidden;
        }
        
        /* Animated Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            position: relative;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Noise Texture Overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E");
            pointer-events: none;
            z-index: 1;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Main Container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1400px;
            position: relative;
            z-index: 2;
        }
        
        /* ========== HEADER SECTION ========== */
        .hero-header {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 24px;
            padding: 3rem 3rem 2.5rem 3rem;
            margin-bottom: 3rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        }
        
        .hero-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(139,92,246,0.3) 0%, transparent 70%);
            border-radius: 50%;
            animation: float 8s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            50% { transform: translate(-20px, 20px) rotate(180deg); }
        }
        
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        .hero-subtitle {
            font-size: 1.25rem;
            color: rgba(255,255,255,0.8);
            font-weight: 300;
            max-width: 700px;
            line-height: 1.6;
            margin-bottom: 2rem;
        }
        
        .stats-bar {
            display: flex;
            gap: 3rem;
            margin-top: 2rem;
            flex-wrap: wrap;
        }
        
        .stat-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .stat-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #ffffff;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* ========== CARDS ========== */
        .premium-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .premium-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, transparent 100%);
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .premium-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(139,92,246,0.3);
            border-color: rgba(139,92,246,0.5);
        }
        
        .premium-card:hover::before {
            opacity: 1;
        }
        
        .card-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .card-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }
        
        /* ========== UPLOAD ZONE ========== */
        .upload-container {
            margin-top: 1.5rem;
        }
        
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(139,92,246,0.5);
            border-radius: 16px;
            padding: 3rem 2rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(139,92,246,0.8);
            background: rgba(139,92,246,0.1);
        }
        
        [data-testid="stFileUploader"] section {
            border: none !important;
        }
        
        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(139,92,246,0.4);
        }
        
        [data-testid="stFileUploader"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(139,92,246,0.6);
        }
        
        /* Image Preview */
        .image-preview-container {
            margin-top: 2rem;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        /* ========== BUTTONS ========== */
        .stButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 20px rgba(139,92,246,0.4);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton > button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(139,92,246,0.6);
        }
        
        /* ========== RESULTS DISPLAY ========== */
        .result-container {
            background: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(99,102,241,0.1) 100%);
            border: 1px solid rgba(139,92,246,0.3);
            border-radius: 16px;
            padding: 2rem;
            margin-top: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .result-container::before {
            content: '"';
            position: absolute;
            top: -20px;
            left: 20px;
            font-family: 'Playfair Display', serif;
            font-size: 8rem;
            color: rgba(139,92,246,0.1);
            line-height: 1;
        }
        
        .caption-text {
            font-family: 'Playfair Display', serif;
            font-size: 1.75rem;
            line-height: 1.6;
            color: #ffffff;
            font-weight: 500;
            position: relative;
            z-index: 1;
            font-style: italic;
        }
        
        /* ========== METRICS ========== */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }
        
        div[data-testid="metric-container"] > label {
            color: rgba(255,255,255,0.7) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        div[data-testid="metric-container"] > div {
            color: #ffffff !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
        }
        
        /* ========== DOWNLOAD BUTTON ========== */
        .download-btn {
            display: inline-block;
            width: 100%;
            text-align: center;
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            color: #ffffff;
            padding: 1rem 2rem;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
            margin-top: 1rem;
        }
        
        .download-btn:hover {
            background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
            border-color: rgba(139,92,246,0.5);
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(139,92,246,0.3);
        }
        
        /* ========== EMPTY STATE ========== */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: rgba(255,255,255,0.4);
        }
        
        .empty-state-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.3;
        }
        
        .empty-state-text {
            font-size: 1.125rem;
            color: rgba(255,255,255,0.5);
        }
        
        /* ========== FOOTER ========== */
        .custom-footer {
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }
        
        .footer-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        
        .footer-link {
            color: rgba(255,255,255,0.6);
            text-decoration: none;
            font-size: 0.875rem;
            transition: color 0.3s ease;
        }
        
        .footer-link:hover {
            color: #8b5cf6;
        }
        
        .footer-text {
            color: rgba(255,255,255,0.4);
            font-size: 0.875rem;
        }
        
        .tech-badge {
            display: inline-block;
            background: rgba(139,92,246,0.2);
            color: rgba(255,255,255,0.8);
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 0.25rem;
            border: 1px solid rgba(139,92,246,0.3);
        }
        
        /* ========== RESPONSIVE ========== */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }
            .hero-subtitle {
                font-size: 1rem;
            }
            .stats-bar {
                gap: 1.5rem;
            }
            .caption-text {
                font-size: 1.25rem;
            }
        }
        
        /* ========== ANIMATIONS ========== */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }
        
        /* ========== LOADING SPINNER ========== */
        .stSpinner > div {
            border-color: #8b5cf6 !important;
            border-right-color: transparent !important;
        }
        
        /* File uploader text color */
        [data-testid="stFileUploader"] label, 
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p {
            color: rgba(255,255,255,0.8) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. SETUP & CACHING
# ------------------------------------------------------------------

DEVICE = torch.device('cpu')

@st.cache_resource(show_spinner="🔮 Initializing Neural Networks...")
def load_resources():
    try:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        return None, None, None

    try:
        embed_size = 512
        hidden_size = 712
        vocab_size = 7737
        num_layers = 5
        
        model = Seq2Seq(embed_size, hidden_size, vocab_size, num_layers)
        state_dict = torch.load('caption_model.pth', map_location=DEVICE)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        feature_extractor.eval()
        
        return model, feature_extractor, vocab
        
    except Exception as e:
        st.error(f"❌ Initialization Error: {str(e)}")
        return None, None, None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------

def generate_caption(model, feature_extractor, vocab, image, max_length=20):
    start_time = time.time()
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0)
        feature = feature_extractor(img_tensor)
        feature = feature.view(1, -1)
        
        h = model.encoder(feature)
        num_layers = model.decoder.lstm.num_layers
        h_state = h.unsqueeze(0).repeat(num_layers, 1, 1)
        c_state = h.unsqueeze(0).repeat(num_layers, 1, 1)
        state = (h_state, c_state)
        
        inputs = torch.tensor([vocab.stoi["<start>"]])
        caption = []
        attention_weights = []
        
        for _ in range(max_length):
            embeddings = model.decoder.embed(inputs).unsqueeze(0)
            lstm_out, state = model.decoder.lstm(embeddings, state)
            outputs = model.decoder.linear(lstm_out.squeeze(1))
            
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            attention_weights.append(confidence.item())
            
            pred_word_idx = pred.item()
            if pred_word_idx == vocab.stoi["<end>"]: break
                
            if pred_word_idx in vocab.itos:
                caption.append(vocab.itos[pred_word_idx])
                inputs = torch.tensor([pred_word_idx])
        
        processing_time = time.time() - start_time
        return caption, attention_weights, processing_time

def get_download_link(caption):
    b64 = base64.b64encode(caption.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="caption.txt" class="download-btn">⬇️ Download Caption</a>'

def init_session_state():
    if 'generated_caption' not in st.session_state:
        st.session_state.generated_caption = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'current_image_name' not in st.session_state:
        st.session_state.current_image_name = None
    if 'image_dimensions' not in st.session_state:
        st.session_state.image_dimensions = None

# ------------------------------------------------------------------
# 5. MAIN APPLICATION
# ------------------------------------------------------------------

def main():
    load_css()
    init_session_state()
    
    # Load resources
    model, feature_extractor, vocab = load_resources()
    
    # ========== HERO HEADER ==========
    st.markdown(f"""
    <div class="hero-header fade-in-up">
        <h1 class="hero-title">Vision AI Studio</h1>
        <p class="hero-subtitle">
            Transform images into intelligent narratives using state-of-the-art deep learning.
            Our advanced neural architecture combines ResNet-50 feature extraction with LSTM sequence generation.
        </p>
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">{'3' if model else '0'}</div>
                <div class="stat-label">Models Active</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">&lt;1s</div>
                <div class="stat-label">Processing Speed</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(vocab) if vocab else '0'}</div>
                <div class="stat-label">Vocabulary Size</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">JPG • PNG • JPEG</div>
                <div class="stat-label">Supported Formats</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models loaded
    if model is None or vocab is None:
        st.error("⚠️ **System Offline**: Critical resources missing. Please ensure 'vocab.pkl' and 'caption_model.pth' are in the application directory.")
        return
    
    # ========== MAIN CONTENT ==========
    col_left, col_right = st.columns([1.3, 1], gap="large")
    
    # LEFT COLUMN - Upload & Input
    with col_left:
        st.markdown("""
        <div class="premium-card fade-in-up">
            <div class="card-title">
                <div class="card-icon">📸</div>
                Image Upload
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=['jpg', 'jpeg', 'png'],
            help="Maximum file size: 200MB • Formats: JPG, JPEG, PNG",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Reset state on new image
            if st.session_state.current_image_name != uploaded_file.name:
                st.session_state.generated_caption = None
                st.session_state.confidence = None
                st.session_state.processing_time = None
                st.session_state.current_image_name = uploaded_file.name
                
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.image_dimensions = image.size
            
            # Display image preview
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image metadata
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.caption(f"📊 {uploaded_file.name} • {image.size[0]}×{image.size[1]}px • {file_size_mb:.2f} MB")
            
            # Generate button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Generate Caption", use_container_width=True):
                with st.spinner("🔮 Analyzing visual features..."):
                    caption_words, att_weights, proc_time = generate_caption(
                        model, feature_extractor, vocab, image
                    )
                    st.session_state.generated_caption = " ".join(caption_words)
                    st.session_state.confidence = np.mean(att_weights) if att_weights else 0
                    st.session_state.processing_time = proc_time
                    st.rerun()
        else:
            # Empty state
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🖼️</div>
                <p class="empty-state-text">Upload an image to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # RIGHT COLUMN - Results & Analysis
    with col_right:
        st.markdown("""
        <div class="premium-card fade-in-up">
            <div class="card-title">
                <div class="card-icon">✨</div>
                Generated Caption
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.generated_caption is not None:
            # Display caption
            st.markdown(f"""
            <div class="result-container">
                <p class="caption-text">{st.session_state.generated_caption.capitalize()}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Confidence", f"{st.session_state.confidence:.1%}")
            
            with metric_col2:
                word_count = len(st.session_state.generated_caption.split())
                st.metric("Tokens", f"{word_count}")
            
            with metric_col3:
                st.metric("Speed", f"{st.session_state.processing_time:.2f}s")
            
            # Download button
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(get_download_link(st.session_state.generated_caption), unsafe_allow_html=True)
            
        else:
            # Empty state
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">💭</div>
                <p class="empty-state-text">Caption will appear here after processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== FOOTER ==========
    st.markdown("""
    <div class="custom-footer">
        <div class="footer-content">
            <span class="tech-badge">PyTorch</span>
            <span class="tech-badge">ResNet-50</span>
            <span class="tech-badge">LSTM</span>
            <span class="tech-badge">Computer Vision</span>
        </div>
        <p class="footer-text">
            Vision AI Studio © 2026 • Built with deep learning excellence
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
