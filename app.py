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
# 2. FIXED CSS STYLING (SIMPLIFIED FOR COMPATIBILITY)
# ------------------------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600;700&display=swap');
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-attachment: fixed;
        }
        
        /* Main Container - CRITICAL FIX */
        .main .block-container {
            padding: 2rem 1rem 4rem 1rem;
            max-width: 1400px;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-family: 'Playfair Display', serif;
        }
        
        p, div, span, label {
            color: rgba(255,255,255,0.9) !important;
            font-family: 'DM Sans', sans-serif;
        }
        
        /* Hero Header */
        .hero-header {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
        }
        
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            font-weight: 800;
            color: #ffffff !important;
            margin-bottom: 1rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
            color: rgba(255,255,255,0.8) !important;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        
        /* Premium Cards */
        .premium-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff !important;
            margin-bottom: 1rem;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(139,92,246,0.1);
            border: 2px dashed rgba(139,92,246,0.5);
            border-radius: 12px;
            padding: 2rem 1rem;
        }
        
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small {
            color: rgba(255,255,255,0.9) !important;
        }
        
        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        
        /* Image Preview */
        .image-preview-container {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.2);
            margin: 1rem 0;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            width: 100%;
            font-size: 0.95rem !important;
        }
        
        .stButton > button:hover {
            box-shadow: 0 4px 20px rgba(139,92,246,0.5);
            transform: translateY(-2px);
        }
        
        /* Secondary button styling */
        .stButton > button[kind="secondary"] {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
        }
        
        /* Caption Result */
        .result-container {
            background: rgba(139,92,246,0.15);
            border: 1px solid rgba(139,92,246,0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .caption-text {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            line-height: 1.6;
            color: #ffffff !important;
            font-style: italic;
        }
        
        /* Metrics */
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            padding: 1rem;
            border-radius: 12px;
        }
        
        div[data-testid="metric-container"] label {
            color: rgba(255,255,255,0.7) !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
        }
        
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }
        
        /* Download Button */
        .download-btn {
            display: block;
            width: 100%;
            text-align: center;
            background: rgba(255,255,255,0.1);
            color: #ffffff !important;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.2);
            margin-top: 1rem;
        }
        
        .download-btn:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(139,92,246,0.5);
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 3rem 1.5rem;
            color: rgba(255,255,255,0.5) !important;
        }
        
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            opacity: 0.4;
        }
        
        /* Footer */
        .custom-footer {
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }
        
        .tech-badge {
            display: inline-block;
            background: rgba(139,92,246,0.2);
            color: rgba(255,255,255,0.9) !important;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            margin: 0.25rem;
            border: 1px solid rgba(139,92,246,0.3);
        }
        
        .footer-text {
            color: rgba(255,255,255,0.5) !important;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
        
        /* Caption text color fix */
        .stMarkdown p {
            color: rgba(255,255,255,0.9) !important;
        }
        
        /* Spinner color */
        .stSpinner > div {
            border-color: #8b5cf6 !important;
            border-right-color: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. SETUP & CACHING
# ------------------------------------------------------------------

DEVICE = torch.device('cpu')

@st.cache_resource(show_spinner="🔮 Loading AI Models...")
def load_resources():
    try:
        # Try to load vocabulary
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ vocab.pkl not found in current directory")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading vocab.pkl: {str(e)}")
        return None, None, None

    try:
        embed_size = 512
        hidden_size = 712
        vocab_size = 7737
        num_layers = 5
        
        # Load caption model
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
        
        # Load ResNet feature extractor
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        feature_extractor.eval()
        
        return model, feature_extractor, vocab
        
    except FileNotFoundError:
        st.error("❌ caption_model.pth not found in current directory")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
            if pred_word_idx == vocab.stoi["<end>"]: 
                break
                
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
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'file_size_mb' not in st.session_state:
        st.session_state.file_size_mb = 0

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
    <div class="hero-header">
        <h1 class="hero-title">Vision AI Studio</h1>
        <p class="hero-subtitle">
            Transform images into intelligent narratives using state-of-the-art deep learning.
            Our advanced neural architecture combines ResNet-50 feature extraction with LSTM sequence generation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models loaded
    if model is None or vocab is None:
        st.error("⚠️ **System Offline**: Critical resources missing. Please ensure 'vocab.pkl' and 'caption_model.pth' are in the application directory.")
        st.info(f"📁 Current directory: {os.getcwd()}")
        st.info(f"📋 Files in directory: {os.listdir('.')}")
        return
    
    # ========== MAIN CONTENT ==========
    col_left, col_right = st.columns([1.3, 1], gap="large")
    
    # LEFT COLUMN - Upload & Input
    with col_left:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📸 Image Upload</div>', unsafe_allow_html=True)
        
        # Only show uploader if no file is uploaded
        if st.session_state.current_image_name is None:
            uploaded_file = st.file_uploader(
                "Drop your image here or click to browse",
                type=['jpg', 'jpeg', 'png'],
                help="Maximum file size: 200MB • Formats: JPG, JPEG, PNG",
                label_visibility="visible",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                # Store in session state
                st.session_state.current_image_name = uploaded_file.name
                st.session_state.current_image = Image.open(uploaded_file).convert('RGB')
                st.session_state.file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.rerun()
        
        # Display image if we have one
        if st.session_state.current_image is not None:
            image = st.session_state.current_image
            
            # Display image preview
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image metadata
            st.caption(f"📊 {st.session_state.current_image_name} • {image.size[0]}×{image.size[1]}px • {st.session_state.file_size_mb:.2f} MB")
            
            # Buttons row
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                # Generate button
                if st.button("🚀 Generate Caption", use_container_width=True):
                    with st.spinner("🔮 Analyzing visual features..."):
                        caption_words, att_weights, proc_time = generate_caption(
                            model, feature_extractor, vocab, image
                        )
                        st.session_state.generated_caption = " ".join(caption_words)
                        st.session_state.confidence = np.mean(att_weights) if att_weights else 0
                        st.session_state.processing_time = proc_time
                        st.rerun()
            
            with btn_col2:
                # Upload different image button
                if st.button("📁 Change Image", use_container_width=True):
                    st.session_state.current_image_name = None
                    st.session_state.current_image = None
                    st.session_state.generated_caption = None
                    st.session_state.confidence = None
                    st.session_state.processing_time = None
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
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✨ Generated Caption</div>', unsafe_allow_html=True)
        
        if st.session_state.generated_caption is not None:
            # Display caption
            st.markdown(f"""
            <div class="result-container">
                <p class="caption-text">{st.session_state.generated_caption.capitalize()}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Confidence", f"{st.session_state.confidence:.1%}")
            
            with metric_col2:
                word_count = len(st.session_state.generated_caption.split())
                st.metric("Tokens", f"{word_count}")
            
            with metric_col3:
                st.metric("Speed", f"{st.session_state.processing_time:.2f}s")
            
            # Download button
            st.markdown(get_download_link(st.session_state.generated_caption), unsafe_allow_html=True)
            
        else:
            # Empty state
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">💭</div>
                <p>Caption will appear here after processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== FOOTER ==========
    st.markdown("""
    <div class="custom-footer">
        <div>
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
