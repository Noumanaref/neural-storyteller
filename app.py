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

# ------------------------------------------------------------------
# 0. PAGE CONFIGURATION (MUST BE FIRST)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Neural Storyteller | AI Image Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
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
# 2. ENTERPRISE CSS STYLING
# ------------------------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Typography & Colors */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1E293B; /* Slate 800 */
        }
        
        /* Background */
        .stApp {
            background-color: #F8FAFC; /* Slate 50 */
        }
        
        /* Hide Default Streamlit Elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Typography */
        h1, h2, h3 {
            color: #0F172A; /* Slate 900 */
            font-weight: 600;
            letter-spacing: -0.025em;
        }
        
        /* Card Layouts */
        .custom-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #E2E8F0;
            margin-bottom: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .custom-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        }

        /* Metrics Styling */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            padding: 16px 24px;
            border-radius: 12px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }

        /* Buttons */
        .stButton > button {
            background-color: #1E3A8A; /* Blue 900 */
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            width: 100%;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #1E40AF; /* Blue 800 */
            color: #FFFFFF;
            box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.3);
        }
        
        /* Secondary Download Button */
        .download-btn {
            display: block;
            width: 100%;
            text-align: center;
            background-color: #F1F5F9;
            color: #334155;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
            border: 1px solid #CBD5E1;
            transition: all 0.2s ease;
            margin-top: 12px;
        }
        .download-btn:hover {
            background-color: #E2E8F0;
            color: #0F172A;
        }
        
        /* Image Preview Box */
        .image-preview {
            border: 2px dashed #CBD5E1;
            border-radius: 12px;
            padding: 8px;
            background: #FFFFFF;
        }
        
        /* Result Caption Text */
        .result-caption {
            font-size: 1.25rem;
            line-height: 1.75rem;
            color: #0F172A;
            font-weight: 500;
            padding: 20px;
            background-color: #F8FAFC;
            border-left: 4px solid #1E3A8A;
            border-radius: 0 8px 8px 0;
            margin: 16px 0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            font-size: 0.875rem;
            color: #64748B;
            padding-top: 32px;
            margin-top: 48px;
            border-top: 1px solid #E2E8F0;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. SETUP & CACHING
# ------------------------------------------------------------------

DEVICE = torch.device('cpu')

@st.cache_resource(show_spinner="Loading AI Models...")
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
        
        return caption, attention_weights

def get_download_link(caption):
    b64 = base64.b64encode(caption.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="generated_caption.txt" class="download-btn">⬇️ Download Caption (.txt)</a>'

def init_session_state():
    if 'generated_caption' not in st.session_state:
        st.session_state.generated_caption = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None
    if 'current_image_name' not in st.session_state:
        st.session_state.current_image_name = None

# ------------------------------------------------------------------
# 5. MAIN APPLICATION
# ------------------------------------------------------------------

def main():
    load_css()
    init_session_state()
    
    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103130.png", width=60) # Placeholder icon
        st.title("Neural Storyteller")
        st.markdown("---")
        
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload** a clear image (JPG/PNG).
        2. Wait for the preview to load.
        3. Click **Generate Caption**.
        4. Review the AI output and confidence metrics.
        """)
        
        st.markdown("### ⚙️ System Status")
        model, feature_extractor, vocab = load_resources()
        if model and vocab:
            st.success("✅ Models Online")
            st.caption(f"Vocabulary Size: {len(vocab)} words")
        else:
            st.error("❌ System Offline (Missing files)")

        st.markdown("---")
        st.caption("v2.0.0 Enterprise Edition | © 2026")

    # --- MAIN CONTENT AREA ---
    st.markdown("## 📷 AI Image Captioning Module")
    st.markdown("Upload imagery to automatically generate descriptive, contextual narratives using deep learning.")
    
    if model is None or vocab is None:
        st.warning("⚠️ Critical resources are missing. Please verify 'vocab.pkl' and 'caption_model.pth' exist in the root directory.")
        return

    # Use a two-column layout for desktop
    col_left, col_right = st.columns([1.2, 1], gap="large")
    
    with col_left:
        st.markdown("#### 1. Input Image")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], help="Max file size 200MB.")
        
        if uploaded_file is not None:
            # Check if new image uploaded, reset state if true
            if st.session_state.current_image_name != uploaded_file.name:
                st.session_state.generated_caption = None
                st.session_state.confidence = None
                st.session_state.current_image_name = uploaded_file.name
                
            image = Image.open(uploaded_file).convert('RGB')
            st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action Button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Analyze & Generate Sequence", use_container_width=True):
                with st.spinner("Processing visual features & querying LSTM..."):
                    caption_words, att_weights = generate_caption(model, feature_extractor, vocab, image)
                    st.session_state.generated_caption = " ".join(caption_words)
                    st.session_state.confidence = np.mean(att_weights) if att_weights else 0
        else:
            # Empty State
            st.info("💡 Awaiting visual input. Please upload an image to begin.")
            
    with col_right:
        st.markdown("#### 2. Generated Output")
        
        if st.session_state.generated_caption is not None:
            # Display Result in Custom Styling
            st.markdown(f'<div class="result-caption">{st.session_state.generated_caption.capitalize()}.</div>', unsafe_allow_html=True)
            
            # Display Metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="System Confidence", value=f"{st.session_state.confidence:.2%}")
            with m_col2:
                word_count = len(st.session_state.generated_caption.split())
                st.metric(label="Sequence Length", value=f"{word_count} tokens")
                
            # Export Action
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 3. Export Data")
            st.markdown(get_download_link(st.session_state.generated_caption), unsafe_allow_html=True)
            
        else:
            # Empty State for Right Column
            st.markdown("""
            <div class="custom-card" style="text-align: center; color: #94A3B8; padding: 40px 20px;">
                <h3 style="color: #CBD5E1; font-size: 3rem;">📝</h3>
                <p>Output will appear here after analysis.</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">Developed for advanced computer vision workflows. Requires GPU-accelerated environments for batch inference.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
