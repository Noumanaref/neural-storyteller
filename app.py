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
# 1. CLASS DEFINITIONS (UPDATED TO MATCH CHECKPOINT DIMENSIONS)
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
# 2. CUSTOM CSS FOR CLEAN UI
# ------------------------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main Header */
        .main-header {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: white;
        }
        
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            letter-spacing: 2px;
        }
        
        /* Hide file uploader details */
        .stFileUploader > div > div > div > small {
            display: none !important;
        }
        
        .stFileUploader > div > div > div > div:nth-child(2) {
            display: none !important;
        }
        
        .stFileUploader > div > div > div > div:nth-child(3) {
            display: none !important;
        }
        
        /* Custom File Uploader */
        .upload-section {
            text-align: center;
            padding: 1rem 0 2rem 0;
        }
        
        /* Browse button styling */
        .stFileUploader > div > button {
            background: white !important;
            color: #667eea !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            padding: 0.75rem 2.5rem !important;
            border: none !important;
            border-radius: 50px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
            margin: 0 auto !important;
        }
        
        .stFileUploader > div > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 30px rgba(0,0,0,0.3) !important;
        }
        
        /* Image Container */
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 1rem 0;
        }
        
        .image-container img {
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            max-width: 90%;
            max-height: 500px;
            object-fit: contain;
        }
        
        /* Generate Button */
        .stButton > button {
            background: white !important;
            color: #667eea !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            padding: 0.75rem 2.5rem !important;
            border: none !important;
            border-radius: 50px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            width: auto !important;
            min-width: 250px !important;
            margin: 1rem auto !important;
            display: block !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 30px rgba(0,0,0,0.3) !important;
        }
        
        /* Caption Box */
        .caption-box {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            animation: slideIn 0.5s ease;
        }
        
        .caption-text {
            font-size: 1.5rem;
            font-weight: 500;
            line-height: 1.6;
            text-align: center;
            color: #333;
        }
        
        .caption-label {
            font-size: 0.9rem;
            color: #667eea;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            text-align: center;
            font-weight: 600;
        }
        
        /* Confidence Score */
        .confidence {
            text-align: center;
            margin-top: 1rem;
            color: #666;
            font-size: 0.9rem;
        }
        
        /* Download Link */
        .download-link {
            text-align: center;
            margin-top: 1rem;
        }
        
        .download-link a {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            text-decoration: none;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            display: inline-block;
        }
        
        .download-link a:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        /* Hide all other elements */
        .stProgress > div > div > div > div {
            display: none;
        }
        
        /* Center everything */
        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. SETUP & CACHING
# ------------------------------------------------------------------

DEVICE = torch.device('cpu')

@st.cache_resource
def load_resources():
    """Load all necessary resources with proper error handling"""
    
    # 1. Load Vocabulary
    try:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ vocab.pkl not found. Please ensure it's in the correct directory.")
        return None, None, None

    # 2. Initialize Model with checkpoint dimensions
    try:
        embed_size = 512
        hidden_size = 712
        vocab_size = 7737
        num_layers = 5
        
        model = Seq2Seq(embed_size, hidden_size, vocab_size, num_layers)
        
        # 3. Load State Dict
        state_dict = torch.load('caption_model.pth', map_location=DEVICE)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # 4. Load ResNet Feature Extractor
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        feature_extractor.eval()
        
        return model, feature_extractor, vocab
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Image Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------

def generate_caption(model, feature_extractor, vocab, image, max_length=20):
    """Generate caption for an image"""
    
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
        
        return caption, attention_weights

def get_download_link(caption):
    """Generate download link for the generated caption"""
    href = f'<a href="data:file/txt;base64,{base64.b64encode(caption.encode()).decode()}" download="caption.txt">📥 Download Caption</a>'
    return href

# ------------------------------------------------------------------
# 5. MAIN STREAMLIT APP
# ------------------------------------------------------------------

def main():
    # Load custom CSS
    load_css()
    
    # Simple Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📸 Neural Storyteller</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    model, feature_extractor, vocab = load_resources()
    
    if model is None or feature_extractor is None or vocab is None:
        st.error("⚠️ Please ensure both 'vocab.pkl' and 'caption_model.pth' are in the application directory.")
        return
    
    # Simple file uploader
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Display image centered
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_column_width=False, width=None)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                generate_button = st.button("✨ Generate Caption", key="generate")
            
            if generate_button:
                with st.spinner("✨ Generating caption..."):
                    # Generate caption
                    caption_words, attention_weights = generate_caption(
                        model, feature_extractor, vocab, image
                    )
                    
                    caption = " ".join(caption_words)
                    
                    # Display caption
                    st.markdown(f"""
                    <div class="caption-box">
                        <div class="caption-label">📝 CAPTION</div>
                        <div class="caption-text">"{caption}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence score (small)
                    avg_confidence = np.mean(attention_weights) if attention_weights else 0
                    st.markdown(f"""
                    <div class="confidence">
                        Confidence: {avg_confidence:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download link
                    st.markdown(
                        f'<div class="download-link">{get_download_link(caption)}</div>',
                        unsafe_allow_html=True
                    )
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()