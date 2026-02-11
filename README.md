# 📸 Neural Storyteller - Image Captioning

<div align="center">

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://neural-storyteller-nunrymehmshrtydpmgbfo6.streamlit.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Dataset-Flickr30k-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/adityajn105/flickr30k)

**Generate natural language descriptions for any image using Seq2Seq deep learning**

</div>

---

## 🚀 **Live Demo**
**👉 [Click Here to Try the App](https://neural-storyteller-nunrymehmshrtydpmgbfo6.streamlit.app)** 👈

Upload any image and get an AI-generated caption in seconds!

---

## 📌 **Quick Overview**

| **Aspect** | **Details** |
|-----------|------------|
| **Task** | Image Captioning (Seq2Seq) |
| **Dataset** | Flickr30k (31k images, 158k captions) |
| **Model** | ResNet50 Encoder + LSTM Decoder |
| **Accuracy** | BLEU-4: 13.5%, F1: 0.33 |
| **Deployment** | Streamlit Cloud (Free) |
| **Training** | Kaggle T4x2 Dual GPU |

---

## 🏗 **Model Architecture (Simple)**

Image → ResNet50 → 2048-dim → Linear(512) → LSTM(512) → Caption


**Hyperparameters:**
- Embed Size: 256 | Hidden Size: 512 | Layers: 2 | Dropout: 0.3
- Vocab Size: 7,737 | Epochs: 25 | Batch: 64 | LR: 1e-4

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/Noumanaref/neural-storyteller.git
cd neural-storyteller

# 2. Install
pip install streamlit torch torchvision pillow numpy rich

# 3. Run
streamlit run app.py
```

That's it! The app will open in your browser.


## 📊 **Results**

| **Sample Image** | **Generated Caption** |
|-----------------|----------------------|
| 🐕 Dog running | "a brown dog runs on grass" |
| ⚽ Kids playing | "children playing soccer on field" |
| 🏄 Surfer | "a surfer rides a big wave" |

**Metrics:** BLEU-4: 13.5% | Precision: 0.41 | Recall: 0.35 | F1: 0.33

---

## 🛠 **Tech Stack**

| **Category** | **Technologies** |
|-------------|-----------------|
| **Deep Learning** | PyTorch, TorchVision, ResNet50, LSTM |
| **Web App** | Streamlit, PIL, NumPy |
| **Deployment** | GitHub, Streamlit Cloud |
| **Training** | Kaggle (T4x2 GPU) |

---

## 📁 **Project Structure**

```
neural-storyteller/
├── app.py                 # Main Streamlit application
├── caption_model.pth      # Trained model weights (117MB)
├── vocab.pkl             # Vocabulary file
├── requirements.txt      # Dependencies
├── .streamlit/           # Streamlit configuration
│   └── config.toml      
└── README.md            # This file
```

---

## 🎯 **Key Features**

✅ **Live Web App** - Deployed and accessible globally  
✅ **Real-time Inference** - Generate captions in seconds  
✅ **Beautiful UI** - Gradient design, smooth animations  
✅ **Download Captions** - Save results as .txt files  
✅ **Confidence Score** - See model prediction confidence  
✅ **Mobile Responsive** - Works on all devices  

---

## 📈 **Performance**

| **Epoch** | **Train Loss** | **Val Loss** | **BLEU-4** |
|-----------|---------------|-------------|------------|
| 10 | 3.68 | 3.92 | 5.3% |
| 20 | 2.95 | 3.41 | 11.2% |
| 25 | 2.78 | 3.32 | 13.5% |

*Note: State-of-the-art on Flickr30k is 22-25% BLEU-4 with 100+ epochs*

---

## 🚨 **Common Issues & Fixes**

| **Issue** | **Solution** |
|----------|------------|
| PyTorch not installing | Use Python 3.11, `pip install torch==2.0.0` |
| Model file too large | Use Git LFS or Google Drive hosting |
| Config.toml error | Keep only `[server]` section, no theme |
| Out of memory | Reduce batch size, use CPU-only PyTorch |
