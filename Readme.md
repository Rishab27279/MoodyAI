# 🎭 Moody.AI - Multimodal Sentiment Analysis System

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-rishab27279%2Fmoody--ai-blue)](https://hub.docker.com/r/rishab27279/moody-ai)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A cutting-edge multimodal AI system that analyzes emotions from video content using computer vision, audio processing, and natural language processing. **Moody.AI** combines state-of-the-art deep learning models to provide comprehensive sentiment analysis with an intuitive web interface.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

    docker run -p 8501:8501 rishab27279/moody-ai

Navigate to `http://localhost:8501` and start analyzing emotions in your videos!

### Option 2: Local Installation

    git clone https://github.com/Rishab27279/MoodyAI.git
    cd MoodyAI
    pip install -r requirements.txt


## 🎯 Key Features

- **🎥 Multimodal Analysis**: Simultaneous processing of vision, audio, and text
- **🧠 Advanced AI Models**: DINOv2, Wav2Vec2, DistilBERT, and Whisper integration
- **📊 Real-time Results**: Interactive emotion classification with confidence scores
- **🎨 Beautiful UI**: Modern glass morphism design with animated visualizations
- **🐳 Production Ready**: Containerized deployment with Docker
- **📈 SOTA Performance**: 61% accuracy on challenging MELD dataset

## 🏗️ System Architecture

                    Video Input
                          ↓
                    Preprocessing
                          ↓
                    Multimodal AI
                          ↓
            ┌────────┐ ┌───────┐ ┌─────────────┐
            │ Vision │ │ Audio │ │ Text        │
            │ Frames │ │ Stream│ │ Transcripts │
            └────────┘ └───────┘ └─────────────┘
                        ↓ ↓ ↓
                  DINOv2 Wav2Vec2 + DistilBERT
                  ViT-B14 Whisper
                        ↓ ↓ ↓
            └─────────────┼───────────────────┘
                          ↓
                    Fusion Model
                    Cross-Attention
                          ↓
                    Emotion Prediction
                      (5 Classes)


## 🎬 Video Processing Pipeline

### **Input Processing**
The system accepts video files (MP4, MOV, MKV) up to 200MB and processes them through a sophisticated multimodal pipeline:

### **1. 🎥 Vision Processing**

    Frame Extraction (from app.py)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps)) # ~1 frame

    frames = []
    frame_count = 0
    while cap.read():
    ret, frame = cap.re
    d() if frame_count % frame_interv
    l == 0: # Face detec
    ion using CVLib face_detected
    cv2.detect_face(
    rame) if face_detected:
    faces, confidences = fac
    _detected # Process highest conf
    dence face

Typical output: 30-60 face frames per video (depending on duration)


**Vision Pipeline Details:**
- **Sampling Rate**: ~1 frame per second from original video
- **Face Detection**: CVLib for robust facial region extraction
- **Preprocessing**: Resize to 518×518, ImageNet normalization
- **Feature Extraction**: DINOv2 ViT-B/14 produces 768-dimensional embeddings
- **Compression**: Vision features compressed to 256 dimensions (noise reduction breakthrough)

### **2. 🔊 Audio Processing**

    Audio Extraction and Processing
    def extract_audio_features(video_path):
# Extract audio using MoviePy
    audio = AudioFileClip(video_path)
    audio_array = audio.to_soundarray(fps=16000) # 16kHz sampling

# Speech-to-Text with Whisper
    transcript = whisper.transcribe(audio_array, language="auto")

# Audio feature extraction with Wav2Vec2

    audio_features = wav2vec2_model(audio_array)  # 768-dim features

    return transcript["text"], audio_features


**Audio Pipeline Details:**
- **Extraction**: MoviePy converts video to 16kHz WAV audio
- **Speech-to-Text**: OpenAI Whisper (base/small models) with multilingual support
- **Feature Extraction**: Wav2Vec2-base-superb-er for emotional audio representations
- **Text Processing**: Automatic language detection and transcription
- **Output**: Both transcript text and 768-dimensional audio embeddings

### **3. 📝 Text Processing**

    Text Feature Extraction
    def extract_text_features(transcript):

# Tokenization with DistilBERT tokenizer

    inputs = tokenizer(transcript,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt")

# Feature extraction with fine-tuned DistilBERT

    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        # Hidden state averaging for sentence representation
        text_features = outputs.last_hidden_state.mean(dim=1)

    return text_features  # 768-dimensional embeddings


**Text Pipeline Details:**
- **Input**: Whisper transcription output
- **Tokenization**: DistilBERT tokenizer with BERT vocabulary
- **Processing**: Maximum sequence length of 512 tokens
- **Feature Extraction**: Fine-tuned DistilBERT for emotion classification
- **Output**: 768-dimensional contextualized text embeddings

### **4. 🔗 Multimodal Fusion**

    Trimodal Fusion Architecture
    class TrimodalFusionModel(nn.Module):
    def init(self):
    super().init()
    self.vision_compress = nn.Linear(768, 256) # Noise reduction
    self.cross_attention = CrossAttentionLayer(768)
    self.fusion_layers = nn.Sequential(
    nn.Linear(768 + 768 + 256, 512),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.Dropout(0.2),
    nn.Linear(128, 5) # 5 emotion classes
    )

    def forward(self, text_feat, audio_feat, vision_feat):
        # Vision compression (breakthrough technique)
        vision_compressed = self.vision_compress(vision_feat)
        
        # Cross-modal attention
        text_attended = self.cross_attention(text_feat, audio_feat, vision_compressed)
        audio_attended = self.cross_attention(audio_feat, text_feat, vision_compressed)
        
        # Feature concatenation and prediction
        fused = torch.cat([text_attended, audio_attended, vision_compressed], dim=1)
        emotion_logits = self.fusion_layers(fused)
        
        return F.softmax(emotion_logits, dim=1)


## 🧠 Model Architecture & Performance

### **Individual Model Components**

| Component | Model | Performance | Details |
|-----------|-------|-------------|---------|
| **Vision** | DINOv2 ViT-B/14 | Feature Extractor ~ 30% | Self-supervised visual representations |
| **Audio** | Wav2Vec2-base-superb-er | 32% on MELD | Fine-tuned for emotion recognition |
| **Text** | DistilBERT-base-uncased | 50% on MELD | Custom emotion classification head |
| **Speech** | OpenAI Whisper | Transcription | Multilingual speech-to-text |

### **Fusion Model Evolution**

| Architecture | Modalities | Accuracy | Key Innovation |
|-------------|------------|----------|----------------|
| **Text Only** | Text | ~50% | Baseline DistilBERT |
| **Bimodal** | Audio + Text | 56-57% | Cross-attention fusion |
| **Initial Trimodal** | All Three | 45-49% | Vision noise issues |
| **Compressed Trimodal** | All Three | **61%** | Vision feature compression |

### **SOTA Comparison on MELD Dataset**
- **Our Result**: **61% accuracy**
- **DialogueRNN (Poria et al., ACL 2019)**: 60.25% F1-score
- **COSMIC (Ghosal et al., EMNLP 2020)**: Previous SOTA benchmark
- **MERC-PLTAF (Nature 2025)**: Recent multimodal approach

## 📊 Experimental Results

### **Dataset Performance Analysis**

#### **MELD Dataset (Primary Evaluation)**
- **Challenge**: Natural conversational videos with complex emotions
- **Our Performance**: **61% accuracy** (competitive with SOTA)
- **Breakthrough**: Vision feature compression eliminated noise from dynamic facial expressions

#### **RAVDESS Dataset (Transfer Learning Study)**
- **Audio Model**: **92% accuracy** on controlled expressions
- **Transfer to MELD**: 17-20% (highlighting domain gap challenges)
- **Insight**: Controlled vs. naturalistic emotion expression differences

### **Ablation Studies**
1. **Vision Feature Compression**: 45-49% → 61% (+12-16% improvement)
2. **Fusion Architecture Comparison**:
   - Base Attention: 53-55%
   - Cross-Attention: 55-57%
   - GRU + Cross-Attention: 56-57%
   - **Compressed Trimodal**: **61%**

## 🎨 User Interface

### **Streamlit Web Application**
- **Framework**: Streamlit with custom CSS styling
- **Design**: Glass morphism UI with animated gradient backgrounds
- **Features**:

      - Drag-and-drop video upload
      - Real-time processing indicators
      - Interactive emotion visualization
      - Confidence score displays
      - Probability distribution charts

### **Processing Modes**
- **Lightweight**: Fast processing for quick analysis
- **Balanced**: Optimal speed-accuracy tradeoff
- **High Fidelity**: Maximum accuracy for detailed analysis

## 🛠️ Technical Implementation

### **Dependencies**

    torch>=1.13.0
    transformers>=4.21.0
    streamlit>=1.24.0
    opencv-python-headless>=4.6.0
    librosa>=0.9.2
    whisper>=1.1.10
    timm>=0.6.7
    moviepy>=1.0.3
    Pillow>=9.2.0
    numpy>=1.21.0


### **Hardware Requirements**
- **Minimum**: 8GB RAM, CPU-only processing
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Optimal**: 32GB RAM, Modern GPU (RTX 3060+)

### **Model Specifications**
- **Text Model**: `best_text_only_model.pth` (262 MB)
- **Fusion Model**: `best_trimodal_model.pth` (409 MB)
- **Total Memory**: ~4GB during inference
- **Processing Time**: 10-30 seconds per video (GPU)

## 📁 Repository Structure


    MoodyAI/
    ├── 📁 Audio_Part/ # Audio model training & evaluation
    │ ├── 📁 Audio_Training/ # Training scripts and utilities
    │ ├── 📁 Confusion_Matrix(s)/ # Progressive training results
    │ ├── 📁 finetuned-audio-model-5class-emotion-model/
    │ ├── 📁 ravdess-wav2vec2-emotion-model/
    │ ├── 🖼️ Audio_Conf_Matrix.png # Final confusion matrix
    │ └── 📄 evaluation_results.json # Performance metrics
    ├── 📁 Text_Part/ # Text processing & DistilBERT
    │ ├── 📁 distilbert_sentiment_model/# Fine-tuned model & checkpoints
    │ ├── 📁 Training_files/ # Training utilities
    │ ├── 🖼️ Text_Conf_Matrix.png # Performance visualization
    │ └── 📄 evaluation_results.json # Detailed metrics
    ├── 📁 Vision_Part/ # Computer vision experiments
    │ ├── 📁 fer2013/ # FER2013 dataset experiments
    │ ├── 🗂️ fer2013_upper_face_.npz # Modified datasets
    │ ├── ⚙️ final_model_weights.h5 # Vision model weights
    │ └── 🖼️ Vision_Conf_Matrix.png # Results visualization
    ├── 📁 Fusion-Modal_Part/ # Multimodal fusion system
    │ ├── 📁 Models/ # All trained models
    │ ├── 📁 Training_Codes/ # Fusion training scripts
    │ ├── 🖼️ confusion_matrix_.png # Bimodal vs Trimodal results
    │ └── 📄 evaluation_results.json # Final performance metrics
    ├── 📁 Multi-Model_Files/ # Combined model utilities
    ├── 🐍 app.py # Main Streamlit application
    ├── 🐳 Dockerfile # Container configuration
    ├── 📋 requirements.txt # Python dependencies
    └── 📖 README.md # This documentation


## 🔬 Research Contributions

### **Technical Innovations**
1. **Vision Feature Compression**: Novel noise reduction technique for conversational video emotion recognition
2. **Progressive Multimodal Architecture**: Systematic evolution from bimodal to trimodal fusion
3. **Cross-Modal Attention**: Advanced fusion mechanisms for multimodal integration
4. **Domain Gap Analysis**: Comprehensive study of transfer learning challenges in emotion recognition

### **Key Research Insights**
- **Vision Compression Efficacy**: Dimensional reduction as an effective noise filtering technique
- **Modality Synergy**: Proper vision integration amplifies audio-text performance significantly
- **Dataset Domain Challenges**: MELD's naturalistic videos require specialized approaches
- **Architecture Evolution**: Systematic approach to multimodal model development

## 🚀 Deployment Options

### **Docker Deployment (Production)**

    Pull and run pre-built image
    docker pull rishab27279/moody-ai
    docker run -p 8501:8501 rishab27279/moody-ai

    Or build locally
    docker build -t moody-ai .
    docker run -p 8501:8501 moody-ai


### **Local Development**


    Clone and setup
    git clone https://github.com/Rishab27279/MoodyAI.git
    cd MoodyAI
    python -m venv moody-env
    source moody-env/bin/activate # Windows: moody-env\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py



### **Cloud Deployment**
- **Streamlit Cloud**: Direct deployment from GitHub
- **AWS/Azure**: Container deployment with GPU support
- **Google Cloud Run**: Serverless container deployment

## 📈 Performance Benchmarks

### **Processing Speed**
- **CPU Only**: 30-60 seconds per video
- **GPU (RTX 3060)**: 10-20 seconds per video
- **High-end GPU**: 5-10 seconds per video

### **Accuracy Metrics**
- **Overall Accuracy**: 61% on MELD dataset
- **Precision**: 0.58-0.64 across emotion classes
- **Recall**: 0.55-0.67 across emotion classes
- **F1-Score**: 0.59-0.63 (macro average)

### **Memory Usage**
- **Model Loading**: ~4GB RAM
- **Video Processing**: +2-3GB per video
- **Peak Usage**: ~6-7GB for large videos

## 🤝 Contributing

I warmly welcome contributions to improve **Moody.AI**! Areas for contribution:

- **Model Improvements**: Better fusion architectures
- **Dataset Integration**: Additional emotion datasets
- **UI Enhancements**: Improved visualizations
- **Performance Optimization**: Faster inference
- **Documentation**: Code documentation and tutorials

## 🙏 Acknowledgments

- **Hugging Face**: Transformers library and model hub
- **OpenAI**: Whisper speech recognition model
- **Meta AI**: DINOv2 vision transformer
- **MELD Dataset**: Multimodal emotion recognition benchmark
- **Streamlit**: Web application framework}


## 🔗 Links

- **Docker Hub**: [rishab27279/moody-ai](https://hub.docker.com/r/rishab27279/moody-ai)
- **Portfolio**: rishab27279.github.io

---

**Built with ❤️ for advancing multimodal AI research community.**

