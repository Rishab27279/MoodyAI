import os
import gc
import math
import tempfile
import streamlit as st
import numpy as np
import torch
import timm
import librosa
import cv2
import cvlib as cv

from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Wav2Vec2ForSequenceClassification
from model_definition import TrimodalFusionModel

# -------------------- App config --------------------
st.set_page_config(page_title='Moody.AI — Multimodal Sentiment', layout='centered')

# -------------------- UI Theming & Animations --------------------
def inject_dynamic_background():
    st.markdown("""
    <style>
    /* Animated gradient background */
    .stApp {
        background: radial-gradient(1200px 800px at 20% -10%, #16344a 0%, rgba(22,52,74,0) 60%),
                    radial-gradient(1000px 600px at 80% 110%, #0f2027 0%, rgba(15,32,39,0) 60%),
                    linear-gradient(120deg, #0f2027, #203a43, #2c5364);
        background-size: 600% 600%;
        animation: gradientShift 18s ease-in-out infinite;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Dim line grid horizon (replaces dots) */
    .bg-lines {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
        perspective: 800px;
    }
    .bg-lines .grid {
        position: absolute;
        inset: -50% -100% -20% -100%;
        transform: rotateX(78deg) translateY(10%);
        transform-origin: center;
        color: rgba(255,255,255,0.08);
        opacity: 0.9;
    }
    /* Horizontal lines */
    .bg-lines .grid .hline {
        position: relative;
        width: 100%;
        height: 2px;
        background: currentColor;
        margin-top: 80px;
        animation: hmove 6s linear infinite;
    }
    @keyframes hmove {
        0% { transform: translateY(0); }
        100% { transform: translateY(80px); }
    }
    /* Vertical lines container */
    .bg-lines .grid .vwrap {
        position: absolute;
        top: 0; left: 50%;
        width: 0; height: 100%;
        display: flex;
        transform: translateX(-50%);
    }
    /* Vertical lines */
    .bg-lines .grid .vline {
        width: 2px; height: 100%;
        background: currentColor;
        margin: 0 40px;
        filter: blur(0.2px);
        animation: vfade 4s ease-in-out infinite alternate;
    }
    @keyframes vfade {
        from { opacity: 0.7; }
        to   { opacity: 0.35; }
    }

    /* Subtle glow on the horizon */
    .bg-lines .glow {
        position: absolute;
        left: 50%; bottom: 15%;
        width: 60vw; height: 60vw;
        transform: translateX(-50%);
        background: radial-gradient(closest-side, rgba(255,255,255,0.10), rgba(255,255,255,0));
        filter: blur(20px);
        opacity: 0.6;
        animation: pulse 5.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.55; transform: translateX(-50%) scale(1.00); }
        50%      { opacity: 0.75; transform: translateX(-50%) scale(1.04); }
    }

    /* Glass cards */
    .glass {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        border-radius: 16px;
        padding: 18px 22px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        position: relative;
        z-index: 1;
    }

    /* Overlay loader with car animation */
    #overlay-loader {
        position: fixed; inset: 0;
        display: none; align-items: center; justify-content: center;
        background: rgba(0,0,0,0.35);
        z-index: 9999;
    }
    .car {
        width: 140px; height: 72px; position: relative;
        animation: drive 2.8s ease-in-out infinite;
    }
    .car-body {
        width: 100%; height: 46px; background: linear-gradient(90deg, #ff6b6b, #feca57);
        border-radius: 12px; position: absolute; top: 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.25);
    }
    .headlight {
        position: absolute; right: -8px; top: 8px;
        width: 16px; height: 10px; border-radius: 6px;
        background: radial-gradient(circle at 0% 50%, #fff8 0%, #fff3 30%, transparent 70%);
        filter: blur(0.5px);
    }
    .wheel {
        width: 30px; height: 30px; background: #2d3436; border-radius: 50%;
        position: absolute; bottom: -10px;
        border: 5px solid #636e72;
        animation: spin 0.9s linear infinite;
    }
    .wheel.left { left: 18px; }
    .wheel.right { right: 18px; }
    .road {
        width: 240px; height: 4px; background: rgba(255,255,255,0.35);
        position: absolute; bottom: -4px; left: -60px;
        border-radius: 2px; animation: roadMove 1.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes drive {
        0% { transform: translateX(-22px); }
        50% { transform: translateX(22px); }
        100% { transform: translateX(-22px); }
    }
    @keyframes roadMove {
        0% { transform: translateX(0); }
        100% { transform: translateX(-90px); }
    }

    /* Primary button style */
    .primary-btn button, .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        border: 0; color: #1e272e; font-weight: 700;
        box-shadow: 0 6px 18px rgba(254,202,87,0.25);
    }
    .primary-btn button:hover, .stButton>button[kind="primary"]:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    /* Improve file uploader contrast on glass */
    .glass .uploadedFile { color: #fff; }
    </style>

    <!-- Dim line grid background structure -->
    <div class="bg-lines">
      <div class="grid">
        <!-- Horizontal lines (generate many via CSS or static repeats) -->
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <div class="hline"></div>
        <!-- Vertical lines wrapper -->
        <div class="vwrap">
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
          <div class="vline"></div>
        </div>
        <div class="glow"></div>
      </div>
    </div>

    <!-- Backend loading overlay -->
    <div id="overlay-loader">
        <div class="car">
            <div class="car-body"></div>
            <div class="headlight"></div>
            <div class="wheel left"></div>
            <div class="wheel right"></div>
            <div class="road"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_overlay_loader(show: bool):
    st.markdown(f"""
        <script>
        const el = window.parent.document.getElementById('overlay-loader');
        if (el) {{ el.style.display = '{'flex' if show else 'none'}'; }}
        </script>
    """, unsafe_allow_html=True)

# -------------------- Memory hygiene --------------------
def _free_cuda():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

def _unload_model(model):
    try:
        if model is not None:
            try:
                model.to('cpu')
            except Exception:
                pass
            del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------- Media preprocessing --------------------
def _extract_audio_wav(video_path):
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(
        wav_path, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None
    )
    clip.close()
    return wav_path

def _run_transcription(wav_path, prefer_device="auto", model_size="base", language=None):
    if prefer_device == "auto":
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"
    elif prefer_device in ("cuda", "cpu"):
        device_choice = prefer_device
    else:
        device_choice = "cpu"
    asr_model = whisper.load_model(model_size, device=device_choice)
    result = asr_model.transcribe(wav_path, language=language)
    text = result.get("text", "").strip()
    _unload_model(asr_model)
    _free_cuda()
    return text

def _persist_video(uploaded_video):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()
    video_path = tfile.name
    tfile.close()
    return video_path

# -------------------- Feature extraction --------------------
@torch.no_grad()
def _extract_text_features(text, tokenizer, text_model, device, max_len=128):
    inputs = tokenizer(
        text, return_tensors='pt', max_length=max_len, truncation=True, padding=True
    ).to(device)
    outputs = text_model.distilbert(**inputs, output_hidden_states=True)
    feats = outputs.hidden_states[-1].mean(dim=1)
    return feats  # [1, H]

@torch.no_grad()
def _extract_audio_features(wav_path, audio_model, device, sr=16000):
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    if audio.size == 0:
        audio = np.zeros(int(sr * 0.5), dtype=np.float32)
    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T]
    outputs = audio_model.wav2vec2(x, output_hidden_states=True)
    feats = outputs.last_hidden_state.mean(dim=1)  # [1, D]
    return feats

@torch.no_grad()
def _extract_vision_features(video_path, vision_model, device, max_seq=10):
    # DINOv2 ViT-B/14 expects 518x518 per model card
    target = 518
    vision_sequence = np.zeros((max_seq, 3, target, target), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 24

    frame_idx, count = 0, 0
    while cap.isOpened() and count < max_seq:
        ret, frame = cap.read()
        if not ret:
            break

        # sample ~1 fps
        if frame_idx % max(1, math.ceil(fps)) == 0:
            detection = cv.detect_face(frame)
            faces = []
            if isinstance(detection, tuple) and len(detection) == 2:
                faces, _ = detection
            elif isinstance(detection, (list, np.ndarray)):
                faces = detection

            if faces and len(faces) > 0:
                face_coords = faces[0] if isinstance(faces[0], (list, tuple, np.ndarray)) else faces
                if len(face_coords) >= 4:
                    x1, y1, x2, y2 = map(int, face_coords[:4])
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        face = frame[y1:y2, x1:x2]
                        if face.size > 0:
                            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            face_resized = cv2.resize(face_rgb, (target, target), interpolation=cv2.INTER_LINEAR)
                            face_norm = (face_resized / 255.0).astype(np.float32)
                            # Normalize to ImageNet stats (common for ViTs)
                            face_norm = (face_norm - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
                            vision_sequence[count] = np.transpose(face_norm, (2, 0, 1))
                            count += 1
        frame_idx += 1

    cap.release()
    vt = torch.from_numpy(vision_sequence).to(device)  # [T, 3, 518, 518]
    
    # CRITICAL FIX: Proper ViT feature extraction to avoid memory allocation issues
    feats = vision_model.forward_features(vt)  # Returns unpooled features
    
    # Handle different timm ViT output structures according to documentation
    if isinstance(feats, dict):
        # Some timm ViTs return dict with various keys
        if 'x_norm_clstoken' in feats:
            feats = feats['x_norm_clstoken']  # [T, hidden_dim]
        elif 'x' in feats:
            # If 'x' is [T, num_tokens, hidden_dim], take cls token (index 0)
            if feats['x'].dim() == 3:
                feats = feats['x'][:, 0]  # [T, hidden_dim]
            else:
                feats = feats['x']  # Already [T, hidden_dim]
        else:
            # Fallback: get first value and handle appropriately
            feats = next(iter(feats.values()))
            if feats.dim() == 3:  # [T, tokens, hidden]
                feats = feats.mean(dim=1)  # [T, hidden]
    else:
        # feats is a tensor
        if feats.dim() == 3:  # [T, num_tokens, hidden_dim]
            # For ViT, typically token 0 is CLS token
            feats = feats[:, 0]  # [T, hidden_dim] 
        # If already [T, hidden_dim], keep as is
    
    # Ensure we have [T, hidden_dim] and average over time
    filled = int((vt.abs().sum(dim=(1, 2, 3)) > 0).sum().item())
    if filled > 0:
        feats = feats[:filled].mean(dim=0, keepdim=True)  # [1, hidden_dim]
    else:
        feats = feats[:1]  # [1, hidden_dim]
    
    # Debug print to verify shape (remove in production)
    # print(f"Vision features shape: {feats.shape}")
    
    return feats

# -------------------- Model loading/unloading --------------------
def _load_inference_stack(device):
    print("--- Building Trimodal Cross-Attention Fusion Model ---")
    
    # Vision: vit_base_patch14_dinov2.lvd142m with fixed 518x518 input
    vision_model = timm.create_model(
        'vit_base_patch14_dinov2.lvd142m',
        pretrained=True,
        in_chans=3,
        img_size=518,  # critical to satisfy PatchEmbed assertion
    ).to(device)
    vision_model.eval()

    # Audio
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
        'superb/wav2vec2-base-superb-er', use_safetensors=True
    ).to(device)
    audio_model.eval()

    # Text
    text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', use_safetensors=True
    ).to(device)
    text_model.eval()

    # Fusion
    fusion_model = TrimodalFusionModel(num_classes=5, d_model=256, nhead=4, num_decoder_layers=2).to(device)
    fusion_model.load_state_dict(torch.load('MoodyAI.pth', map_location=device))
    fusion_model.eval()

    return fusion_model, vision_model, audio_model, text_tokenizer, text_model

def _unload_inference_stack(*models):
    for m in models:
        _unload_model(m)
    _free_cuda()

# -------------------- App --------------------
def main():
    inject_dynamic_background()

    st.title('😎 Moody.AI — Multimodal Sentiment')

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader('📤 Upload')
    video_file = st.file_uploader('Upload a video (mp4/mov/mkv)', type=['mp4', 'mov', 'mkv'])

    col1, col2 = st.columns(2)
    with col1:
        processing_profile = st.selectbox('🔧 Processing profile', ['Balanced', 'Lightweight', 'High fidelity'], index=0)
    with col2:
        language_hint = st.text_input('🌐 Language hint (optional, e.g., "en")', value='')
    st.markdown('</div>', unsafe_allow_html=True)

    start_col, _ = st.columns([1, 3])
    with start_col:
        go = st.button('🎬 Analyze with Moody.AI', type='primary', use_container_width=True)

    if go:
        show_overlay_loader(True)
        fusion_model = None
        vision_model = None
        audio_model = None
        text_model = None
        video_path = None
        wav_path = None
        
        try:
            if not video_file:
                st.error('❌ A video is required to proceed.')
                return

            # Map profile
            if processing_profile == 'Lightweight':
                asr_size, prefer_device = "tiny", "auto"
            elif processing_profile == 'High fidelity':
                asr_size, prefer_device = "small", "auto"
            else:
                asr_size, prefer_device = "base", "auto"

            with st.spinner('🎬 Preparing media…'):
                video_path = _persist_video(video_file)

            with st.spinner('🎵 Processing audio…'):
                wav_path = _extract_audio_wav(video_path)

            with st.spinner('📝 Generating transcript…'):
                transcript = _run_transcription(
                    wav_path,
                    prefer_device=prefer_device,
                    model_size=asr_size,
                    language=(language_hint if language_hint.strip() else None),
                )

            if transcript:
                st.markdown(f'**📋 Preview:** {transcript[:200]}{"..." if len(transcript) > 200 else ""}')

            with st.spinner('🤖 Loading models…'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                fusion_model, vision_model, audio_model, text_tokenizer, text_model = _load_inference_stack(device)

            with st.spinner('👁️ Analyzing visual cues…'):
                vis = _extract_vision_features(video_path, vision_model, device)

            with st.spinner('🎧 Processing audio features…'):
                aud = _extract_audio_features(wav_path, audio_model, device)

            with st.spinner('📖 Processing text features…'):
                txt = _extract_text_features(transcript if transcript else "", text_tokenizer, text_model, device)

            with st.spinner('🧠 Running Moody.AI analysis…'):
                outputs = fusion_model(txt, aud, vis)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                labels = {0: 'anger', 1: 'joy', 2: 'melancholy', 3: 'neutral', 4: 'surprise'}
                pred = labels[pred_idx]
                conf = probs[0, pred_idx].item()

            # Results display with enhanced formatting
            st.markdown('---')
            st.success(f'🎭 **Predicted sentiment:** **{pred.upper()}** • **confidence:** {conf:.3f}')
            
            st.markdown('**📊 Class probabilities:**')
            for i, name in labels.items():
                prob = probs[0, i].item()
                # Create a visual bar
                bar_length = int(prob * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                st.markdown(f'- **{name}**: {prob:.3f} `{bar}`')

        except Exception as e:
            st.error(f'❌ An error occurred: {str(e)}')
            
        finally:
            show_overlay_loader(False)
            if 'fusion_model' in locals():
                _unload_inference_stack(fusion_model, vision_model, audio_model, text_model)
            try:
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

if __name__ == '__main__':
    main()
