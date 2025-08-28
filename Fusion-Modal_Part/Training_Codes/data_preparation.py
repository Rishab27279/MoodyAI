import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import timm
import librosa
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Wav2Vec2ForSequenceClassification
import shutil
import cv2
import cvlib as cv
import math

# --- Configuration ---
BASE_DATA_PATH = r'D:\Sentiment.AI\Data\MELD-RAW\MELD'
FINAL_TRIMODAL_DIR = 'final_processed_trimodal_data'
ORIGINAL_AUDIO_MODEL_NAME = "superb/wav2vec2-base-superb-er"
ORIGINAL_TEXT_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 10
SAMPLE_RATE = 16000
TEXT_MAX_LENGTH = 128

class FeatureExtractors:
    """A class to hold all the pre-trained models for feature extraction."""
    def __init__(self, device):
        self.device = device
        
        print("Loading DINOv2 Vision Encoder...")
        self.vision_model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m', 
            pretrained=True, 
            in_chans=3, 
            img_size=224
        ).to(self.device)
        self.vision_model.eval()
        print("✅ Vision Encoder loaded.")

        print(f"Loading original Audio Encoder from Hub: {ORIGINAL_AUDIO_MODEL_NAME}...")
        self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            ORIGINAL_AUDIO_MODEL_NAME, use_safetensors=True
        ).to(self.device)
        self.audio_model.eval()
        print("✅ Audio Encoder loaded.")

        print(f"Loading original Text Encoder from Hub: {ORIGINAL_TEXT_MODEL_NAME}...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_TEXT_MODEL_NAME)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            ORIGINAL_TEXT_MODEL_NAME, use_safetensors=True
        ).to(self.device)
        self.text_model.eval()
        print("✅ Text Encoder loaded.")

    @torch.no_grad()
    def get_vision_features(self, video_path):
        vision_sequence = np.zeros((MAX_SEQUENCE_LENGTH, 3, 224, 224), dtype=np.float32)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 24
        frame_idx, count = 0, 0
        while cap.isOpened() and count < MAX_SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % math.ceil(fps) == 0:
                faces, _ = cv.detect_face(frame)
                if faces and faces[0][2] > faces[0][0] and faces[0][3] > faces[0][1]:
                    (x1, y1, x2, y2) = faces[0]
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (224, 224))
                        face_normalized = (face_resized / 255.0).astype(np.float32)
                        vision_sequence[count] = np.transpose(face_normalized, (2, 0, 1))
                count += 1
            frame_idx += 1
        cap.release()
        
        vision_tensor = torch.from_numpy(vision_sequence).to(self.device)
        features = self.vision_model.forward_features(vision_tensor)
        features = features.view(2, 5, -1).mean(dim=1)
        return features.cpu()

    @torch.no_grad()
    def get_audio_features(self, video_path):
        raw_audio, _ = librosa.load(video_path, sr=SAMPLE_RATE, mono=True)
        inputs = torch.tensor(raw_audio).unsqueeze(0).to(self.device)
        outputs = self.audio_model.wav2vec2(inputs, output_hidden_states=True)
        return outputs.last_hidden_state.mean(dim=1).cpu()

    @torch.no_grad()
    def get_text_features(self, text):
        inputs = self.text_tokenizer(text, return_tensors="pt", max_length=TEXT_MAX_LENGTH, truncation=True, padding=True).to(self.device)
        outputs = self.text_model.distilbert(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1).cpu()

def process_and_save_all_features(data_df, clips_dir, feature_extractors, output_prefix):
    output_dir = os.path.join(FINAL_TRIMODAL_DIR, output_prefix)
    text_dir = os.path.join(output_dir, 'text')
    audio_dir = os.path.join(output_dir, 'audio')
    vision_dir = os.path.join(output_dir, 'vision')
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(vision_dir, exist_ok=True)

    emotion_map = {'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'fear': 5, 'disgust': 6}
    saved_labels = []
    successful_indices = [] # <-- FIX: Track successful indices
    
    print(f"Processing and saving all features for '{output_prefix}' set...")
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
        try:
            video_path = os.path.join(clips_dir, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")
            if not os.path.exists(video_path): continue

            text = row['Utterance']
            
            text_features = feature_extractors.get_text_features(text)
            audio_features = feature_extractors.get_audio_features(video_path)
            vision_features = feature_extractors.get_vision_features(video_path)

            # Use the original dataframe index as the filename
            torch.save(text_features, os.path.join(text_dir, f"{idx}.pt"))
            torch.save(audio_features, os.path.join(audio_dir, f"{idx}.pt"))
            torch.save(vision_features, os.path.join(vision_dir, f"{idx}.pt"))
            
            saved_labels.append(emotion_map[row['Emotion']])
            successful_indices.append(idx) # <-- FIX: Save the index if successful
        except Exception as e:
            # print(f"Skipping index {idx} due to error: {e}")
            pass
            
    np.save(os.path.join(output_dir, 'labels.npy'), np.array(saved_labels))
    np.save(os.path.join(output_dir, 'indices.npy'), np.array(successful_indices)) # <-- FIX: Save the indices file
    print(f"✅ Saved {len(saved_labels)} features and labels for '{output_prefix}' set.")

if __name__ == "__main__":
    if os.path.exists(FINAL_TRIMODAL_DIR):
        print(f"Removing old trimodal data directory: {FINAL_TRIMODAL_DIR}")
        shutil.rmtree(FINAL_TRIMODAL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractors = FeatureExtractors(device)

    train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train/train_sent_emo.csv'))
    dev_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'dev_sent_emo.csv'))
    
    train_clips = os.path.join(BASE_DATA_PATH, 'train/train_splits')
    dev_clips = os.path.join(BASE_DATA_PATH, 'dev/dev_splits_complete')

    process_and_save_all_features(train_df, train_clips, feature_extractors, 'train')
    process_and_save_all_features(dev_df, dev_clips, feature_extractors, 'val')

    print("\n✅ All trimodal feature extraction complete!")
