import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import shutil

# --- Configuration ---
SOURCE_PROCESSED_DIR = 'final_processed_audio'
FINAL_HIDDEN_STATE_DIR = 'emotion_enhanced_processed_audio_hidden_states'  # 🎯 New directory name
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # 🎯 EMOTION-SPECIFIC MODEL
BATCH_SIZE = 8  # 🎯 Reduced batch size for larger model

# --- Custom Dataset for Pre-computed Features ---
class PrecomputedAudioDataset(Dataset):
    """Loads pre-computed audio features and labels."""
    def __init__(self, data_prefix):
        self.features_dir = os.path.join(SOURCE_PROCESSED_DIR, data_prefix)
        self.labels_df = pd.read_csv(os.path.join(self.features_dir, 'labels.csv'))
        self.num_samples = len(self.labels_df)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature_path = os.path.join(self.features_dir, f'features_{idx}.pt')
        features = torch.load(feature_path, weights_only=True)
        return features

def extract_and_save_hidden_states(dataset, model, device, output_prefix):
    """
    Runs the emotion-enhanced base model on the features and saves the last hidden state.
    """
    output_dir = os.path.join(FINAL_HIDDEN_STATE_DIR, output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)  # 🎯 Reduced workers
    
    print(f"🎯 Extracting emotion-enhanced hidden states for '{output_prefix}' set...")
    saved_count = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Processing {output_prefix}")):
            inputs = batch.to(device)
            
            # 🎯 Extract from emotion-specific model
            outputs = model.wav2vec2(inputs)
            last_hidden_state = outputs.last_hidden_state.cpu()
            
            # Save each sample's hidden state individually
            for j in range(last_hidden_state.size(0)):
                # --- Keep FP16 for memory efficiency ---
                hidden_state_fp16 = last_hidden_state[j].to(torch.float16)
                torch.save(hidden_state_fp16, os.path.join(output_dir, f'hidden_state_{saved_count}.pt'))
                saved_count += 1
                
                # Progress update every 1000 samples
                if saved_count % 1000 == 0:
                    print(f"✅ Processed {saved_count} samples...")

    # Copy the labels file, as it doesn't need to change
    shutil.copy(
        os.path.join(SOURCE_PROCESSED_DIR, output_prefix, 'labels.csv'),
        os.path.join(output_dir, 'labels.csv')
    )
    print(f"✅ Saved {saved_count} emotion-enhanced hidden states for '{output_prefix}' set.")


if __name__ == "__main__":
    print(f"🎯 Starting Emotion-Enhanced Hidden State Extraction")
    print(f"🤖 Using Model: {MODEL_NAME}")
    
    if os.path.exists(FINAL_HIDDEN_STATE_DIR):
        print(f"🧹 Removing old hidden state directory: {FINAL_HIDDEN_STATE_DIR}")
        shutil.rmtree(FINAL_HIDDEN_STATE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} for hidden state extraction")

    # 🎯 Load the emotion-specific Wav2Vec2 model
    print(f"📥 Loading emotion-specific model: {MODEL_NAME}")
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32  # Use FP16 on GPU
        ).to(device)
        model.eval()  # Set to evaluation mode
        print("✅ Emotion-specific model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Fallback: Using original model...")
        MODEL_NAME = "superb/wav2vec2-base-superb-er"
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        model.eval()

    # Create datasets from the feature files
    print("📂 Loading datasets...")
    train_dataset = PrecomputedAudioDataset('train')
    eval_dataset = PrecomputedAudioDataset('validation')
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(eval_dataset)}")

    # Process and save the hidden states for both splits
    extract_and_save_hidden_states(train_dataset, model, device, 'train')
    extract_and_save_hidden_states(eval_dataset, model, device, 'validation')

    # Save processing info
    info = {
        'model_used': MODEL_NAME,
        'batch_size': BATCH_SIZE,
        'device_used': str(device),
        'train_samples': len(train_dataset),
        'validation_samples': len(eval_dataset),
        'hidden_state_precision': 'float16',
        'extraction_layer': 'last_hidden_state'
    }
    
    info_path = os.path.join(FINAL_HIDDEN_STATE_DIR, 'processing_info.txt')
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    print("\n🎉 Emotion-Enhanced Hidden State Extraction Complete!")
    print(f"💾 Saved to: {FINAL_HIDDEN_STATE_DIR}")
    print(f"🎯 Ready for emotion-enhanced training!")
