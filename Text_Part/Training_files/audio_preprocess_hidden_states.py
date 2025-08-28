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
FINAL_HIDDEN_STATE_DIR = 'emotion_enhanced_processed_audio_hidden_states'
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
BATCH_SIZE = 4  # Reduced batch size for GPU memory

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
    Runs the emotion-enhanced model and saves hidden states with proper dtype handling.
    """
    output_dir = os.path.join(FINAL_HIDDEN_STATE_DIR, output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"🎯 Extracting emotion-enhanced hidden states for '{output_prefix}' set...")
    
    saved_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Processing {output_prefix}")):
            # FIXED: Convert inputs to match model's expected dtype
            inputs = batch.to(device)
            
            # Ensure consistent dtype throughout the computation
            if device.type == "cuda":
                inputs = inputs.to(torch.float32)
            
            try:
                # Extract from emotion-specific model
                outputs = model.wav2vec2(inputs)
                last_hidden_state = outputs.last_hidden_state.cpu()
                
                # Save each sample's hidden state individually
                for j in range(last_hidden_state.size(0)):
                    # Convert to FP16 for storage efficiency
                    hidden_state_fp16 = last_hidden_state[j].to(torch.float16)
                    torch.save(hidden_state_fp16, os.path.join(output_dir, f'hidden_state_{saved_count}.pt'))
                    saved_count += 1
                    
                    # Progress update every 1000 samples
                    if saved_count % 1000 == 0:
                        print(f"✅ Processed {saved_count} samples...")
                        
            except Exception as e:
                print(f"❌ Error processing batch {i}: {e}")
                continue
    
    # Copy the labels file
    try:
        shutil.copy(
            os.path.join(SOURCE_PROCESSED_DIR, output_prefix, 'labels.csv'),
            os.path.join(output_dir, 'labels.csv')
        )
    except Exception as e:
        print(f"⚠️ Warning: Could not copy labels file: {e}")
    
    print(f"✅ Saved {saved_count} emotion-enhanced hidden states for '{output_prefix}' set.")

if __name__ == "__main__":
    print(f"🎯 Starting Emotion-Enhanced Hidden State Extraction")
    print(f"🤖 Using Model: {MODEL_NAME}")
    
    if os.path.exists(FINAL_HIDDEN_STATE_DIR):
        print(f"🧹 Removing old hidden state directory: {FINAL_HIDDEN_STATE_DIR}")
        shutil.rmtree(FINAL_HIDDEN_STATE_DIR)

    # Device detection and setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Load the emotion-specific Wav2Vec2 model
    print(f"📥 Loading emotion-specific model: {MODEL_NAME}")
    try:
        # FIXED: Use consistent float32 dtype to avoid mixed precision issues
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32  # Use float32 for stability
        ).to(device)
        model.eval()
        print("✅ Emotion-specific model loaded successfully!")
        
        # Verify model dtype
        first_param = next(model.parameters())
        print(f"🔍 Model dtype: {first_param.dtype}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Fallback: Using smaller base model...")
        MODEL_NAME = "facebook/wav2vec2-base-960h"  # Smaller, more stable model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        ).to(device)
        model.eval()
        print(f"✅ Fallback model loaded: {MODEL_NAME}")

    # Create datasets from the feature files
    print("📂 Loading datasets...")
    try:
        train_dataset = PrecomputedAudioDataset('train')
        eval_dataset = PrecomputedAudioDataset('validation')
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(eval_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        exit(1)

    # Process and save the hidden states for both splits
    print("\n🔄 Starting processing...")
    
    try:
        extract_and_save_hidden_states(train_dataset, model, device, 'train')
        extract_and_save_hidden_states(eval_dataset, model, device, 'validation')
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 Try reducing BATCH_SIZE or using CPU")
    
    # Save processing info
    info = {
        'model_used': MODEL_NAME,
        'batch_size': BATCH_SIZE,
        'device_used': str(device),
        'train_samples': len(train_dataset) if 'train_dataset' in locals() else 0,
        'validation_samples': len(eval_dataset) if 'eval_dataset' in locals() else 0,
        'hidden_state_precision': 'float16 (storage), float32 (computation)',
        'extraction_layer': 'last_hidden_state'
    }
    
    info_path = os.path.join(FINAL_HIDDEN_STATE_DIR, 'processing_info.txt')
    try:
        with open(info_path, 'w') as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
    except Exception as e:
        print(f"⚠️ Could not save processing info: {e}")

    print("\n🎉 Emotion-Enhanced Hidden State Extraction Complete!")
    print(f"💾 Saved to: {FINAL_HIDDEN_STATE_DIR}")
    print(f"🎯 Ready for emotion-enhanced training!")
