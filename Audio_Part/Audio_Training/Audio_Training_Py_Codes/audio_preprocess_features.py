import os
import pandas as pd
from tqdm import tqdm
import shutil
from datasets import load_dataset, Audio
from transformers import Wav2Vec2FeatureExtractor
import torch # <-- ADDED THIS IMPORT

# --- Configuration ---
PROCESSED_AUDIO_DIR = 'processed_audio_for_finetuning'
FINAL_PROCESSED_DIR = 'final_processed_audio'
MODEL_NAME = "superb/wav2vec2-base-superb-er"

def preprocess_and_save_features(dataset, feature_extractor, output_prefix):
    """
    Applies the feature extractor to the audio data and saves the results.
    """
    output_dir = os.path.join(FINAL_PROCESSED_DIR, output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    all_labels = []
    
    print(f"Processing and saving features for '{output_prefix}' set...")
    for i, example in enumerate(tqdm(dataset)):
        try:
            # Extract audio array and label
            audio_input = example['path']['array']
            label = example['emotion']
            
            # Process with feature extractor
            inputs = feature_extractor(
                audio_input, 
                sampling_rate=16000, 
                max_length=int(16000 * 8.0), # 8 seconds max
                truncation=True,
                padding='max_length' # Pad to a consistent length
            )
            
            # Save the processed input_values tensor
            processed_tensor = torch.tensor(inputs['input_values'][0])
            torch.save(processed_tensor, os.path.join(output_dir, f'features_{i}.pt'))
            
            # Store the label
            all_labels.append(label)

        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")
            pass
            
    # Save all labels to a single file
    labels_df = pd.DataFrame(all_labels, columns=['emotion'])
    labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
    print(f"✅ Saved {len(all_labels)} features and labels for '{output_prefix}' set.")

if __name__ == "__main__":
    if os.path.exists(FINAL_PROCESSED_DIR):
        print(f"Removing old final processed data directory: {FINAL_PROCESSED_DIR}")
        shutil.rmtree(FINAL_PROCESSED_DIR)

    # Load the feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    # Load the metadata CSVs into a dataset dictionary
    data_files = {
        "train": os.path.join(PROCESSED_AUDIO_DIR, 'train/metadata.csv'),
        "validation": os.path.join(PROCESSED_AUDIO_DIR, 'validation/metadata.csv'),
    }
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Process and save the features for both splits
    preprocess_and_save_features(dataset['train'], feature_extractor, 'train')
    preprocess_and_save_features(dataset['validation'], feature_extractor, 'validation')

    print("\n✅ All audio feature extraction complete!")
