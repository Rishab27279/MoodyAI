import os
import pandas as pd
from tqdm import tqdm
import shutil
import subprocess

# --- Configuration ---
BASE_DATA_PATH = r'D:\Sentiment.AI\Data\MELD-RAW\MELD'
PROCESSED_AUDIO_DIR = 'processed_audio_for_finetuning'
SAMPLE_RATE = 16000 # The sample rate required by the Wav2Vec2 model

def create_filepath_label_pairs(csv_path, clips_dir):
    """Maps CSV rows to their corresponding video file paths and labels."""
    df = pd.read_csv(csv_path)
    data_pairs = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Mapping {os.path.basename(csv_path)}"):
        video_path = os.path.join(clips_dir, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")
        if os.path.exists(video_path):
            data_pairs.append((video_path, row['Emotion']))
    return data_pairs

def extract_and_save_audio(data_pairs, output_prefix):
    """
    Extracts audio from video files and saves it in a structured directory.
    Also creates a metadata CSV file for the Hugging Face dataset loader.
    """
    output_dir = os.path.join(PROCESSED_AUDIO_DIR, output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    
    print(f"Extracting audio for '{output_prefix}' set...")
    for i, (video_path, emotion_label) in enumerate(tqdm(data_pairs)):
        try:
            output_filename = f"audio_{i}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Use ffmpeg to extract audio, resample to 16kHz, and convert to mono
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn', # No video
                '-acodec', 'pcm_s16le', # Standard WAV format
                '-ar', str(SAMPLE_RATE), # Resample
                '-ac', '1', # Mono channel
                '-y', # Overwrite output file if it exists
                '-loglevel', 'error', # Suppress verbose output
                output_path
            ]
            subprocess.run(command, check=True)
            
            # Add to metadata for later use
            metadata.append({'path': output_path, 'emotion': emotion_label})
            
        except Exception as e:
            print(f"Skipping {video_path} due to error: {e}")
            pass
            
    # Save metadata to a CSV file
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    print(f"✅ Saved metadata for '{output_prefix}' set.")


if __name__ == "__main__":
    if os.path.exists(PROCESSED_AUDIO_DIR):
        print(f"Removing old processed audio directory: {PROCESSED_AUDIO_DIR}")
        shutil.rmtree(PROCESSED_AUDIO_DIR)

    # Load data paths
    train_csv = os.path.join(BASE_DATA_PATH, 'train/train_sent_emo.csv')
    dev_csv = os.path.join(BASE_DATA_PATH, 'dev_sent_emo.csv')
    train_clips = os.path.join(BASE_DATA_PATH, 'train/train_splits')
    dev_clips = os.path.join(BASE_DATA_PATH, 'dev/dev_splits_complete')
    
    train_pairs = create_filepath_label_pairs(train_csv, train_clips)
    val_pairs = create_filepath_label_pairs(dev_csv, dev_clips)

    # Process and save the datasets
    extract_and_save_audio(train_pairs, 'train')
    extract_and_save_audio(val_pairs, 'validation')

    print("\n✅ All audio pre-processing complete!")
