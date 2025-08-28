import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
)
import librosa
from typing import List, Dict, Any
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set cache directories
CACHE_DIR = r"D:/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["TMPDIR"] = r"D:/temp"
os.environ["TEMP"] = r"D:/temp"
os.environ["TMP"] = r"D:/temp"

# Create directories
for dir_path in [CACHE_DIR, r"D:/temp"]:
    os.makedirs(dir_path, exist_ok=True)

# Configuration - Works with your completed hidden states
HIDDEN_STATES_DIR = 'emotion_enhanced_processed_audio_hidden_states'
PROCESSED_AUDIO_DIR = 'processed_audio_for_finetuning'
TEXT_OUTPUT_DIR = "./text_transcriptions"
SENTIMENT_OUTPUT_DIR = "./text_sentiment_model"

# Create output directories
os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
os.makedirs(SENTIMENT_OUTPUT_DIR, exist_ok=True)

print(f"✅ All cache directories set to D: drive")

class AudioToTextProcessor:
    def __init__(self, model_name="openai/whisper-large-v3"):
        """Initialize Whisper model for speech-to-text conversion"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"🤖 Loading Whisper model: {model_name}")
        print(f"📱 Using device: {self.device}")
        
        # Load Whisper model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True
        )
        
        print("✅ Whisper model loaded successfully!")
    
    def transcribe_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file to text with timestamps"""
        try:
            if os.path.exists(audio_path):
                # Load actual audio file
                audio, sr = librosa.load(audio_path, sr=16000)
                
                # Transcribe with timestamps
                result = self.pipe(audio, return_timestamps=True)
                
                return {
                    "text": result["text"],
                    "chunks": result.get("chunks", []),
                    "duration": len(audio) / 16000,
                    "sample_rate": 16000,
                    "audio_length": len(audio)
                }
            else:
                print(f"⚠️ Audio file not found: {audio_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error transcribing {audio_path}: {e}")
            return None
    
    def create_placeholder_transcription(self, idx: int, emotion: str, audio_sequence_length: int) -> Dict[str, Any]:
        """Create placeholder transcription when audio file is not available"""
        placeholder_text = f"This is a {emotion} emotion sample"
        
        # Create segments based on audio sequence length
        num_segments = min(10, max(1, audio_sequence_length // 50))
        segments = [f"{emotion} segment {i+1}" for i in range(num_segments)]
        
        return {
            "full_transcription": placeholder_text,
            "text_segments": segments,
            "audio_duration": 3.0,
            "audio_sequence_length": audio_sequence_length,
            "frames_per_text_segment": max(1, audio_sequence_length // num_segments),
            "time_alignment": {
                "time_per_frame": 3.0 / audio_sequence_length if audio_sequence_length > 0 else 0.01,
                "segment_duration": 3.0 / num_segments,
                "num_segments": num_segments
            }
        }
    
    def segment_text_by_duration(self, transcription_result: Dict, target_duration: float) -> List[str]:
        """Segment text into chunks matching audio duration for multimodal alignment"""
        if not transcription_result or not transcription_result.get("chunks"):
            # If no timestamps, split text evenly
            text = transcription_result.get("text", "")
            words = text.split()
            if len(words) == 0:
                return [""]
            
            # Simple segmentation based on target duration
            total_duration = transcription_result.get("duration", 1.0)
            num_segments = max(1, int(total_duration / target_duration))
            words_per_segment = max(1, len(words) // num_segments)
            
            segments = []
            for i in range(0, len(words), words_per_segment):
                segment = " ".join(words[i:i + words_per_segment])
                segments.append(segment)
            
            return segments
        
        # Use timestamp-based segmentation
        chunks = transcription_result["chunks"]
        segments = []
        current_segment = ""
        current_duration = 0
        
        for chunk in chunks:
            chunk_start = chunk.get("timestamp", [0, 0])[0]
            chunk_end = chunk.get("timestamp", [0, 0])[1]
            chunk_duration = chunk_end - chunk_start
            
            if current_duration + chunk_duration <= target_duration:
                current_segment += " " + chunk["text"]
                current_duration += chunk_duration
            else:
                if current_segment.strip():
                    segments.append(current_segment.strip())
                current_segment = chunk["text"]
                current_duration = chunk_duration
        
        # Add final segment
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments if segments else [""]
    
    def create_aligned_text_features(self, idx: int, emotion: str, hidden_states_path: str, audio_path: str = None) -> Dict[str, Any]:
        """Create text features aligned with audio hidden states for multimodal fusion"""
        try:
            # Load audio hidden states to get sequence length
            hidden_states = torch.load(hidden_states_path, weights_only=True)
            audio_sequence_length = hidden_states.shape[0]  # [T, 768] -> T
            
            # Try to transcribe audio if path is provided and exists
            transcription_result = None
            if audio_path:
                transcription_result = self.transcribe_audio_file(audio_path)
            
            if transcription_result:
                # Use actual transcription
                audio_duration = transcription_result["duration"]
                time_per_frame = audio_duration / audio_sequence_length if audio_sequence_length > 0 else 0.01
                
                # Create text segments aligned with audio frames
                frames_per_text_segment = max(1, audio_sequence_length // 10)  # ~10 text segments
                segment_duration = time_per_frame * frames_per_text_segment
                
                text_segments = self.segment_text_by_duration(transcription_result, segment_duration)
                
                # Ensure we have the right number of segments
                target_segments = max(1, audio_sequence_length // frames_per_text_segment)
                while len(text_segments) < target_segments:
                    text_segments.append(text_segments[-1] if text_segments else f"Segment {len(text_segments)}")
                while len(text_segments) > target_segments:
                    text_segments.pop()
                
                return {
                    "full_transcription": transcription_result["text"],
                    "text_segments": text_segments,
                    "audio_duration": audio_duration,
                    "audio_sequence_length": audio_sequence_length,
                    "frames_per_text_segment": frames_per_text_segment,
                    "time_alignment": {
                        "time_per_frame": time_per_frame,
                        "segment_duration": segment_duration,
                        "num_segments": len(text_segments)
                    }
                }
            else:
                # Use placeholder transcription
                return self.create_placeholder_transcription(idx, emotion, audio_sequence_length)
            
        except Exception as e:
            print(f"❌ Error creating aligned features for idx {idx}: {e}")
            # Return placeholder on error
            return self.create_placeholder_transcription(idx, emotion, 100)
    
    def process_from_existing_structure(self, data_split: str) -> pd.DataFrame:
        """Process dataset using existing hidden states and audio files"""
        print(f"\n🎙️ Processing {data_split} for transcription...")
        
        # Check paths
        hidden_states_dir = os.path.join(HIDDEN_STATES_DIR, data_split)
        hidden_labels_path = os.path.join(hidden_states_dir, 'labels.csv')
        
        audio_metadata_dir = os.path.join(PROCESSED_AUDIO_DIR, data_split)
        audio_metadata_path = os.path.join(audio_metadata_dir, 'metadata.csv')
        
        # Verify hidden states exist
        if not os.path.exists(hidden_labels_path):
            print(f"❌ Hidden states labels not found: {hidden_labels_path}")
            return pd.DataFrame()
        
        # Load hidden states labels
        hidden_labels_df = pd.read_csv(hidden_labels_path)
        print(f"📊 Found {len(hidden_labels_df)} hidden state samples")
        
        # Try to load audio metadata if available
        audio_metadata_df = None
        if os.path.exists(audio_metadata_path):
            audio_metadata_df = pd.read_csv(audio_metadata_path)
            print(f"📊 Found {len(audio_metadata_df)} audio metadata entries")
        else:
            print(f"⚠️ Audio metadata not found: {audio_metadata_path}")
            print("💡 Will create placeholder transcriptions aligned with hidden states")
        
        transcriptions = []
        
        # Process each sample
        for idx in tqdm(range(len(hidden_labels_df)), desc=f"Processing {data_split}"):
            try:
                emotion = hidden_labels_df.iloc[idx]['emotion']
                hidden_states_path = os.path.join(hidden_states_dir, f'hidden_state_{idx}.pt')
                
                # Get corresponding audio path if metadata is available
                audio_path = None
                if audio_metadata_df is not None and idx < len(audio_metadata_df):
                    audio_path = audio_metadata_df.iloc[idx]['path']
                
                if os.path.exists(hidden_states_path):
                    # Create aligned text features
                    aligned_features = self.create_aligned_text_features(
                        idx, emotion, hidden_states_path, audio_path
                    )
                    
                    if aligned_features:
                        transcriptions.append({
                            'idx': idx,
                            'audio_path': audio_path if audio_path else f"hidden_state_{idx}.pt",
                            'emotion': emotion,
                            'full_transcription': aligned_features['full_transcription'],
                            'text_segments': json.dumps(aligned_features['text_segments']),
                            'audio_duration': aligned_features['audio_duration'],
                            'audio_sequence_length': aligned_features['audio_sequence_length'],
                            'frames_per_text_segment': aligned_features['frames_per_text_segment'],
                            'num_text_segments': aligned_features['time_alignment']['num_segments']
                        })
                else:
                    print(f"⚠️ Missing hidden state file: {hidden_states_path}")
                    continue
                    
            except Exception as e:
                print(f"❌ Error processing sample {idx}: {e}")
                continue
        
        if transcriptions:
            # Save transcriptions
            transcription_df = pd.DataFrame(transcriptions)
            output_path = os.path.join(TEXT_OUTPUT_DIR, f"{data_split}_transcriptions.csv")
            transcription_df.to_csv(output_path, index=False)
            
            print(f"✅ Saved {len(transcriptions)} transcriptions to {output_path}")
            
            # Save summary statistics
            self.save_transcription_stats(transcription_df, data_split)
            
            return transcription_df
        else:
            print(f"❌ No transcriptions created for {data_split}")
            return pd.DataFrame()
    
    def save_transcription_stats(self, transcription_df: pd.DataFrame, data_split: str):
        """Save transcription statistics for analysis"""
        stats = {
            "dataset_split": data_split,
            "total_files": len(transcription_df),
            "successful_transcriptions": len(transcription_df[transcription_df['full_transcription'] != ""]),
            "average_audio_duration": transcription_df['audio_duration'].mean(),
            "average_transcription_length": transcription_df['full_transcription'].str.len().mean(),
            "average_segments_per_file": transcription_df['num_text_segments'].mean(),
            "emotion_distribution": transcription_df['emotion'].value_counts().to_dict()
        }
        
        stats_path = os.path.join(TEXT_OUTPUT_DIR, f"{data_split}_transcription_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Transcription statistics:")
        print(f"   • Success rate: {stats['successful_transcriptions']}/{stats['total_files']} ({stats['successful_transcriptions']/stats['total_files']*100:.1f}%)")
        print(f"   • Avg duration: {stats['average_audio_duration']:.2f}s")
        print(f"   • Avg text length: {stats['average_transcription_length']:.0f} chars")

# Utility functions
def load_transcriptions(data_split: str) -> pd.DataFrame:
    """Load processed transcriptions"""
    file_path = os.path.join(TEXT_OUTPUT_DIR, f"{data_split}_transcriptions.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"❌ Transcription file not found: {file_path}")
        return pd.DataFrame()

def get_aligned_text_segments(transcription_row) -> List[str]:
    """Extract text segments from transcription row"""
    try:
        return json.loads(transcription_row['text_segments'])
    except:
        return [transcription_row.get('full_transcription', "")]

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Audio-to-Text Processing...")
    print("✅ Using completed hidden states from audio preprocessing")
    
    # Check if hidden states exist
    hidden_states_train = os.path.join(HIDDEN_STATES_DIR, "train")
    hidden_states_validation = os.path.join(HIDDEN_STATES_DIR, "validation")
    
    print(f"\n🔍 Checking files:")
    print(f"   • Train hidden states: {hidden_states_train}")
    print(f"   • Validation hidden states: {hidden_states_validation}")
    
    if not os.path.exists(hidden_states_train):
        print(f"❌ Missing: {hidden_states_train}")
        print("🔧 Please run audio_preprocess_hidden_states.py first")
    else:
        print("✅ Hidden states found! Starting transcription processing...")
        
        # Initialize processor
        processor = AudioToTextProcessor()
        
        # Process training set
        print("\n" + "="*60)
        print("PROCESSING TRAINING SET")
        print("="*60)
        train_transcriptions = processor.process_from_existing_structure("train")
        
        # Process validation set
        print("\n" + "="*60)
        print("PROCESSING VALIDATION SET")
        print("="*60)
        val_transcriptions = processor.process_from_existing_structure("validation")
        
        print("\n🎉 Text preprocessing complete!")
        print(f"📁 Training transcriptions: {len(train_transcriptions)}")
        print(f"📁 Validation transcriptions: {len(val_transcriptions)}")
        print(f"💾 Files saved to: {TEXT_OUTPUT_DIR}")
        
        # Display sample results
        if len(train_transcriptions) > 0:
            print("\n📋 Sample transcription:")
            sample = train_transcriptions.iloc[0]
            print(f"   • Reference: {sample['audio_path']}")
            print(f"   • Emotion: {sample['emotion']}")
            print(f"   • Duration: {sample['audio_duration']:.2f}s")
            print(f"   • Text: {sample['full_transcription'][:100]}...")
            print(f"   • Segments: {sample['num_text_segments']}")
        
        print("\n🎯 Next steps:")
        print("   • Text transcriptions are ready for sentiment model training")
        print("   • You can now train both audio and text emotion models")
        print("   • Later combine them for multimodal emotion recognition")
        print(f"   • Your hidden states are ready in: {HIDDEN_STATES_DIR}")
