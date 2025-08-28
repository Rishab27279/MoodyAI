import os
import torch
import pandas as pd
import numpy as np
import json
import glob
from typing import Dict, List
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import warnings
warnings.filterwarnings("ignore")

# Configuration
MODEL_OUTPUT_DIR = "./distilbert_sentiment_model"
CACHE_DIR = r"D:/huggingface_cache"

# Set cache directories
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def find_meld_csv_files():
    """Find MELD CSV files in various locations"""
    print("🔍 Auto-detecting MELD CSV files...")
    
    # Search patterns for MELD CSV files
    search_patterns = [
        r'D:\Sentiment.AI\Data\**\*train_sent_emo.csv',
        r'D:\Sentiment.AI\**\*train_sent_emo.csv',
        r'D:\Desktop\**\*train_sent_emo.csv',
        './meld_csv_data/train_sent_emo.csv',
        './**/train_sent_emo.csv',
        '../**/train_sent_emo.csv'
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            train_file = matches[0]
            base_dir = os.path.dirname(train_file)
            
            # Check if dev file also exists
            dev_file = os.path.join(base_dir, 'dev_sent_emo.csv')
            if os.path.exists(dev_file):
                print(f"✅ Found MELD CSV files in: {base_dir}")
                return train_file, dev_file
    
    print("❌ MELD CSV files not found. Please download them.")
    return None, None

class MELDSentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DistilBERTSentimentTrainer:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.id2label = {}
        self.label2id = {}
        
    def create_emotion_mapping(self) -> Dict[str, str]:
        emotion_mapping = {
            'sadness': 'melancholy',
            'fear': 'melancholy', 
            'disgust': 'melancholy',
            'anger': 'anger',
            'joy': 'joy',
            'neutral': 'neutral', 
            'surprise': 'surprise'
        }
        return emotion_mapping
        
    def load_original_meld_data(self) -> tuple:
        print("📂 Loading original MELD CSV files...")
        
        # Auto-detect CSV file locations
        train_path, val_path = find_meld_csv_files()
        
        if not train_path or not val_path:
            print("💡 Downloading MELD CSV files from GitHub...")
            # Download files if not found
            import requests
            
            base_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
            download_dir = "./meld_csv_data"
            os.makedirs(download_dir, exist_ok=True)
            
            files = ["train_sent_emo.csv", "dev_sent_emo.csv"]
            
            for filename in files:
                url = f"{base_url}/{filename}"
                local_path = os.path.join(download_dir, filename)
                
                if not os.path.exists(local_path):
                    print(f"   📥 Downloading {filename}...")
                    try:
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()
                        
                        with open(local_path, 'wb') as f:
                            f.write(response.content)
                        print(f"   ✅ Downloaded: {local_path}")
                    except Exception as e:
                        print(f"   ❌ Failed to download {filename}: {e}")
                        raise
            
            train_path = os.path.join(download_dir, "train_sent_emo.csv")
            val_path = os.path.join(download_dir, "dev_sent_emo.csv")
        
        print(f"📂 Loading from: {train_path}")
        print(f"📂 Loading from: {val_path}")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        print(f"📊 Loaded {len(train_df)} training samples")
        print(f"📊 Loaded {len(val_df)} validation samples")
        
        # Use actual dialogue text from 'Utterance' column
        train_texts = train_df['Utterance'].astype(str).tolist()
        train_emotions_raw = train_df['Emotion'].str.lower().tolist()
        
        val_texts = val_df['Utterance'].astype(str).tolist()
        val_emotions_raw = val_df['Emotion'].str.lower().tolist()
        
        # Apply emotion mapping
        emotion_mapping = self.create_emotion_mapping()
        
        train_emotions = [emotion_mapping.get(emotion, emotion) for emotion in train_emotions_raw]
        val_emotions = [emotion_mapping.get(emotion, emotion) for emotion in val_emotions_raw]
        
        print(f"🔄 Applied emotion mapping:")
        print(f"   • sadness, fear, disgust → melancholy")
        print(f"   • Original classes: {sorted(set(train_emotions_raw))}")
        print(f"   • Mapped classes: {sorted(set(train_emotions))}")
        
        # Encode labels
        all_emotions = train_emotions + val_emotions
        self.label_encoder.fit(all_emotions)
        
        train_labels = self.label_encoder.transform(train_emotions)
        val_labels = self.label_encoder.transform(val_emotions)
        
        # Create label mappings
        self.id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        self.label2id = {label: i for i, label in self.id2label.items()}
        
        print(f"🏷️ Final emotion classes ({len(self.label_encoder.classes_)}):")
        for i, emotion in enumerate(self.label_encoder.classes_):
            count = train_emotions.count(emotion)
            print(f"   {i}: {emotion} ({count} samples)")
        
        # Show sample real dialogues
        print("\n💬 Sample real dialogue texts:")
        for i in range(5):
            print(f"   {i+1}: [{train_emotions[i]}] {train_texts[i][:100]}...")
        
        return (train_texts, train_labels.tolist(), val_texts, val_labels.tolist())
    
    def create_datasets(self, train_texts: List[str], train_labels: List[int],
                       val_texts: List[str], val_labels: List[int]) -> tuple:
        print("🔧 Creating datasets...")
        
        train_dataset = MELDSentimentDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer
        )
        
        val_dataset = MELDSentimentDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=self.tokenizer
        )
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset):
        print("🤖 Loading DistilBERT model...")
        
        num_labels = len(self.label_encoder.classes_)
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        training_args = TrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(MODEL_OUTPUT_DIR, 'logs'),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=3,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("🏋️ Starting training with real MELD dialogue data...")
        trainer.train()
        
        print("💾 Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        
        return trainer
    
    def evaluate_model(self, trainer: Trainer, val_dataset: Dataset):
        print("\n📊 Evaluating model...")
        
        predictions = trainer.predict(val_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"✅ Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy

def main():
    print("🚀 Starting DistilBERT Fine-tuning for MELD Sentiment Classification")
    print("📝 Using REAL MELD dialogue transcripts with auto-download")
    print("🔄 With Emotion Mapping: sad, fear, disgust → melancholy")
    print("="*70)
    
    trainer_class = DistilBERTSentimentTrainer()
    
    try:
        train_texts, train_labels, val_texts, val_labels = trainer_class.load_original_meld_data()
        train_dataset, val_dataset = trainer_class.create_datasets(
            train_texts, train_labels, val_texts, val_labels
        )
        
        trainer = trainer_class.train_model(train_dataset, val_dataset)
        accuracy = trainer_class.evaluate_model(trainer, val_dataset)
        
        print("\n🎉 Training completed successfully!")
        print(f"🎯 Final accuracy: {accuracy*100:.2f}%")
        
        if accuracy > 0.70:
            print("🌟 Outstanding! Exceeded SOTA performance!")
        elif accuracy > 0.65:
            print("✅ Excellent! Achieved top-tier MELD performance")
        elif accuracy > 0.60:
            print("✅ Great! Achieved good MELD performance")
        else:
            print("🔄 Decent performance. Consider hyperparameter tuning")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
