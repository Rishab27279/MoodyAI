import os, warnings, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
# Set HuggingFace cache to a directory with sufficient space
# ============= MUST BE FIRST - BEFORE ANY OTHER IMPORTS =============
# Set HuggingFace cache to D: drive (124 GB free space)
CACHE_DIR = r"D:/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub") 
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")

# Also set system temp directory to D: drive to avoid temp file issues
os.environ["TMPDIR"] = r"D:/temp"
os.environ["TEMP"] = r"D:/temp" 
os.environ["TMP"] = r"D:/temp"

# Create all directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(r"D:/temp", exist_ok=True)
print(f"✅ All cache directories set to D: drive")
# ====================================================================

warnings.filterwarnings("ignore")


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# --- Configuration ---
PROCESSED_AUDIO_DIR = 'emotion_enhanced_processed_audio_hidden_states'
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUTPUT_DIR = './finetuned-audio-model-5class-emotion-focal-attention'
NUM_EPOCHS = 25
LEARNING_RATE = 5e-5

# --- 7-to-5 Class Emotion Mapping ---
def map_emotions_to_5_class(original_emotion):
    emotion_mapping = {
        'neutral': 'neutral',
        'surprise': 'surprise', 
        'fear': 'melancholy',
        'sadness': 'melancholy',
        'disgust': 'melancholy',
        'joy': 'joy',
        'anger': 'anger'
    }
    return emotion_mapping[original_emotion]

# --- Class-Balanced Focal Loss ---
class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for handling imbalanced datasets"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Attention Pooling Module ---
class AttentionPooling(nn.Module):
    """Enhanced attention pooling for temporal sequences"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        # Use mean pooling as query for attention
        query = x.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
        
        # Self-attention with query-key-value
        attn_output, attention_weights = self.attention(query, x, x)
        
        # Add residual connection and layer norm
        output = self.layer_norm(attn_output + query)
        output = self.dropout(output)
        
        return output.squeeze(1)  # [batch_size, embed_dim]

# --- Enhanced Emotion Classifier with Focal Loss and Attention ---
class FocalAttentionEmotionClassifier(nn.Module):
    """Emotion classifier with focal loss and attention pooling"""
    def __init__(self, full_model, hidden_dim=768, dropout=0.1):
        super().__init__()
        
        # Enhanced attention pooling
        self.attention_pool = AttentionPooling(hidden_dim, num_heads=8, dropout=dropout)
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5)
        )
    
    def forward(self, hidden_states, labels=None):
        # Apply attention pooling
        pooled_features = self.attention_pool(hidden_states)
        
        # Classification
        logits = self.classifier(pooled_features)
        return (None, logits)

# --- Dataset ---
class PrecomputedHiddenStateDataset5Class(Dataset):
    def __init__(self, data_prefix, label2id):
        self.features_dir = os.path.join(PROCESSED_AUDIO_DIR, data_prefix)
        self.labels_df = pd.read_csv(os.path.join(self.features_dir, 'labels.csv'))
        self.label2id = label2id
        self.num_samples = len(self.labels_df)
        
        self.labels_df['emotion_5class'] = self.labels_df['emotion'].apply(map_emotions_to_5_class)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature_path = os.path.join(self.features_dir, f'hidden_state_{idx}.pt')
        features = torch.load(feature_path, weights_only=True).to(torch.float32)
        
        label_str = self.labels_df.iloc[idx]['emotion_5class']
        label_id = self.label2id[label_str]
        
        return {'hidden_states': features, 'labels': torch.tensor(label_id, dtype=torch.long)}

# --- Enhanced Trainer with Focal Loss ---
class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs[1]
        
        # Use focal loss instead of cross-entropy
        if self.focal_loss_fn is not None:
            loss = self.focal_loss_fn(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)
        
        return (loss, {"logits": logits}) if return_outputs else loss

# --- Evaluation Metrics ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    return {
        'accuracy': acc, 
        'f1': f1, 
        'f1_macro': f1_macro,
        'precision': precision, 
        'precision_macro': precision_macro,
        'recall': recall, 
        'recall_macro': recall_macro,
        'kappa': kappa,
        'f1_per_class': f1_per_class.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist()
    }

def plot_confusion_matrix(trainer, eval_dataset, id2label, save_path="confusion_matrix_focal_attention.png"):
    print("\nGenerating confusion matrix...")
    preds_output = trainer.predict(eval_dataset)
    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_true = preds_output.label_ids
    
    cm = confusion_matrix(y_true, y_preds)
    labels = list(id2label.values())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted') 
    plt.title('Focal Loss + Attention Pooling Results\n(Class-Balanced Learning)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    print("🚀 Starting FOCAL LOSS + ATTENTION POOLING Training...")
    print(f"🎯 Enhanced Setup: {MODEL_NAME}")
    print("🔧 Features: Class-Balanced Focal Loss + Attention Pooling")
    
    # Create 5-class label mappings
    unique_emotions_5class = ['anger', 'joy', 'melancholy', 'neutral', 'surprise']
    label2id = {label: i for i, label in enumerate(unique_emotions_5class)}
    id2label = {i: label for i, label in enumerate(unique_emotions_5class)}
    
    train_dataset = PrecomputedHiddenStateDataset5Class('train', label2id)
    eval_dataset = PrecomputedHiddenStateDataset5Class('validation', label2id)
    
    print(f"📈 Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")
    
    # Check class distribution for focal loss weights
    train_emotion_counts = Counter(train_dataset.labels_df['emotion_5class'])
    print(f"📊 5-Class Distribution: {dict(train_emotion_counts)}")
    
    # Calculate class-balanced weights for focal loss
    total_samples = len(train_dataset)
    class_weights = []
    for emotion in unique_emotions_5class:
        weight = total_samples / train_emotion_counts[emotion]
        class_weights.append(weight)
    
    # Normalize weights
    class_weights = [w / sum(class_weights) * len(class_weights) for w in class_weights]
    print(f"🎯 Class weights for focal loss: {dict(zip(unique_emotions_5class, class_weights))}")
    
    # Load emotion-specific model and create enhanced classifier
    print(f"🤖 Loading emotion model: {MODEL_NAME}")
    full_model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), ignore_mismatched_sizes=True
    )
    model = FocalAttentionEmotionClassifier(full_model)
    print("✅ Created focal loss + attention emotion classifier!")
    
    # Initialize focal loss with class weights
    focal_loss_fn = ClassBalancedFocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    print("✅ Class-balanced focal loss initialized with γ=2.0")
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
        dataloader_num_workers=0,
        lr_scheduler_type='linear',
        fp16=True if torch.cuda.is_available() else False,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.005
    )

    # Create focal loss trainer
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        focal_loss_fn=focal_loss_fn,
        callbacks=[early_stopping_callback],
    )

    print("\n🎯 Starting FOCAL LOSS + ATTENTION Training...")
    print("🔧 Enhancements: Class-Balanced Focal Loss (γ=2.0) + Multi-Head Attention Pooling")
    print("📊 Expected: Better handling of class imbalance + improved temporal modeling")

    # Train the model
    trainer.train()
    print("\n✅ Focal loss + attention training complete!")

    # Final evaluation
    print("\n📊 Final Evaluation on Validation Set:")
    eval_results = trainer.evaluate()

    print("\n📈 FOCAL LOSS + ATTENTION RESULTS:")
    print(f"🎯 **ACCURACY: {eval_results['eval_accuracy']:.1%}**")
    print(f"🎯 Weighted F1: {eval_results['eval_f1']:.4f}")
    print(f"🎯 Macro F1: {eval_results['eval_f1_macro']:.4f}")
    print(f"🎯 Cohen's Kappa: {eval_results['eval_kappa']:.4f}")

    # Per-class results
    if 'eval_f1_per_class' in eval_results:
        print("\n📊 Per-class F1 scores:")
        for i, (emotion, f1_score) in enumerate(zip(unique_emotions_5class, eval_results['eval_f1_per_class'])):
            print(f"  {emotion}: {f1_score:.4f}")

    # Generate confusion matrix
    plot_confusion_matrix(trainer, eval_dataset, id2label)

    # Save the model
    trainer.save_model(os.path.join(OUTPUT_DIR, "focal_attention_model"))
    print(f"\n💾 Model saved to {os.path.join(OUTPUT_DIR, 'focal_attention_model')}")

    print("\n🎉 Focal Loss + Attention training completed!")
    print("📈 This should show improved performance on minority classes (joy, melancholy)!")

