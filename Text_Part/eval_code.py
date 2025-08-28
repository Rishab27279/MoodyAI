import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix, 
    accuracy_score,
    cohen_kappa_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration ---
PROCESSED_DATA_DIR = 'final_processed_trimodal_data'
BATCH_SIZE = 16
ORIGINAL_TEXT_MODEL_NAME = "distilbert-base-uncased"

class TextOnlyDataset(Dataset):
    """Loads only text features for emotion classification."""
    def __init__(self, data_prefix):
        self.data_dir = os.path.join(PROCESSED_DATA_DIR, data_prefix)
        original_labels = np.load(os.path.join(self.data_dir, 'labels.npy'))
        self.indices = np.load(os.path.join(self.data_dir, 'indices.npy'))
        
        # Map 7 classes to 5 classes
        self.seven_to_five_map = {
            0: 3, # neutral -> neutral
            1: 1, # joy -> joy
            2: 2, # sadness -> melancholy
            3: 0, # anger -> anger
            4: 4, # surprise -> surprise
            5: 2, # fear -> melancholy
            6: 2  # disgust -> melancholy
        }
        
        self.labels = np.array([self.seven_to_five_map.get(label, -1) for label in original_labels])
        self.num_samples = len(self.labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        
        # Load only text features
        text_path = os.path.join(self.data_dir, 'text', f"{file_idx}.pt")
        text_features = torch.load(text_path, weights_only=True)
        label = self.labels[idx]
        
        return text_features.squeeze(0), torch.tensor(label, dtype=torch.long)

class DistilBERTTextClassifier(nn.Module):
    """Text-only emotion classifier using DistilBERT features with simple classifier head."""
    
    def __init__(self, num_classes=5):
        super(DistilBERTTextClassifier, self).__init__()
        
        print(f"Loading DistilBERT Text Encoder: {ORIGINAL_TEXT_MODEL_NAME}...")
        # Use the same model as in your data_preparation.py
        self.text_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_TEXT_MODEL_NAME)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            ORIGINAL_TEXT_MODEL_NAME, use_safetensors=True
        )
        self.text_model.eval()
        
        # Freeze DistilBERT parameters - we only want inference
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        print("✅ DistilBERT Text Encoder loaded.")
        
        # DistilBERT feature dimension is 768
        self.feature_dim = 768
        
        # Simple but effective classifier (similar to your audio model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        print(f"✅ Classifier created for DistilBERT features (dim: {self.feature_dim})")
    
    def forward(self, text_features):
        """Forward pass using pre-computed DistilBERT features"""
        # text_features shape: (batch_size, 768)
        output = self.classifier(text_features)
        return output

def train_text_model(device):
    """Train the text-only model with early stopping at 50% accuracy"""
    print("📊 Loading datasets...")
    train_dataset = TextOnlyDataset('train')
    val_dataset = TextOnlyDataset('val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("🤖 Creating DistilBERT text-only model...")
    model = DistilBERTTextClassifier(num_classes=5).to(device)
    
    # Calculate class weights
    from collections import Counter
    label_counts = Counter(train_dataset.labels)
    total_samples = len(train_dataset.labels)
    class_weights = torch.tensor([total_samples / label_counts.get(i, 1) for i in range(5)], dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("🏃 Training text-only model...")
    print("🎯 Will stop training as soon as validation accuracy reaches 50%")
    
    best_val_acc = 0.0
    num_epochs = 50
    target_accuracy = 0.50  # Stop at 50%
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for text_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            text_batch, labels_batch = text_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(text_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for text_batch, labels_batch in val_loader:
                text_batch, labels_batch = text_batch.to(device), labels_batch.to(device)
                
                outputs = model(text_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels_batch.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_text_only_model.pth')
            print("✅ New best model saved!")
        
        # Early stopping at target accuracy
        if val_acc >= target_accuracy:
            print(f"🎯 Target accuracy {target_accuracy:.1%} reached at epoch {epoch+1}!")
            print("⏹️  Stopping training early as requested.")
            break
    
    print(f"🏁 Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model

def evaluate_text_distilbert():
    """Evaluate text-only model using DistilBERT features"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        raise FileNotFoundError(f"❌ Processed data not found at: {PROCESSED_DATA_DIR}")
    
    print("📊 Loading validation dataset...")
    val_dataset = TextOnlyDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("🤖 Creating/Training DistilBERT text-only model...")
    
    # Try to load pre-trained model, otherwise train new one
    if os.path.exists('best_text_only_model.pth'):
        model = DistilBERTTextClassifier(num_classes=5).to(device)
        model.load_state_dict(torch.load('best_text_only_model.pth', map_location=device, weights_only=True))
        print("✅ Loaded pre-trained text model")
    else:
        print("🏃 No pre-trained model found, training new one...")
        model = train_text_model(device)
        model.load_state_dict(torch.load('best_text_only_model.pth', map_location=device, weights_only=True))
    
    model.eval()
    
    print("🔍 Running DistilBERT text-only evaluation...")
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Evaluation loop
    with torch.no_grad():
        for text_batch, labels_batch in tqdm(val_loader, desc="DistilBERT Text Evaluation"):
            text_batch = text_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            outputs = model(text_batch)
            
            # Get predictions and probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results
            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Calculate comprehensive metrics
    print("📈 Calculating metrics...")
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    kappa = cohen_kappa_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    class_names = {0: 'anger', 1: 'joy', 2: 'melancholy', 3: 'neutral', 4: 'surprise'}
    
    per_class_metrics = {}
    for i, class_name in class_names.items():
        per_class_metrics[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
    
    evaluation_results = {
        'model_info': {
            'model_type': 'DistilBERT Text-Only Emotion Classifier',
            'backbone': ORIGINAL_TEXT_MODEL_NAME,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'feature_dimension': 768,
            'total_samples': len(all_labels),
            'num_classes': 5,
            'class_mapping': class_names,
            'early_stopping': 'Stopped at 50% validation accuracy'
        },
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_score_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_score_macro': float(f1_macro),
            'cohen_kappa': float(kappa)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': {
            'matrix': conf_matrix.tolist(),
            'class_order': list(class_names.values())
        }
    }
    
    # Save results
    results_filename = f"distilbert_text_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    # Print results
    print("\n" + "="*60)
    print("📝 DISTILBERT TEXT-ONLY EVALUATION RESULTS")
    print("="*60)
    print(f"📊 Total Samples: {len(all_labels)}")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"⚖️  Precision (Weighted): {precision_weighted:.4f}")
    print(f"🔍 Recall (Weighted): {recall_weighted:.4f}")
    print(f"🏆 F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"📏 Cohen's Kappa: {kappa:.4f}")
    print()
    print("📋 Per-Class Performance:")
    print("-" * 40)
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name.capitalize():>12}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1_score']:.3f} (n={metrics['support']})")
    
    print(f"\n💾 Results saved to: {results_filename}")
    
    # Plot confusion matrix
    plot_confusion_matrix_text(all_labels, all_preds)
    
    return evaluation_results

def plot_confusion_matrix_text(labels, preds, save_path="distilbert_text_confusion_matrix.png"):
    """Plot confusion matrix for DistilBERT text model"""
    emotion_map = {0: 'anger', 1: 'joy', 2: 'melancholy', 3: 'neutral', 4: 'surprise'}
    cm = confusion_matrix(labels, preds, labels=list(emotion_map.keys()))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=emotion_map.values(), 
                yticklabels=emotion_map.values())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('DistilBERT Text-Only Model Confusion Matrix\n(Training stopped at 50% accuracy)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.show()

def generate_classification_report_text():
    """Generate detailed classification report"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = TextOnlyDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = DistilBERTTextClassifier(num_classes=5).to(device)
    model.load_state_dict(torch.load('best_text_only_model.pth', map_location=device, weights_only=True))
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for text_batch, labels_batch in tqdm(val_loader, desc="Generating Classification Report"):
            text_batch = text_batch.to(device)
            
            outputs = model(text_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    class_names = ['anger', 'joy', 'melancholy', 'neutral', 'surprise']
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    report_filename = f"distilbert_text_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write("DISTILBERT TEXT-ONLY EMOTION CLASSIFIER - CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {ORIGINAL_TEXT_MODEL_NAME}\n")
        f.write(f"Training: Stopped at 50% validation accuracy\n")
        f.write(f"Total Samples: {len(all_labels)}\n\n")
        f.write(report)
    
    print(f"📄 Classification report saved to: {report_filename}")
    return report

if __name__ == "__main__":
    try:
        print("📝 Starting DistilBERT Text-Only Evaluation...")
        print("🔧 Using the same DistilBERT model as your data preparation")
        print("🎯 Training will stop as soon as 50% validation accuracy is reached")
        
        results = evaluate_text_distilbert()
        
        # Generate classification report
        generate_classification_report_text()
        
        print("\n✅ DistilBERT text-only evaluation completed!")
        print("📝 This uses the exact same text features as your trimodal model")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
