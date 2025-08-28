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

# --- Configuration ---
PROCESSED_DATA_DIR = 'final_processed_trimodal_data'
BATCH_SIZE = 16

class AudioOnlyDataset(Dataset):
    """Loads only the audio features for emotion classification."""
    def __init__(self, data_prefix):
        self.data_dir = os.path.join(PROCESSED_DATA_DIR, data_prefix)
        original_labels = np.load(os.path.join(self.data_dir, 'labels.npy'))
        self.indices = np.load(os.path.join(self.data_dir, 'indices.npy'))
        self.num_samples = len(self.indices)
        
        valid_original_labels = original_labels
        
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
        
        self.labels = np.array([self.seven_to_five_map.get(label, -1) for label in valid_original_labels])
        self.num_samples = len(self.labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        
        # Load only audio features
        audio_path = os.path.join(self.data_dir, 'audio', f"{file_idx}.pt")
        audio_features = torch.load(audio_path, weights_only=True)
        label = self.labels[idx]
        
        return audio_features.squeeze(0), torch.tensor(label, dtype=torch.long)

class AudioOnlyClassifier(nn.Module):
    """Simple audio-only emotion classifier using pre-computed audio embeddings."""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=5, dropout=0.3):
        super(AudioOnlyClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            # Input projection
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
    def forward(self, audio_features):
        return self.classifier(audio_features)

def train_audio_model(device):
    """Train the audio-only model"""
    print("📊 Loading datasets...")
    train_dataset = AudioOnlyDataset('train')
    val_dataset = AudioOnlyDataset('val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("🤖 Creating audio-only model...")
    model = AudioOnlyClassifier(input_dim=768, num_classes=5).to(device)
    
    # Calculate class weights
    from collections import Counter
    label_counts = Counter(train_dataset.labels)
    total_samples = len(train_dataset.labels)
    class_weights = torch.tensor([total_samples / label_counts.get(i, 1) for i in range(5)], dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    print("🏃 Training audio-only model...")
    best_val_acc = 0.0
    num_epochs = 50
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for audio_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            audio_batch, labels_batch = audio_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio_batch)
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
            for audio_batch, labels_batch in val_loader:
                audio_batch, labels_batch = audio_batch.to(device), labels_batch.to(device)
                
                outputs = model(audio_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels_batch.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_audio_only_model.pth')
            print("✅ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏰ Early stopping triggered!")
                break
    
    return model

def evaluate_audio_model():
    """Comprehensive evaluation of audio-only model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Check if processed data exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        raise FileNotFoundError(f"❌ Processed data not found at: {PROCESSED_DATA_DIR}")
    
    # Load validation dataset
    print("📊 Loading validation dataset...")
    val_dataset = AudioOnlyDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create and load model
    print("🤖 Loading/Training audio-only model...")
    model = AudioOnlyClassifier(input_dim=768, num_classes=5).to(device)
    
    # Try to load pre-trained model, otherwise train new one
    if os.path.exists('best_audio_only_model.pth'):
        model.load_state_dict(torch.load('best_audio_only_model.pth', map_location=device, weights_only=True))
        print("✅ Loaded pre-trained audio model")
    else:
        print("🏃 No pre-trained model found, training new one...")
        model = train_audio_model(device)
        model.load_state_dict(torch.load('best_audio_only_model.pth', map_location=device, weights_only=True))
    
    model.eval()
    
    print("🔍 Running evaluation...")
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Evaluation loop
    with torch.no_grad():
        for audio_batch, labels_batch in tqdm(val_loader, desc="Evaluating Audio Model"):
            audio_batch = audio_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            outputs = model(audio_batch)
            
            # Get predictions and probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results
            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Calculate comprehensive metrics
    print("📈 Calculating metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Precision, Recall, F1 - both weighted and per-class
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Class names mapping
    class_names = {0: 'anger', 1: 'joy', 2: 'melancholy', 3: 'neutral', 4: 'surprise'}
    
    # Prepare per-class metrics
    per_class_metrics = {}
    for i, class_name in class_names.items():
        per_class_metrics[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
    
    # Compile all results
    evaluation_results = {
        'model_info': {
            'model_type': 'Audio-Only Emotion Classifier',
            'model_path': 'best_audio_only_model.pth',
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'total_samples': len(all_labels),
            'num_classes': 5,
            'class_mapping': class_names
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
        },
        'detailed_stats': {
            'total_predictions': len(all_preds),
            'correct_predictions': int(sum(1 for x, y in zip(all_preds, all_labels) if x == y)),
            'class_distribution': {
                class_names[i]: int(np.sum(np.array(all_labels) == i)) 
                for i in range(5)
            }
        }
    }
    
    # Save results to JSON
    results_filename = f"audio_only_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("🎵 AUDIO-ONLY EMOTION CLASSIFIER EVALUATION RESULTS")
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
    
    print(f"\n💾 Detailed results saved to: {results_filename}")
    print("="*60)
    
    # Plot confusion matrix
    plot_confusion_matrix_audio(all_labels, all_preds)
    
    return evaluation_results

def plot_confusion_matrix_audio(labels, preds, save_path="audio_only_confusion_matrix.png"):
    """Plot and save confusion matrix for audio-only model"""
    emotion_map = {0: 'anger', 1: 'joy', 2: 'melancholy', 3: 'neutral', 4: 'surprise'}
    cm = confusion_matrix(labels, preds, labels=list(emotion_map.keys()))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_map.values(), 
                yticklabels=emotion_map.values())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Audio-Only Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.show()

def generate_classification_report_audio():
    """Generate detailed classification report for audio model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = AudioOnlyDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = AudioOnlyClassifier(input_dim=768, num_classes=5).to(device)
    model.load_state_dict(torch.load('best_audio_only_model.pth', map_location=device, weights_only=True))
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for audio_batch, labels_batch in tqdm(val_loader, desc="Generating Classification Report"):
            audio_batch = audio_batch.to(device)
            
            outputs = model(audio_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Generate classification report
    class_names = ['anger', 'joy', 'melancholy', 'neutral', 'surprise']
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    # Save report
    report_filename = f"audio_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write("AUDIO-ONLY EMOTION CLASSIFIER - CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Audio-Only Classifier\n")
        f.write(f"Total Samples: {len(all_labels)}\n\n")
        f.write(report)
    
    print(f"📄 Classification report saved to: {report_filename}")
    return report

if __name__ == "__main__":
    try:
        # Run comprehensive evaluation
        print("🎵 Starting Audio-Only Model Evaluation...")
        results = evaluate_audio_model()
        
        # Generate additional classification report
        generate_classification_report_audio()
        
        print("\n✅ Audio-only evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
