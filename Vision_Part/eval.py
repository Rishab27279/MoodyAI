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
import timm

# --- Configuration ---
PROCESSED_DATA_DIR = 'final_processed_trimodal_data'
BATCH_SIZE = 8  # Small batch size for memory efficiency

class VisionOnlyDataset(Dataset):
    """Loads only vision features for evaluation."""
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
        vision_path = os.path.join(self.data_dir, 'vision', f"{file_idx}.pt")
        vision_features = torch.load(vision_path, weights_only=True)
        label = self.labels[idx]
        
        return vision_features.squeeze(0), torch.tensor(label, dtype=torch.long)

class DINOv2VisionClassifier(nn.Module):
    """Vision-only emotion classifier using DINOv2 features with simple classifier head."""
    
    def __init__(self, num_classes=5):
        super(DINOv2VisionClassifier, self).__init__()
        
        print("Loading DINOv2 Vision Encoder...")
        # Use the same model as in your data_preparation.py
        self.vision_model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            in_chans=3,
            img_size=224
        )
        self.vision_model.eval()
        
        # Freeze DINOv2 parameters - we only want inference
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        print("✅ DINOv2 Vision Encoder loaded.")
        
        # Get DINOv2 feature dimension (768 for vit_base)
        # Based on your data_preparation.py, features are processed as: features.view(2, 5, -1).mean(dim=1)
        # This means you have 2 samples per video, each with some feature dim
        # Let's determine the actual feature size from your saved features
        self.feature_dim = None  # Will be set dynamically
        self.classifier = None   # Will be created after determining feature size
        
    def _create_classifier(self, feature_dim):
        """Create classifier based on actual feature dimensions"""
        self.feature_dim = feature_dim
        
        # Simple but effective classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)  # 5 emotion classes
        )
        
        print(f"✅ Classifier created for feature dimension: {feature_dim}")
    
    def forward(self, vision_features):
        """Forward pass using pre-computed DINOv2 features"""
        
        # Determine feature size and create classifier if needed
        if self.classifier is None:
            if len(vision_features.shape) == 3:
                # Shape: (batch, seq_len, features) - flatten sequence dimension
                feature_dim = vision_features.shape[1] * vision_features.shape[2]
            else:
                # Shape: (batch, features)
                feature_dim = vision_features.shape[1]
            
            self._create_classifier(feature_dim)
            self.classifier = self.classifier.to(vision_features.device)
        
        # Flatten vision features if needed
        if len(vision_features.shape) > 2:
            batch_size = vision_features.shape[0]
            vision_features = vision_features.view(batch_size, -1)
        
        # Pass through classifier
        output = self.classifier(vision_features)
        return output

def evaluate_vision_dinov2():
    """Evaluate vision-only model using DINOv2 features"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        raise FileNotFoundError(f"❌ Processed data not found at: {PROCESSED_DATA_DIR}")
    
    print("📊 Loading validation dataset...")
    val_dataset = VisionOnlyDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("🤖 Creating DINOv2 vision-only model...")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    model = DINOv2VisionClassifier(num_classes=5).to(device)
    model.eval()
    
    print("🔍 Running DINOv2 vision-only evaluation...")
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Evaluation loop
    with torch.no_grad():
        for i, (vision_batch, labels_batch) in enumerate(tqdm(val_loader, desc="DINOv2 Vision Evaluation")):
            
            # Clear cache periodically
            if i % 20 == 0:
                torch.cuda.empty_cache()
            
            vision_batch = vision_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            try:
                # Forward pass
                outputs = model(vision_batch)
                
                # Get predictions and probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store results
                all_labels.extend(labels_batch.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  Skipping batch {i} due to memory issue")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
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
            'model_type': 'DINOv2 Vision-Only Emotion Classifier',
            'backbone': 'vit_base_patch14_dinov2.lvd142m',
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'feature_dimension': model.feature_dim,
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
        }
    }
    
    # Save results
    results_filename = f"dinov2_vision_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    # Print results
    print("\n" + "="*60)
    print("👁️  DINOV2 VISION-ONLY EVALUATION RESULTS")
    print("="*60)
    print(f"📊 Total Samples: {len(all_labels)}")
    print(f"🔍 Feature Dimension: {model.feature_dim:,}")
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
    plot_confusion_matrix_dinov2(all_labels, all_preds)
    
    return evaluation_results

def plot_confusion_matrix_dinov2(labels, preds, save_path="dinov2_vision_confusion_matrix.png"):
    """Plot confusion matrix for DINOv2 vision model"""
    emotion_map = {0: 'anger', 1: 'joy', 2: 'melancholy', 3: 'neutral', 4: 'surprise'}
    cm = confusion_matrix(labels, preds, labels=list(emotion_map.keys()))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=emotion_map.values(), 
                yticklabels=emotion_map.values())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('DINOv2 Vision-Only Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.show()

def generate_classification_report_dinov2():
    """Generate detailed classification report"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = VisionOnlyDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = DINOv2VisionClassifier(num_classes=5).to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for vision_batch, labels_batch in tqdm(val_loader, desc="Generating Classification Report"):
            vision_batch = vision_batch.to(device)
            
            outputs = model(vision_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    class_names = ['anger', 'joy', 'melancholy', 'neutral', 'surprise']
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    report_filename = f"dinov2_vision_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write("DINOV2 VISION-ONLY EMOTION CLASSIFIER - CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: DINOv2 vit_base_patch14_dinov2.lvd142m\n")
        f.write(f"Total Samples: {len(all_labels)}\n\n")
        f.write(report)
    
    print(f"📄 Classification report saved to: {report_filename}")
    return report

if __name__ == "__main__":
    try:
        print("👁️  Starting DINOv2 Vision-Only Evaluation...")
        print("🔧 Using the same DINOv2 model as your data preparation")
        print("🚫 Pure inference - no training required!")
        
        results = evaluate_vision_dinov2()
        
        # Generate classification report
        generate_classification_report_dinov2()
        
        print("\n✅ DINOv2 vision-only evaluation completed!")
        print("📝 This uses the exact same vision features as your trimodal model")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
