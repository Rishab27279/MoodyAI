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
from datetime import datetime

# Import the model definition and dataset
from model_definition import TrimodalFusionModel

# --- Configuration ---
PROCESSED_DATA_DIR = 'final_processed_trimodal_data'
BATCH_SIZE = 16
MODEL_PATH = 'best_trimodal_model.pth'

class TrimodalDataset(Dataset):
    """Loads the final pre-computed text, audio, and vision features."""
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
        
        text_path = os.path.join(self.data_dir, 'text', f"{file_idx}.pt")
        audio_path = os.path.join(self.data_dir, 'audio', f"{file_idx}.pt")
        vision_path = os.path.join(self.data_dir, 'vision', f"{file_idx}.pt")
        
        text_features = torch.load(text_path, weights_only=True)
        audio_features = torch.load(audio_path, weights_only=True)
        vision_features = torch.load(vision_path, weights_only=True)
        label = self.labels[idx]
        
        return text_features.squeeze(0), audio_features.squeeze(0), vision_features.squeeze(0), torch.tensor(label, dtype=torch.long)

def load_model_with_vision_features(model_path, device):
    """
    Load model and handle dynamic vision projection layer creation
    """
    # Load the saved state dict
    saved_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Check if vision_projection exists in saved state
    vision_projection_exists = any(key.startswith('vision_projection') for key in saved_state_dict.keys())
    
    if vision_projection_exists:
        # Get vision projection input size from saved weights
        vision_weight = saved_state_dict['vision_projection.weight']
        vision_input_size = vision_weight.shape[1]
        print(f"🔍 Found vision projection layer with input size: {vision_input_size}")
    else:
        # Default vision input size (you may need to adjust this)
        vision_input_size = 197376
        print(f"⚠️  No vision projection found, using default size: {vision_input_size}")
    
    # Create model
    model = TrimodalFusionModel(num_classes=5).to(device)
    
    # Create vision projection layer manually if it exists in saved state
    if vision_projection_exists:
        model.vision_projection = nn.Linear(vision_input_size, 256).to(device)
        model.vision_input_size = vision_input_size
    
    # Load state dict with strict=False to handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(saved_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys in model: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys in state_dict: {unexpected_keys}")
    
    return model

def evaluate_model():
    """
    Comprehensive model evaluation function that calculates all metrics
    and saves results in JSON format.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Check if processed data exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        raise FileNotFoundError(f"❌ Processed data not found at: {PROCESSED_DATA_DIR}")
    
    # Check if trained model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Trained model not found at: {MODEL_PATH}")
    
    print("📊 Loading validation dataset...")
    val_dataset = TrimodalDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("🤖 Loading trained model...")
    model = load_model_with_vision_features(MODEL_PATH, device)
    model.eval()
    
    print("🔍 Running evaluation...")
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Inference loop
    with torch.no_grad():
        for text_batch, audio_batch, vision_batch, labels_batch in tqdm(val_loader, desc="Evaluating"):
            text_batch = text_batch.to(device)
            audio_batch = audio_batch.to(device) 
            vision_batch = vision_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            outputs = model(text_batch, audio_batch, vision_batch)
            
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
            'model_path': MODEL_PATH,
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
    results_filename = f"trimodal_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 TRIMODAL FUSION MODEL EVALUATION RESULTS")
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
    
    return evaluation_results

def generate_classification_report():
    """Generate and save a detailed classification report"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and model
    val_dataset = TrimodalDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = load_model_with_vision_features(MODEL_PATH, device)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for text_batch, audio_batch, vision_batch, labels_batch in tqdm(val_loader, desc="Generating Classification Report"):
            text_batch = text_batch.to(device)
            audio_batch = audio_batch.to(device)
            vision_batch = vision_batch.to(device)
            
            outputs = model(text_batch, audio_batch, vision_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Generate classification report
    class_names = ['anger', 'joy', 'melancholy', 'neutral', 'surprise']
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    # Save report
    report_filename = f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write("TRIMODAL FUSION MODEL - CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Total Samples: {len(all_labels)}\n\n")
        f.write(report)
    
    print(f"📄 Classification report saved to: {report_filename}")
    return report

def quick_test_model():
    """Quick test to verify model loading works"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Testing model loading on device: {device}")
    
    try:
        model = load_model_with_vision_features(MODEL_PATH, device)
        print("✅ Model loaded successfully!")
        
        # Test with dummy data
        val_dataset = TrimodalDataset('val')
        sample_text, sample_audio, sample_vision, sample_label = val_dataset[0]
        
        # Add batch dimension
        sample_text = sample_text.unsqueeze(0).to(device)
        sample_audio = sample_audio.unsqueeze(0).to(device)
        sample_vision = sample_vision.unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(sample_text, sample_audio, sample_vision)
            prediction = torch.argmax(output, dim=1)
            
        print(f"✅ Forward pass successful!")
        print(f"   Sample prediction: {prediction.item()}")
        print(f"   Actual label: {sample_label.item()}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Quick test first
        print("🧪 Running quick model test...")
        if not quick_test_model():
            print("❌ Model test failed. Please check your model file.")
            exit(1)
        
        print("\n" + "="*50)
        
        # Run comprehensive evaluation
        results = evaluate_model()
        
        # Generate additional classification report
        generate_classification_report()
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
