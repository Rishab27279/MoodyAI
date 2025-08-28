import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import the new Bimodal model from the Canvas
from model_definition2 import BimodalFusionModel

# --- Configuration ---
PROCESSED_DATA_DIR = 'final_processed_trimodal_data'
BATCH_SIZE = 16 
EPOCHS = 30 
LEARNING_RATE = 1e-4

class BimodalDataset(Dataset):
    """Loads the final pre-computed text and audio features, ignoring vision."""
    def __init__(self, data_prefix):
        self.data_dir = os.path.join(PROCESSED_DATA_DIR, data_prefix)
        original_labels = np.load(os.path.join(self.data_dir, 'labels.npy'))
        self.indices = np.load(os.path.join(self.data_dir, 'indices.npy'))
        self.num_samples = len(self.indices)

        # Map 7 classes to 5 classes for a more balanced problem
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
        # Filter out any labels that weren't in the map (if any) and their corresponding indices
        self.indices = self.indices[self.labels != -1]
        self.labels = self.labels[self.labels != -1]
        self.num_samples = len(self.indices)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use the successful index to get the correct file name
        file_idx = self.indices[idx]
        
        text_path = os.path.join(self.data_dir, 'text', f"{file_idx}.pt")
        audio_path = os.path.join(self.data_dir, 'audio', f"{file_idx}.pt")
        
        text_features = torch.load(text_path, weights_only=True)
        audio_features = torch.load(audio_path, weights_only=True)
        label = self.labels[idx]

        # Squeeze to remove unnecessary batch dimension from pre-processing
        return text_features.squeeze(0), audio_features.squeeze(0), torch.tensor(label, dtype=torch.long)

def plot_confusion_matrix(labels, preds, save_path="confusion_matrix_bimodal.png"):
    # Updated emotion map for 5 classes
    emotion_map = {0: 'anger', 1: 'joy', 2: 'melancholy', 3: 'neutral', 4: 'surprise'}
    cm = confusion_matrix(labels, preds, labels=list(emotion_map.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_map.values(), yticklabels=emotion_map.values())
    plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.title('Bimodal (Text + Audio) Confusion Matrix')
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(PROCESSED_DATA_DIR):
        raise FileNotFoundError(f"Processed data not found. Please run the data_preparation.py script first.")

    train_dataset = BimodalDataset('train')
    val_dataset = BimodalDataset('val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nCalculating class weights for 5 classes...")
    label_counts = Counter(train_dataset.labels)
    total_samples = len(train_dataset.labels)
    num_classes = 5
    class_weights = torch.tensor([total_samples / label_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float).to(device)
    print("Class weights calculated.")

    model = BimodalFusionModel(num_classes=num_classes).to(device)
    
    print("\n--- Model Architecture ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("--------------------------\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for text_batch, audio_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            text_batch, audio_batch, labels_batch = text_batch.to(device), audio_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(text_batch, audio_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * text_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for text_batch, audio_batch, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                text_batch, audio_batch, labels_batch = text_batch.to(device), audio_batch.to(device), labels_batch.to(device)
                outputs = model(text_batch, audio_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * text_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels_batch.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        val_acc = sum(1 for x,y in zip(all_preds, all_labels) if x == y) / len(all_labels)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_bimodal_model.pth')
            print("✅ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("Early stopping triggered.")
                break
    
    print("✅ Training complete.")

    print("\nLoading best model for visualization...")
    model.load_state_dict(torch.load('best_bimodal_model.pth', weights_only=True))
    
    model.eval()
    all_labels_cm, all_preds_cm = [], []
    with torch.no_grad():
        for text_batch, audio_batch, labels_batch in tqdm(val_loader, desc="Generating predictions for Confusion Matrix"):
            text_batch, audio_batch = text_batch.to(device), audio_batch.to(device)
            outputs = model(text_batch, audio_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_labels_cm.extend(labels_batch.cpu().numpy())
            all_preds_cm.extend(predicted.cpu().numpy())
    plot_confusion_matrix(all_labels_cm, all_preds_cm)
