import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
PROCESSED_AUDIO_DIR = 'final_processed_audio' # <-- Use the new directory with pre-computed features
MODEL_NAME = "superb/wav2vec2-base-superb-er"
OUTPUT_DIR = './finetuned-audio-model'
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# --- Custom Dataset for Pre-computed Features ---
class PrecomputedAudioDataset(Dataset):
    """Loads pre-computed audio features and labels."""
    def __init__(self, data_prefix, label2id):
        self.features_dir = os.path.join(PROCESSED_AUDIO_DIR, data_prefix)
        self.labels_df = pd.read_csv(os.path.join(self.features_dir, 'labels.csv'))
        self.label2id = label2id
        self.num_samples = len(self.labels_df)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature_path = os.path.join(self.features_dir, f'features_{idx}.pt')
        # Add weights_only=True to address the FutureWarning
        features = torch.load(feature_path, weights_only=True)
        
        label_str = self.labels_df.iloc[idx]['emotion']
        label_id = self.label2id[label_str]
        
        # The Trainer expects a dictionary
        return {'input_values': features, 'labels': torch.tensor(label_id, dtype=torch.long)}

# --- Evaluation Metrics ---
def compute_metrics(pred):
    """Calculates and returns a dictionary of evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'kappa': kappa
    }

def plot_confusion_matrix(trainer, eval_dataset, id2label, save_path="confusion_matrix_audio.png"):
    """Generates and saves a confusion matrix plot."""
    print("\nGenerating confusion matrix...")
    preds_output = trainer.predict(eval_dataset)
    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_true = [item['labels'].item() for item in eval_dataset]
    
    cm = confusion_matrix(y_true, y_preds)
    labels = list(id2label.values())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Audio-Only Confusion Matrix')
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    # 1. Create Label Mappings from the saved labels file
    train_labels_df = pd.read_csv(os.path.join(PROCESSED_AUDIO_DIR, 'train/labels.csv'))
    unique_emotions = sorted(train_labels_df['emotion'].unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_emotions)}
    id2label = {i: label for i, label in enumerate(unique_emotions)}
    
    # 2. Create Datasets from pre-computed features
    train_dataset = PrecomputedAudioDataset('train', label2id)
    eval_dataset = PrecomputedAudioDataset('validation', label2id)

    # 3. Initialize Model for Fine-tuning
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    
    # 4. Strategic Freezing
    for param in model.base_model.parameters():
        param.requires_grad = False
    print("\nFeature extractor and base transformer layers are FROZEN.")
    print("Only the classifier head is TRAINABLE.")

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
        # --- MODIFICATION: Enable multi-process data loading ---
        dataloader_num_workers=4 
    )

    # 6. Create and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n--- Starting Audio Model Fine-tuning ---")
    trainer.train()
    print("\n✅ Training complete.")

    # 7. Final Evaluation
    print("\n--- Final Evaluation on Validation Set ---")
    eval_results = trainer.evaluate()
    print(eval_results)

    # 8. Plot Confusion Matrix
    plot_confusion_matrix(trainer, eval_dataset, id2label)

    # 9. Save the final model
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print(f"\n✅ Final model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")
