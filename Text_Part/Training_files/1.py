import pandas as pd

# Check for data leakage
train_df = pd.read_csv("./text_transcriptions/train_transcriptions.csv")
val_df = pd.read_csv("./text_transcriptions/validation_transcriptions.csv")

# Check for duplicate transcriptions
common_texts = set(train_df['full_transcription']) & set(val_df['full_transcription'])
print(f"Overlapping transcriptions between train and validation: {len(common_texts)}")

# Check emotion distribution
print("\nTraining set emotion distribution:")
print(train_df['emotion'].value_counts())

print("\nValidation set emotion distribution:")
print(val_df['emotion'].value_counts())

# Check for suspicious patterns in transcriptions
print("\nSample training transcriptions:")
for i, text in enumerate(train_df['full_transcription'].head(10)):
    emotion = train_df.iloc[i]['emotion']
    print(f"{i}: [{emotion}] {text}")

print("\nSample validation transcriptions:")
for i, text in enumerate(val_df['full_transcription'].head(10)):
    emotion = val_df.iloc[i]['emotion']
    print(f"{i}: [{emotion}] {text}")

# Check if transcriptions are all placeholders
placeholder_pattern = train_df['full_transcription'].str.contains("This is a .* emotion sample")
print(f"\nPlaceholder transcriptions in training: {placeholder_pattern.sum()}/{len(train_df)}")

placeholder_pattern_val = val_df['full_transcription'].str.contains("This is a .* emotion sample")
print(f"Placeholder transcriptions in validation: {placeholder_pattern_val.sum()}/{len(val_df)}")
