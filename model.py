import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset, random_split
import torch
# Use the PyTorch implementation of AdamW instead of the one from transformers
from torch.optim import AdamW


# Load your dataset
data = pd.read_csv('data.csv')

# Ensure 'label' column is integer type and handle potential NaN values
data['label'] = data['label'].fillna(0).astype(int)  # Replace NaN with a default value like 0

# Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the input texts
tokens = tokenizer(list(data['text']), padding=True, truncation=True, return_tensors="pt")
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# Custom dataset
class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create the dataset
dataset = HateSpeechDataset(tokens, data['label'].values)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define the number of epochs
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluation loop
    model.eval()
    total_eval_accuracy = 0
    for batch in val_loader:
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        total_eval_accuracy += (predictions == batch['labels'].to(model.device)).sum().item()

    avg_val_accuracy = total_eval_accuracy / len(val_dataset)
    print(f"Validation accuracy: {avg_val_accuracy:.4f}")

# Save the trained model
model.save_pretrained('./my_moderation_model')
