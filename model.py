import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, random_split
import torch

# Load your dataset
data = pd.read_csv('data.csv')

# Convert labels to numerical format if they aren't already
# Assuming 'hate' is labeled as 'hate' and non-hate as 'notgiven', you might do:
data['label'] = data['label'].map({'hate': 1, 'notgiven': 0})

# Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the input texts
tokens = tokenizer(list(data['text']), padding=True, truncation=True, return_tensors="pt")
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# Prepare dataset
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

dataset = HateSpeechDataset(tokens, list(data['label']))

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define a training loop
num_epochs = 3  # Define the number of epochs
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluate the model on the validation set
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
