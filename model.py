import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset, random_split
import torch
# Use the PyTorch implementation of AdamW instead of the one from transformers
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix




# Load your dataset
data = pd.read_csv('data.csv')

# Ensure 'label' column is integer type and handle potential NaN values
data['label'] = data['label'].fillna(0).astype(int)  # Replace NaN with a default value like 0

# Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move your model to the selected device
model = model.to(device)

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define the number of epochs
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
    true_labels = []
    pred_labels = []

    # Evaluation loop
    model.eval()
    total_eval_accuracy = 0
    
    # Inside the evaluation loop:
    for batch in val_loader:
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu()  # Move predictions to CPU
        correct_predictions = predictions.eq(batch['labels'].to(predictions.device))
        total_eval_accuracy += correct_predictions.sum().item()

        true_labels.extend(batch['labels'].cpu().numpy())  # Move true labels to CPU
        pred_labels.extend(predictions.numpy())  # predictions are already on CPU

        
    avg_val_accuracy = total_eval_accuracy / len(val_dataset)
    print(f"Validation accuracy: {avg_val_accuracy:.4f}")
    
    # Save the trained model
    model.save_pretrained('./my_moderation_model')
    tokenizer.save_pretrained('./my_moderation_model')
    
    # Calculate metrics
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print("Validation Confusion Matrix:")
    print(conf_matrix)


 

# Load the test data
test_data = pd.read_csv('data2.csv')

# Process the test data as you did the training and validation data
test_tokens = tokenizer(list(test_data['text']), padding=True, truncation=True, return_tensors="pt")
test_dataset = HateSpeechDataset(test_tokens, test_data['label'].values)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

# Collect all labels and predictions for test set
true_test_labels = []
pred_test_labels = []

model.eval()  # Make sure the model is in evaluation mode
for batch in test_loader:
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    true_test_labels.extend(batch['labels'].cpu().numpy())
    pred_test_labels.extend(predictions)

# Calculate test metrics
test_precision = precision_score(true_test_labels, pred_test_labels)
test_recall = recall_score(true_test_labels, pred_test_labels)
test_f1 = f1_score(true_test_labels, pred_test_labels)
test_conf_matrix = confusion_matrix(true_test_labels, pred_test_labels)

print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("Test Confusion Matrix:")
print(test_conf_matrix)

