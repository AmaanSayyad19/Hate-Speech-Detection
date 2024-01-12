import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Custom dataset class
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

# Load the trained model and tokenizer
model_path = './my_moderation_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load and prepare the test data
test_data = pd.read_csv('data2.csv')
test_tokens = tokenizer(list(test_data['text']), padding=True, truncation=True, return_tensors="pt")
test_dataset = HateSpeechDataset(test_tokens, test_data['label'].values)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

# Test loop
model.eval()  # Set the model to evaluation mode
true_test_labels = []
pred_test_labels = []

for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
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
