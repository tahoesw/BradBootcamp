import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv('sentences.csv')

# Split the data into features (X) and labels (y)
X = df['Sentence']
y = df['Source']

# Check class distribution
print(df['Source'].value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['Source'].unique()))

# Tokenize and encode the sentences
X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt')
X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='pt')

# Create a LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels for training data
y_train_encoded = label_encoder.fit_transform(y_train)

# Create a PyTorch tensor for training labels
train_dataset = TensorDataset(
    X_train_tokens['input_ids'],
    X_train_tokens['attention_mask'],
    torch.tensor(y_train_encoded)
)

# Transform the labels for testing data using the same LabelEncoder
y_test_encoded = label_encoder.transform(y_test)

# Create a PyTorch tensor for testing labels
test_dataset = TensorDataset(
    X_test_tokens['input_ids'],
    X_test_tokens['attention_mask'],
    torch.tensor(y_test_encoded)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Set up training parameters with adjusted learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3

for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
# Evaluation
model.eval()
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

# Calculate accuracy
accuracy = accuracy_score(y_test, all_preds)
print(f'BERT Accuracy: {accuracy * 100:.2f}%')
