import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


data = pd.read_csv('D:/Downloads/english.csv')

data = data[['processed_crime_info', 'category', 'sub_category']].dropna()
label_encoder_cat = LabelEncoder()
label_encoder_subcat = LabelEncoder()
data['category_label'] = label_encoder_cat.fit_transform(data['category'])
data['sub_category_label'] = label_encoder_subcat.fit_transform(data['sub_category'])
# Split data
X_train, X_test, y_train_cat, y_test_cat, y_train_subcat, y_test_subcat = train_test_split(
    data['processed_crime_info'],
    data['category_label'],
    data['sub_category_label'],
    test_size=0.2,
    random_state=42
)


class CrimeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=128)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        label = torch.tensor(self.labels[idx])
        inputs['labels'] = label
        return inputs
model_cat = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_cat.classes_))
model_subcat = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_subcat.classes_))
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    recall = recall(labels, preds)
    precision = precision(labels, preds)
    return {'accuracy': acc, 'f1': f1 ,'recall': recall ,'precision': Precision }
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train_cat = y_train_cat.reset_index(drop=True).tolist()
y_test_cat = y_test_cat.reset_index(drop=True).tolist()
y_train_subcat = y_train_subcat.reset_index(drop=True).tolist()
y_test_subcat = y_test_subcat.reset_index(drop=True).tolist()# Prepare the datasets
train_dataset_cat = CrimeDataset(X_train, y_train_cat)
test_dataset_cat = CrimeDataset(X_test, y_test_cat)
train_dataset_subcat = CrimeDataset(X_train, y_train_subcat)
test_dataset_subcat = CrimeDataset(X_test, y_test_subcat)
trainer_cat = Trainer(
    model=model_cat,
    args=training_args,
    train_dataset=train_dataset_cat,
    eval_dataset=test_dataset_cat,
    compute_metrics=compute_metrics
)

trainer_subcat = Trainer(
    model=model_subcat,
    args=training_args,
    train_dataset=train_dataset_subcat,
    eval_dataset=test_dataset_subcat,
    compute_metrics=compute_metrics
)
print("Training Category Model...")
trainer_cat.train()
print("Training Sub-category Model...")
trainer_subcat.train()

print("\nCategory classification evaluation:")
cat_eval_results = trainer_cat.evaluate()
print(cat_eval_results)

print("\nSub-category classification evaluation:")
subcat_eval_results = trainer_subcat.evaluate()
print(subcat_eval_results)

