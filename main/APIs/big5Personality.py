from pymongo import MongoClient
import gridfs
import torch
import torch.nn as nn
import numpy as np
import pickle
import io, os
from transformers import DistilBertTokenizer, DistilBertModel

# === MongoDB Setup (no django.conf import needed) ===
MONGO_URI = "mongodb://localhost:27017/trait_classification_model_DB"

client = MongoClient(MONGO_URI)
db = client["trait_classification_model_DB"] # picks DB from URI path
fs = gridfs.GridFS(db)


def load_file_from_gridfs(filename):
    file = fs.find_one({"filename": filename})
    if not file:
        raise FileNotFoundError(f"{filename} not found in GridFS")
    return io.BytesIO(file.read())

# === Model Definition ===
class DistilBertTraitClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for name, param in self.bert.named_parameters():
            if "transformer.layer.4" not in name and "transformer.layer.5" not in name:
                param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]
        cls_token = self.dropout(cls_token)
        return self.classifier(cls_token)

# === Cached Loading ===
_cached_model = None
_cached_tokenizer = None
_cached_labels = None

def load_model_and_tokenizer():
    global _cached_model, _cached_tokenizer, _cached_labels
    if _cached_model and _cached_tokenizer and _cached_labels:
        return _cached_model, _cached_tokenizer, _cached_labels

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    label_columns = pickle.load(load_file_from_gridfs("Big5Personality/big5_labels.pkl"))
    model = DistilBertTraitClassifier(num_labels=len(label_columns))
    model.load_state_dict(torch.load(load_file_from_gridfs("Big5Personality/distilbert_big5.pt"), map_location="cpu"))
    # for GPU based machines use map_location=torch.device("cuda")) --- IMPORTANT-NOTE
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_labels = label_columns

    return model, tokenizer, label_columns

# === Prediction Function ===
def predict_big5(text, num_passes=30):
    # some text, one API hit 30 times on that single text to get better results and more accurate results
    model, tokenizer, labels = load_model_and_tokenizer()

    # Enable dropout at inference
    def enable_dropout(m):
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    enable_dropout(model)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    all_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).squeeze(0).numpy() * 100  # Convert to [0, 100] scale
            all_probs.append(probs)

    all_probs = np.array(all_probs)
    # Openness to Experience
    # Conscientiousness
    # Extraversion
    # Agreeableness
    # Neuroticism
    mean_probs = np.mean(all_probs, axis=0)
    std_devs = np.std(all_probs, axis=0)

    def calculate_moe(mean, std):
        moe = round(1.96 * std / np.sqrt(num_passes), 2)
        capped_moe = min(moe, mean, 100 - mean)
        lower = round(mean - capped_moe, 2)
        upper = round(mean + capped_moe, 2)
        return round(capped_moe, 2), f"{lower}% - {upper}%"

    traits = []
    for label, mean, std in zip(labels, mean_probs, std_devs):
        moe, confidence_range = calculate_moe(mean, std)
        traits.append({
            "trait": label,
            "value": round(mean, 2),
            "margin_of_error": moe,
            "range": confidence_range
        })

    return traits

# def print_big5_output(traits):
#     print("\n--- Big Five Personality Prediction ---\n")
#     for trait in traits:
#         print(f"Trait: {trait['trait']}")
#         print(f"  Value            : {round(trait['value'], 2)}%")
#         print(f"  Margin of Error  : +/- {round(trait['margin_of_error'], 2)}%")
#         print(f"  Confidence Range : {trait['range']}")
#         print("-" * 40)
        
# traits = predict_big5("I enjoy philosophical conversations and abstract thinking.")
# print_big5_output(traits)

