# APIs/mbti.py

from pymongo import MongoClient
import gridfs
import torch
import torch.nn as nn
import numpy as np
import io
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn.functional as F

MONGO_URI = "mongodb://localhost:27017/trait_classification_model_DB"

client = MongoClient(MONGO_URI)
db = client["trait_classification_model_DB"] # picks DB from URI path
fs = gridfs.GridFS(db)

# === MBTI Labels ===
label_names = [
    "ENFJ", "ENFP", "ENTJ", "ENTP", "ESFJ", "ESFP", "ESTJ", "ESTP",
    "INFJ", "INFP", "INTJ", "INTP", "ISFJ", "ISFP", "ISTJ", "ISTP"
]
# I --> INFJ, INFP, INTP etc, sum up all the probalities, or accuracy
# E --> INFJ, INFP, INTP etc, sum up all the probalities, or accuracy
# J --> INFJ, INFP, INTP etc, sum up all the probalities, or accuracy


def load_file_from_gridfs(filename):
    file = fs.find_one({"filename": filename})
    if not file:
        raise FileNotFoundError(f"{filename} not found in GridFS")
    return io.BytesIO(file.read())

# === Model Definition ===
class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(DistilBertClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)

# === Cache ===
_cached_model = None
_cached_tokenizer = None

def load_model_and_tokenizer():
    global _cached_model, _cached_tokenizer
    if _cached_model and _cached_tokenizer:
        return _cached_model, _cached_tokenizer

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertClassifier(num_labels=16)
    model.load_state_dict(torch.load(load_file_from_gridfs("MBTI/distilbert_mbti_classifier.pt"), map_location="cpu"))
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    return model, tokenizer


def predict_mbti(text, num_passes=30):
    model, tokenizer = load_model_and_tokenizer()

    # Enable dropout for uncertainty
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
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            all_probs.append(probs)

    all_probs = np.array(all_probs)  # shape: (num_passes, 16)
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)

    def sum_stat(fn, stat_array):
        return sum(p for label, p in zip(label_names, stat_array) if fn(label))

    def calculate_moe(mean, std):
        moe = round(1.96 * std / np.sqrt(num_passes), 2)
        capped_moe = min(moe, mean, 100 - mean)
        lower = round(mean - capped_moe, 2)
        upper = round(mean + capped_moe, 2)
        return round(capped_moe, 2), f"{lower}% - {upper}%"

    traits = []
    for trait_label, trait_filter in [
        ("Introversion (I)", lambda l: l[0] == "I"),
        ("Extraversion (E)", lambda l: l[0] == "E"),
        ("Sensing (S)",      lambda l: l[1] == "S"),
        ("Intuition (N)",    lambda l: l[1] == "N"),
        ("Feeling (F)",      lambda l: l[2] == "F"),
        ("Thinking (T)",     lambda l: l[2] == "T"),
        ("Judging (J)",      lambda l: l[3] == "J"),
        ("Perceiving (P)",   lambda l: l[3] == "P"),
    ]:
        mean_score = round(sum_stat(trait_filter, mean_probs) * 100, 2)
        std_score = sum_stat(trait_filter, std_probs) * 100
        moe, conf_range = calculate_moe(mean_score, std_score)

        traits.append({
            "trait": trait_label,
            "value": mean_score,
            "margin_of_error": moe,
            "range": conf_range
        })

    return traits


# def print_mbti_output(traits):
#     print("\n--- MBTI Trait Predictions ---\n")
#     for trait in traits:
#         print(f"Trait: {trait['trait']}")
#         print(f"  Value            : {trait['value']}%")
#         print(f"  Margin of Error  : ±{trait['margin_of_error']}%")
#         print(f"  Confidence Range : {trait['range']}")
#         print("-" * 40)

# traits = predict_mbti("I prefer deep conversations over small talk.")
# print_mbti_output(traits)

