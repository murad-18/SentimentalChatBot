# APIs/moods.py
from pymongo import MongoClient
import gridfs
import torch
import torch.nn as nn
import numpy as np
import pickle
import io
from transformers import DistilBertTokenizer

# === MongoDB Setup ===
MONGO_URI = "mongodb://localhost:27017/trait_classification_model_DB"

client = MongoClient(MONGO_URI)
db = client["trait_classification_model_DB"] # picks DB from URI path
fs = gridfs.GridFS(db)

def enable_dropout(self):
    """ Force dropout to remain active during inference. """
    for m in self.modules():
        if isinstance(m, nn.Dropout):
            m.train()



def load_file_from_gridfs(filename):
    file = fs.find_one({"filename": filename})
    if not file:
        raise FileNotFoundError(f"{filename} not found in GridFS")
    return io.BytesIO(file.read())

# === Model Definition ===
class DistilBertMultiLabel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]
        return self.classifier(cls_token)

# === Cached Loading ===
_cached_mood_model = None
_cached_tokenizer = None
_cached_labels = None

def load_model_and_tokenizer():
    global _cached_mood_model, _cached_tokenizer, _cached_labels
    if _cached_mood_model and _cached_tokenizer and _cached_labels:
        return _cached_mood_model, _cached_tokenizer, _cached_labels

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    label_columns = pickle.load(load_file_from_gridfs("Moods/emotion_labels.pkl"))
    model = DistilBertMultiLabel(num_labels=len(label_columns))
    model.load_state_dict(torch.load(load_file_from_gridfs("Moods/distilbert_emotion.pt"), map_location="cpu"))
    model.eval()

    _cached_mood_model = model
    _cached_tokenizer = tokenizer
    _cached_labels = label_columns

    return model, tokenizer, label_columns


def predict_moods(text, num_passes=30):
    model, tokenizer, labels = load_model_and_tokenizer()

    # Enable dropout during inference
    def enable_dropout(m):
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    enable_dropout(model)

    encodings = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    all_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(0).numpy() * 100  # [0–100]
            all_probs.append(probs)

    all_probs = np.array(all_probs)
    mean_probs = np.mean(all_probs, axis=0)
    std_devs = np.std(all_probs, axis=0)

    traits = []
    for label, mean, std in zip(labels, mean_probs, std_devs):
        moe = round(1.96 * std / np.sqrt(num_passes), 2)  # 95% CI
        capped_moe = min(moe, mean, 100 - mean)
        lower = round(mean - capped_moe, 2)
        upper = round(mean + capped_moe, 2)

        traits.append({
            "trait": label,
            "value": round(mean, 2),
            "margin_of_error": round(capped_moe, 2),
            "range": f"{lower}% - {upper}%"
        })

    return traits


# moodsRes = predict_moods("I am glad you made it this far. Hope you for the best. Goodbye!")
# print(moodsRes)
