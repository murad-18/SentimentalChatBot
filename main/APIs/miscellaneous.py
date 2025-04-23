# APIs/miscellaneous.py

from pymongo import MongoClient
import gridfs
import torch
import torch.nn as nn
import numpy as np
import io
import pickle
from transformers import DistilBertTokenizer, DistilBertModel

# === MongoDB Setup ===
MONGO_URI = "mongodb://localhost:27017/trait_classification_model_DB"

client = MongoClient(MONGO_URI)
db = client["trait_classification_model_DB"] # picks DB from URI path
fs = gridfs.GridFS(db)

def load_file_from_gridfs(filename):
    file = fs.find_one({"filename": filename})
    if not file:
        raise FileNotFoundError(f"{filename} not found in GridFS")
    return io.BytesIO(file.read())

# === Trait Config ===
BINARY_COLS = [
    'Human', 'Language_skill', 'Formality', 'Attractiveness', 'Intelligence',
    'Promiscuousness', 'Social_Attitudes', 'Socioeconomic_Status',
    'Listening_Skills', 'Sense_of_Humor', 'Stress_Levels', 'Coy'
]
MULTICLASS_COL = 'Language_name'

# === Model ===
class MultiTaskDistilBERT(nn.Module):
    def __init__(self, num_binary, num_multiclass):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.binary_head = nn.Linear(768, num_binary)
        self.multiclass_head = nn.Linear(768, num_multiclass)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return torch.sigmoid(self.binary_head(pooled)), torch.softmax(self.multiclass_head(pooled), dim=1)

# === Caching ===
_cached_model = None
_cached_tokenizer = None
_cached_label_encoders = None

def load_model_and_tokenizer():
    global _cached_model, _cached_tokenizer, _cached_label_encoders
    if _cached_model and _cached_tokenizer and _cached_label_encoders:
        return _cached_model, _cached_tokenizer, _cached_label_encoders

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    label_encoders = pickle.load(load_file_from_gridfs("Miscellaneous/label_encoders.pkl"))
    language_labels = label_encoders[MULTICLASS_COL].classes_

    model = MultiTaskDistilBERT(len(BINARY_COLS), len(language_labels))
    model.load_state_dict(torch.load(load_file_from_gridfs("Miscellaneous/multitask_distilbert.pt"), map_location="cpu"))
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_label_encoders = label_encoders

    return model, tokenizer, label_encoders

def predict_miscellaneous(text, num_passes=30):
    model, tokenizer, encoders = load_model_and_tokenizer()

    # Enable dropout during inference
    def enable_dropout(m):
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    enable_dropout(model)

    input_data = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]

    binary_results = []
    multiclass_results = []

    with torch.no_grad():
        for _ in range(num_passes):
            binary_out, multiclass_out = model(input_ids, attention_mask)
            binary_results.append(binary_out.squeeze(0).numpy() * 100)
            multiclass_results.append(multiclass_out.squeeze(0).numpy() * 100)

    binary_results = np.array(binary_results)        # shape: (num_passes, num_binary_traits)
    multiclass_results = np.array(multiclass_results)  # shape: (num_passes, num_languages)

    binary_means = np.mean(binary_results, axis=0)
    binary_stds = np.std(binary_results, axis=0)

    multiclass_means = np.mean(multiclass_results, axis=0)
    multiclass_stds = np.std(multiclass_results, axis=0)

    def calculate_moe(mean, std):
        moe = round(1.96 * std / np.sqrt(num_passes), 2)
        capped_moe = min(moe, mean, 100 - mean)
        lower = round(mean - capped_moe, 2)
        upper = round(mean + capped_moe, 2)
        return round(capped_moe, 2), f"{lower}% - {upper}%"

    traits = []

    # # Binary traits
    for trait, mean, std in zip(BINARY_COLS, binary_means, binary_stds):
        moe, confidence_range = calculate_moe(mean, std)
        traits.append({
            "trait": trait,
            "value": round(mean, 2),
            "margin_of_error": moe,
            "range": confidence_range
        })

    # Multiclass language prediction
    lang_labels = encoders[MULTICLASS_COL].classes_
    for lang, mean, std in zip(lang_labels, multiclass_means, multiclass_stds):
        moe, confidence_range = calculate_moe(mean, std)
        traits.append({
            "trait": f"Language_{lang}",
            "value": round(mean, 2),
            "margin_of_error": moe,
            "range": confidence_range
        })

    return traits


# def print_miscellaneous_output(traits):
#     print("\n--- Miscellaneous Trait Predictions ---\n")
#     for trait in traits:
#         print(f"Trait: {trait['trait']}")
#         print(f"  Value            : {trait['value']}%")
#         print(f"  Margin of Error  : ±{trait['margin_of_error']}%")
#         print(f"  Confidence Range : {trait['range']}")
#         print("-" * 40)

# traits = predict_miscellaneous("I'm so glad you're here, it's been ages!")
# print_miscellaneous_output(traits)


