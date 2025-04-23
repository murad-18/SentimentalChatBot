# APIs/interest_confidence.py

from pymongo import MongoClient
import gridfs
import torch
import torch.nn as nn
import io
import numpy as np
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

# === Model ===
class DistilBertRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 2)  # [Interest, Confidence]

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.regressor(cls_token)

# === Cache ===
_cached_model = None
_cached_tokenizer = None

def load_model_and_tokenizer():
    global _cached_model, _cached_tokenizer
    if _cached_model and _cached_tokenizer:
        return _cached_model, _cached_tokenizer

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertRegressor()
    model.load_state_dict(torch.load(load_file_from_gridfs("ChangingTraits/distilbert_interest_confidence.pt"), map_location="cpu"))
    # when you host it on a server with GPU --> change it map_location=torch.device("cuda")
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    return model, tokenizer

# === Prediction ===
def predict_interest_confidence(text, num_passes=30):
    model, tokenizer = load_model_and_tokenizer()

    # Enable dropout at inference
    def enable_dropout(m):
        for module in m.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    enable_dropout(model)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    all_outputs = []

    with torch.no_grad():
        for _ in range(num_passes):
            raw_outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(0).numpy()
            scaled_outputs = raw_outputs * 50 + 50  # Scale to [0, 100]
            clipped_outputs = np.clip(scaled_outputs, 0, 100)
            all_outputs.append(clipped_outputs)

    all_outputs = np.array(all_outputs)  # shape: (n, 2)
    means = np.mean(all_outputs, axis=0)
    stds = np.std(all_outputs, axis=0)

    def calculate_moe(mean, std):
        moe = round(1.96 * std / np.sqrt(num_passes), 2)
        capped_moe = min(moe, mean, 100 - mean)
        lower = round(mean - capped_moe, 2)
        upper = round(mean + capped_moe, 2)
        return round(capped_moe, 2), f"{lower}% - {upper}%"

    traits = []
    for trait_name, mean, std in zip(["Interest Level", "Confidence Level"], means, stds):
        moe, conf_range = calculate_moe(mean, std)
        traits.append({
            "trait": trait_name,
            "value": round(mean, 2),
            "margin_of_error": moe,
            "range": conf_range
        })

    return traits

# def print_interest_confidence_output(traits):
#     print("\n--- Interest & Confidence Prediction ---\n")
#     for trait in traits:
#         print(f"Trait: {trait['trait']}")
#         print(f"  Value            : {trait['value']}%")
#         print(f"  Margin of Error  : ±{trait['margin_of_error']}%")
#         print(f"  Confidence Range : {trait['range']}")
#         print("-" * 40)

# traits = predict_interest_confidence("I think I'm pretty good at explaining complex things.")
# print_interest_confidence_output(traits)
