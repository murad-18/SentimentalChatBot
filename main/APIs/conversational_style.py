from pymongo import MongoClient
import gridfs
import torch
import torch.nn as nn
import numpy as np
import io
from transformers import DistilBertTokenizer, DistilBertModel

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/trait_classification_model_DB"

client = MongoClient(MONGO_URI)
db = client["trait_classification_model_DB"] # picks DB from URI path
fs = gridfs.GridFS(db)

def load_file_from_gridfs(filename):
    file = fs.find_one({"filename": filename})
    if not file:
        raise FileNotFoundError(f"{filename} not found in GridFS")
    return io.BytesIO(file.read())

# Label names for conversational styles
label_names = ["Assertive", "Competitive", "Cooperative", "Passive", "Passive-Aggressive"]

# Classifier Model
class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(DistilBertClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)

# Add global cache
_cached_model = None
_cached_tokenizer = None

def load_model_and_tokenizer():
    global _cached_model, _cached_tokenizer
    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertClassifier(num_labels=5)
    model.load_state_dict(torch.load(load_file_from_gridfs("conversationalStyle/distilbert_communication_style.pt"), map_location="cpu"))
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    return model, tokenizer


# Prediction function
def predict_conversational_style(text, num_passes=1):
    model, tokenizer = load_model_and_tokenizer()

    # Enable dropout during inference
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
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).numpy()
            all_probs.append(probs * 100)  # scale to 0–100%

    all_probs = np.array(all_probs)  # shape: (n, 5)
    mean_probs = np.mean(all_probs, axis=0)
    std_devs = np.std(all_probs, axis=0)

    def calculate_moe(mean, std):
        moe = round(1.96 * std / np.sqrt(num_passes), 2)
        capped_moe = min(moe, mean, 100 - mean)
        lower = round(mean - capped_moe, 2)
        upper = round(mean + capped_moe, 2)
        return round(capped_moe, 2), f"{lower}% - {upper}%"

    traits = []
    for label, mean, std in zip(label_names, mean_probs, std_devs):
        moe, conf_range = calculate_moe(mean, std)
        traits.append({
            "trait": label,
            "value": round(mean, 2),
            "margin_of_error": moe,
            "range": conf_range
        })

    # Pick the top predicted trait
    top_trait = max(traits, key=lambda x: x["value"])

    return {
        "predicted_trait": top_trait["trait"],
        "accuracy": top_trait["value"],
        "traits": traits
    }

# def print_conversational_style_output(result):
#     print("\n--- Conversational Style Prediction ---\n")
#     print(f"Predicted Style : {result['predicted_trait']}")
#     print(f"Confidence       : {round(result['accuracy'], 2)}%\n")
#     print("Detailed Breakdown:")
#     print("-" * 40)
#     for trait in result["traits"]:
#         print(f"Trait: {trait['trait']}")
#         print(f"  Value            : {round(trait['value'], 2)}%")
#         print(f"  Margin of Error  : +/- {round(trait['margin_of_error'], 2)}%")
#         print(f"  Confidence Range : {trait['range']}")
#         print("-" * 40)


# result = predict_conversational_style("He is a good boy, and he does his homework neatly.")
# print(result)
