# main/utils.py
import json, time, os, uuid
import numpy as np
from pymongo import MongoClient
import gridfs
from django.conf import settings
from pymongo import MongoClient
import gridfs

MONGO_URI = "mongodb://localhost:27017/trait_classification_model_DB"

client = MongoClient(MONGO_URI)
db = client["trait_classification_model_DB"] # picks DB from URI path
fs = gridfs.GridFS(db)

BATCH_SIZE = 50         # upload every 50 exchanges


# def _log_path(conv_id):
#     return settings.TRAIT_LOG_DIR / f"{conv_id}.jsonl"


# def _line_count(path):
#     return sum(1 for _ in open(path, "r", encoding="utf-8")) if path.exists() else 0


# def _upload_and_rotate(conv_id, path):
#     with open(path, "rb") as f:
#         fs.put(f, filename=f"logs/{conv_id}_{int(time.time())}.jsonl",
#                metadata={"conv_id": conv_id})
#     path.unlink(missing_ok=True)      # start fresh file
#     print(f"Uploaded {conv_id} batch to MongoDB and rotated file.")


def _to_py(obj):
    """Recursively convert NumPy scalars to native Python types."""
    if isinstance(obj, (np.generic,)):          # float32, int64, etc.
        return obj.item()
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    return obj


def append_trait_log(conv_id: str, data: dict):
    """
    Append one exchange record and auto-upload every BATCH_SIZE lines.
    """
    path = settings.TRAIT_LOG_DIR / f"{conv_id}.jsonl"

    # 🔑 ensure everything is JSON‑serializable
    record = _to_py({"ts": time.time(), **data})

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    if sum(1 for _ in open(path)) >= BATCH_SIZE:
        with open(path, "rb") as infile:
            fs.put(infile,
                   filename=f"logs/{conv_id}_{int(time.time())}.jsonl",
                   metadata={"conv_id": conv_id})
        path.unlink(missing_ok=True)
        print(f"Uploaded {conv_id} batch to MongoDB and rotated file.")
        
# Add at the bottom of utils.py
def safe_scalar(v):
    """Return built-in types for JSON serialisation."""
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
    except ModuleNotFoundError:
        pass
    return v
