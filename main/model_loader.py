# models_loader.py
# This file goes in your main Django app (e.g., beside views.py)

from main.APIs.conversational_style import load_model_and_tokenizer as load_convo
from main.APIs.moods import load_model_and_tokenizer as load_moods
from main.APIs.big5Personality import load_model_and_tokenizer as load_big5
from main.APIs.mbti import load_model_and_tokenizer as load_mbti
from main.APIs.miscellaneous import load_model_and_tokenizer as load_misc
from main.APIs.interest_level import load_model_and_tokenizer as load_interest
import threading


# Global flag
models_loaded = False


def preload_models():
    global models_loaded

    try:
        print("Preloading all trait models...")
        load_convo()
        load_moods()
        load_big5()
        load_mbti()
        load_misc()
        load_interest()

        models_loaded = True
        print("All models loaded and ready.")
    except Exception as e:
        print("Model loading failed:", e)
        models_loaded = False


def start_model_loading_async():
    """Run model loading in background thread so server starts instantly."""
    thread = threading.Thread(target=preload_models)
    thread.start()
