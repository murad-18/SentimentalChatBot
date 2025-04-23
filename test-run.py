# test_run.py  (place anywhere inside your venv / project root)

import uuid, time
from pprint import pprint
from main.APIs.conversational_style import predict_conversational_style
from main.APIs.moods                import predict_moods
from main.APIs.big5Personality      import predict_big5
from main.APIs.mbti                 import predict_mbti
from main.APIs.miscellaneous        import predict_miscellaneous
from main.APIs.interest_level       import predict_interest_confidence

# --- quick clip helper identical to earlier ---
import numpy as np
def clip_trait(t):
    v = round(t["value"], 2)
    p = max(min(v / 100.0, 0.9999), 1e-4)
    moe = round(min(1.96 * np.sqrt(p * (1 - p)) * 100, v, 100 - v), 2)
    low, hi = max(0, v - moe), min(100, v + moe)
    return dict(trait=t["trait"], value=v, margin_of_error=moe,
                range=f"{low:.2f}-{hi:.2f}")

# --- import your LLM call ---
from main.views import call_deepseek_llm   # reuse same helper

from sample_messages import USER_EXCHANGES  # or paste the list above

def run_simulated_chat():
    conv_id = str(uuid.uuid4())
    log = []


    for turn, user_text in enumerate(USER_EXCHANGES, 1):
        # per‑turn predictions
        style = predict_conversational_style(user_text)
        mood  = predict_moods(user_text)
        big5  = predict_big5(user_text)
        mbti  = predict_mbti(user_text)
        misc  = predict_miscellaneous(user_text)
        ic    = predict_interest_confidence(user_text)

        trait_groups = {
            "Conversational Style": [clip_trait(t) for t in style["traits"]],
            "Moods":               [clip_trait(t) for t in mood],
            "Big Five":            [clip_trait(t) for t in big5],
            "MBTI":                [clip_trait(t) for t in mbti],
            "Miscellaneous":       [clip_trait(t) for t in misc],
            "Interest/Confidence": [clip_trait(t) for t in ic],
        }

        bot_reply = call_deepseek_llm(prompt=user_text)

        log.append({
            "turn": turn,
            "user": user_text,
            "bot":  bot_reply,
            "traits": trait_groups
        })

    return log

def print_log(log):
    for entry in log:
        print(f"\n--- Turn {entry['turn']} ---")
        print("User:", entry["user"])
        print("Bot :", entry["bot"][:120])  # truncate for brevity
        for group, traits in entry["traits"].items():
            print(f"  {group}:")
            for t in traits:
                print(f"    - {t['trait']}: {t['value']}% ±{t['margin_of_error']}% ({t['range']})")

if __name__ == "__main__":
    start = time.time()
    chat_log = run_simulated_chat()
    print_log(chat_log)
    print("\nFinished in", round(time.time() - start, 1), "sec")
