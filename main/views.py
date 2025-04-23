# main/views.py
from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
import requests
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import model_loader
from .APIs.big5Personality import predict_big5
from .APIs.conversational_style import predict_conversational_style
from .APIs.interest_level import predict_interest_confidence
from .APIs.mbti import predict_mbti
from .APIs.miscellaneous import predict_miscellaneous
from .APIs.moods import predict_moods
from .utils import append_trait_log

# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
def _to_py(v: Any) -> Any:
    """Convert NumPy scalar → Python float/int so JsonResponse is happy."""
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, list):
        return [_to_py(i) for i in v]
    if isinstance(v, dict):
        return {k: _to_py(i) for k, i in v.items()}
    return v


def call_deepseek_llm(prompt: str, max_tokens: int = 128) -> str:
    url = "https://71c6-34-169-234-207.ngrok-free.app/predict"
    payload = {"text": prompt, "max_tokens": max_tokens}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("response", "[no response]")
    except requests.RequestException as e:
        return f"[server error] {e}"


# ────────────────────────────────────────────────────────────────────
# AJAX endpoint – returns BOTH LLM reply *and* trait data
# ────────────────────────────────────────────────────────────────────
@require_POST
@csrf_exempt
def ajax_chat(request):
    body = json.loads(request.body or "{}")
    text = body.get("text", "").strip()
    if not text:
        return JsonResponse({"error": "empty"}, status=400)

    # run all models
    style = predict_conversational_style(text)
    trait_groups = {
        "conversation_styles": style['traits'],
        "moods": predict_moods(text),
        "big_five": predict_big5(text),
        "mbti": predict_mbti(text),
        "misc": predict_miscellaneous(text),
        "changing_traits": predict_interest_confidence(text),
    }
    chatBotReply = call_deepseek_llm(text)
    # log
    conv_id = request.session.get("conv_id") or str(uuid.uuid4())
    request.session["conv_id"] = conv_id
    append_trait_log(conv_id, {"user_message": text, "bot_response" : chatBotReply, "trait_groups": trait_groups})

    return JsonResponse(
        _to_py(
            {
                "response": chatBotReply,
                "trait_groups": trait_groups,
            }
        )
    )


# ────────────────────────────────────────────────────────────────────
# page view – only serves the initial template
# ────────────────────────────────────────────────────────────────────
def chat_view(request):
    if not model_loader.models_loaded:
        return render(request, "loading.html")

    # ensure session objects exist
    if "conv_id" not in request.session:
        request.session["conv_id"] = str(uuid.uuid4())
    if "chat_history" not in request.session:
        request.session["chat_history"] = []

    return render(request, "chat.html")
