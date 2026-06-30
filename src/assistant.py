"""AI স্বাস্থ্য সহকারী — Claude API দিয়ে ফলাফল ব্যাখ্যা ও প্রশ্নোত্তর।

পূর্বাভাসের প্রসঙ্গ (রোগ, সম্ভাবনা, প্রভাবক ফ্যাক্টর) নিয়ে রোগীর প্রশ্নের
উত্তর দেয় — সহজ ভাষায়, সতর্কতার সাথে, রোগ নির্ণয় না করে।

ব্যবহারের জন্য পরিবেশে `ANTHROPIC_API_KEY` সেট থাকতে হবে। না থাকলে সহকারী
নিষ্ক্রিয় থাকে এবং অ্যাপের বাকি অংশ স্বাভাবিকভাবে কাজ করে।

মডেল নির্বাচন: পরিবেশ ভেরিয়েবল `MEDIPREDICT_MODEL` (ডিফল্ট নিচে)।
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are MediPredict's careful health-information assistant. "
    "You explain machine-learning risk predictions in plain, supportive language. "
    "IMPORTANT RULES:\n"
    "- You are NOT a doctor and must NOT give a diagnosis or prescribe treatment.\n"
    "- The prediction comes from a demo model trained on SYNTHETIC data — say so if relevant.\n"
    "- Encourage consulting a qualified physician for real concerns.\n"
    "- Reply in the SAME language the user asked in (Bengali or English).\n"
    "- Be concise (under ~150 words) and end with a short safety reminder."
)


def is_available() -> bool:
    """ANTHROPIC_API_KEY সেট থাকলে সহকারী ব্যবহারযোগ্য।"""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def build_context_text(context: dict | None) -> str:
    """পূর্বাভাসের প্রসঙ্গকে সংক্ষিপ্ত টেক্সটে রূপান্তর করে।"""
    if not context:
        return "No prediction context was provided."
    parts = [
        f"Disease: {context.get('disease_name_en') or context.get('disease', 'unknown')}",
        f"Predicted risk: {context.get('probability_percent', '?')}% "
        f"({context.get('risk_level_en') or context.get('risk_level', '')})",
    ]
    factors = context.get("explanation") or []
    if factors:
        top = ", ".join(
            f"{f.get('label_en', f.get('feature'))}={f.get('value')}"
            for f in factors[:5]
        )
        parts.append(f"Top contributing factors: {top}")
    return "\n".join(parts)


def build_messages(question: str, context: dict | None) -> list[dict]:
    """Claude-এর জন্য মেসেজ তালিকা তৈরি করে (নেটওয়ার্ক ছাড়াই টেস্টযোগ্য)।"""
    user_content = (
        f"Prediction context:\n{build_context_text(context)}\n\n"
        f"User question: {question}"
    )
    return [{"role": "user", "content": user_content}]


def ask(question: str, context: dict | None = None, model: str | None = None) -> str:
    """প্রশ্নের উত্তর Claude থেকে নিয়ে আসে। API key না থাকলে RuntimeError।"""
    if not question or not question.strip():
        raise ValueError("প্রশ্ন খালি হতে পারে না।")
    if not is_available():
        raise RuntimeError(
            "AI সহকারী নিষ্ক্রিয় — পরিবেশে ANTHROPIC_API_KEY সেট করুন।"
        )

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model or os.environ.get("MEDIPREDICT_MODEL", DEFAULT_MODEL),
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, context),
    )
    return "".join(block.text for block in response.content if block.type == "text")
