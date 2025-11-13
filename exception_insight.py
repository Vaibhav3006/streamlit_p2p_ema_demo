import json
import pandas as pd
from typing import Callable
import os
from openai import OpenAI
import openai
import streamlit as st

api_key = st.secrets["OPENAI_API_KEY"]

# 1) Prompt builder for one row
def build_exception_prompt(row: pd.Series) -> str:
    """
    Build a 2–3 line explanation prompt for a single exception row.
    Expected columns: exception_type, message, context_json, invoice_number, po_number (optional).
    """
    exception_type = row.get("exception_type", "")
    message = row.get("message", "")
    context_json = row.get("context_json", "")
    invoice_number = row.get("invoice_number", "")
    po_number = row.get("po_number", "")

    # Make context pretty & safe
    try:
        ctx = json.loads(context_json) if isinstance(context_json, str) else context_json
        ctx_str = json.dumps(ctx, ensure_ascii=False)
    except Exception:
        ctx_str = str(context_json)

    prompt = f"""
You are a financial reconciliation assistant.
Your task is to summarize *why* an exception occurred in 2–3 clear lines for a business analyst.

Information about the exception:
- Invoice number: {invoice_number}
- PO number: {po_number}
- Exception type: {exception_type}
- System message: {message}
- Context (JSON): {ctx_str}

Instructions:
1. Explain the root cause in simple, business-friendly language.
2. Mention which values differ (Invoice vs PO vs GRN) when relevant.
3. Include key numbers (amounts, percentages, quantities) if available.
4. Do not repeat column names; write it as a natural mini-explanation.
5. Output 2–3 short sentences, plain text only (no bullet points, no JSON).
"""
    return prompt.strip()


# 2) LLM caller – plug in your own model here
# Example using OpenAI's Python client (you can replace with any LLM you use):
#
# from openai import OpenAI
client = OpenAI(api_key=api_key)
#
def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()

# def call_llm(prompt: str) -> str:
#     """
#     Placeholder: replace body with your actual LLM call.
#     For now it just echoes the prompt tail so you don't crash while wiring.
#     """
#     return "[LLM_OUTPUT_PLACEHOLDER] " + prompt[:200] + " ..."


# 3) Apply across dataframe
def add_llm_exception_descriptions(
    df: pd.DataFrame,
    llm_fn: Callable[[str], str] = call_llm,
    prompt_builder: Callable[[pd.Series], str] = build_exception_prompt,
) -> pd.DataFrame:
    """
    For each row in df, call LLM and add column 'Final_Exception_Description'.
    """
    descriptions = []

    for _, row in df.iterrows():
        prompt = prompt_builder(row)
        explanation = llm_fn(prompt)
        descriptions.append(explanation)

    df = df.copy()
    df["Final_Exception_Description"] = descriptions
    return df
