import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
import re
from typing import List

import numpy as np
import streamlit as st
from pypdf import PdfReader


DATA_PATH = Path(__file__).parent / "data" / "japan_travel_guide.txt"
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL_PRIORITY = (
    "qwen2.5-coder:7b",
    "glm-5.1:cloud",
    "kimi-k2.7-code:cloud",
    "minimax-m2:cloud",
    "glm-4.6:cloud",
    "llama3.2:3b",
)
STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "for", "how", "i", "in", "is",
    "it", "my", "of", "on", "or", "the", "this", "to", "what", "with",
}


def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.getvalue().decode("utf-8", errors="replace")


def split_text(text: str, chunk_size: int = 900) -> List[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def ollama_request(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Start Ollama and check its URL "
            f"({base_url}). Details: {error}"
        ) from error


def ollama_models(base_url: str) -> List[str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return [model["name"] for model in json.loads(response.read())["models"]]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read models from Ollama at {base_url}. Details: {error}"
        ) from error


def choose_models(base_url: str) -> tuple[str, str | None]:
    installed = ollama_models(base_url)
    chat_model = next(
        (model for model in CHAT_MODEL_PRIORITY if model in installed),
        installed[0] if installed else "",
    )
    embedding_model = next(
        (
            model
            for model in installed
            if any(name in model.lower() for name in ("embed", "bge", "nomic", "mxbai"))
        ),
        None,
    )
    return chat_model, embedding_model


def embed_texts(config: dict, texts: List[str]) -> np.ndarray:
    if config["provider"] == "ollama":
        response = ollama_request(
            config["base_url"],
            "/api/embed",
            {"model": config["embedding_model"], "input": texts},
        )
        vectors = response["embeddings"]
    else:
        response = config["client"].models.embed_content(
            model=config["embedding_model"],
            contents=texts,
        )
        vectors = [embedding.values for embedding in response.embeddings]
    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-12)
    return matrix


def build_index(config: dict, chunks: List[str]):
    if config["embedding_model"] is None:
        return None
    vectors = embed_texts(config, chunks)
    return vectors


def retrieve(config: dict, index, chunks: List[str], question: str, limit: int = 4) -> List[str]:
    if index is None:
        query_words = {
            word for word in re.findall(r"[a-z0-9]+", question.lower())
            if word not in STOP_WORDS and len(word) > 2
        }
        ranked = sorted(
            chunks,
            key=lambda chunk: sum(word in chunk.lower() for word in query_words),
            reverse=True,
        )
        return ranked[:limit]
    question_vector = embed_texts(config, [question])
    scores = index @ question_vector[0]
    positions = np.argsort(scores)[::-1][:limit]
    return [chunks[position] for position in positions]


def generate_answer(config: dict, question: str, sources: List[str]) -> str:
    context = "\n\n--- SOURCE ---\n\n".join(sources)
    prompt = f"""You are TravelTruth AI, a careful travel-information assistant.
Answer the user's question using only the supplied context.
If the context does not contain the answer, say that clearly and recommend checking
an official source. Never invent visa, safety, health, cost, or legal information.
For time-sensitive information, explicitly remind the user to verify it with an
official source. Keep the answer concise and label the result as Myth, Fact, or
Travel guidance when appropriate.

Context:
{context}

Question: {question}
"""
    if config["provider"] == "ollama":
        response = ollama_request(
            config["base_url"],
            "/api/chat",
            {
                "model": config["generation_model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        return response["message"]["content"]

    response = config["client"].models.generate_content(
        model=config["generation_model"],
        contents=prompt,
    )
    return response.text


