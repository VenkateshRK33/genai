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


def generate_itinerary(config: dict, trip: dict, sources: List[str]) -> str:
    context = "\n\n--- SOURCE ---\n\n".join(sources)
    prompt = f"""You are TravelTruth AI, a careful Japan trip planner.
Create a practical, realistic itinerary using only the supplied context for travel
facts. Do not invent current visa, prices, opening hours, or transport schedules.
Clearly label assumptions and tell the traveler what must be verified officially.

Traveler preferences:
{json.dumps(trip, indent=2)}

Knowledge-base context:
{context}

Return the plan with these sections:
1. Trip snapshot
2. Day-by-day itinerary with morning, afternoon, and evening
3. Transport approach
4. Accommodation area suggestions (not made-up hotels)
5. Food and culture ideas
6. Packing and preparation checklist
7. Safety and official verification reminders
Keep it useful and concise. Avoid cramming too many activities into one day.
"""
    if config["provider"] == "ollama":
        response = ollama_request(
            config["base_url"],
            "/api/chat",
            {
                "model": config["generation_model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3},
            },
        )
        return response["message"]["content"]

    response = config["client"].models.generate_content(
        model=config["generation_model"],
        contents=prompt,
    )
    return response.text


@st.cache_resource(show_spinner=False)
def prepare_knowledge_base(config_key: tuple, uploaded_contents: tuple[str, ...]):
    provider, base_url, embedding_model, generation_model, api_key = config_key
    config = {
        "provider": provider,
        "base_url": base_url,
        "embedding_model": embedding_model,
        "generation_model": generation_model,
    }
    if provider == "gemini":
        from google import genai

        config["client"] = genai.Client(api_key=api_key)
    text = DATA_PATH.read_text(encoding="utf-8")
    text += "\n\n" + "\n\n".join(uploaded_contents)
    chunks = split_text(text)
    return config, chunks, build_index(config, chunks)


def main() -> None:
    st.set_page_config(
        page_title="TravelTruth AI",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .hero { padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
                background: linear-gradient(120deg, #102a43, #1d4e89);
                color: white; }
        .hero h1 { margin: 0; font-size: 2.6rem; }
        .hero p { margin: .5rem 0 0; color: #d9eaf7; font-size: 1.1rem; }
        .card { padding: 1rem 1.2rem; border: 1px solid #d9e2ec; border-radius: 14px;
                background: #f7fafc; min-height: 100px; }
        .fact { padding: .8rem 1rem; border-radius: 12px; background: #eef6ff;
                border-left: 4px solid #2f80ed; min-height: 80px; }
        .muted { color: #627d98; font-size: .9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><h1>✈️ TravelTruth AI</h1>'
        "<p>Ask smarter questions. Get answers grounded in your Japan travel documents.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Travel setup")
        st.caption("Your local-first RAG assistant")
    provider = st.sidebar.selectbox("AI provider", ["Ollama (local)", "Gemini (cloud)"])
    if provider.startswith("Ollama"):
        base_url = st.sidebar.text_input(
            "Ollama URL", value=os.getenv("OLLAMA_URL", OLLAMA_URL)
        )
        try:
            detected_chat_model, detected_embedding_model = choose_models(base_url)
        except RuntimeError as error:
            st.error(str(error))
            return
        embedding_model = detected_embedding_model
        generation_model = detected_chat_model
        st.sidebar.success(f"Using local model: {generation_model}")
        if embedding_model:
            st.sidebar.caption(f"Auto-selected embedding model: `{embedding_model}`")
        else:
            st.sidebar.caption("No embedding model installed; using local keyword retrieval.")
        api_key = ""
        provider_name = "ollama"
    else:
        api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input(
            "Gemini API key", type="password"
        )
        base_url = ""
        embedding_model = "gemini-embedding-001"
        generation_model = "gemini-2.0-flash"
        provider_name = "gemini"
    uploaded_files = st.sidebar.file_uploader(
        "Add your travel documents (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if provider_name == "gemini" and not api_key:
        st.info("Add a Gemini API key or select Ollama to run locally.")
        return

    uploaded_contents = tuple(read_uploaded_file(file) for file in uploaded_files)
    config_key = (
        provider_name,
        base_url,
        embedding_model,
        generation_model,
        api_key,
    )
with st.spinner("Preparing the travel knowledge base..."):
        config, chunks, index = prepare_knowledge_base(config_key, uploaded_contents)

    overview, ask_tab, plan, planner_tab = st.tabs(
        ["🌸 Japan overview", "💬 Ask TravelTruth", "🧳 Checklist", "🗺️ Trip planner"]
    )

    with overview:
        st.subheader("Start with the essentials")
        facts = [
            ("Best for", "City breaks, food, culture, temples, and efficient public transport"),
            ("Popular route", "Tokyo → Kyoto → Osaka for a balanced first visit"),
            ("Remember", "Visa rules, weather, transport schedules, and entry requirements change"),
        ]
        fact_columns = st.columns(3)
        for column, (title, text) in zip(fact_columns, facts):
            with column:
                st.markdown(
                    f'<div class="fact"><strong>{title}</strong><br>'
                    f'<span class="muted">{text}</span></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("#### Useful questions to ask")
        question_examples = [
            "What should I pack for Japan in autumn?",
            "How should I plan transport between Tokyo and Kyoto?",
            "What etiquette should I know before visiting a temple?",
        ]
        for example in question_examples:
            st.markdown(f"- {example}")

    with ask_tab:
        left, right = st.columns([1.5, 1])
        with left:
            st.subheader("What do you want to verify?")
            category = st.selectbox(
                "Question category",
                ["Myth or fact", "Visa and entry", "Culture and etiquette", "Transport", "Safety", "Planning and packing"],
                label_visibility="collapsed",
            )
            examples = {
                "Myth or fact": "Is tipping required in Japan?",
                "Visa and entry": "What should I check before applying for a visa?",
                "Culture and etiquette": "What etiquette should I know at a temple?",
                "Transport": "How should I travel between Tokyo and Kyoto?",
                "Safety": "What should I verify before traveling?",
                "Planning and packing": "What should I pack for a seven-day Japan trip?",
            }
            st.caption(f"Try: {examples[category]}")
            question = st.text_input(
                "Your question",
                placeholder=examples[category],
                key="question",
            )
            ask = st.button("🔎 Check with TravelTruth", type="primary", use_container_width=True)
        with right:
            st.markdown(
                '<div class="card"><strong>Reliable answers, step by step</strong><br><br>'
                '<span class="muted">1. Add trusted travel documents<br>'
                "2. Ask in your own words<br>"
                "3. Review the answer and source excerpts<br>"
                "4. Verify time-sensitive advice officially</span></div>",
                unsafe_allow_html=True,
            )
        if ask and not question.strip():
            st.warning("Please enter a travel question first.")
            return
        if ask:
            try:
                with st.spinner("Checking your travel documents..."):
                    sources = retrieve(config, index, chunks, question, retrieval_depth)
                    answer = generate_answer(config, question, sources)
                st.subheader("TravelTruth result")
                st.info(answer)
                with st.expander(f"📚 View {len(sources)} retrieved source excerpts"):
                    for number, source in enumerate(sources, start=1):
                        st.markdown(f"**Source {number}**")
                        st.write(source)
            except (RuntimeError, KeyError, ValueError) as error:
                st.error(f"We could not complete that check: {error}")

    with plan:
        st.subheader("Japan pre-trip checklist")
        checklist = [
            "Verify passport, visa, and entry requirements",
            "Save accommodation addresses in English and Japanese",
            "Arrange connectivity and download offline maps",
            "Check transport routes, reservations, and luggage rules",
            "Pack for the season and bring comfortable walking shoes",
            "Save insurance, medicine, and emergency information",
        ]
        for item in checklist:
            st.checkbox(item, key=f"check_{item}")

    with planner_tab:
        st.subheader("Build your Japan trip")
        st.caption(
            "Answer a few questions and TravelTruth will create a personalized "
            "day-by-day plan grounded in your travel documents."
        )
        with st.form("trip_planner_form"):
            destination = st.multiselect(
                "Where would you like to go?",
                ["Tokyo", "Kyoto", "Osaka", "Nara", "Hokkaido", "Okinawa"],
                default=["Tokyo", "Kyoto", "Osaka"],
            )
            duration = st.slider("How many days?", min_value=3, max_value=21, value=7)
            travel_month = st.selectbox(
                "When are you traveling?",
                ["Not decided", "Spring", "Summer", "Autumn", "Winter"],
            )
            traveler_type = st.selectbox(
                "Who are you traveling with?",
                ["Solo", "Couple", "Family with children", "Friends", "Senior-friendly trip"],
            )
            interests = st.multiselect(
                "What do you enjoy?",
                ["Food", "Culture and temples", "Nature", "Shopping", "Anime and games",
                 "History", "Nightlife", "Photography"],
                default=["Food", "Culture and temples"],
            )
            pace = st.select_slider(
                "Preferred pace",
                options=["Relaxed", "Balanced", "Packed"],
                value="Balanced",
            )
            budget = st.selectbox(
                "Budget preference",
                ["Value-conscious", "Comfortable", "Premium"],
            )
            accessibility = st.text_input(
                "Special needs or constraints",
                placeholder="Example: vegetarian food, limited walking, allergy",
            )
            planner_submitted = st.form_submit_button(
                "✨ Create my travel plan",
                type="primary",
                use_container_width=True,
            )
        if planner_submitted:
            if not destination:
                st.warning("Choose at least one destination.")
            else:
                trip = {
                    "destinations": destination,
                    "duration_days": duration,
                    "travel_season": travel_month,
                    "travel_group": traveler_type,
                    "interests": interests,
                    "pace": pace,
                    "budget_preference": budget,
                    "special_needs": accessibility or "None provided",
                }
                planning_query = (
                    "Create Japan trip planning guidance for "
                    f"{duration} days in {', '.join(destination)}. "
                    f"Focus on {', '.join(interests)} and {pace.lower()} pacing."
                )
                try:
                    with st.spinner("Designing your trip from the source documents..."):
                        planner_sources = retrieve(config, index, chunks, planning_query, retrieval_depth)
                        itinerary = generate_itinerary(config, trip, planner_sources)
                    st.success("Your personalized plan is ready.")
                    st.markdown(itinerary)
                    with st.expander("📚 View planning sources"):
                        for number, source in enumerate(planner_sources, start=1):
                            st.markdown(f"**Source {number}**")
                            st.write(source)
                except (RuntimeError, KeyError, ValueError) as error:
                    st.error(f"We could not create the plan: {error}")


if __name__ == "__main__":
    main()
