
# =====================================================
# SYNAPTIQ AI - APP.PY
# =====================================================

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import shutil
import gc

from rag.rag_pipeline import (
    process_pdf,
    retrieve_context,
    load_vectorstore
)

from quiz.quiz_generator import (
    generate_quiz
)

from summary.summary_generator import (
    generate_summary
)

from flashcards.flashcard_generator import (
    generate_flashcards
)

# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()

# =====================================================
# INITIALIZE GROQ
# =====================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Synaptiq AI",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #050816;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #1a1a2e,
            #16213e
        );
    }

    .stButton button {
        border-radius: 14px;
        background: linear-gradient(
            90deg,
            #4facfe,
            #00f2fe
        );
        color: white;
        border: none;
        font-weight: bold;
        padding: 0.6rem 1rem;
    }

    .stButton button:hover {
        transform: scale(1.03);
        transition: 0.2s ease-in-out;
    }

    div[data-testid="stExpander"] {
        border-radius: 15px;
        border: 1px solid #2d3748;
        background-color: #111827;
    }

    .stTextInput input {
        border-radius: 12px;
    }

    .chat-card {
        background-color: #111827;
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 15px;
        border: 1px solid #1f2937;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SESSION STATE
# =====================================================

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "answered_questions" not in st.session_state:
    st.session_state.answered_questions = {}

if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================
# TITLE
# =====================================================

st.title("🧠 Synaptiq AI")

st.subheader(
    "Adaptive AI Learning Companion"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "📄 Upload Study Material"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =====================================================
# CLEAR KNOWLEDGE BASE
# =====================================================

if st.sidebar.button(
    "🗑 Clear Knowledge Base"
):

    try:

        load_vectorstore.cache_clear()

        gc.collect()

        if os.path.exists(
            "vectorstore/faiss_index"
        ):

            shutil.rmtree(
                "vectorstore/faiss_index",
                ignore_errors=True
            )

        st.session_state.quiz_data = None
        st.session_state.quiz_score = 0
        st.session_state.answered_questions = {}
        st.session_state.quiz_feedback = {}
        st.session_state.chat_history = []

        st.sidebar.success(
            "Knowledge base cleared!"
        )

    except Exception as e:

        st.sidebar.error(
            f"Error: {e}"
        )

# =====================================================
# PDF PROCESSING
# =====================================================

if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    st.sidebar.success(
        "PDF uploaded successfully!"
    )

    with st.spinner(
        "Processing PDF..."
    ):

        result = process_pdf(
            pdf_path
        )

    load_vectorstore.cache_clear()

    st.sidebar.success(result)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Ask AI",
    "📝 Quiz",
    "📚 Summary",
    "🧠 Flashcards"
])

# =====================================================
# TAB 1 — ASK AI
# =====================================================

with tab1:

    st.header("📖 Ask Questions")

    user_input = st.text_input(
        "Ask your study question:",
        placeholder="Example: Explain clustering"
    )

    if st.button("Generate Response"):

        if user_input.strip() != "":

            with st.spinner(
                "Synaptiq AI is thinking..."
            ):

                try:

                    context = retrieve_context(
                        user_input
                    )

                    rag_prompt = f"""
                    You are Synaptiq AI.

                    Answer ONLY using
                    the provided context.

                    Do NOT hallucinate information.

                    Context:
                    {context}

                    Question:
                    {user_input}

                    Give:
                    - beginner-friendly explanation
                    - concise answer
                    - clear examples
                    """

                    completion = (
                        client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": rag_prompt,
                                }
                            ],
                            model="llama-3.1-8b-instant",
                        )
                    )

                    response = (
                        completion
                        .choices[0]
                        .message.content
                    )

                    st.session_state.chat_history.append(
                        {
                            "question": user_input,
                            "response": response
                        }
                    )

                    st.success(
                        "Response Generated!"
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

    # CHAT HISTORY

    if st.session_state.chat_history:

        st.markdown("## 💬 Chat History")

        for chat in reversed(
            st.session_state.chat_history
        ):

            st.markdown(
                f"""
<div class="chat-card">

### ❓ Question

{chat['question']}

### 🤖 Response

{chat['response']}

</div>
""",
                unsafe_allow_html=True
            )

# =====================================================
# TAB 2 — QUIZ
# =====================================================

with tab2:

    st.header(
        "📝 Interactive Quiz Generator"
    )

    quiz_topic = st.text_input(
        "Enter topic for quiz generation:",
        placeholder="Example: GIS"
    )

    if st.button("Generate Quiz"):

        if quiz_topic.strip() != "":

            with st.spinner(
                "Generating quiz..."
            ):

                try:

                    quiz_data = generate_quiz(
                        quiz_topic
                    )

                    st.session_state.quiz_data = (
                        quiz_data
                    )

                    st.session_state.quiz_score = 0

                    st.session_state.answered_questions = {}

                    st.session_state.quiz_feedback = {}

                    st.success(
                        "Quiz Generated!"
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

    if st.session_state.quiz_data:

        for i, q in enumerate(
            st.session_state.quiz_data
        ):

            st.markdown(
                f"## Question {i+1}"
            )

            st.write(
                q["question"]
            )

            selected_option = st.radio(
                "Choose your answer:",
                q["options"],
                key=f"quiz_option_{i}"
            )

            if st.button(
                f"Submit Question {i+1}",
                key=f"submit_{i}"
            ):

                if (
                    i
                    not in
                    st.session_state
                    .answered_questions
                ):

                    st.session_state.answered_questions[i] = (
                        selected_option
                    )

                    correct_option = (
                        q["options"][
                            q["correct_answer"]
                        ]
                    )

                    if (
                        selected_option
                        ==
                        correct_option
                    ):

                        st.session_state.quiz_score += 1

                        st.session_state.quiz_feedback[i] = {
                            "result": "correct",
                            "answer": correct_option
                        }

                    else:

                        st.session_state.quiz_feedback[i] = {
                            "result": "wrong",
                            "answer": correct_option
                        }

            # FEEDBACK

            if i in st.session_state.quiz_feedback:

                result = (
                    st.session_state
                    .quiz_feedback[i]["result"]
                )

                correct_answer = (
                    st.session_state
                    .quiz_feedback[i]["answer"]
                )

                if result == "correct":

                    st.success(
                        "✅ Correct!"
                    )

                else:

                    st.error(
                        "❌ Wrong Answer"
                    )

                    st.info(
                        f"Correct Answer: "
                        f"{correct_answer}"
                    )

            st.markdown("---")

        total_questions = len(
            st.session_state.quiz_data
        )

        st.markdown(
            f"""
            # 🏆 Quiz Score:
            {st.session_state.quiz_score}
            / {total_questions}
            """
        )

# =====================================================
# TAB 3 — SUMMARY
# =====================================================

with tab3:

    st.header("📚 PDF Summary Generator")

    summary_topic = st.text_input(
        "Enter topic for summary:",
        placeholder="Example: Data Visualization"
    )

    if st.button("Generate Summary"):

        with st.spinner(
            "Generating summary..."
        ):

            try:

                summary = generate_summary(
                    summary_topic
                )

                st.success(
                    "Summary Generated!"
                )

                st.write(summary)

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )

# =====================================================
# TAB 4 — FLASHCARDS
# =====================================================

with tab4:

    st.header("🧠 Flashcard Generator")

    flashcard_topic = st.text_input(
        "Enter topic for flashcards:",
        placeholder="Example: Scatterplots"
    )

    if st.button("Generate Flashcards"):

        with st.spinner(
            "Generating flashcards..."
        ):

            try:

                flashcards = (
                    generate_flashcards(
                        flashcard_topic
                    )
                )

                for i, card in enumerate(
                    flashcards
                ):

                    with st.expander(
                        f"Flashcard {i+1}"
                    ):

                        st.markdown(
                            f"### ❓ "
                            f"{card['question']}"
                        )

                        st.markdown(
                            f"### ✅ "
                            f"{card['answer']}"
                        )

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )

