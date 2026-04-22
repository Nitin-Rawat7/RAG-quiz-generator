import os
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF

from rag_pipeline import build_index, chunk_text, generate_quiz

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="RAG Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CUSTOM CSS
# -------------------------
def load_custom_css(theme="dark"):
    if theme == "dark":
        bg = "linear-gradient(135deg, #0f172a, #111827, #1e293b)"
        text = "white"
        subtext = "#cbd5e1"
        card_bg = "rgba(255,255,255,0.06)"
        card_border = "rgba(255,255,255,0.08)"
        hero_bg = "linear-gradient(135deg, rgba(37,99,235,0.18), rgba(124,58,237,0.18))"
        input_bg = "rgba(255,255,255,0.08)"
        quiz_bg = "rgba(15, 23, 42, 0.88)"
        sidebar_bg = "linear-gradient(180deg, #111827, #0f172a)"
        footer_text = "#94a3b8"
    else:
        bg = "linear-gradient(135deg, #f8fafc, #e2e8f0, #ffffff)"
        text = "#0f172a"
        subtext = "#334155"
        card_bg = "rgba(255,255,255,0.85)"
        card_border = "rgba(15,23,42,0.08)"
        hero_bg = "linear-gradient(135deg, rgba(59,130,246,0.10), rgba(168,85,247,0.10))"
        input_bg = "rgba(255,255,255,0.95)"
        quiz_bg = "rgba(255,255,255,0.96)"
        sidebar_bg = "linear-gradient(180deg, #f8fafc, #e2e8f0)"
        footer_text = "#475569"

    st.markdown(f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text};
    }}

    .block-container {{
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    .hero-box {{
        background: {hero_bg};
        border: 1px solid {card_border};
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }}

    .hero-title {{
        font-size: 42px;
        font-weight: 800;
        color: {text};
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        font-size: 17px;
        color: {subtext};
        line-height: 1.6;
    }}

    .custom-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }}

    .card-title {{
        font-size: 20px;
        font-weight: 700;
        color: {text};
        margin-bottom: 10px;
    }}

    .card-text {{
        color: {subtext};
        font-size: 15px;
        line-height: 1.6;
    }}

    .quiz-box {{
        background: {quiz_bg};
        border: 1px solid rgba(59,130,246,0.35);
        border-left: 5px solid #3b82f6;
        border-radius: 18px;
        padding: 24px;
        margin-top: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
        color: {text};
        line-height: 1.8;
        white-space: pre-wrap;
    }}

    .stButton > button {{
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 16px;
        font-weight: 700;
        transition: 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59,130,246,0.3);
    }}

    .stTextInput input, .stTextArea textarea {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 1px solid {card_border};
    }}

    .footer-text {{
        text-align: center;
        color: {footer_text};
        font-size: 14px;
        margin-top: 30px;
    }}
    </style>
    """, unsafe_allow_html=True)
# -------------------------
# SESSION STATE
# -------------------------
if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = ""

if "answer_result" not in st.session_state:
    st.session_state.answer_result = ""

if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "uploaded_text_length" not in st.session_state:
    st.session_state.uploaded_text_length = 0

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

# -------------------------
# FILE READING
# -------------------------
def read_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def read_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text


def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return read_txt(uploaded_file)
    elif file_name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        return read_docx(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload TXT, PDF, or DOCX.")


# -------------------------
# EXPORT HELPERS
# -------------------------
def create_txt_download(content):
    return content.encode("utf-8")


def create_pdf_download(content):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    safe_text = content.encode("latin-1", "replace").decode("latin-1")
    for line in safe_text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf_output = pdf.output(dest="S").encode("latin-1")
    return pdf_output


def remove_answers_from_quiz(text):
    lines = text.splitlines()
    filtered_lines = []
    skip_next = False

    for line in lines:
        stripped = line.strip().lower()

        if stripped.startswith("answer:"):
            continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines)


# -------------------------
# UI
# -------------------------
def app():
    load_custom_css(
    theme="dark" if st.session_state.theme_mode == "Dark" else "light"
)

    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">📚 RAG Quiz Generator</div>
        <div class="hero-subtitle">
            Upload your study material, build a FAISS-powered knowledge index, and generate professional quizzes
            with different quiz types, export options, and answer visibility control.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🎨 Appearance")

        theme_mode = st.radio(
        "Theme Mode",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        horizontal=True
    )

        st.session_state.theme_mode = theme_mode

        st.markdown("---")
        st.markdown("## ⚙️ Quiz Settings")

        difficulty = st.selectbox(
            "Difficulty Level",
            ["Beginner", "Intermediate", "Advanced"]
        )

        quiz_type = st.selectbox(
            "Quiz Type",
            ["MCQ", "True/False", "Short Answer", "Mixed"]
        )

        num_questions = st.slider(
            "Number of Questions",
            min_value=2,
            max_value=10,
            value=5
        )

        show_answers = st.checkbox("Generate with Answers", value=True)

        st.markdown("---")
        st.markdown("### 📌 Upload Notes")
        uploaded_file = st.file_uploader(
            "Upload TXT, PDF, or DOCX",
            type=["txt", "pdf", "docx"]
        )

        build_btn = st.button("📂 Build Knowledge Index")

        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.metric("Index Ready", "Yes" if st.session_state.index_ready else "No")
        st.metric("Uploaded Text Length", st.session_state.uploaded_text_length)

        st.markdown("---")
        st.markdown("### ✨ Portfolio Features")
        st.markdown("""
        - File upload
        - Dynamic FAISS indexing
        - Multiple quiz types
        - Answer visibility control
        - TXT/PDF export
        """)

    if uploaded_file and build_btn:
        try:
            with st.spinner("Reading file and building vector index..."):
                text = extract_text_from_file(uploaded_file)
                chunks = chunk_text(text, chunk_size=400)

                if not chunks:
                    st.error("No readable content found in the uploaded file.")
                else:
                    build_index(chunks)
                    st.session_state.index_ready = True
                    st.session_state.uploaded_text_length = len(text)
                    st.success(f"Index built successfully from {uploaded_file.name}.")
        except Exception as e:
            st.error(f"File processing error: {str(e)}")

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🎯 Generate Quiz</div>
            <div class="card-text">
                Enter a topic based on your uploaded notes and generate a customized quiz.
            </div>
        </div>
        """, unsafe_allow_html=True)

        topic = st.text_input(
            "Enter Topic",
            placeholder="e.g. Machine Learning, RAG, Embeddings, NLP"
        )

        generate_btn = st.button("🚀 Generate Quiz")

        if generate_btn:
            if not st.session_state.index_ready:
                st.warning("Please upload a file and build the index first.")
            elif not topic.strip():
                st.warning("Please enter a topic.")
            else:
                try:
                    with st.spinner("Generating quiz..."):
                        full_result = generate_quiz(
                            topic=topic,
                            difficulty=difficulty,
                            num_questions=num_questions,
                            quiz_type=quiz_type,
                            include_answers=show_answers
                        )

                        st.session_state.answer_result = full_result

                        if show_answers:
                            st.session_state.quiz_result = full_result
                        else:
                            st.session_state.quiz_result = remove_answers_from_quiz(full_result)

                        st.session_state.quiz_history.insert(0, {
                            "topic": topic,
                            "difficulty": difficulty,
                            "quiz_type": quiz_type,
                            "content": st.session_state.quiz_result
                        })

                except Exception as e:
                    st.error(f"Quiz generation error: {str(e)}")

        if st.session_state.quiz_result:
            st.markdown("### 📝 Generated Quiz")
            st.markdown(
                f'<div class="quiz-box">{st.session_state.quiz_result}</div>',
                unsafe_allow_html=True
            )

            col_txt, col_pdf = st.columns(2)

            with col_txt:
                st.download_button(
                    label="⬇️ Download as TXT",
                    data=create_txt_download(st.session_state.quiz_result),
                    file_name="generated_quiz.txt",
                    mime="text/plain"
                )

            with col_pdf:
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=create_pdf_download(st.session_state.quiz_result),
                    file_name="generated_quiz.pdf",
                    mime="application/pdf"
                )

            if show_answers and st.session_state.answer_result:
                with st.expander("👁 View Answer Version"):
                    st.text(st.session_state.answer_result)

    with right_col:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">📘 Overview</div>
            <div class="card-text">
                This app uses Retrieval-Augmented Generation with FAISS and sentence embeddings
                to create topic-based quizzes directly from uploaded study materials.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">💡 Suggested Usage</div>
            <div class="card-text">
                1. Upload your notes<br>
                2. Build the FAISS index<br>
                3. Enter a topic<br>
                4. Select quiz type and difficulty<br>
                5. Generate and export your quiz
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🧠 Supported Quiz Types</div>
            <div class="card-text">
                • MCQ<br>
                • True/False<br>
                • Short Answer<br>
                • Mixed
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.quiz_history:
        st.markdown("## 🕘 Recent Quiz History")

        for item in st.session_state.quiz_history[:5]:
            with st.expander(f"{item['topic']} • {item['quiz_type']} • {item['difficulty']}"):
                st.text(item["content"])

    st.markdown(
        '<div class="footer-text">Built with Streamlit • Powered by FAISS + Sentence Transformers + FLAN-T5</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    app()