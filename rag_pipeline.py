import os
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# CONFIG
# -------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"
FAISS_DIR = "faiss_index"
FAISS_PATH = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "chunks.npy")

# -------------------------
# LOAD MODELS
# -------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model.to(device)

# -------------------------
# HELPERS
# -------------------------
def load_index():
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)
    return None


def load_chunks():
    if os.path.exists(CHUNKS_PATH):
        return np.load(CHUNKS_PATH, allow_pickle=True).tolist()
    return []


def chunk_text(text, chunk_size=400):
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# -------------------------
# BUILD INDEX
# -------------------------
def build_index(documents):
    if not documents:
        raise ValueError("No documents found to build index.")

    os.makedirs(FAISS_DIR, exist_ok=True)

    embeddings = embed_model.encode(
        documents,
        convert_to_numpy=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    np.save(CHUNKS_PATH, np.array(documents, dtype=object))


# -------------------------
# RETRIEVE
# -------------------------
def retrieve(query, top_k=4):
    index = load_index()
    chunks = load_chunks()

    if index is None:
        raise RuntimeError("FAISS index not found. Upload a file or build the index first.")

    if not chunks:
        raise RuntimeError("No chunks found.")

    if index.ntotal != len(chunks):
        raise RuntimeError("Index and chunks count mismatch.")

    top_k = min(top_k, len(chunks))

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [chunks[i] for i in indices[0]]

    return "\n\n".join(retrieved_docs)


# -------------------------
# PROMPT BUILDER
# -------------------------
def build_prompt(topic, context, difficulty, num_questions, quiz_type, include_answers):
    if quiz_type == "MCQ":
        instructions = f"""
Generate {num_questions} multiple choice questions.
Each question must have 4 options: A, B, C, D.
"""
        format_block = """
Q1:
Question:
A)
B)
C)
D)
"""
        if include_answers:
            format_block += "Answer:\n"

    elif quiz_type == "True/False":
        instructions = f"""
Generate {num_questions} True/False questions.
Each question must clearly end with True or False options.
"""
        format_block = """
Q1:
Statement:
Options: True / False
"""
        if include_answers:
            format_block += "Answer:\n"

    elif quiz_type == "Short Answer":
        instructions = f"""
Generate {num_questions} short answer questions.
Each question should require a brief educational answer.
"""
        format_block = """
Q1:
Question:
"""
        if include_answers:
            format_block += "Answer:\n"

    else:  # Mixed
        instructions = f"""
Generate a mixed quiz with {num_questions} questions.
Include a balanced mix of:
- Multiple choice questions
- True/False questions
- Short answer questions
"""
        format_block = """
Q1:
Type:
Question:
Options (if needed):
"""
        if include_answers:
            format_block += "Answer:\n"

    answer_rule = (
        "Provide the correct answer after each question."
        if include_answers
        else "Do NOT provide answers."
    )

    prompt = f"""
You are an expert quiz generator.

Study Material:
{context}

Task:
Create a {difficulty.lower()} level quiz on the topic: {topic}

Requirements:
1. {instructions.strip()}
2. Use only the study material provided above.
3. Keep the language clear, structured, and educational.
4. {answer_rule}
5. Make the output clean and well-formatted.

Output Format Example:
{format_block}
"""
    return prompt


# -------------------------
# GENERATE QUIZ
# -------------------------
def generate_quiz(topic, difficulty="Beginner", num_questions=5, quiz_type="MCQ", include_answers=True):
    context = retrieve(f"important study information about {topic}")

    prompt = build_prompt(
        topic=topic,
        context=context,
        difficulty=difficulty,
        num_questions=num_questions,
        quiz_type=quiz_type,
        include_answers=include_answers
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=450,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result