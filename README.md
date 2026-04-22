# 📚 RAG Quiz Generator

A **Retrieval-Augmented Generation (RAG) based Quiz Generator** that allows users to upload study materials and automatically generate quizzes using AI.

This project combines **FAISS vector search, sentence embeddings, and LLM-based generation** to create dynamic, topic-based quizzes with a clean and interactive UI.

---

## 🚀 Features

* 📂 **File Upload Support**

  * Upload `.txt`, `.pdf`, or `.docx` files
  * Automatically extracts and processes text

* 🧠 **RAG Pipeline**

  * Uses **FAISS** for semantic search
  * Retrieves relevant context before generating quizzes

* 📝 **Multiple Quiz Types**

  * MCQ (Multiple Choice Questions)
  * True / False
  * Short Answer
  * Mixed Mode

* 🎯 **Custom Quiz Controls**

  * Select difficulty (Beginner, Intermediate, Advanced)
  * Choose number of questions
  * Topic-based quiz generation

* 👁 **Answer Visibility Control**

  * Generate quiz with or without answers
  * Toggle answer view

* 📤 **Export Options**

  * Download quiz as `.txt`
  * Download quiz as `.pdf`

* 🎨 **Modern UI**

  * Dark / Light theme toggle
  * Dashboard-style layout
  * Sidebar controls and metrics

* 🕘 **Quiz History**

  * View previously generated quizzes

---

## 🧩 Tech Stack

* **Frontend/UI:** Streamlit
* **Vector Search:** FAISS
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **LLM:** FLAN-T5 (`google/flan-t5-base`)
* **File Processing:** PyPDF, python-docx
* **Export:** FPDF

---

## 📂 Project Structure

```bash
rag-quiz-generator/
│
├── app.py                # Main Streamlit UI
├── rag_pipeline.py       # RAG logic (retrieval + generation)
├── faiss_index/
│   ├── index.faiss
│   └── chunks.npy
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-quiz-generator.git
cd rag-quiz-generator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Upload your study material (PDF, DOCX, TXT)
2. Text is extracted and split into chunks
3. Chunks are converted into embeddings
4. FAISS index is built for semantic retrieval
5. User enters a topic
6. Relevant context is retrieved
7. LLM generates quiz based on retrieved content

---

## 📸 Screenshots (Add these)

> Add screenshots of:

* Upload + index build
* Quiz generation UI
* Dark/Light theme
* Export buttons

---

## 💡 Future Improvements

* Multiple file upload support
* Quiz scoring system (interactive answering)
* Better LLM integration (OpenAI / Groq / Gemini)
* Source citation for generated questions
* Deployment on Streamlit Cloud

---

## 👨‍💻 Author

**Nitin Rawat**

* 🔗 GitHub: https://github.com/your-username
* 🔗 LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ Contribution

Feel free to fork this repo, improve features, and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.
