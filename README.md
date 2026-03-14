# 🎓 Agentic AI Career Guidance Assistant

An **AI-powered career guidance system** built using **LangGraph, Groq LLM, ChromaDB, and Streamlit**.
This application allows users to upload career-related PDF documents and ask questions to receive **context-aware career advice** powered by **Retrieval-Augmented Generation (RAG)** and **agentic decision making**.

---

## 🚀 Features

• Upload **career-related PDF documents**
• Automatically **extract and chunk document content**
• Store embeddings in a **ChromaDB vector database**
• **Agentic decision node** determines if document retrieval is required
• Context-aware answers using **Groq LLM (Llama-3.1-8B)**
• Interactive **Streamlit web interface**
• Uses **LangGraph workflow** to orchestrate reasoning steps

---

## 🧠 System Architecture

The application follows an **Agentic RAG Workflow** using LangGraph:

1. **Agent Decision Node**

   * Determines whether document retrieval is required.

2. **Retrieval Node**

   * Searches relevant document chunks from ChromaDB.

3. **Response Generation Node**

   * Generates final career guidance using Groq LLM.

Workflow structure:

```
START
  ↓
Agent Decision Node
  ↓
Retrieval Node
  ↓
Response Generation Node
  ↓
END
```

---

## 🛠 Tech Stack

| Technology                  | Purpose                      |
| --------------------------- | ---------------------------- |
| **Python**                  | Backend language             |
| **Streamlit**               | Web interface                |
| **LangGraph**               | Agent workflow orchestration |
| **LangChain**               | LLM message structure        |
| **Groq LLM (Llama-3.1-8B)** | Fast inference               |
| **ChromaDB**                | Vector database              |
| **Sentence Transformers**   | Embedding model              |
| **PyPDF2**                  | PDF text extraction          |

---

## 📂 Project Structure

```
career_guidance/
│
├── CAREER_GUIDANCE.py      # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/career_guidance.git
cd career_guidance
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv genai_env
source genai_env/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```
pip install streamlit langgraph langchain chromadb sentence-transformers PyPDF2 langchain-groq langchain-text-splitters
```

---

### 4️⃣ Set Groq API Key

Set your Groq API key as an environment variable.

Mac / Linux:

```
export GROQ_API_KEY="your_api_key_here"
```

Windows:

```
set GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```
python -m streamlit run CAREER_GUIDANCE.py
```

Then open the URL shown in the terminal (usually):

```
http://localhost:8501
```

---

## 💡 Example Usage

1. Upload a **career-related PDF** such as:

   * Resume writing guides
   * Career development books
   * Job preparation materials

2. Ask questions like:

```
How can I prepare for a machine learning career?
```

```
What skills are required for data science jobs?
```

```
How should I structure my resume for AI roles?
```

3. The system will:

   * Decide whether document retrieval is required
   * Retrieve relevant context
   * Generate personalized career guidance

---

## 🔐 Security Note

Never commit API keys directly into code.
Use **environment variables** or a `.env` file.

---

## 📈 Future Improvements

• Multi-document retrieval support
• Resume analysis and feedback
• Job role recommendation system
• Career roadmap generation
• Chat history and memory
• Support for multiple LLM providers

---

## 👩‍💻 Author

**Manasa Kommineni**

AI & Machine Learning Student
Focused on building **GenAI applications and intelligent systems**
