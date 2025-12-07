# **SmartHire AI Chatbot 🤖**

An intelligent AI-powered interview assistant designed to help recruiters and job seekers. The chatbot can ask role-specific interview questions, analyze candidate answers using NLP techniques, and provide feedback based on **relevance, depth, and clarity**.

---

## **✨ Features**

* **Role & Level Selection** – Choose job role (e.g., Data Scientist, Software Engineer) and experience level (Junior/Mid/Senior).
* **Interactive Interview** – The bot asks **technical and behavioral** questions.
* **Follow-up Questions** – Dynamically generated based on user answers.
* **Answer Analysis Engine** – Uses **BERT/Transformers** to evaluate answers and score them.
* **Entity & Keyword Extraction** – Highlights important concepts from answers.
* **Streamlit Frontend** – User-friendly interface with chat-like design.

---

## **📦 Tech Stack**

* **Frontend:** Streamlit
* **Backend:** Python
* **AI Models:** GROQ, HuggingFace Transformers (BERT/RoBERTa)
* **Answer Scoring:** Semantic similarity + relevance analysis
* **Entity Extraction:** spaCy / Transformers NER

---

## **🛠 Dependencies**

Install these using pip:

```bash
pip install streamlit
pip install groq
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install spacy
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install python-dotenv
```

*(Optional for GPU support: Install the CUDA version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/))*

---

## **⚙ Setup Instructions**

1. **Clone the repository**

```bash
git clone https://github.com/HamzaArshad-27/Smart-hire.git
cd Smart-hire
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Download spaCy model** (if using spaCy for entity extraction)

```bash
python -m spacy download en_core_web_sm
```

---

## **🚀 Run the Application**

```bash
streamlit run app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)** to interact with the chatbot.

---

## **📂 Project Structure**

```
📁 smarthire-ai-chatbot
 ├── app.py                # Main Streamlit app
 ├── answer_analysis.py    # BERT-based answer scoring logic
 ├── question_bank.py      # Predefined technical & behavioral questions
 ├── requirements.txt      # Project dependencies
 ├── .env                  # Environment variables
 ├── README.md             # Project documentation
 └── assets/               # Images, icons, etc.
```

---

## **🧠 How It Works**

1. **User selects role & level** → Chatbot loads relevant questions.
2. **Bot asks a question** → User responds.
3. **Answer Analysis Engine**:

   * Tokenizes and embeds answer using **BERT**.
   * Compares against ideal answers for **semantic similarity**.
   * Scores based on **relevance, clarity, and completeness**.
   * Extracts **keywords/entities**.
4. **Bot provides feedback** → Asks follow-up questions.

---

## **📌 Future Improvements**

* Add **voice input/output** for spoken interviews.
* Integrate **resume parsing** for tailored question generation.
* Store interview history in a **database**.
* Add **multi-language support**.
