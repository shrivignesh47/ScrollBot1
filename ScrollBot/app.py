# from flask import Flask, render_template, request, redirect, url_for, session, flash
# from dotenv import load_dotenv
# import os
# import re
# import json
# import logging
# from markupsafe import Markup, escape
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# import google.generativeai as genai

# # =============== 1. CONFIGURATION ===============
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# app = Flask(__name__)
# app.secret_key = os.getenv("SECRET_KEY", "your_very_strong_secret_key_2025")

# # Paths
# FAISS_INDEX_PATH = "faiss_index"
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# # Logging
# logging.basicConfig(
#     filename=os.path.join(LOG_DIR, "medical_bot.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # Dummy users
# users = {}

# # Symptom synonyms
# SYNONYM_MAP = {
#     "stomach pain": ["tummy ache", "belly pain", "gastric pain", "abdominal discomfort"],
#     "fever": ["high temperature", "running a fever", "hot body", "chills"],
#     "rash": ["skin rash", "red spots", "itchy skin"],
#     "headache": ["head pain", "migraine", "throbbing head"],
#     "nausea": ["feeling sick", "want to vomit", "queasy"]
# }

# def standardize_symptom(symptom: str) -> str:
#     s = symptom.lower().strip()
#     for std, variants in SYNONYM_MAP.items():
#         if s in variants or s == std:
#             return std
#     return s


# # =============== 2. MEDICAL INTENT DETECTION ===============
# def detect_medical_question(q: str) -> bool:
#     keywords = [
#         "pain", "fever", "sick", "allergy", "rash", "hurt", "symptom",
#         "medicine", "headache", "cough", "vomiting", "nausea", "dizzy",
#         "fatigue", "chest pain", "shortness of breath", "bleeding",
#         "doctor", "treatment", "infection", "swelling", "burning", "itch"
#     ]
#     q = q.lower()
#     return any(k in q for k in keywords)


# # =============== 3. DETECT GENERAL MEDICAL KNOWLEDGE QUERY ===============
# def is_general_medical_query(question: str) -> bool:
#     """Detect if user is asking a broad medical knowledge question."""
#     general_phrases = [
#         "what is", "treatment for", "symptoms of", "causes of", "define",
#         "explain", "how to treat", "management of", "diagnosis of"
#     ]
#     specific_conditions = [
#         "dvt", "deep venous thrombosis", "cardiovascular", "ulcerative colitis",
#         "diabetes", "hypertension", "asthma", "arthritis", "cancer", "infection",
#         "pneumonia", "migraine", "stroke", "heart attack", "angina"
#     ]
#     q = question.lower().strip()
#     return (
#         any(phrase in q for phrase in general_phrases) or
#         any(cond in q for cond in specific_conditions)
#     )


# # =============== 4. PDF-BASED RESPONSE ===============
# def get_pdf_response(user_question):
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.load_local(
#             FAISS_INDEX_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )
#         retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#         docs = retriever.get_relevant_documents(user_question)

#         if not docs:
#             return "I couldn't find relevant information in the documents."

#         prompt_template = """
# Answer the medical question using **only** the context below.
# If not relevant, say: 'I can't find specific details about this in the current knowledge base.'

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#         prompt = ChatPromptTemplate.from_template(prompt_template)
#         model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#         chain = create_stuff_documents_chain(llm=model, prompt=prompt)
#         response = chain.invoke({"context": docs, "question": user_question})

#         references = [
#             f'<sup>[{i+1}]</sup> p.{doc.metadata["page"]} ({os.path.basename(doc.metadata["source"])})'
#             for i, doc in enumerate(docs)
#         ]
#         final = response.strip()
#         if references:
#             final += "<br><br><strong>Sources:</strong> " + " ".join(references)

#         return final

#     except Exception as e:
#         logging.error(f"PDF Error: {str(e)}")
#         return "I'm having trouble accessing the knowledge base right now."


# # =============== 5. AI MEDICAL ADVICE (JSON) ===============
# def get_medical_advice_model():
#     # ✅ Fixed: {{ }} escapes JSON schema to avoid INVALID_PROMPT_INPUT
#     prompt_template = """
# You are a virtual health assistant. Generate a structured medical guidance response in **strict JSON format only**.
# Do not add any text before or after.

# Use this schema:
# {{  "greeting": "Personalized thank-you message with name and age",
#     "summary": "Brief summary of possible causes (non-diagnostic)",
#     "recommendations": ["List", "of", "actionable", "steps"],
#     "action": "When to consult a doctor"
# }}

# Guidelines:
# - Be empathetic and professional.
# - Never claim to diagnose.
# - Keep recommendations safe and general.

# Patient Info:
# Name: {name}
# Age: {age}
# Symptoms: {symptoms}

# Respond ONLY with valid JSON.
# """
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, convert_system_message_to_human=True)
#     return prompt | model


# def generate_medical_advice(name, age, symptoms):
#     try:
#         chain = get_medical_advice_model()
#         standardized_symptoms = ", ".join(standardize_symptom(s.strip()) for s in symptoms.split(",") if s.strip())
        
#         response = chain.invoke({
#             "name": escape(name),
#             "age": age,
#             "symptoms": standardized_symptoms
#         })
#         text = response.content.strip()

#         if text.startswith("```json"):
#             text = text[7:-3] if text.endswith("```") else text[7:]
#         elif text.startswith("```"):
#             text = text[3:-3]

#         data = json.loads(text)

#         html = f"<b>🩺 {escape(data['greeting'])}</b><br><br>"
#         if data.get("summary"):
#             html += f"<i>{escape(data['summary'])}</i><br><br>"
#         if data.get("recommendations"):
#             html += "<b>💡 What You Can Do:</b><ul>"
#             for item in data["recommendations"]:
#                 html += f"<li>{escape(item)}</li>"
#             html += "</ul><br>"
#         if data.get("action"):
#             html += f"<b>⚠️ Important:</b> {escape(data['action'])}"

#         return Markup(html)

#     except Exception as e:
#         logging.error(f"Medical AI Error: {str(e)}")
#         return Markup("⚠️ I'm currently unable to assist. Please consult a healthcare provider.")


# # =============== 6. SESSION MANAGEMENT ===============
# def get_medical_data():
#     if "medical_data" not in session:
#         session["medical_data"] = {"name": None, "age": None, "symptoms": []}
#         session.modified = True

#     data = session["medical_data"]
#     if not isinstance(data, dict):
#         data = {"name": None, "age": None, "symptoms": []}

#     data.setdefault("name", None)
#     data.setdefault("age", None)
#     if not isinstance(data.get("symptoms"), list):
#         data["symptoms"] = []

#     session["medical_data"] = data
#     session.modified = True
#     return data
# def handle_medical_question(question):
#     data = get_medical_data()
#     lower_q = question.strip().lower()

#     # 🔥 Always check for general knowledge queries first
#     if is_general_medical_query(lower_q):
#         # Exit personal flow and use PDF knowledge base
#         session.pop("medical_data", None)
#         session.pop("last_medical_trigger", None)
#         session.modified = True
#         return get_pdf_response(question)

#     # ✅ If full data already collected (name, age, symptoms)
#     if data["name"] and data["age"] and data["symptoms"]:
#         symptoms_str = ", ".join(data["symptoms"])

#         if lower_q in ["repeat", "show", "details"]:
#             return (
#                 f"<b>📋 Your Details:</b><br>"
#                 f"• <b>Name:</b> {data['name']}<br>"
#                 f"• <b>Age:</b> {data['age']}<br>"
#                 f"• <b>Symptoms:</b> {symptoms_str}<br><br>"
#                 "Say <b>update</b> to change, or <b>no</b> if done."
#             )

#         if lower_q in ["update", "change"]:
#             session["medical_data"] = {"name": None, "age": None, "symptoms": []}
#             session.modified = True
#             return "Let's start over. What's your name?"

#         if lower_q == "no":
#             session.pop("medical_data", None)
#             return "Got it. Feel free to ask anything else!"

#         if detect_medical_question(question):
#             std_symptom = standardize_symptom(question)
#             existing_std = [standardize_symptom(s) for s in data["symptoms"]]
#             if std_symptom not in existing_std:
#                 data["symptoms"].append(question.strip())
#                 session["medical_data"] = data
#                 session.modified = True
#             return generate_medical_advice(
#                 data["name"], data["age"], ", ".join(data["symptoms"])
#             )

#         # If it's not a symptom and not a general medical query
#         return "⚠️ I can only assist with medical follow-ups or general medical knowledge."

#     # ✅ Step 1: Get name
#     if data["name"] is None:
#         name = question.strip()
#         if len(name) < 2 or not re.match(r"^[A-Za-z\s]+$", name):
#             return "Please enter a valid name (letters only)."
#         data["name"] = name
#         session["medical_data"] = data
#         session.modified = True
#         return "Thank you! What's your age?"

#     # ✅ Step 2: Get age
#     if data["age"] is None:
#         if re.match(r"^\d+$", question.strip()):
#             age_val = int(question.strip())
#             if 1 <= age_val <= 120:
#                 data["age"] = age_val
#                 session["medical_data"] = data
#                 session.modified = True
#                 return "Got it. Please describe your symptoms clearly (e.g., fever, nausea)."
#             else:
#                 return "Please enter a realistic age (1–120)."
#         return "Enter a number."

#     # ✅ Step 3: Collect symptoms
#     std_symptom = standardize_symptom(question)
#     existing_std = [standardize_symptom(s) for s in data["symptoms"]]
#     if std_symptom not in existing_std:
#         data["symptoms"].append(question.strip())
#         session["medical_data"] = data
#         session.modified = True

#     return generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))

# # =============== 7. ROUTES ===============
# @app.route('/')
# def home():
#     return redirect(url_for('login'))


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users and users[uname] == pwd:
#             session['user'] = uname
#             session['chat_history'] = []
#             flash("Logged in successfully!", "success")
#             return redirect(url_for('chatbot'))
#         else:
#             error = "Invalid username or password"
#     return render_template('login.html', error=error)


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users:
#             return "User already exists. <a href='/login'>Login</a>."
#         users[uname] = pwd
#         flash("Registration successful!", "info")
#         return redirect(url_for('login'))
#     return render_template('register.html')

# @app.route('/chatbot', methods=['GET', 'POST'])
# def chatbot():
#     if 'user' not in session:
#         return redirect(url_for('login'))

#     if 'chat_history' not in session:
#         session['chat_history'] = []

#     if request.method == 'POST':
#         question = request.form.get('question', '').strip()
#         if not question:
#             answer = "Please enter a message."
#         else:
#             lower_q = question.lower().strip()

#             if lower_q in ["hi", "hello", "hey"]:
#                 answer = "Hi 👋 I'm your Medical Assistant. You can ask about symptoms or medical conditions."

#             # ✅ Check general query BEFORE starting medical flow
#             elif is_general_medical_query(lower_q):
#                 session.pop('medical_data', None)
#                 session.pop('last_medical_trigger', None)
#                 session.modified = True
#                 answer = get_pdf_response(question)

#             elif detect_medical_question(lower_q) or "medical_data" in session:
#                 session["last_medical_trigger"] = question.strip()
#                 answer = handle_medical_question(question)  # This now has escape logic

#             else:
#                 answer = get_pdf_response(question)

#         session['chat_history'].append({
#             'question': escape(question),
#             'answer': answer
#         })
#         session.modified = True

#     return render_template('chatbot.html', chat_history=session['chat_history'])

# @app.route('/clear_chat')
# def clear_chat():
#     session.pop('chat_history', None)
#     session.pop('medical_data', None)
#     session.pop('last_medical_trigger', None)
#     flash("Chat cleared! 🧹", "info")
#     return redirect(url_for('chatbot'))


# @app.route('/logout')
# def logout():
#     username = session.get('user', 'User')
#     session.clear()
#     flash(f"See you later, {username}! 👋", "info")
#     return redirect(url_for('login'))


# if __name__ == '__main__':
#     print("✅ Medical Chatbot running on http://127.0.0.1:5000")
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv
import os
import re
import json
import logging
import difflib
from markupsafe import Markup, escape
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

# =============== 1. CONFIGURATION ===============
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_very_strong_secret_key_2025")

# Paths
FAISS_INDEX_PATH = "faiss_index"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "medical_bot.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Dummy users storage
users = {}

# ================= 1.a Knowledge / Vocabulary =================
# Known conditions to help detect + autocorrect general queries.
KNOWN_CONDITIONS = [
    # Add any condition names relevant to your PDFs/KB here
    "ulcerative colitis",
    "crohn disease",
    "atopic dermatitis",
    "giant cell arteritis",
    "psoriatic arthritis",
    "rheumatoid arthritis",
    "ankylosing spondylitis",
    "non-radiographic axial spondyloarthritis",
    "psoriasis",
    "asthma",
    "diabetes",
    "hypertension",
    "migraine",
    "stroke",
    "pneumonia",
    "deep venous thrombosis",
    "dvt",
    "heart attack",
    "angina",
    "infection"
]

# Symptom synonyms map
SYNONYM_MAP = {
    "stomach pain": ["tummy ache", "belly pain", "gastric pain", "abdominal discomfort"],
    "fever": ["high temperature", "running a fever", "hot body", "chills", "feverish"],
    "rash": ["skin rash", "red spots", "itchy skin"],
    "headache": ["head pain", "migraine", "throbbing head"],
    "nausea": ["feeling sick", "want to vomit", "queasy"]
}


def fuzzy_best_match(s: str, candidates: list[str], cutoff: float = 0.8) -> str | None:
    """Return best fuzzy match from candidates if above cutoff, else None."""
    s = (s or "").strip().lower()
    if not s:
        return None
    # Use difflib to find best close match
    best = difflib.get_close_matches(s, candidates, n=1, cutoff=cutoff)
    return best[0] if best else None


def standardize_symptom(symptom: str) -> str:
    """
    Convert variations of a symptom to a standard form.
    Uses exact + fuzzy matching across keys and their variants.
    """
    s = symptom.lower().strip()
    # Exact map
    for std, variants in SYNONYM_MAP.items():
        if s == std or s in variants:
            return std

    # Fuzzy against all terms
    all_terms = []
    owner = {}
    for std, variants in SYNONYM_MAP.items():
        all_terms.append(std)
        owner[std] = std
        for v in variants:
            all_terms.append(v)
            owner[v] = std

    m = fuzzy_best_match(s, all_terms, cutoff=0.75)
    if m:
        return owner[m]

    return s


# =============== 2. MEDICAL INTENT DETECTION ===============
def detect_medical_question(q: str) -> bool:
    keywords = [
        "pain", "fever", "sick", "allergy", "rash", "hurt", "symptom",
        "medicine", "headache", "cough", "vomiting", "nausea", "dizzy",
        "fatigue", "chest pain", "shortness of breath", "bleeding",
        "doctor", "treatment", "infection", "swelling", "burning", "itch"
    ]
    q = q.lower()
    return any(k in q for k in keywords)


# =============== 3. GENERAL MEDICAL KNOWLEDGE DETECTION ===============
def correct_condition_if_any(text: str) -> str | None:
    """
    Try to detect if the user text is referring to a known condition,
    even with typos (e.g., 'ulcerative colotos' -> 'ulcerative colitis').
    Returns the corrected condition name or None.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    # Direct contains check
    for cond in KNOWN_CONDITIONS:
        if cond in t:
            return cond

    # Fuzzy check against whole condition names
    m = fuzzy_best_match(t, KNOWN_CONDITIONS, cutoff=0.75)
    if m:
        return m

    # Token-wise: try to repair multiword conditions
    tokens = re.split(r"[^a-z0-9]+", t)
    # Build candidate phrases of up to 4 tokens
    for size in [4, 3, 2, 1]:
        for i in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[i:i+size]).strip()
            if not phrase:
                continue
            m2 = fuzzy_best_match(phrase, KNOWN_CONDITIONS, cutoff=0.75)
            if m2:
                return m2

    return None


def is_general_medical_query(question: str) -> bool:
    """Detect if the question is a broad general knowledge query."""
    general_phrases = [
        "what is", "treatment for", "symptoms of", "causes of", "define",
        "explain", "how to treat", "management of", "diagnosis of"
    ]
    q = question.lower().strip()
    if any(p in q for p in general_phrases):
        return True
    # If it looks like (possibly misspelled) condition, treat as general
    return correct_condition_if_any(q) is not None


# =============== 4. PDF-BASED RESPONSE ===============
def get_pdf_response(user_question: str, corrected_condition: str | None = None):
    """Search FAISS knowledge base for general queries. Optionally include a correction banner."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        query = corrected_condition if corrected_condition else user_question
        docs = retriever.get_relevant_documents(query)

        if not docs:
            banner = ""
            if corrected_condition and corrected_condition.lower() != user_question.lower():
                banner = f"<i>Showing results for <b>{escape(corrected_condition)}</b>.</i><br><br>"
            return banner + "I couldn't find relevant information in the documents."

        prompt_template = """
Answer the medical question using **only** the context below.
If not relevant, say: 'I can't find specific details about this in the current knowledge base.'

Context:
{context}

Question:
{question}

Answer:
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        chain = create_stuff_documents_chain(llm=model, prompt=prompt)
        response = chain.invoke({"context": docs, "question": query})

        # References for citation
        references = [
            f'<sup>[{i+1}]</sup> p.{escape(str(doc.metadata.get("page","?")))} ({escape(os.path.basename(doc.metadata.get("source","?")))})'
            for i, doc in enumerate(docs)
        ]
        final = response.strip()
        # If we corrected the user's condition, show a friendly banner
        if corrected_condition and corrected_condition.lower() != user_question.lower():
            final = f"<i>Showing results for <b>{escape(corrected_condition)}</b> (corrected from '{escape(user_question)}').</i><br><br>" + final

        if references:
            final += "<br><br><strong>Sources:</strong> " + " ".join(references)

        return final
    except Exception as e:
        logging.error(f"PDF Error: {str(e)}")
        return "⚠️ I'm having trouble accessing the knowledge base right now."


# =============== 5. AI MEDICAL ADVICE (JSON) ===============
def get_medical_advice_model():
    """Return structured advice chain (strict JSON)."""
    prompt_template = """
You are a virtual health assistant. Generate a structured medical guidance response in **strict JSON format only**.
Do not add any text before or after.

Schema:
{{  "greeting": "Personalized thank-you message with name and age",
    "summary": "Brief summary of possible causes (non-diagnostic)",
    "recommendations": ["List", "of", "actionable", "steps"],
    "action": "When to consult a doctor"
}}

Guidelines:
- Be empathetic and professional.
- Never claim to diagnose.
- Keep recommendations safe and general.

Patient Info:
Name: {name}
Age: {age}
Symptoms: {symptoms}

Respond ONLY with valid JSON.
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.4, convert_system_message_to_human=True
    )
    return prompt | model


def generate_medical_advice(name, age, symptoms):
    """Generate structured safe advice in HTML."""
    try:
        chain = get_medical_advice_model()
        standardized_symptoms = ", ".join(
            standardize_symptom(s.strip()) for s in symptoms.split(",") if s.strip()
        )

        response = chain.invoke({"name": escape(name), "age": age, "symptoms": standardized_symptoms})
        text = response.content.strip()

        # Strip markdown formatting
        if text.startswith("```json"):
            text = text[7:-3] if text.endswith("```") else text[7:]
        elif text.startswith("```"):
            text = text[3:-3]

        data = json.loads(text)

        # Convert JSON to safe HTML
        html = f"<b>🩺 {escape(data.get('greeting', 'Thank you'))}</b><br><br>"
        if data.get("summary"):
            html += f"<i>{escape(data['summary'])}</i><br><br>"
        if data.get("recommendations"):
            html += "<b>💡 What You Can Do:</b><ul>"
            for item in data["recommendations"]:
                html += f"<li>{escape(item)}</li>"
            html += "</ul><br>"
        if data.get("action"):
            html += f"<b>⚠️ Important:</b> {escape(data['action'])}"

        return Markup(html)
    except Exception as e:
        logging.error(f"Medical AI Error: {str(e)}")
        return Markup("⚠️ I'm currently unable to assist. Please consult a healthcare provider.")


# =============== 6. SESSION MANAGEMENT ===============
def get_medical_data():
    """Ensure session data exists and is valid."""
    if "medical_data" not in session:
        session["medical_data"] = {"name": None, "age": None, "symptoms": []}
        session.modified = True

    data = session["medical_data"]
    if not isinstance(data, dict):
        data = {"name": None, "age": None, "symptoms": []}

    data.setdefault("name", None)
    data.setdefault("age", None)
    if not isinstance(data.get("symptoms"), list):
        data["symptoms"] = []

    session["medical_data"] = data
    session.modified = True
    return data


def handle_medical_question(question):
    """Handle personal symptom-based conversation flow with graceful handover to general knowledge."""
    data = get_medical_data()
    lower_q = question.strip().lower()

    # If the user actually asked a general medical question (or misspelled condition), switch flows
    corrected_condition = correct_condition_if_any(lower_q)
    if is_general_medical_query(lower_q) or corrected_condition:
        # Friendly handover banner if already in personal flow
        banner = ""
        if data.get("name") or data.get("age") or data.get("symptoms"):
            banner = "Got it 👍 Switching from your personal symptoms to general medical information.<br><br>"
        session.pop("medical_data", None)
        session.pop("last_medical_trigger", None)
        session.modified = True
        kb_answer = get_pdf_response(question, corrected_condition=corrected_condition)
        return banner + kb_answer

    # If name, age, and symptoms already collected
    if data["name"] and data["age"] and data["symptoms"]:
        symptoms_str = ", ".join(data["symptoms"])

        if lower_q in ["repeat", "show", "details"]:
            return (
                f"<b>📋 Your Details:</b><br>"
                f"• <b>Name:</b> {escape(data['name'])}<br>"
                f"• <b>Age:</b> {escape(str(data['age']))}<br>"
                f"• <b>Symptoms:</b> {escape(symptoms_str)}<br><br>"
                "Say <b>update</b> to change, or <b>no</b> if done."
            )

        if lower_q in ["update", "change"]:
            session["medical_data"] = {"name": None, "age": None, "symptoms": []}
            session.modified = True
            return "Let's start over. What's your name?"

        if lower_q == "no":
            session.pop("medical_data", None)
            return "Got it. Feel free to ask anything else!"

        if detect_medical_question(question):
            std_symptom = standardize_symptom(question)
            existing_std = [standardize_symptom(s) for s in data["symptoms"]]
            if std_symptom not in existing_std:
                data["symptoms"].append(question.strip())
                session["medical_data"] = data
                session.modified = True
            return generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))

        # Before giving up, one last attempt to see if it's a condition-like text
        corrected_condition2 = correct_condition_if_any(lower_q)
        if corrected_condition2:
            banner = "I think you’re asking about a medical condition. Switching to general information…<br><br>"
            session.pop("medical_data", None)
            session.modified = True
            return banner + get_pdf_response(question, corrected_condition=corrected_condition2)

        return ("Hmm 🤔 I’m not sure if this is a <b>symptom</b> or a <b>general medical question</b>.<br>"
                "If it’s about your health, tell me another symptom. "
                "If it’s about a condition or a medicine, try asking “what is …” or “treatment for …”.")

    # Step 1: Ask name
    if data["name"] is None:
        name = question.strip()
        if len(name) < 2 or not re.match(r"^[A-Za-z\s]+$", name):
            return "Please enter a valid name (letters only)."
        data["name"] = name
        session["medical_data"] = data
        session.modified = True
        return "Thank you! What's your age?"

    # Step 2: Ask age
    if data["age"] is None:
        if re.match(r"^\d+$", question.strip()):
            age_val = int(question.strip())
            if 1 <= age_val <= 120:
                data["age"] = age_val
                session["medical_data"] = data
                session.modified = True
                return "Got it. Please describe your symptoms clearly (e.g., fever, nausea)."
            else:
                return "Please enter a realistic age (1–120)."
        return "Enter a number."

    # Step 3: Collect symptoms
    std_symptom = standardize_symptom(question)
    existing_std = [standardize_symptom(s) for s in data["symptoms"]]
    if std_symptom not in existing_std:
        data["symptoms"].append(question.strip())
        session["medical_data"] = data
        session.modified = True

    return generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))


# =============== 7. ROUTES ===============
@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname] == pwd:
            session['user'] = uname
            session['chat_history'] = []
            flash("Logged in successfully!", "success")
            return redirect(url_for('chatbot'))
        else:
            error = "Invalid username or password"
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users:
            return "User already exists. <a href='/login'>Login</a>."
        users[uname] = pwd
        flash("Registration successful!", "info")
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            answer = "Please enter a message."
        else:
            lower_q = question.lower().strip()

            # Quick correction attempt at top-level (helps outside personal flow too)
            corrected_condition = correct_condition_if_any(lower_q)

            if lower_q in ["hi", "hello", "hey"]:
                answer = "Hi 👋 I'm your Medical Assistant. You can ask about symptoms or medical conditions."

            # Check general query or corrected condition BEFORE starting/continuing medical flow
            elif is_general_medical_query(lower_q) or corrected_condition:
                # Clear personal flow if present and show a handover banner
                banner = ""
                if "medical_data" in session and session["medical_data"]:
                    banner = "Got it 👍 Switching from your personal symptoms to general medical information.<br><br>"
                session.pop('medical_data', None)
                session.pop('last_medical_trigger', None)
                session.modified = True
                answer = banner + get_pdf_response(question, corrected_condition=corrected_condition)

            elif detect_medical_question(lower_q) or "medical_data" in session:
                session["last_medical_trigger"] = question.strip()
                answer = handle_medical_question(question)

            else:
                # Not obviously personal symptom or general phrase — try KB anyway
                answer = get_pdf_response(question)

        session['chat_history'].append({
            'question': escape(question),
            'answer': answer
        })
        session.modified = True

    return render_template('chatbot.html', chat_history=session['chat_history'])


@app.route('/clear_chat')
def clear_chat():
    session.pop('chat_history', None)
    session.pop('medical_data', None)
    session.pop('last_medical_trigger', None)
    flash("Chat cleared! 🧹", "info")
    return redirect(url_for('chatbot'))


@app.route('/logout')
def logout():
    username = session.get('user', 'User')
    session.clear()
    flash(f"See you later, {username}! 👋", "info")
    return redirect(url_for('login'))


if __name__ == '__main__':
    print("✅ Medical Chatbot running on http://127.0.0.1:5000")
    app.run(debug=True)
